from collections import OrderedDict
import numpy as np
import onnx
from polygraphy.backend.trt import CreateConfig, Profile
from polygraphy.backend.trt import engine_from_bytes, engine_from_network, network_from_onnx_path, save_engine
from polygraphy.backend.trt import util as trt_util
from polygraphy import cuda
from polygraphy.backend.common import bytes_from_path
import onnx_graphsurgeon as gs
import tensorrt as trt
import torch
import requests
from io import BytesIO
from cuda import cudart

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8      : torch.uint8,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float64,
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

# Map of torch dtype -> numpy dtype
torch_to_numpy_dtype_dict = {value : key for (key, value) in numpy_to_torch_dtype_dict.items()}

def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
         raise RuntimeError(f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t")
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None
class Engine():
    def __init__(
        self,
        engine_path,
    ):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.latent_h = 32
        self.latent_w = 48
        self.batch_size = 1
        self.cuda_graph_instance = None # cuda graph
    def check_dims(self, batch_size, image_height, image_width):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        assert image_height % 8 == 0 or image_width % 8 == 0
        latent_height = image_height // 8
        latent_width = image_width // 8
        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)
    def clip_model_shape_dict(self,batch_size, text_maxlen, embedding_dim):
        return {
            'input_ids': (batch_size, text_maxlen),
            'last_hidden_state': (batch_size, text_maxlen, embedding_dim)
        }
    def controlunet_model_shape_dict(self):
        return {
            'sample': (2*batch_size, self.unet_dim, latent_height, latent_width),
            'encoder_hidden_states': (2*batch_size, self.text_maxlen, self.embedding_dim),
            'latent': (2*batch_size, 4, latent_height, latent_width)
        }
    def control_model_shape_dict(self):
        return {
            "x_noisy": (self.batch_size, 4, self.latent_h, self.latent_w)
        }
    def decoder_model_shape_dict(self):
        return {
            'latent': (self.batch_size, 4, self.latent_h, self.latent_w),
            'images': (self.batch_size, 3, self.latent_h * 8, self.latent_w * 8)
        }
    def __del__(self):
        [buf.free() for buf in self.buffers.values() if isinstance(buf, cuda.DeviceArray) ]
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def load(self):
        print(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self, reuse_device_memory=None):
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
            self.context.device_memory = reuse_device_memory
        else:
            self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device='cuda'):
        for idx in range(trt_util.get_bindings_per_profile(self.engine)):
            binding = self.engine[idx]
            if shape_dict and binding in shape_dict:
                shape = shape_dict[binding]
            else:
                shape = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            if self.engine.binding_is_input(binding):
                self.context.set_binding_shape(idx, shape)
            tensor = torch.empty(tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]).to(device=device)
            self.tensors[binding] = tensor
    def get_engine_infor(self):
        nInput = np.sum([self.engine.binding_is_input(i) for i in range(self.engine.num_bindings)])
        nOutput = self.engine.num_bindings - nInput
        input_infor = dict((self.engine.get_tensor_name(i), self.context.get_binding_shape(i))  for i in range(nInput))
        ouput_infor = dict((self.engine.get_tensor_name(nInput + i), self.context.get_binding_shape(nInput + i))  for i in range(nOutput))
        print("TensorRT engine infors -----------------")
        print("engin nInput: ", nInput, ", Input shape: ", input_infor)
        print("engin nOutput: ", nOutput, ", Outpu shape: ", ouput_infor)

    def infer(self, feed_dict, stream=None, use_cuda_graph=False):
        # import pdb; pdb.set_trace()
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        if use_cuda_graph:
            if self.cuda_graph_instance is not None:
                CUASSERT(cudart.cudaGraphLaunch(self.cuda_graph_instance, stream.ptr))
                CUASSERT(cudart.cudaStreamSynchronize(stream.ptr))
            else:
                # do inference before CUDA graph capture
                noerror = self.context.execute_async_v3(stream.ptr)
                if not noerror:
                    raise ValueError(f"ERROR: inference failed.")
                # capture cuda graph
                CUASSERT(cudart.cudaStreamBeginCapture(stream.ptr, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal))
                self.context.execute_async_v3(stream.ptr)
                self.graph = CUASSERT(cudart.cudaStreamEndCapture(stream.ptr))
                self.cuda_graph_instance = CUASSERT(cudart.cudaGraphInstantiate(self.graph, 0))
        else:
            if stream:
                noerror = self.context.execute_async_v3(stream.ptr)
            else:
                noerror = self.context.execute_async_v3(0)
            if not noerror:
                raise ValueError(f"ERROR: inference failed.")

        return self.tensors


