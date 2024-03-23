#!/usr/bin/env python3

import tensorrt as trt
import os

from cuda import cudart
import numpy as np

import pdb

class EntropyCalibrator2(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_path, cache_file = "int8.cache"):
        super().__init__()

        self.current_index = 0

        print("start read " + data_path)
        self.data_list = []

        file_list = os.listdir(data_path)

        # 遍历文件列表并处理每个文件
        for file_name in file_list:
            file_path = os.path.join(data_path, file_name)
            data = np.load(file_path)
            self.data_list.append(data)

        self.batch_size = 1
        self.cache_file = cache_file

        self.num_inputs = len(self.data_list)
        print("read " + data_path + " done, len = " + str(self.num_inputs))

        _, self.x_noisy = cudart.cudaMalloc(self.data_list[0]["x_noisy"].nbytes)
        _, self.hint = cudart.cudaMalloc(self.data_list[0]["hint"].nbytes)
        _, self.timestep = cudart.cudaMalloc(self.data_list[0]["timestep"].nbytes)
        _, self.context = cudart.cudaMalloc(self.data_list[0]["context"].nbytes)

    def free(self):
        pass

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):

        if self.current_index + self.batch_size > self.num_inputs:
            print("Calibrating index {:} batch size {:} exceed max input limit {:} sentences".format(self.current_index, self.batch_size, self.num_inputs))
            return None

        # import pdb; pdb.set_trace()
        data  = self.data_list[self.current_index]
        cudart.cudaMemcpy(self.x_noisy, np.ascontiguousarray(data["x_noisy"]).data,
                          data["x_noisy"].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        cudart.cudaMemcpy(self.hint, np.ascontiguousarray(data["hint"]).data,
                          data["hint"].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        cudart.cudaMemcpy(self.timestep, np.ascontiguousarray(data["timestep"]).data,
                          data["timestep"].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        cudart.cudaMemcpy(self.context, np.ascontiguousarray(data["context"]).data,
                          data["context"].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        cuda_arrs = []
        cuda_arrs.append(self.x_noisy)
        cuda_arrs.append(self.hint)
        cuda_arrs.append(self.timestep)
        cuda_arrs.append(self.context)

        self.current_index += self.batch_size
        return cuda_arrs

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            f.flush()
            os.fsync(f)

    def get_quantile(self):
        return 0.9999

    def get_regression_cutoff(self):
        return 1.0

    def read_histogram_cache(self, length):
        return None

    def write_histogram_cache(self, ptr, length):
        return None


def onnx2trt(onnxFile, plan_name, min_shapes, opt_shapes, max_shapes,
             max_workspace_size = None, use_fp16=False, use_int8=False, int8_calib_data_path=None,
             builder_opt_evel=None):
    logger = trt.Logger(trt.Logger.VERBOSE)

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    if max_workspace_size:
        config.max_workspace_size = max_workspace_size
    else:
        config.max_workspace_size = 10<<30

    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnxFile):
        print("Failed finding onnx file!")
        exit()
    print("Succeeded finding onnx file!")
    with open(onnxFile, 'rb') as model:
        # import pdb; pdb.set_trace()
        (onnx_path, _) = os.path.split(onnxFile)
        if not parser.parse(model.read(), path=onnxFile):
            print("Failed parsing ONNX file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
    print("Succeeded parsing ONNX file!")

    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        plan_name = plan_name.replace(".plan", "_fp16.plan")

    if use_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        plan_name = plan_name.replace(".plan", "_int8.plan")
        config.int8_calibrator = EntropyCalibrator2(int8_calib_data_path)

    builder_opt_evel = 0
    if builder_opt_evel:
        config.builder_optimization_level = builder_opt_evel

    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    # set profile
    assert network.num_inputs == len(min_shapes)
    assert network.num_inputs == len(opt_shapes)
    assert network.num_inputs == len(max_shapes)

    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        profile.set_shape(input_tensor.name, tuple(min_shapes[i]), tuple(opt_shapes[i]), tuple(max_shapes[i]))

    for i in range(network.num_layers):
        layer = network.get_layer(i)
        layer_name = layer.name
        layer_type = layer.type
        # print(f"Layer Name: {layer_name}, Layer Type: {layer_type}")
        if layer_type == trt.LayerType.MATRIX_MULTIPLY or layer_type == trt.LayerType.CONVOLUTION:
            if "emb_layers" in layer_name or \
               "attn2/to_k" in layer_name or \
               "attn2/to_v" in layer_name or \
               "attn2/to_out" in layer_name or \
               "time_embed" in layer_name or \
               "in_layers" in layer_name or \
               "out_layers" in layer_name or \
               "ff/net/net.2" in layer_name:
                layer.set_output_type(0, trt.float16)
                print(f"Layer Name: {layer_name}, Layer Type: {layer_type}")

    # import pdb; pdb.set_trace()

    config.add_optimization_profile(profile)

    engine = builder.build_engine(network, config)
    if not engine:
        raise RuntimeError("build_engine failed")
    print("Succeeded building engine!")

    print("Serializing Engine...")
    serialized_engine = engine.serialize()
    if serialized_engine is None:
        raise RuntimeError("serialize failed")

    (plan_path, _) = os.path.split(plan_name)
    os.makedirs(plan_path, exist_ok=True)
    with open(plan_name, "wb") as fout:
        fout.write(serialized_engine)

def export_control_net_model():
    def get_shapes(B, S):
        return [(B, 4, 32, 48), (B, 3, 256, 384), tuple([B]), (B, S, 768)]

    # onnx_path = "./onnx/ControlNet.onnx"
    # plan_path = "./engine/ControlNet.plan"

    onnx_path = "./onnx/ControlNet_smooth.onnx"
    plan_path = "./engine/ControlNet_smooth.plan"

    onnx2trt(onnx_path, plan_path,
             get_shapes(1, 77),
             get_shapes(1, 77),
             get_shapes(1, 77),
             use_fp16=True, use_int8=True, int8_calib_data_path="./calib_data/ControlNet/")

    print("======================= ControlNet onnx2trt done!")

if __name__ == '__main__':
    export_control_net_model()
