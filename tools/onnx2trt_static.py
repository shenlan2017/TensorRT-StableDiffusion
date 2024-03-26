import numpy as np
import os
import ctypes

import tensorrt as trt

trt_plugin = "plugin/build/libplugin.so"
handle = ctypes.CDLL(trt_plugin, mode=ctypes.RTLD_GLOBAL)
if not handle:
    raise RuntimeError("Could not load plugin library. Is " + trt_plugin + " on your LD_LIBRARY_PATH?")


def onnx2trt(onnxFile, plan_name, min_shapes, opt_shapes, max_shapes,
             max_workspace_size = None, use_fp16=False, builder_opt_evel=None):
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

    if builder_opt_evel:
        config.builder_optimization_level = builder_opt_evel

    # set profile
    assert network.num_inputs == len(min_shapes)
    assert network.num_inputs == len(opt_shapes)
    assert network.num_inputs == len(max_shapes)

    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        profile.set_shape(input_tensor.name, tuple(min_shapes[i]), tuple(opt_shapes[i]), tuple(max_shapes[i]))

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

def export_clip_model(onnx_path, plan_path, batch=1, use_fp16=False):

    onnx2trt(onnx_path, plan_path, [(batch, 77)], [(batch, 77)], [(batch, 77)], use_fp16=use_fp16)
    print("======================= CLIP onnx2trt done!")

def export_control_net_model(onnx_path, plan_path, batch=1, use_fp16=False):
    def get_shapes(B, S):
        return [(B, 4, 32, 48), (B, 3, 256, 384), tuple([B]), (B, S, 768)]

    # import pdb; pdb.set_trace()
    onnx2trt(onnx_path, plan_path,
             get_shapes(batch, 77),
             get_shapes(batch, 77),
             get_shapes(batch, 77),
             use_fp16=use_fp16)

    print("======================= ControlNet onnx2trt done!")

def export_controlled_unet_model(onnx_path, plan_path, batch=1, use_fp16=False):
    def get_shapes(B, S):
        return [(B, 4, 32, 48), tuple([B]), (B, S, 768),
                (B, 320, 32, 48),
                (B, 320, 32, 48),
                (B, 320, 32, 48),
                (B, 320, 16, 24),
                (B, 640, 16, 24),
                (B, 640, 16, 24),
                (B, 640, 8, 12),
                (B, 1280, 8, 12),
                (B, 1280, 8, 12),
                (B, 1280, 4, 6),
                (B, 1280, 4, 6),
                (B, 1280, 4, 6),
                (B, 1280, 4, 6)]

    onnx2trt(onnx_path, plan_path,
             get_shapes(batch, 77),
             get_shapes(batch, 77),
             get_shapes(batch, 77),
             use_fp16=use_fp16)

    print("======================= ControlNet onnx2trt done!")

def export_decoder_model(onnx_path, plan_path, batch=1, use_fp16=False):

    onnx2trt(onnx_path, plan_path,
            [(batch, 4, 32, 48)], [(batch, 4, 32, 48)], [(batch, 4, 32, 48)],
             use_fp16=use_fp16)

    print("======================= Decoder  onnx2trt done!")

def onnxs2trts(use_fp16):
    onnx_path = "./onnx/CLIP.onnx"
    plan_path = "./engine/CLIP.plan"
    export_clip_model(onnx_path, plan_path, use_fp16=use_fp16)

    onnx_path = "./onnx/ControlNet.onnx"
    plan_path = "./engine/ControlNet.plan"
    export_control_net_model(onnx_path, plan_path, use_fp16=use_fp16)

    onnx_path = "./onnx/ControlledUnet/ControlledUnet.onnx"
    plan_path = "./engine/ControlledUnet.plan"
    export_controlled_unet_model(onnx_path, plan_path, use_fp16=use_fp16)

    onnx_path = "./onnx/Decoder.onnx"
    plan_path = "./engine/Decoder.plan"
    export_decoder_model(onnx_path, plan_path, use_fp16=use_fp16)

def onnxs2trts_opt(use_fp16):

    onnx_path = "./onnx/CLIP_opt.onnx"
    plan_path = "./engine/CLIP_opt.plan"
    export_clip_model(onnx_path, plan_path, use_fp16=use_fp16)

    onnx_path = "./onnx/ControlNet_opt.onnx"
    plan_path = "./engine/ControlNet_opt.plan"
    export_control_net_model(onnx_path, plan_path, use_fp16=use_fp16)

    onnx_path = "./onnx/ControlledUnet_opt/ControlledUnet_opt.onnx"
    plan_path = "./engine/ControlledUnet_opt.plan"
    export_controlled_unet_model(onnx_path, plan_path, use_fp16=use_fp16)

    onnx_path = "./onnx/Decoder_opt.onnx"
    plan_path = "./engine/Decoder_opt.plan"
    export_decoder_model(onnx_path, plan_path, use_fp16=use_fp16)

def onnxs2trts_opt_batch(use_fp16):
    onnx_path = "./onnx/CLIP_opt_batch.onnx"
    plan_path = "./engine/CLIP_opt_batch.plan"
    export_clip_model(onnx_path, plan_path, batch=2, use_fp16=use_fp16)

    onnx_path = "./onnx/ControlNet_opt_batch.onnx"
    plan_path = "./engine/ControlNet_opt_batch.plan"
    export_control_net_model(onnx_path, plan_path, batch=2, use_fp16=use_fp16)

    onnx_path = "./onnx/ControlledUnet_opt_batch/ControlledUnet_opt_batch.onnx"
    plan_path = "./engine/ControlledUnet_opt_batch.plan"
    export_controlled_unet_model(onnx_path, plan_path, batch=2, use_fp16=use_fp16)

if __name__ == '__main__':
    # onnxs2trts(use_fp16=False)
    onnxs2trts(use_fp16=True)
    onnxs2trts_opt(use_fp16=True)
    onnxs2trts_opt_batch(use_fp16=True)
