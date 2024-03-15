import numpy as np
import os
import tensorrt as trt

def onnx2trt(onnxFile, plan_name, min_shapes, opt_shapes, max_shapes, max_workspace_size = None, use_fp16=False, builder_opt_evel=None):
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

def export_clip_model():
    onnx_path = "./onnx/CLIP.onnx"
    plan_path = "./engine/CLIP.plan"

    # onnx2trt(onnx_path, plan_path, [(1, 77)], [(1, 77)], [(1, 77)])
    onnx2trt(onnx_path, plan_path, [(1, 77)], [(1, 77)], [(1, 77)], use_fp16=True)
    print("======================= CLIP onnx2trt done!")

def export_control_net_model():
    def get_shapes(B, S):
        return [(B, 4, 32, 48), (B, 3, 256, 384), tuple([B]), (B, S, 768)]

    onnx_path = "./onnx/ControlNet.onnx"
    plan_path = "./engine/ControlNet.plan"

    # onnx2trt(onnx_path, plan_path,
             # get_shapes(1, 77),
             # get_shapes(1, 77),
             # get_shapes(1, 77))

    onnx2trt(onnx_path, plan_path,
             get_shapes(1, 77),
             get_shapes(1, 77),
             get_shapes(1, 77),
             use_fp16=True)

    print("======================= ControlNet onnx2trt done!")

def export_controlled_unet_model():
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

    onnx_path = "./onnx/ControlledUnet"
    onnx_path = onnx_path + "/ControlledUnet.onnx"

    plan_path = "./engine/ControlledUnet.plan"

    # onnx2trt(onnx_path, plan_path,
             # get_shapes(1, 77),
             # get_shapes(1, 77),
             # get_shapes(1, 77))

    onnx2trt(onnx_path, plan_path,
             get_shapes(1, 77),
             get_shapes(1, 77),
             get_shapes(1, 77),
             use_fp16=True)

    print("======================= ControlNet onnx2trt done!")

def export_decoder_model():
    onnx_path = "./onnx/Decoder.onnx"
    plan_path = "./engine/Decoder.plan"

    # onnx2trt(onnx_path, plan_path,
            # [(1, 4, 32, 48)], [(1, 4, 32, 48)], [(1, 4, 32, 48)])

    onnx2trt(onnx_path, plan_path,
            [(1, 4, 32, 48)], [(1, 4, 32, 48)], [(1, 4, 32, 48)],
             use_fp16=True)

    print("======================= Decoder  onnx2trt done!")

def main():
    export_clip_model()
    export_control_net_model()
    export_controlled_unet_model()
    export_decoder_model()

if __name__ == '__main__':
    main()
