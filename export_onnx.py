import numpy as np
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
import cv2
import datetime
from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import os

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from onnx import shape_inference
import onnx_graphsurgeon as gs
import onnx
import onnxruntime as rt

def optimize(onnx_path, opt_onnx_path):
    from onnxsim import simplify
    model = onnx.load(onnx_path)
    graph = gs.import_onnx(model)
    print(f"{onnx_path} simplify start !")
    # self.info("init", graph)
    model_simp, check = simplify(model)
    # self.info("opt", gs.import_onnx(model_simp))
    onnx.save(model_simp, opt_onnx_path, save_as_external_data=True)
    assert check, "Simplified ONNX model could not be validated"
    print(f"{onnx_path} simplify done !")

def onnxruntime_check(onnx_path, input_dicts, torch_outputs):
    onnx_model = onnx.load(onnx_path)
    # onnx.checker.check_model(onnx_model)
    sess = rt.InferenceSession(onnx_path)
    # outputs = self.get_output_names()
    # latent input
    # data = np.zeros((4, 77), dtype=np.int32)
    result = sess.run(None, input_dicts)

    for i in range(0, len(torch_outputs)):
        ret = np.allclose(result[i], torch_outputs[i].detach().numpy(), rtol=1e-03, atol=1e-05, equal_nan=False)
        if ret is False:
            print("Error onnxruntime_check")
            # import pdb; pdb.set_trace()


class hackathon():
    def initialize(self):
        self.apply_canny = CannyDetector()
        self.model = create_model('./models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cpu'))
        # self.model.load_state_dict(load_state_dict('/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda'))
        self.model = self.model.cpu()
        self.model.eval()
        self.ddim_sampler = DDIMSampler(self.model)

hk = hackathon()
hk.initialize()

def export_clip_model():
    clip_model = hk.model.cond_stage_model

    import types

    def forward(self, tokens):
        outputs = self.transformer(
            input_ids=tokens, output_hidden_states=self.layer == "hidden"
        )
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    clip_model.forward = types.MethodType(forward, clip_model)

    onnx_path = "./onnx/CLIP.onnx"

    tokens = torch.zeros(1, 77, dtype=torch.int32)
    input_names = ["input_ids"]
    output_names = ["last_hidden_state"]
    dynamic_axes = {"input_ids": {1: "S"}, "last_hidden_state": {1: "S"}}

    torch.onnx.export(
        clip_model,
        (tokens),
        onnx_path,
        verbose=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print("======================= CLIP model export onnx done!")

    # verify onnx model
    output = clip_model(tokens)
    input_dicts = {"input_ids": tokens.numpy()}
    onnxruntime_check(onnx_path, input_dicts, [output])
    print("======================= CLIP onnx model verify done!")

    # opt_onnx_path = "./onnx/CLIP.opt.onnx"
    # optimize(onnx_path, opt_onnx_path)


def export_control_net_model():
    control_net = hk.model.control_model


def export_controlled_unet_model():
    controlled_unet_mdoel = hk.model.model.diffusion_model

def export_decoder_model():
    # control_net = hk.model.control_model

    decode_model = hk.model.first_stage_model
    decode_model.forward = decode_model.decode

def main():
    export_clip_model()
    export_control_net_model()
    export_controlled_unet_model()
    export_decoder_model()

if __name__ == '__main__':
    main()
