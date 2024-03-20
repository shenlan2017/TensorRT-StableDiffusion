import os

import torch
import torch.nn as nn

import functools

from functools import partial
import numpy as np

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from onnx import shape_inference

from ldm.modules.attention import *


class hackathon():
    def initialize(self):
        self.apply_canny = CannyDetector()
        self.model = create_model('./models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cpu'))
        self.model = self.model.cpu()
        self.model.eval()
        self.ddim_sampler = DDIMSampler(self.model)


def get_act_scales(model, dataset_path):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )

    file_list = os.listdir(dataset_path)

    for file_name in file_list:
        file_path = os.path.join(dataset_path, file_name)
        data = np.load(file_path)

        model(torch.from_numpy(data["x_noisy"]), torch.from_numpy(data["hint"]),
              torch.from_numpy(data["timestep"]), torch.from_numpy(data["context"]))

    for h in hooks:
        h.remove()

    return act_scales

@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    ln.weight.div_(scales)
    ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

@torch.no_grad()
def smooth_controlnet(model, scales, alpha=0.5):
    for name, module in model.named_modules():

        if isinstance(module, BasicTransformerBlock):
            # import pdb; pdb.set_trace()
            # print(name)
            # attn1
            attn_ln = module.norm1
            qkv = [
                module.attn1.to_q,
                module.attn1.to_k,
                module.attn1.to_v,
            ]
            qkv_input_scales = scales[name + ".attn1.to_q"]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            # attn2
            attn_ln = module.norm2
            qkv = [
                module.attn2.to_q,
            ]
            qkv_input_scales = scales[name + ".attn2.to_q"]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            # ff
            attn_ln = module.norm3
            fc1 = module.ff.net[0].proj
            qkv_input_scales = scales[name + ".ff.net.0.proj"]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

def export_control_net_model(control_net, onnx_path):

    x_noisy = torch.randn(1, 4, 32, 48, dtype=torch.float32)
    hint = torch.randn(1, 3, 256, 384, dtype=torch.float32)
    timestep = torch.tensor([1], dtype=torch.int32)
    context = torch.randn(1, 77, 768, dtype=torch.float32)

    input_names = ["x_noisy", "hint", "timestep", "context"]
    output_names = ["latent"]

    # onnx_path = "./onnx/ControlNet.onnx"

    torch.onnx.export(
        control_net,
        (x_noisy, hint, timestep, context),
        onnx_path,
        verbose=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=input_names,
        keep_initializers_as_inputs=True
    )

def main():
    hk = hackathon()
    hk.initialize()
    control_net = hk.model.control_model.cpu()

    act_scales_path = "./smoothquant/control_net_act_scales"
    if os.path.exists(act_scales_path):
        act_scales = torch.load(act_scales_path)
    else:
        act_scales = get_act_scales(control_net, "./calib_data/ControlNet/")
        os.makedirs(os.path.dirname(act_scales_path), exist_ok=True)
        torch.save(act_scales, act_scales_path)

    smooth_controlnet(control_net, act_scales, alpha=0.5)

    export_control_net_model(control_net, "./onnx/ControlNet_smooth.onnx")

if __name__ == '__main__':
    main()
