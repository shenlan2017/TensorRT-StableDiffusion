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
import torch.nn as nn
import random
import os
import functools
from functools import partial

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from onnx import shape_inference
import onnx_graphsurgeon as gs
import onnx
import onnxruntime as rt
import matplotlib.pyplot as plt


class hackathon():
    def initialize(self):
        self.apply_canny = CannyDetector()
        self.model = create_model('./models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cpu'))
        # self.model.load_state_dict(load_state_dict('/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda'))
        self.model = self.model.cpu()
        self.model.eval()
        self.ddim_sampler = DDIMSampler(self.model)

def control_net_weighs2histogram(control_net):

    save_folder = 'histogram/weights'
    os.makedirs(save_folder, exist_ok=True)

    # 遍历state_dict并绘制权重的直方图并保存
    for name, m in control_net.named_modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            # import pdb; pdb.set_trace()
            weights = m.weight.flatten().detach().cpu().numpy()

            # 绘制权重的直方图
            plt.hist(weights, bins=50)
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.title(f'Histogram of {name} Weights')

            # 保存直方图为图像文件
            save_path = os.path.join(save_folder, f'{name}.png')
            plt.savefig(save_path)
            plt.close()  # 关闭当前图形，以便绘制下一个直方图

def control_net_acts2histogram(control_net, input_path):

    save_folder = 'histogram/acts'
    os.makedirs(save_folder, exist_ok=True)

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        # stat_tensor(name, x)

        x = x.flatten().detach().numpy()
        # 绘制权重的直方图
        plt.hist(x, bins=50)
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {name} Acts')

        # 保存直方图为图像文件
        save_path = os.path.join(save_folder, f'{name}.png')
        plt.savefig(save_path)
        plt.close()  # 关闭当前图形，以便绘制下一个直方图

    hooks = []
    for name, m in control_net.named_modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )

    data = np.load(input_path)

    control_net(torch.from_numpy(data["x_noisy"]), torch.from_numpy(data["hint"]), torch.from_numpy(data["timestep"]), torch.from_numpy(data["context"]))


    for h in hooks:
        h.remove()

def main():
    hk = hackathon()
    hk.initialize()

    control_net = hk.model.control_model.cpu()
    control_net.eval()

    control_net_weighs2histogram(control_net)
    control_net_acts2histogram(control_net, "./calib_data/ControlNet/0.npz")

if __name__ == '__main__':
    main()
