import numpy as np
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
import cv2
import datetime
# from share import *
# import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import os

from pytorch_lightning import seed_everything
# from annotator.util import resize_image, HWC3
# from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from onnx import shape_inference
import onnx_graphsurgeon as gs
import onnx
import onnxruntime as rt


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

def control_net_weighs2histogram():
    control_net = hk.model.control_model.cpu()

    save_folder = 'histogram'
    os.makedirs(save_folder, exist_ok=True)

    # 获取模型的state_dict
    state_dict = control_net.state_dict()

    # 遍历state_dict并绘制权重的直方图并保存
    for key, value in state_dict.items():
        if 'weight' in key:  # 只选择包含'weight'的键
            weights = value.flatten().cpu().numpy()

            # 绘制权重的直方图
            plt.hist(weights, bins=50)
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.title(f'Histogram of {key} Weights')

            # 保存直方图为图像文件
            save_path = os.path.join(save_folder, f'{key}_histogram.png')
            plt.savefig(save_path)
            plt.close()  # 关闭当前图形，以便绘制下一个直方图


def main():
    control_net_weighs2histogram()

if __name__ == '__main__':
    main()
