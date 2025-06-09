# -*- coding: utf-8 -*-
# @Time    : 2025/5/18 21:45
# @Author  : sjh
# @Site    : 
# @File    : TinyHITNet_stereonet.py
# @Comment :
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.nn.modules.conv import Conv
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积：减少计算量3-5倍"""

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False)
        self.pointwise = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class LightResBlock(nn.Module):
    """轻量残差块：用深度可分离卷积替代标准卷积"""

    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(c, c),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            DepthwiseSeparableConv(c, c),
            nn.BatchNorm2d(c)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.conv(x))


class TinyStereoNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 特征提取（总下采样8倍）
        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv(3, 16, 2),  # 1/2
            LightResBlock(16)
        )
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(16, 32, 2),  # 1/4
            LightResBlock(32)
        )
        self.conv3 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, 2),  # 1/8
            LightResBlock(64)
        )

        # 快速代价聚合
        self.cost_agg = nn.Sequential(
            nn.Conv3d(1, 8, 3, 1, 1),  # 3D卷积参数量优化
            nn.ReLU(),
            nn.Conv3d(8, 1, 3, 1, 1)
        )

    def forward(self, left, right):
        # 特征提取
        l1 = self.conv1(left)  # [N,16,H/2,W/2]
        l2 = self.conv2(l1)  # [N,32,H/4,W/4]
        l3 = self.conv3(l2)  # [N,64,H/8,W/8]

        r3 = self.conv3(self.conv2(self.conv1(right)))


        # 快速代价聚合

        # 上采样恢复分辨率
        return r3



if __name__ == "__main__":
    from thop import profile

    left = torch.rand(1, 3, 480, 640)
    right = torch.rand(1, 3, 480, 640)
    model = TinyStereoNet()
    model.eval()
    print(model(left, right).size())

    total_ops, total_params = profile(
        model,
        (
            left,
            right,
        ),
    )
    print(
        "{:.4f} MACs(G)\t{:.4f} Params(M)".format(
            total_ops / (1000 ** 3), total_params / (1000 ** 2)
        )
    )
    torch.onnx.export(
        model,
        (left, right),
        "stereoplus_aanet.onnx",
        opset_version=16,  # ONNX opset 版本
        do_constant_folding=True,  # 是否进行常量折叠优化
        input_names=["left", "right"],  # 输入名称
        output_names=["output"],  # 输出名称
        dynamic_axes=None  # 动态维度（可选）
    )