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
def conv_3x3(in_c, out_c, s=1, d=1):
    return nn.Sequential(
        Conv(in_c, out_c, k=3, s=s, d=d),
    )

class ResBlock(nn.Module):
    def __init__(self, c0, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            conv_3x3(c0, c0, d=dilation),
            conv_3x3(c0, c0, d=dilation),
        )

    def forward(self, input):
        x = self.conv(input)
        return x + input
import timm


class StereoNet(nn.Module):
    def __init__(self, backbone='mobilenetv3_small_100', pretrained=True, feature_index=2):
        super().__init__()

        # 使用 timm 提供的 features_only 模式，提取中间层特征
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True
        )

        # 指定使用哪一层特征（例如 -1 表示最后一层，-2 表示倒数第二层）
        self.feature_index = feature_index

        # 对齐要求（以 2^k 为单位）
        self.k = 3
        self.align = 2 ** self.k
        self.max_disp = (192 + 1) // self.align // 4

    def feature_extractor_fn(self, x):
        # 获取中间特征层列表
        feats = self.backbone(x)  # 返回的是 List[Tensor]，例如 5 层下采样输出
        return feats[self.feature_index]  # 选择需要的某一层输出

    def forward(self, left_img, right_img):
        # 提取特征
        lf_feat = self.feature_extractor_fn(left_img)
        rf_feat = self.feature_extractor_fn(right_img)

        # 返回特征层的加和或后续模块处理（这里只是简单加法演示）
        return lf_feat + rf_feat


if __name__ == "__main__":
    from thop import profile

    left = torch.rand(1, 3, 480, 640)
    right = torch.rand(1, 3, 480, 640)
    model = StereoNet()
    model.eval()
    print(model(left, right).size())

    torch.onnx.export(
        model,
        (left, right),
        "stereoplus_aanet.onnx",
        do_constant_folding=True,  # 是否进行常量折叠优化
        opset_version=16,  # ONNX opset 版本
        input_names=["left", "right"],  # 输入名称
        output_names=["output"],  # 输出名称
        dynamic_axes=None  # 动态维度（可选）
    )