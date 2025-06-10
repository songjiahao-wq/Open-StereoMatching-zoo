# -*- coding: utf-8 -*-
# @Time    : 2025/6/10 下午4:07
# @Author  : sjh
# @Site    : 
# @File    : neck.py
# @Comment :
from models.models.stereoplus_dpx.headplus import StereoNetHeadPlus
from models.models.stereoplus_dpx.FPN import FPN
from models.models.stereoplus_dpx.post_process import StereoNetPostProcessPlus
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union


class StereoPlusPipeline(nn.Module):
    def __init__(self, B=8, height=544, width=960):
        super(StereoPlusPipeline, self).__init__()
        print("This is the neck module for StereoPlus DPX model.")
        self.B = B
        self.height = height
        self.width = width

        # 配置参数
        self.in_strides = [8, 16, 32]
        self.in_channels = [64, 96, 16]
        self.out_strides = [8, 16, 32]
        self.out_channels = [16, 16, 16]
        self.fix_out_channel = None
        self.maxdisp = 192
        self.low_max_stride = 8

        # 初始化模块
        self.fpn = FPN(
            in_strides=self.in_strides,
            in_channels=self.in_channels,
            out_strides=self.out_strides,
            out_channels=self.out_channels,
            fix_out_channel=self.fix_out_channel,
        )
        self.fpn._init_weights()

        self.stereo_head = StereoNetHeadPlus(
            maxdisp=self.maxdisp,
            refine_levels=4,
            bn_kwargs={'momentum': 0.1, 'affine': True},
            max_stride=32,
            num_costvolume=3,
            num_fusion=6,
            hidden_dim=16,
            in_channels=[32, 32, 16, 16, 16]
        )

        self.postprocess = StereoNetPostProcessPlus(
            maxdisp=self.maxdisp,
            low_max_stride=self.low_max_stride
        )

    def forward(self, inputs: List[torch.Tensor], gt_disp: torch.Tensor = None) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        前向推理流程，供外部网络调用。

        Args:
            inputs: 包含高分辨率特征 feat1, feat2 和 FPN 输入特征（按 in_channels 配置）
            gt_disp: 训练时的 ground truth disparity，可选

        Returns:
            推理结果（训练模式返回列表，评估模式返回单个 Tensor）
        """
        feat1, feat2 = inputs[0], inputs[1]
        fpn_inputs = inputs[2:]
        outputs = self.fpn(fpn_inputs)
        inputs_for_head = [feat1, feat2] + outputs

        pred0, pred0_unfold, spx_pred = self.stereo_head(inputs_for_head)
        modelouts = [pred0, pred0_unfold, spx_pred]

        if self.training:
            return self.postprocess(modelouts, gt_disp)
        else:
            return self.postprocess(modelouts)

    def run(self):
        # 创建输入特征图
        fpn_inputs = [
            torch.randn(self.B, c, self.height // s, self.width // s)
            for c, s in zip(self.in_channels, self.in_strides)
        ]

        # 添加高分辨率特征
        feat1 = torch.randn(self.B, 32, self.height // 2, self.width // 2)
        feat2 = torch.randn(self.B, 32, self.height // 4, self.width // 4)
        inputs = [feat1, feat2] + fpn_inputs

        for i, out in enumerate(inputs):
            print(f"Input {i + 1} shape: {out.shape}")

        # 模拟训练模式
        self.train()
        gt_disp = torch.rand(self.B, self.height, self.width)
        outputs_train = self.forward(inputs, gt_disp)
        print("Train mode outputs:")
        for out in outputs_train:
            print(out.shape)

        # 模拟评估模式
        self.eval()
        with torch.no_grad():
            output_eval = self.forward(inputs)
        print("Eval mode output:", output_eval.shape)


if __name__ == "__main__":
    pipeline = StereoPlusPipeline()
    pipeline.run()



#
# if __name__ == "__main__":
#     print("This is the neck module for StereoPlus DPX model.")
#     # 输入参数
#     B = 8
#     in_strides = [8, 16, 32]  # 输入特征图的 strides
#     in_channels = [64, 96, 16]  # 输入特征图的通道数
#     out_strides = [8, 16, 32]  # 输出特征图的 strides
#     out_channels = [16, 16, 16]  # 输出特征图的通道数
#     fix_out_channel = None  # 固定输出通道数，如果设置为 None，则不进行转换
#
#     # 创建 FPN 模型
#     model = FPN(
#         in_strides=in_strides,
#         in_channels=in_channels,
#         out_strides=out_strides,
#         out_channels=out_channels,
#         fix_out_channel=fix_out_channel,
#     )
#
#     # 初始化权重
#     model._init_weights()
#
#     # 创建模拟输入数据（每个输入特征图的尺寸为 [batch_size, channels, height, width]）
#     batch_size = B
#     height = 544
#     width = 960
#     # 根据 in_strides 创建不同分辨率的输入数据
#     inputs = [
#         torch.randn(batch_size, in_channels[i], height // (stride), width // (stride)) for i, stride in enumerate(in_strides)
#     ]
#     # 将输入数据传入模型进行前向传播
#     outputs = model(inputs)
#     # 打印输出的形状以检查
#     for i, out in enumerate(outputs):
#         print(f"Output {i + 1} shape: {out.shape}")
#     '******************************************************************************************************************'
#
#     feat1 = torch.randn(B, 32, 272, 480)  # 最大分辨率特征
#     feat2 = torch.randn(B, 32, 136, 240)  # 中等分辨率特征
#
#     # 将其插入到输出列表开头
#     inputs = [feat1, feat2] + outputs
#     print(len(inputs))
#     for i, out in enumerate(inputs):
#         print(f"Output {i + 1} shape: {out.shape}")
#     model = StereoNetHeadPlus(
#         maxdisp=192,
#         refine_levels=4,
#         bn_kwargs={'momentum': 0.1, 'affine': True},
#         max_stride=32,
#         num_costvolume=3,
#         num_fusion=6,
#         hidden_dim=16,
#         in_channels=[32, 32, 16, 16, 16]  # 这里根据你的特征通道数设置，最后两个16是示例，可根据代码修改
#     )
#     model.eval()  # 设为eval模式
#     # 运行前向
#     with torch.no_grad():
#         pred0, pred0_unfold, spx_pred = model(inputs)
#
#     # 输出形状
#     print("pred0 shape:", pred0.shape)  # 预测低分辨率视差图
#     print("pred0_unfold shape:", pred0_unfold.shape)
#     print("spx_pred shape:", spx_pred.shape)  # 权重预测图
#     '******************************************************************************************************************'
#     maxdisp = 192
#     low_max_stride = 8
#     model = StereoNetPostProcessPlus(maxdisp=maxdisp, low_max_stride=low_max_stride)
#     modelouts = [pred0, pred0_unfold, spx_pred]
#     # 训练模式
#     model.train()
#     gt_disp = torch.rand(B, height, width)
#     outputs = model(modelouts, gt_disp)
#     print("Train mode outputs:")
#     for out in outputs:
#         print(out.shape)  # [B, H, W]
#
#     # 评估模式
#     model.eval()
#     with torch.no_grad():
#         output_eval = model(modelouts)
#         print("Eval mode output:", output_eval.shape)  # [B, H, W]