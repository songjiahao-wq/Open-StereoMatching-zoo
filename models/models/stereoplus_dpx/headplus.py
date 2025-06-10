# -*- coding: utf-8 -*-
# @Time    : 2025/6/6 上午11:29
# @Author  : sjh
# @Site    : 
# @File    : headplus.py
# @Comment :
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
from torch import Tensor
from typing import Optional, Tuple, Union, Dict, List
from models import Conv ,ConvTranspose
class DeConvResModule(nn.Module):
    """
    A basic module for deconv shortcut.

    Args:
        in_channels: The channels of inputs.
        out_channels:  The channels of outputs.
        bn_kwargs: Dict for BN layer.
        kernel: The kernel_size of deconv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_kwargs: Dict = None,
        kernel: int = 4,
    ):
        super(DeConvResModule, self).__init__()
        # 反卷积层，stride=2，padding=1，上采样x2
        self.conv1 = nn.Sequential(
            ConvTranspose(
                in_channels,
                out_channels,
                k=kernel,
                s=2,
                act=nn.ReLU(inplace=True),

            ),
        )

        # 普通卷积层，3x3卷积
        self.conv2 = nn.Sequential(
            Conv(
                out_channels,
                out_channels,
                k=3,
                s=1,
                act=nn.ReLU(inplace=True),
            ),
        )

    def forward(self, x: torch.Tensor, rem: torch.Tensor) -> torch.Tensor:
        """Forward with shortcut add."""

        x = self.conv1(x)
        x = x + rem  # 普通加法替代量化的FloatFunctional.add
        x = self.conv2(x)
        return x
class BasicResBlock(nn.Module):
    """
    A basic residual block using Conv-BN-ReLU modules.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        bn_kwargs (dict): Ignored (kept for compatibility).
        stride (int): Stride of the first conv layer.
        bias (bool): Ignored, bias=False in Conv by default.
        expansion (int): Ignored here (kept for compatibility).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,      # unused in Basic block
    ):
        super(BasicResBlock, self).__init__()

        self.conv = nn.Sequential(
            Conv(in_channels, out_channels, k=3, s=stride, p=1),
            Conv(out_channels, out_channels, k=3, s=1, p=1, act=False),
        )

        # downsample for channel/stride mismatch
        self.downsample = (
            Conv(in_channels, out_channels, k=1, s=stride, act=False)
            if (stride != 1 or in_channels != out_channels)
            else nn.Identity()
        )

        self.relu = nn.ReLU(inplace=True)
        self.short_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x
        out = self.conv(x)
        identity = self.downsample(identity)
        out = self.short_add.add(out, identity)
        return self.relu(out)

class ConvModule2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=False, norm_layer=None, act_layer=None):
        super(ConvModule2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.norm = norm_layer if norm_layer is not None else nn.Identity()
        self.act = act_layer if act_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class AdaptiveAggregationModule(nn.Module):
    """
    Adaptive aggregation module for optimizing disparity.

    Args:
        num_scales: The num of cost volume.
        num_output_branches:  The num branch for outputs.
        max_disp: The max value of disparity.
        num_blocks: The num of block.
    """

    def __init__(
            self,
            num_scales: int,
            num_output_branches: int,
            max_disp: int,
            num_blocks: int = 1,
    ):
        super(AdaptiveAggregationModule, self).__init__()

        self.num_scales = num_scales
        self.num_output_branches = num_output_branches
        self.max_disp = max_disp
        self.num_blocks = num_blocks

        self.branches = nn.ModuleList()

        # Adaptive intra-scale aggregation
        for i in range(self.num_scales):
            num_candidates = max_disp // (2 ** i)
            branch = nn.ModuleList()
            for _ in range(num_blocks):
                branch.append(
                    BasicResBlock(num_candidates, num_candidates)
                )
            self.branches.append(nn.Sequential(*branch))

        self.fuse_layers = nn.ModuleList()

        # Adaptive cross-scale aggregation
        # For each output branch
        for i in range(self.num_output_branches):
            self.fuse_layers.append(nn.ModuleList())
            # For each branch (different scale)
            for j in range(self.num_scales):
                if i == j:
                    # Identity
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            ConvModule2d(
                                in_channels=max_disp // (2 ** j),
                                out_channels=max_disp // (2 ** i),
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False,
                                norm_layer=nn.BatchNorm2d(
                                    max_disp // (2 ** i)
                                ),
                                act_layer=None,
                            )
                        )
                    )
                elif i > j:
                    layers = nn.ModuleList()
                    for _ in range(i - j - 1):
                        layers.append(
                            nn.Sequential(
                                ConvModule2d(
                                    in_channels=max_disp // (2 ** j),
                                    out_channels=max_disp // (2 ** j),
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    bias=False,
                                    norm_layer=nn.BatchNorm2d(
                                        max_disp // (2 ** j)
                                    ),
                                    act_layer=nn.ReLU(inplace=True),
                                )
                            )
                        )

                    layers.append(
                        nn.Sequential(
                            ConvModule2d(
                                in_channels=max_disp // (2 ** j),
                                out_channels=max_disp // (2 ** i),
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False,
                                norm_layer=nn.BatchNorm2d(
                                    max_disp // (2 ** i)
                                ),
                                act_layer=None,
                            )
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*layers))

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def update_idx(self, idx: int) -> int:
        """Update the idx."""
        return idx + 1

    def interpolate_exchange(self, x_fused, exchange, i, idx):
        """Helper function for fusing features."""
        # Assuming this was doing some kind of addition operation
        # In PyTorch we can just use + operator
        if exchange.size()[2:] != x_fused[i].size()[2:]:
            exchange = F.interpolate(
                exchange,
                size=x_fused[i].size()[2:],
                mode="bilinear",
                align_corners=False,
            )
        return x_fused[i] + exchange

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        """Perform the forward pass of the model.

        Args:
            x: The inputs pyramid costvolume.

        Returns:
            x_fused: The fused pyramid costvolume.
        """
        assert len(self.branches) == len(x)

        for i in range(len(self.branches)):
            branch = self.branches[i]
            for j in range(self.num_blocks):
                dconv = branch[j]
                x[i] = dconv(x[i])

        x_fused = []
        idx = 0
        for i in range(len(self.fuse_layers)):
            for j in range(len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    exchange = self.fuse_layers[i][j](x[j])
                    x_fused[i] = self.interpolate_exchange(
                        x_fused, exchange, i, idx
                    )
                    idx = self.update_idx(idx)

        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])

        return x_fused
class UnfoldConv(nn.Module):
    """
    A unfold module using conv.

    Args:
        in_channels: The channels of inputs.
        kernel_size: The kernel_size of unfold.
    """

    def __init__(self, in_channels: int = 1, kernel_size: int = 2):
        super(UnfoldConv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.kernel_size ** 2,
            kernel_size=self.kernel_size,
            stride=1,
            bias=False,
        )
        self.pad = nn.ZeroPad2d(padding=(1, 0, 1, 0))
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize the weights of head module."""

        weight_new = torch.zeros(
            self.conv.weight.size(), dtype=self.conv.weight.dtype
        )
        for i in range(self.kernel_size ** 2):
            wx = i % self.kernel_size
            wy = i // self.kernel_size

            if wx < self.kernel_size / 2:
                if wy < self.kernel_size / 2:
                    weight_new[i, :, 0, 0] = 1
                else:
                    weight_new[i, :, 1, 0] = 1
            else:
                if wy < self.kernel_size / 2:
                    weight_new[i, :, 0, 1] = 1
                else:
                    weight_new[i, :, 1, 1] = 1

        self.conv.weight = torch.nn.Parameter(weight_new, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass of the model."""

        x = self.pad(x)
        x = self.conv(x)
        return x

    def fix_weight_qscale(self) -> None:
        """Fix the qscale of conv weight when calibration or qat stage."""

        self.conv.weight_fake_quant.disable_observer()
        self.conv.weight_fake_quant.set_qparams(
            torch.ones(
                self.conv.weight.shape[0], device=self.conv.weight.device
            )
        )
class StereoNetHeadPlus(nn.Module):
    def __init__(
        self,
        maxdisp=192,
        refine_levels=4,
        bn_kwargs=None,
        max_stride=32,
        num_costvolume=3,
        num_fusion=6,
        hidden_dim=16,
        in_channels=(32, 32, 16, 16, 16),
    ):
        super().__init__()
        if bn_kwargs is None:
            bn_kwargs = {}

        self.maxdisp = maxdisp
        self.refine_levels = refine_levels
        self.num_costvolume = num_costvolume
        self.D = self.maxdisp // max_stride
        self.num_fusion = num_fusion
        self.hidden_dim = hidden_dim

        self.gc_pad = nn.ModuleList()
        self.gc_mean = nn.ModuleList()
        self.gc_mul = nn.ModuleList()

        for k in range(num_costvolume):
            scale_tmp = 2 ** k
            for i in range(self.D * scale_tmp):
                self.gc_pad.append(nn.ZeroPad2d((i, 0, 0, 0)))
                # 原先的 FloatFunctional 这里直接用普通张量运算
                self.gc_mean.append(None)  # placeholder
                self.gc_mul.append(None)   # placeholder

        self.gc_cat_final = nn.ModuleList([None] * refine_levels)  # placeholder

        self.softmax2 = nn.Softmax(dim=1)
        # self.softmax2.min_sub_out = -12.0  # 量化相关，暂时删掉

        low_disp_max = self.D * (2 ** (self.num_costvolume - 1))

        # 你这里的AdaptiveAggregationModule如果没改，保持不变
        self.fusions = nn.ModuleList()
        for i in range(self.num_fusion):
            num_out_branches = 1 if i == self.num_fusion - 1 else 3
            self.fusions.append(
                AdaptiveAggregationModule(
                    self.num_costvolume, num_out_branches, low_disp_max
                )
            )

        self.final_conv = Conv(
            low_disp_max,
            low_disp_max,
            k=1,
            s=1,
            act=False,
        )

        # 量化相关，先去掉
        # self.disp_mul_op = hpp.nn.quantized.FloatFunctional()
        # self.disp_sum_op = hpp.nn.quantized.FloatFunctional()
        # self.quant_dispvalue = QuantStub()
        self.softmax = nn.Softmax(dim=1)
        # self.softmax.min_sub_out = -12.0

        self.disp_values = nn.Parameter(
            torch.arange(low_disp_max, dtype=torch.float32).view(1, low_disp_max, 1, 1) / low_disp_max,
            requires_grad=False,
        )

        # 下采样部分，用你自己的模块或用普通Conv+Upsample代替
        self.spx_8 = nn.Sequential(
            Conv(
                hidden_dim,
                hidden_dim,
                k=3,
                s=1,
                act=nn.ReLU(inplace=True),
            ),
            Conv(
                hidden_dim,
                hidden_dim,
                k=3,
                s=1,
                act=False,
            ),
        )

        # 你自己实现或保持不变
        self.spx_4 = DeConvResModule(hidden_dim, hidden_dim, bn_kwargs)
        self.spx_2 = DeConvResModule(hidden_dim, hidden_dim, bn_kwargs)

        self.spx = ConvTranspose(
            hidden_dim,
            4,
            k=4,
            s=2,
            act=nn.ReLU(inplace=True),
        )

        self.spx_conv3x3 = Conv(
            4,
            4,
            k=3,
            s=1,
        )

        self.mod1 = Conv(
            in_channels[0],
            hidden_dim,
            k=3,
            s=1,
            act=False,
        )

        self.mod2 = Conv(
            in_channels[1],
            hidden_dim,
            k=3,
            s=1,
            act=False,
        )

        # 量化去掉
        # self.unfold = UnfoldConv()
        # self.dequant = DeQuantStub()

    def get_l_img(self, img: torch.Tensor, B: int) -> torch.Tensor:
        return img[: B // 2]

    def dis_mul(self, x: torch.Tensor) -> torch.Tensor:
        # 用普通乘法替代量化FloatFunctional
        return x * self.disp_values

    def dis_sum(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1, keepdim=True)

    def build_aanet_volume(self, refimg_fea, maxdisp, offset, idx):
        B, C, H, W = refimg_fea.shape
        num_sample = B // 2
        tmp_volume = []

        for i in range(maxdisp):
            if i > 0:
                cost = refimg_fea[:num_sample, :, :, i:] * refimg_fea[num_sample:, :, :, :-i]
                tmp = cost.mean(dim=1, keepdim=True)
                tmp = F.pad(tmp, (i, 0, 0, 0))
                tmp_volume.append(tmp)
            else:
                cost = refimg_fea[:num_sample, :, :, :] * refimg_fea[num_sample:, :, :, :]
                tmp = cost.mean(dim=1, keepdim=True)
                tmp_volume.append(tmp)

        volume = torch.cat(tmp_volume, dim=1).view(num_sample, maxdisp, H, W)
        return volume

    def get_offset(self, offset: int, idx: int) -> int:
        return offset + self.D * (2 ** idx)

    def forward(self, features_inputs: list):
        features_inputs[0] = self.mod1(features_inputs[0])
        features_inputs[1] = self.mod2(features_inputs[1])

        features = features_inputs[-3:][::-1]
        aanet_volumes = []
        offset = 0
        for i in range(len(features)):
            aanet_volume = self.build_aanet_volume(
                features[i], self.D * (2 ** i), offset, i
            )
            offset = self.get_offset(offset, i)
            aanet_volumes.append(aanet_volume)

        aanet_volumes = aanet_volumes[::-1]
        for fusion in self.fusions:
            aanet_volumes = fusion(aanet_volumes)

        cost0 = self.final_conv(aanet_volumes[0])

        pred0 = self.softmax(cost0)
        pred0 = self.dis_mul(pred0)
        pred0 = self.dis_sum(pred0)

        # unfold相关需要自己实现或者用普通tensor操作代替
        # pred0_unfold = self.unfold(pred0)

        B = features_inputs[0].shape[0]

        xspx = self.spx_8(self.get_l_img(features_inputs[2], B))
        feature1_l = self.get_l_img(features_inputs[1], B)
        xspx = self.spx_4(xspx, feature1_l)
        feature0_l = self.get_l_img(features_inputs[0], B)
        xspx = self.spx_2(xspx, feature0_l)

        spx_pred = self.spx(xspx)
        spx_pred = self.spx_conv3x3(spx_pred)
        spx_pred = self.softmax2(spx_pred)

        return pred0, None, spx_pred  # unfold暂时设为None

if __name__ == '__main__':
    # 初始化模型
    model = StereoNetHeadPlus(
        maxdisp=192,
        refine_levels=4,
        bn_kwargs={'momentum': 0.1, 'affine': True},
        max_stride=32,
        num_costvolume=3,
        num_fusion=6,
        hidden_dim=16,
        in_channels=[16, 96, 64, 16, 16]  # 这里根据你的特征通道数设置，最后两个16是示例，可根据代码修改
    )
    model.eval()  # 设为eval模式

    # 构造三层特征输入
    feat1 = torch.randn(2, 16, 17, 30)  # 最小分辨率
    feat2 = torch.randn(2, 96, 34, 60)
    feat3 = torch.randn(2, 64, 68, 120)  # 最大分辨率

    inputs = [feat1, feat2, feat3]

    # 运行前向
    with torch.no_grad():
        pred0, pred0_unfold, spx_pred = model(inputs)

    # 输出形状
    print("pred0 shape:", pred0.shape)  # 预测低分辨率视差图
    print("pred0_unfold shape:", pred0_unfold.shape)
    print("spx_pred shape:", spx_pred.shape)  # 权重预测图