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
from models import Conv

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
                # if simple_bottleneck:
                branch.append(
                    BasicResBlock(num_candidates, num_candidates, bn_kwargs={})
                )
            self.branches.append(nn.Sequential(*branch))

        self.fuse_layers = nn.ModuleList()

        # Adaptive cross-scale aggregation
        # For each output branch
        for i in range(self.num_output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.num_scales):
                if i == j:
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            Conv(
                                c1=max_disp // (2 ** j),
                                c2=max_disp // (2 ** i),
                                k=1,
                                s=1,
                                p=0,
                                act=False,  # 原 act_layer=None
                            )
                        )
                    )
                elif i > j:
                    layers = []
                    for _ in range(i - j - 1):
                        layers.append(
                            Conv(
                                c1=max_disp // (2 ** j),
                                c2=max_disp // (2 ** j),
                                k=3,
                                s=2,
                                p=1,
                                act=nn.ReLU(inplace=True),
                            )
                        )
                    layers.append(
                        Conv(
                            c1=max_disp // (2 ** j),
                            c2=max_disp // (2 ** i),
                            k=3,
                            s=2,
                            p=1,
                            act=False,
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*layers))

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.fuse_add = nn.ModuleList()
        # for _ in range(len(self.fuse_layers) * len(self.branches)):
        #     self.fuse_add.append(hpp.nn.quantized.FloatFunctional())

    def update_idx(self, idx: int) -> int:
        """Update the idx."""

        return idx + 1

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

    def interpolate_exchange(
        self, x_fused: Tensor, exchange: Tensor, i: int, idx: int
    ) -> Tensor:
        """Unsample costvolume and fuse."""

        if exchange.size()[2:] != x_fused[i].size()[2:]:
            exchange = F.interpolate(
                exchange,
                size=x_fused[i].size()[2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.fuse_add[idx].add(exchange, x_fused[i])
class stereoplus_headplus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(stereoplus_headplus, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
    def build_aanet_volume(self, refimg_fea, maxdisp, offset, idx):
        """
        Build the cost volume using the same approach as AANet.

        Args:
            refimg_fea: Featuremaps.
            maxdisp: Maximum disparity value.
            offset: The offset of gc_mul and gc_mean floatFunctional.
            idx: The idx of cat floatFunctional.

        Returns:
            volume: Costvolume.
        """

        B, C, H, W = refimg_fea.shape
        num_sample = B // 2
        tmp_volume = []
        for i in range(maxdisp):
            if i > 0:
                cost = self.gc_mul[i + offset].mul(
                    refimg_fea[:num_sample, :, :, i:],
                    refimg_fea[num_sample:, :, :, :-i],
                )
                tmp = self.gc_mean[i + offset].mean(cost, dim=1)
                tmp_volume.append(self.gc_pad[i + offset](tmp))
            else:
                cost = self.gc_mul[i + offset].mul(
                    refimg_fea[:num_sample, :, :, :],
                    refimg_fea[num_sample:, :, :, :],
                )
                tmp = self.gc_mean[i + offset].mean(cost, dim=1)
                tmp_volume.append(tmp)

        volume = (
            self.gc_cat_final[idx]
            .cat(tmp_volume, dim=1)
            .view(num_sample, maxdisp, H, W)
        )
        return volume
    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        return x