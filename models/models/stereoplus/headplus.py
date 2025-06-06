# -*- coding: utf-8 -*-
# @Time    : 2025/6/6 上午11:29
# @Author  : sjh
# @Site    : 
# @File    : headplus.py
# @Comment :
import torch
import torch.nn as nn
import torch.nn.functional as F
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