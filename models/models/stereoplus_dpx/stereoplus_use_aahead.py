# -*- coding: utf-8 -*-
# @Time    : 2025/5/18 21:45
# @Author  : sjh
# @Site    : 
# @File    : TinyHITNet_stereonet.py
# @Comment :
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.models.stereoplus_dpx.headplus import AdaptiveAggregationModule, UnfoldConv
def make_cost_volume(left, right, max_disp):
    cost_volume = torch.ones(
        (left.size(0), left.size(1), max_disp, left.size(2), left.size(3)),
        dtype=left.dtype,
        device=left.device,
    )

    cost_volume[:, :, 0, :, :] = left - right
    for d in range(1, max_disp):
        cost_volume[:, :, d, :, d:] = left[:, :, :, d:] - right[:, :, :, :-d]

    return cost_volume
def build_aanet_volume(left, right, max_disp):
    """构建 cost volume：left * right 点乘（互相关）"""
    B, C, H, W = left.shape
    cost_volume = []

    for d in range(max_disp):
        if d > 0:
            cost = left[:, :, :, d:] * right[:, :, :, :-d]
            cost = F.pad(cost, (d, 0, 0, 0))
        else:
            cost = left * right
        cost = torch.mean(cost, dim=1, keepdim=True)
        cost_volume.append(cost)

    cost_volume = torch.cat(cost_volume, dim=1)  # [B, D, H, W]
    return cost_volume
class CostAggregation2D(nn.Module):
    def __init__(self, max_disp):
        super().__init__()
        self.max_disp = max_disp
        self.encoder = nn.Sequential(
            conv_3x3(max_disp, 64),
            conv_3x3(64, 64),
            conv_3x3(64, 64),
        )
        self.decoder = nn.Sequential(
            conv_3x3(64, 32),
            conv_3x3(32, 32),
            nn.Conv2d(32, max_disp, 3, 1, 1)
        )

    def forward(self, x):  # x: [B, D, H, W]
        x = self.encoder(x)
        x = self.decoder(x)
        return x  # [B, D, H, W]
def conv_3x3(in_c, out_c, s=1, d=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, s, d, dilation=d, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )


def conv_1x1(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
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


class RefineNet(nn.Module):
    def __init__(self):
        super().__init__()
        d = [1, 2, 4, 8, 1, 1]
        self.conv0 = nn.Sequential(
            conv_3x3(4, 32),
            *[ResBlock(32, d[i]) for i in range(6)],
            nn.Conv2d(32, 1, 3, 1, 1),
        )

    def forward(self, disp, rgb):
        disp = (
            F.interpolate(disp, scale_factor=2, mode="bilinear", align_corners=False)
            * 2
        )
        rgb = F.interpolate(
            rgb, (disp.size(2), disp.size(3)), mode="bilinear", align_corners=False
        )
        x = torch.cat((disp, rgb), dim=1)
        x = self.conv0(x)
        return F.relu(disp + x)
from models.models.stereoplus_dpx.neck import StereoPlusPipeline

class StereoNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.k = 3
        self.align = 2 ** self.k
        self.max_disp = (192 + 1) // (2 ** self.k)

        # Feature extractor with 5 levels of downsampling
        self.conv1 = nn.Sequential(conv_3x3(3, 32, 2), ResBlock(32))  # 1/2
        self.conv2 = nn.Sequential(conv_3x3(32, 32, 2), ResBlock(32))  # 1/4
        self.conv3 = nn.Sequential(conv_3x3(32, 32, 2), ResBlock(32))  # 1/8
        self.conv4 = nn.Sequential(conv_3x3(32, 32, 2), ResBlock(32))  # 1/16
        self.conv5 = nn.Sequential(conv_3x3(32, 32, 2), ResBlock(32))  # 1/32
        self.last_conv = nn.Conv2d(32, 32, 3, 1, 1)

        self.StereoPlusPipeline = StereoPlusPipeline(B=8, height=544, width=960)
        self.StereoPlusPipeline.in_channels = [32, 32, 32]  # Adjusted for StereoPlus

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            if isinstance(m, nn.SyncBatchNorm):
                m.eval()
    def feature_extractor(self, x):
        feat1 = self.conv1(x)        # 1/2 scale
        feat2 = self.conv2(feat1)    # 1/4 scale
        feat3 = self.conv3(feat2)    # 1/8 scale
        feat4 = self.conv4(feat3)    # 1/16 scale
        feat5 = self.conv5(feat4)    # 1/32 scale
        feat5 = self.last_conv(feat5)
        
        return {
            "s2": feat1,
            "s4": feat2,
            "s8": feat3,
            "s16": feat4,
            "s32": feat5,
            "out": feat5  # Main output for backward compatibility
        }
        
    def forward(self, left_img, right_img, gt_disp=None):
        n, c, h, w = left_img.size()
        w_pad = (self.align - (w % self.align)) % self.align
        h_pad = (self.align - (h % self.align)) % self.align

        left_img = F.pad(left_img, (0, w_pad, 0, h_pad))
        right_img = F.pad(right_img, (0, w_pad, 0, h_pad))
        
        # Extract multi-scale features
        lf_features = self.feature_extractor(left_img)
        rf_features = self.feature_extractor(right_img)
        inputs = [
            torch.cat([lf_features[key], rf_features[key]], dim=0)
            for key in ["s2", "s4", "s8", "s16", "s32"]
        ]
        outputs_train = self.StereoPlusPipeline.forward(inputs, gt_disp)
        #disp_low, disp_1
        # loss_weights = [1 / 3, 2 / 3]
        if self.training:
            return {
                "disp_low": outputs_train[0],
                "disp_1": outputs_train[1],
            }
        else:
            return outputs_train[-1]



if __name__ == "__main__":
    from thop import profile

    left = torch.rand(8, 3, 544, 960)
    right = torch.rand(8, 3, 544, 960)
    gt_disp = torch.rand(8, 544, 960)
    model = StereoNet(cfg=None)

    print(model(left, right, gt_disp)["disp"].size())

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