# -*- coding: utf-8 -*-
# @Time    : 2025/5/18 21:45
# @Author  : sjh
# @Site    : 
# @File    : TinyHITNet_stereonet.py
# @Comment :
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union

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


def make_cost_volume_aanet(left_feat, right_feat, max_disp):
    B, C, H, W = left_feat.size()
    cost_volume = []

    for d in range(max_disp):
        if d > 0:
            cost = left_feat[:, :, :, d:] * right_feat[:, :, :, :-d]  # shape [B, C, H, W-d]
            cost = F.pad(cost, (d, 0, 0, 0))  # pad left d
        else:
            cost = left_feat * right_feat

        # 确保最终宽度一致（W），不论之前怎么裁剪和 pad
        if cost.shape[-1] != W:
            cost = F.pad(cost, (0, W - cost.shape[-1], 0, 0))

        cost_volume.append(cost)

    cost_volume = torch.stack(cost_volume, dim=1)  # [B, D, C, H, W]
    cost_volume = cost_volume.mean(2)  # reduce over channel → [B, D, H, W]
    return cost_volume


def make_cost_volume_abs_diff(left, right, max_disp):
    B, C, H, W = left.size()
    cost_volume = torch.zeros(B, max_disp, H, W, device=left.device)
    for d in range(max_disp):
        if d > 0:
            diff = left[:, :, :, d:] - right[:, :, :, :-d]
            cost_volume[:, d, :, d:] = diff.abs().mean(1)
        else:
            cost_volume[:, 0, :, :] = (left - right).abs().mean(1)
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


class UnfoldConv(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 3):
        super(UnfoldConv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * kernel_size * kernel_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False
        )

        # 初始化为展开行为（固定权重）
        self.init_weights()

    def init_weights(self):
        # 初始化权重为 unfold 展开效果
        with torch.no_grad():
            identity_kernel = torch.zeros_like(self.conv.weight)  # [C*k², 1, k, k]
            c = self.conv.in_channels
            k = self.kernel_size
            for i in range(k):
                for j in range(k):
                    idx = i * k + j
                    for ch in range(c):
                        identity_kernel[ch * k * k + idx, ch, i, j] = 1
            self.conv.weight.copy_(identity_kernel)

    def forward(self, x):  # x: [B, C, H, W]
        return self.conv(x)  # [B, C*k², H, W]


class AdaptiveAggregation2D(nn.Module):
    def __init__(self, in_channels=1, kernel_size=3):
        super().__init__()
        self.unfold_conv = UnfoldConv(in_channels, kernel_size)
        self.kernel_size = kernel_size
        self.weight_net = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, kernel_size * kernel_size, 1)
        )

    def forward(self, cost_volume):  # [B, D, H, W]
        B, D, H, W = cost_volume.shape
        cost_volume_reshaped = cost_volume.view(B * D, 1, H, W)  # each disparity as input

        patches = self.unfold_conv(cost_volume_reshaped)  # [B*D, k², H, W]
        patches = patches.view(B, D, self.kernel_size * self.kernel_size, H, W)

        # 生成权重（使用输入视图或 cost 也可）
        weights = self.weight_net(cost_volume.mean(1, keepdim=True))  # [B, k², H, W]
        weights = F.softmax(weights, dim=1).unsqueeze(1)  # [B, 1, k², H, W]

        out = (patches * weights).sum(dim=2)  # [B, D, H, W]
        return out


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

class AdaptiveAggregationModule(nn.Module):
    def __init__(self, in_channels_list):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=1),
                nn.ReLU(inplace=True)
            ) for c in in_channels_list
        ])
        self.weight_layer = nn.Sequential(
            nn.Conv2d(32 * len(in_channels_list), len(in_channels_list), kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, cost_volumes):
        # 每个 cost volume shape: [B, D, H, W] -> reshape 为 [B*D, 1, H, W]
        # reshaped = [cv.view(-1, 1, *cv.shape[2:]) for cv in cost_volumes]

        projected = [conv(x) for conv, x in zip(self.convs, cost_volumes)]  # 每个变成 [BD, 32, H, W]

        fused = torch.cat(projected, dim=1)  # [BD, 32 * N, H, W]
        weights = self.weight_layer(fused)   # [BD, N, H, W]

        # 权重分配到原 cost volume 上
        weights = weights.unsqueeze(2)  # [BD, N, 1, H, W]
        cost_stack = torch.stack(cost_volumes, dim=1)  # [BD, N, 1, H, W]
        fused_cost = (weights * cost_stack).sum(dim=1)  # [BD, 1, H, W]

        # reshape 回 [B, D, H, W]
        B, D = cost_volumes[0].shape[:2]
        fused_cost = fused_cost.view(B, D, *cost_volumes[0].shape[2:])
        return fused_cost
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

        self.cost_filter = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 1, 3, 1, 1),
        )
        self.cost_filter_aanetori = CostAggregation2D(self.max_disp)
        self.cost_filter_aanet = AdaptiveAggregation2D(kernel_size=3)

        self.refine_layer = nn.ModuleList([RefineNet() for _ in range(self.k)])

        self.cost_fusion = AdaptiveAggregationModule([24,24,24])  # s8, s16, s32
        self.conv1_1x1 = nn.Conv2d(12, 24, 1)
        self.conv2_1x1 = nn.Conv2d(6, 24, 1)
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            if isinstance(m, nn.SyncBatchNorm):
                m.eval()

    def feature_extractor(self, x):
        feat1 = self.conv1(x)  # 1/2 scale
        feat2 = self.conv2(feat1)  # 1/4 scale
        feat3 = self.conv3(feat2)  # 1/8 scale
        feat4 = self.conv4(feat3)  # 1/16 scale
        feat5 = self.conv5(feat4)  # 1/32 scale
        feat5 = self.last_conv(feat5)

        return {
            "s2": feat1,
            "s4": feat2,
            "s8": feat3,
            "s16": feat4,
            "s32": feat5,
            "out": feat5  # Main output for backward compatibility
        }

    def forward(self, left_img, right_img, iters=None, test_mode=False):
        n, c, h, w = left_img.size()
        w_pad = (self.align - (w % self.align)) % self.align
        h_pad = (self.align - (h % self.align)) % self.align

        left_img = F.pad(left_img, (0, w_pad, 0, h_pad))
        right_img = F.pad(right_img, (0, w_pad, 0, h_pad))

        # Extract multi-scale features
        lf_features = self.feature_extractor(left_img)
        rf_features = self.feature_extractor(right_img)

        # Use the main output for cost volume computation
        scales = ["s8", "s16", "s32"]
        cost_volumes = []

        for scale in scales:
            scale_val = int(scale[1:])
            assert scale_val >= 8, f"Invalid scale: {scale}"  # 确保合法
            scale_factor = scale_val // 1
            max_disp_scale = 192 // scale_factor

            lf = lf_features[scale]
            rf = rf_features[scale]
            cost = make_cost_volume_aanet(lf, rf, max_disp_scale)  # [B, D, H, W]
            cost = -cost
            cost = self.cost_filter_aanet(cost)
            cost_volumes.append(cost)

        # 将所有 cost volume 上采样到 s8 分辨率
        target_shape = cost_volumes[0].shape[-2:]  # s8 分辨率
        cost_volumes = [F.interpolate(cv, target_shape, mode='bilinear', align_corners=False) for cv in cost_volumes]

        # 融合所有代价体（简单平均或使用可学习融合模块）
        cost_volumes[1] = self.conv1_1x1(cost_volumes[1])  # s16 -> s8
        cost_volumes[2] = self.conv2_1x1(cost_volumes[2]) # s32 -> s8
        fused_cost_volume = self.cost_fusion(cost_volumes)  # [B, D, H, W]

        prob_volume = F.softmax(fused_cost_volume, dim=1)
        disp_values = torch.arange(0, self.max_disp, device=prob_volume.device).view(1, -1, 1, 1)
        disp = torch.sum(prob_volume * disp_values, dim=1, keepdim=True)

        multi_scale = []
        for refine in self.refine_layer:
            x = refine(disp, left_img)
            scale = left_img.size(3) / x.size(3)
            full_res = F.interpolate(x * scale, left_img.shape[2:])[:, :, :h, :w]
            multi_scale.append(full_res)
        if self.training:
            return {
                "disp": multi_scale[-1],
                "multi_scale": multi_scale,
                "features": lf_features  # Return multi-scale features
            }
        else:
            return multi_scale[-1]

if __name__ == "__main__":
    from thop import profile

    left = torch.rand(1, 3, 544, 960)
    model = StereoNet(cfg=None)

    print(model(left, left)["disp"].size())

    total_ops, total_params = profile(
        model,
        (
            left,
            left,
        ),
    )
    print(
        "{:.4f} MACs(G)\t{:.4f} Params(M)".format(
            total_ops / (1000 ** 3), total_params / (1000 ** 2)
        )
    )