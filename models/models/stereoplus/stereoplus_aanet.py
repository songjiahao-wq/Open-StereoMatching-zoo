# -*- coding: utf-8 -*-
# @Time    : 2025/5/18 21:45
# @Author  : sjh
# @Site    : 
# @File    : TinyHITNet_stereonet.py
# @Comment :
import torch
import torch.nn as nn
import torch.nn.functional as F


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
# 修改make_cost_volume_aanet（返回绝对值差异）
def make_cost_volume_aanet2(left, right, max_disp):
    B, C, H, W = left.shape
    cost = torch.zeros((B, max_disp, H, W), device=left.device)
    for d in range(max_disp):
        if d > 0:
            cost[:, d, :, d:] = torch.abs(left[:, :, :, d:] - right[:, :, :, :-d]).mean(1)
        else:
            cost[:, 0] = torch.abs(left - right).mean(1)
    return cost

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
        self.cost_filter_aanet = CostAggregation2D(self.max_disp)
        self.refine_layer = nn.ModuleList([RefineNet() for _ in range(self.k)])
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
    def compute_cost_volume(self, lf, rf, max_disp):

        # Compute cost volume
        # (1,32,24,68,120) self.max_disp=24
        cost_volume = make_cost_volume(lf, rf, self.max_disp)
        # cost_volume=(1,32,24,68,120)
        cost_volume = self.cost_filter(cost_volume).squeeze(1)
        x = F.softmax(cost_volume, dim=1)
        d = torch.arange(0, self.max_disp, device=x.device, dtype=x.dtype)
        x = torch.sum(x * d.view(1, -1, 1, 1), dim=1, keepdim=True)
        return x
    def compute_cost_volume1(self, lf, rf, max_disp):
        # Compute cost volume aanet style
        cost_volume = make_cost_volume_aanet(lf, rf, self.max_disp)  # [B, D, H, W]
        # 2D aggregation
        cost_volume = self.cost_filter_aanet(cost_volume)  # cost_filter now uses 2D conv
        # Disparity regression
        prob_volume = F.softmax(cost_volume, dim=1)
        disp_values = torch.arange(0, self.max_disp, device=prob_volume.device, dtype=prob_volume.dtype)
        disp = torch.sum(prob_volume * disp_values.view(1, -1, 1, 1), dim=1, keepdim=True)  # [B, 1, H, W]
        return disp
    def compute_cost_volume2(self, lf, rf, max_disp):
        # 修改后的forward片段
        cost_volume = make_cost_volume_aanet(lf, rf, max_disp)  # [B, D, H, W]

        # 2D聚合后转为差异度量（关键修改）
        cost_volume = self.cost_filter_aanet(cost_volume)  # [B, D, H, W]
        cost_volume = -cost_volume  # 将相似度转换为差异度量

        # 视差回归
        prob_volume = F.softmax(cost_volume, dim=1)  # 现在差异越小概率越大
        disp_values = torch.arange(0, self.max_disp, device=prob_volume.device)
        disp = torch.sum(prob_volume * disp_values.view(1, -1, 1, 1), dim=1, keepdim=True)
        return disp
    def compute_cost_volume3(self, lf, rf, max_disp):
        # forward保持不变
        cost_volume = make_cost_volume_aanet2(lf, rf, max_disp)
        cost_volume = self.cost_filter_aanet(cost_volume)
        prob_volume = F.softmax(-cost_volume, dim=1)  # 差异越小概率越大
        disp_values = torch.arange(0, self.max_disp, device=prob_volume.device)
        disp = torch.sum(prob_volume * disp_values.view(1, -1, 1, 1), dim=1, keepdim=True)
        return disp
    def forward(self, left_img, right_img,iters=None,test_mode=False):
        n, c, h, w = left_img.size()
        w_pad = (self.align - (w % self.align)) % self.align
        h_pad = (self.align - (h % self.align)) % self.align

        left_img = F.pad(left_img, (0, w_pad, 0, h_pad))
        right_img = F.pad(right_img, (0, w_pad, 0, h_pad))
        
        # Extract multi-scale features
        lf_features = self.feature_extractor(left_img)
        rf_features = self.feature_extractor(right_img)
        
        # Use the main output for cost volume computation
        lf = lf_features["s8"]
        rf = rf_features["s8"]

        disp = self.compute_cost_volume3(lf, rf, self.max_disp)


        multi_scale = []
        for refine in self.refine_layer:
            x = refine(disp, left_img)
            scale = left_img.size(3) / x.size(3)
            full_res = F.interpolate(x * scale, left_img.shape[2:])[:, :, :h, :w]
            multi_scale.append(full_res)
        if test_mode:
            return multi_scale[-1]
        else:
            return {
                "disp": multi_scale[-1],
                "multi_scale": multi_scale,
                "features": lf_features  # Return multi-scale features
            }


if __name__ == "__main__":
    from thop import profile

    left = torch.rand(1, 3, 544, 960)
    right = torch.rand(1, 3, 544, 960)
    model = StereoNet(cfg=None)

    print(model(left, right)["disp"].size())

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