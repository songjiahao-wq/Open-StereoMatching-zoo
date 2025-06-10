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
        # self.cost_filter_aanet = AdaptiveAggregation2D(kernel_size=3)

        self.refine_layer = nn.ModuleList([RefineNet() for _ in range(self.k)])

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
        # 1. 构建 AANet 风格的相似度 cost volume（B, D, H, W）
        cost_volume = make_cost_volume_aanet(lf, rf, max_disp)

        # 2. 归一化 cost volume（均值 0，方差 1）或 min-max 缩放
        cost_volume = cost_volume - cost_volume.mean(dim=1, keepdim=True)

        # 3. 取负号 → 相似度 → 差异度（softmax 会更偏向相似度大的）
        cost_volume = -cost_volume

        # 4. 进入 2D aggregation 模块（CostAggregation2D）
        cost_volume = self.cost_filter_aanetori(cost_volume)
        cost_volume = self.cost_filter_aanet(cost_volume)

        # 5. softmax + 视差回归
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
        cost_volume = make_cost_volume_abs_diff(lf, rf, max_disp)
        cost_volume = self.cost_filter_aanet(cost_volume)
        prob_volume = F.softmax(-cost_volume, dim=1)  # 差异越小概率越大
        disp_values = torch.arange(0, self.max_disp, device=prob_volume.device)
        disp = torch.sum(prob_volume * disp_values.view(1, -1, 1, 1), dim=1, keepdim=True)
        return disp

    def compute_cost_volume4(self, lf, rf, max_disp):
        # forward保持不变
        cost_volume = make_cost_volume_abs_diff(lf, rf, max_disp)
        cost_volume = self.cost_filter_aanetori(cost_volume)
        temperature = 0.5
        prob_volume = F.softmax(-cost_volume / temperature, dim=1)
        disp_values = torch.arange(0, self.max_disp, device=prob_volume.device).view(1, -1, 1, 1)
        disp = torch.sum(prob_volume * disp_values, dim=1, keepdim=True)
        return disp

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
        lf = lf_features["s8"]
        rf = rf_features["s8"]

        disp = self.compute_cost_volume4(lf, rf, self.max_disp)

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