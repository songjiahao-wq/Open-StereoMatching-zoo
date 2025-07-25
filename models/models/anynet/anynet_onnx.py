# -*- coding: utf-8 -*-
# @Time    : 2025/6/11 下午3:05
# @Author  : sjh
# @Site    : 
# @File    : anynet.py
# @Comment :
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
# from models.models.anynet.submodule import *
from submodule import *


class AnyNet(nn.Module):
    def __init__(self, args):
        super(AnyNet, self).__init__()
        self.with_refine = args.with_refine
        self.feature_extraction = feature_extraction()
        self.conv3d_1 = make_conv3d_block(in_channels=1, hidden_channels=16, out_channels=1, num_layers=6)
        self.conv3d_2 = make_conv3d_block(in_channels=1, hidden_channels=4, out_channels=1, num_layers=6)
        self.conv3d_3 = make_conv3d_block(in_channels=1, hidden_channels=4, out_channels=1, num_layers=6)
        self.volume_regularization = nn.ModuleList([self.conv3d_1, self.conv3d_2, self.conv3d_3])
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def _build_volume(self, refimg, targetimg, maxdisp):
        B, C, H, W = refimg.shape
        cost = torch.zeros(B, 1, maxdisp, H, W, device='cuda')
        for i in range(maxdisp):
            cost[:, :, i, :, :i] = torch.norm(refimg[:, :, :, :i], p=1, dim=1, keepdim=True)
            if i == 0:
                cost[:, :, i, :, :] = torch.norm(refimg[:, :, :, :] - targetimg[:, :, :, :], p=1, dim=1, keepdim=True)
            else:
                cost[:, :, i, :, i:] = torch.norm(refimg[:, :, :, i:] - targetimg[:, :, :,
                                                                        :-i], p=1, dim=1, keepdim=True)
        return cost.contiguous()

    def _bulid_residual_volume(self, refimg, targetimg, maxdisp, disp):
        B, C, H, W = refimg.shape
        cost = torch.zeros(B, 1, 2 * maxdisp + 1, H, W, device='cuda')
        for i in range(-maxdisp, maxdisp + 1):
            new_disp = disp + i
            reconimg = self._warp(targetimg, new_disp)
            cost[:, :, i + maxdisp, :, :] = torch.norm(refimg[:, :, :, :] - reconimg[:, :, :,
                                                                            :], p=1, dim=1, keepdim=True)
        return cost.contiguous()

    def _warp(self, x, disp):
        '''
        Warp an image tensor right image to left image, according to disparity
        x: [B, C, H, W] right image
        disp: [B, 1, H, W] horizontal shift
        '''
        B, C, H, W = x.shape
        # mesh grid
        '''
        for example: H=4, W=3
        xx =         yy =
        [[0 1 2],    [[0 0 0],    
         [0 1 2],     [1 1 1],
         [0 1 2],     [2 2 2],
         [0 1 2]]     [3 3 3]]
        '''
        xx = torch.arange(0, W, device='cuda').view(1, -1).repeat(H, 1)  # [H, W]
        yy = torch.arange(0, H, device='cuda').view(-1, 1).repeat(1, W)  # [H, W]
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)  # [B, 1, H, W]
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)  # [B, 1, H, W]
        vgrid = torch.cat((xx, yy), dim=1).float()  # [B, 2, H, W]

        # the correspondence between left and right is that left (i, j) = right (i-d, j)
        vgrid[:, :1, :, :] = vgrid[:, :1, :, :] - disp
        # scale to [-1, 1]
        vgrid[:, 0, :, :] = vgrid[:, 0, :, :] * 2.0 / (W - 1) - 1.0
        vgrid[:, 1, :, :] = vgrid[:, 1, :, :] * 2.0 / (H - 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(x, vgrid, align_corners=True)
        return output

    def forward(self, left, right):
        refimg = self.feature_extraction(left)  # [1/16, 1/8, 1/4]
        targetimg = self.feature_extraction(right)  # [1/16, 1/8, 1/4]
        scales = len(refimg)


        pred = []
        for i in range(scales):
            print(f"refimg[{i}].shape: {refimg[i].shape}")

            if i == 0:
                cost = self._build_volume(refimg[i], targetimg[i], maxdisp=12)
            else:
                warp_disp = F.interpolate(
                    pred[i - 1],
                    size=(refimg[i].shape[2], refimg[i].shape[3]),  # 直接使用目标特征图大小
                    mode='bilinear',
                    align_corners=True
                )
                # 因为在 onnx 中不能使用 float(shape1 / shape2)，此处默认缩放比例为特征图相对于原图大小的整数倍（例如 2, 4, 8）
                scale = left.shape[2] // refimg[i].shape[2]
                warp_disp = warp_disp / scale

                cost = self._bulid_residual_volume(refimg[i], targetimg[i], maxdisp=2, disp=warp_disp)

            cost = self.volume_regularization[i](cost)
            cost = cost.squeeze(1)

            # 插值恢复为原图大小
            upsample_size = (left.shape[2], left.shape[3])

            if i == 0:
                pred_low_res = F.softmax(-cost, dim=1)
                pred_low_res = disparityregression(start=0, maxdisp=12)(pred_low_res)
                pred_low_res = pred_low_res * (left.shape[2] // refimg[i].shape[2])

                pred_high_res = F.interpolate(pred_low_res, size=upsample_size, mode='bilinear', align_corners=True)
                pred.append(pred_high_res)
            else:
                pred_low_res = F.softmax(-cost, dim=1)
                pred_low_res = disparityregression(start=-2, maxdisp=3)(pred_low_res)
                pred_low_res = pred_low_res * (left.shape[2] // refimg[i].shape[2])

                pred_high_res = F.interpolate(pred_low_res, size=upsample_size, mode='bilinear', align_corners=True)
                pred.append(pred_high_res + pred[i - 1])

        return pred[-1]

class Args:
    def __init__(self):
        self.with_refine = True  # 或 False
if __name__ == '__main__':
    device = 'cuda'
    args = Args()
    # summary(AnyNet(args).to(device), [(3, 544, 960), (3, 544, 960)])

    model = AnyNet(args).to(device)
    left = torch.rand(1, 3, 544, 960).to(device)
    right = torch.rand(1, 3, 544, 960).to(device)
    # print(model(left, right)[-1].shape)  # 输出 stage1 的预测视差图形状
    model.eval()
    torch.onnx.export(
        model,
        (left, right),
        "AnyNet.onnx",
        opset_version=16,  # ONNX opset 版本
        input_names=["left", "right"],  # 输入名称
        output_names=["output"],  # 输出名称
        dynamic_axes=None,  # 动态维度（可选）
        verbose = True
    )