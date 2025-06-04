# -*- coding: utf-8 -*-
# @Time    : 2025/5/18 22:26
# @Author  : sjh
# @Site    : 
# @File    : my_stereonet.py
# @Comment :
import torch
import torch.nn.functional as F
from torch import Tensor
def build_concat_volume(

        refimg_fea: Tensor,
        targetimg_fea: Tensor,
        maxdisp: int,
) -> Tensor:
    """
    Build the concat cost volume.

    Args:
        refimg_fea: Left image feature.
        targetimg_fea: Right image feature.
        maxdisp: Maximum disparity value.

    Returns:
        volume: Concatenated cost volume.
    """

    B, C, H, W = refimg_fea.shape
    C = 2 * C
    tmp_volume = []
    for i in range(maxdisp):
        if i > 0:
            tmp = torch.cat(
                (refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i]),
                dim=1,
            )
            tmp_volume.append(F.pad(tmp,(0,i,0,0)).view(-1, 1, H, W))
        else:
            tmp_volume.append(
                torch
                .cat((refimg_fea, targetimg_fea), dim=1)
                .view(-1, 1, H, W)
            )
    volume = torch.cat(tmp_volume, dim=1).view(
        B, C, maxdisp, H, W
    )
    return volume
input = torch.FloatTensor(1, 3, 544, 960).zero_().cuda()

# 连续三次缩小（缩小比例为原始的 1/2 -> 1/4 -> 1/8）
x1 = F.interpolate(input, scale_factor=0.5, mode='bilinear', align_corners=False)
x2 = F.interpolate(x1, scale_factor=0.5, mode='bilinear', align_corners=False)
x3 = F.interpolate(x2, scale_factor=0.5, mode='bilinear', align_corners=False)
k=3
max_disp = (192 + 1) // pow(2, k)
x3 = torch.zeros(1, 32, 544, 960).cuda()
cost_volume = build_concat_volume(x3, x3, max_disp)
print(cost_volume.shape)