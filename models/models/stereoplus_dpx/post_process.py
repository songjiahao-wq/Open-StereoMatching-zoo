from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


__all__ = ["StereoNetPostProcessPlus"]


class StereoNetPostProcessPlus(nn.Module):
    """
    An advanced post process for StereoNet.

    Args:
        maxdisp: The max value of disparity.
        low_max_stride: The max stride of lowest disparity.
    """

    def __init__(self, maxdisp: int = 192, low_max_stride: int = 8):
        super(StereoNetPostProcessPlus, self).__init__()
        self.maxdisp = maxdisp
        self.low_max_stride = low_max_stride

    def forward(
        self,
        modelouts: List[Tensor],
        gt_disps: List[Tensor] = None,
    ) -> Union[Tensor, List[Tensor]]:
        """Perform the forward pass of the model.

        Args:
            modelouts: The model outputs.
            gt_disps: The gt disparitys.

        """

        if len(modelouts) == 3:
            disp_low = modelouts[0]
        else:
            disp_low = None

        disp_low_unfold = modelouts[-2]
        spg = modelouts[-1]

        disp_1 = F.interpolate(
            disp_low_unfold, scale_factor=self.low_max_stride, mode="nearest"
        )

        disp_1 = (spg * disp_1).sum(1)
        disp_1 = disp_1.squeeze(1) * self.maxdisp

        if self.training:
            disp_low = F.interpolate(
                disp_low, size=gt_disps.shape[1:], mode="bilinear"
            )

            disp_low = disp_low.squeeze(1) * self.maxdisp
            return [disp_low, disp_1]
        else:
            return disp_1.squeeze(1)
if __name__ == "__main__":
    # Example usage
    torch.manual_seed(0)

    B, H, W = 1, 544, 960
    maxdisp = 192
    low_max_stride = 8
    K = 9  # 假设有 9 个 disparity 候选

    # 模拟模型输出
    disp_low = torch.rand(B, K, H // 8, W // 8)  # coarse disp
    disp_low_unfold = torch.rand(B, K, H // 8, W // 8)  # unfolded disp
    spg = torch.softmax(torch.rand(B, K, H , W ), dim=1)  # prob for each candidate

    modelouts = [disp_low, disp_low_unfold, spg]

    # 模型推理
    model = StereoNetPostProcessPlus(maxdisp=maxdisp, low_max_stride=low_max_stride)

    # 训练模式
    model.train()
    gt_disp = torch.rand(B, H, W)
    outputs = model(modelouts, gt_disp)
    print("Train mode outputs:")
    for out in outputs:
        print(out.shape)  # [B, H, W]

    # 评估模式
    model.eval()
    with torch.no_grad():
        output_eval = model(modelouts)
        print("Eval mode output:", output_eval.shape)  # [B, H, W]