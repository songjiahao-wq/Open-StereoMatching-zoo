import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

__all__ = ["FPN"]

class FPN(nn.Module):
    def __init__(
        self,
        in_strides: List[int],
        in_channels: List[int],
        out_strides: List[int],
        out_channels: List[int],
        fix_out_channel: Optional[int] = None,
        bn_kwargs: Optional[Dict] = None,
    ):
        """FPN neck.

        Args:
            in_strides (list): strides of each input feature map
            in_channels (list): channels of each input feature map,
                the length of in_channels should be equal to in_strides
            out_strides (list): strides of each output feature map,
                should be a subset of in_strides, and continuous (any
                subsequence of 2, 4, 8, 16, 32, 64 ...). The largest
                stride in in_strides and out_strides should be equal
            out_channels (list): channels of each output feature maps
                the length of out_channels should be equal to out_strides
            fix_out_channel (:obj:`int`, optional): if set, there will be
                a 1x1 conv following each output feature map so that each
                final output has fix_out_channel channels
            bn_kwargs (dict): Dict for Bn layer. No Bn layer if
                bn_kwargs=None
        """

        super(FPN, self).__init__()
        self._valid_strides = [2, 4, 8, 16, 32, 64, 128, 256]
        self.bn_kwargs = bn_kwargs

        # in_strides check
        assert len(in_strides) == len(in_channels)
        for stride_i in in_strides:
            assert stride_i in self._valid_strides

        min_idx = self._valid_strides.index(in_strides[0])
        max_idx = self._valid_strides.index(in_strides[-1])

        assert tuple(in_strides) == tuple(self._valid_strides[min_idx:max_idx + 1]), \
            "Input strides must be continuous and in ascending order"
        self.in_strides = in_strides

        # out_strides check
        assert len(out_strides) == len(out_channels)

        min_idx = self._valid_strides.index(out_strides[0])
        max_idx = self._valid_strides.index(out_strides[-1])

        assert tuple(out_strides) == tuple(self._valid_strides[min_idx:max_idx + 1]), \
            "Output strides must be continuous"

        assert all([stride in in_strides for stride in out_strides]), \
            "All output strides must be in input strides"

        assert out_strides[-1] == in_strides[-1], \
            "The largest stride in in_strides and out_strides should be equal"

        self.out_strides = out_strides
        self.src_min_stride_idx = self.in_strides.index(self.out_strides[0])

        # Init modules
        self.conv_extract = nn.ModuleList()
        self.conv_add = nn.ModuleList()
        self.upscale = nn.ModuleList()
        self.conv1x1_up = nn.ModuleList()

        for idx in range(len(out_channels)):
            if idx == 0:
                self.conv_extract.append(
                    nn.Conv2d(
                        in_channels=in_channels[-1],
                        out_channels=out_channels[-1],
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True
                    )
                )
            else:
                if len(out_channels) > 1:
                    self.conv1x1_up.append(
                        nn.Conv2d(
                            in_channels=out_channels[-idx],
                            out_channels=out_channels[-1 - idx],
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True
                        )
                    )

                self.upscale.append(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                )

                self.conv_extract.append(
                    nn.Conv2d(
                        in_channels=in_channels[-1 - idx],
                        out_channels=out_channels[-1 - idx],
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True
                    )
                )
            self.conv_add.append(nn.Module())

        # Optionally map the output feature maps to fix_out_channel
        if fix_out_channel is not None:
            self.conv1x1 = nn.ModuleList()
            for idx, _stride in enumerate(self.out_strides[::-1]):
                self.conv1x1.append(
                    nn.Conv2d(
                        in_channels=out_channels[-1 - idx],
                        out_channels=fix_out_channel,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True
                    )
                )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(features) == len(self.in_strides)

        # Slice features
        in_features = features[self.src_min_stride_idx:][::-1]
        strides = self.in_strides[self.src_min_stride_idx:][::-1]

        fpn_fuse = {}
        for idx, stride in enumerate(strides):
            cur_feat = (
                self.upscale[idx - 1](
                    self.conv1x1_up[idx - 1](fpn_fuse[strides[idx - 1]]))
                if idx > 0
                else None
            )

            fpn_fuse[stride] = self.conv_extract[idx](in_features[idx])
            if idx > 0:
                fpn_fuse[stride] = fpn_fuse[stride] + cur_feat

        if hasattr(self, "conv1x1"):
            for idx, stride in enumerate(strides):
                fpn_fuse[stride] = self.conv1x1[idx](fpn_fuse[stride])

        return [fpn_fuse[stride] for stride in self.out_strides]

    def _init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def fuse_model(self):
        # Simplified fusion with PyTorch
        for module in self.conv1x1_up:
            module.fuse_model()
        self.conv_extract[0].fuse_model()
        for idx in range(1, len(self.out_strides)):
            torch.quantization.fuse_modules(
                self,
                [
                    f"conv_extract.{idx}.0",  # conv
                    f"conv_add.{idx}",  # add
                ],
                inplace=True
            )
        if hasattr(self, "conv1x1"):
            for module in self.conv1x1:
                module.fuse_model()

if __name__ == '__main__':
    # from fpn import FPN

    # 输入参数
    in_strides = [8, 16, 32]  # 输入特征图的 strides
    in_channels = [64, 96, 16]  # 输入特征图的通道数
    out_strides = [8, 16, 32]  # 输出特征图的 strides
    out_channels = [16, 16, 16]  # 输出特征图的通道数
    fix_out_channel = None  # 固定输出通道数，如果设置为 None，则不进行转换

    # 创建 FPN 模型
    model = FPN(
        in_strides=in_strides,
        in_channels=in_channels,
        out_strides=out_strides,
        out_channels=out_channels,
        fix_out_channel=fix_out_channel,
    )

    # 初始化权重
    model._init_weights()

    # 创建模拟输入数据（每个输入特征图的尺寸为 [batch_size, channels, height, width]）
    batch_size = 2
    height = 544
    width = 960

    # 根据 in_strides 创建不同分辨率的输入数据
    inputs = [
        torch.randn(batch_size, in_channels[i], height // (stride), width // (stride)) for i, stride in enumerate(in_strides)
    ]

    # 将输入数据传入模型进行前向传播
    outputs = model(inputs)

    # 打印输出的形状以检查
    for i, out in enumerate(outputs):
        print(f"Output {i + 1} shape: {out.shape}")