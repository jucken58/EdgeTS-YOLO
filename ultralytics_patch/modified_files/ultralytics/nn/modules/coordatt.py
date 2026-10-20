# EdgeTS-YOLO addition (not part of upstream Ultralytics 8.4.166).
"""Coordinate Attention (Hou, Zhou & Feng, CVPR 2021, https://arxiv.org/abs/2103.02907)."""

from __future__ import annotations  # Python 3.8 (Jetson NX) compatibility for `int | None`

import torch
import torch.nn as nn

__all__ = ("CoordAtt", "CoordAttRes")


class CoordAtt(nn.Module):
    """Coordinate attention block.

    Global pooling is factorised into two 1-D poolings along H and W. The two direction-aware descriptors are
    concatenated, passed through a shared 1x1 conv + BN + Hardswish with reduction ratio r, split again, and turned
    into two sigmoid gates g^h (C x H x 1) and g^w (C x 1 x W). Output: y = x * g^h * g^w (same shape as x).
    Only pooling, 1x1 conv, BN, Hardswish, sigmoid and element-wise products are used (all TensorRT-native).

    Args:
        c1 (int): input channels (inserted automatically by parse_model).
        c2 (int | None): output channels; must equal c1 (kept only for YAML symmetry).
        reduction (int): channel reduction ratio r; hidden width = max(8, c1 // r) as in the original paper.
    """

    def __init__(self, c1: int, c2: int | None = None, reduction: int = 32):
        super().__init__()
        assert c2 is None or c2 == c1, f"CoordAtt keeps the channel count (c1={c1}, c2={c2})"
        mip = max(8, c1 // reduction)
        self.conv1 = nn.Conv2d(c1, mip, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, c1, 1, 1, 0)
        self.conv_w = nn.Conv2d(mip, c1, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply coordinate attention to x of shape (N, C, H, W)."""
        _, _, h, w = x.shape
        x_h = x.mean(dim=3, keepdim=True)  # (N, C, H, 1)  -- pooling along width
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # (N, C, W, 1) -- pooling along height
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        a_h = self.conv_h(y_h).sigmoid()  # (N, C, H, 1)
        a_w = self.conv_w(y_w.permute(0, 1, 3, 2)).sigmoid()  # (N, C, 1, W)
        return x * a_h * a_w


class CoordAttRes(CoordAtt):
    """Residual, identity-initialised coordinate attention (EdgeTS-YOLO variant A3b).

    y = x + gamma * (x * g^h * g^w), with a per-channel scale gamma initialised to 0 (ReZero / LayerScale style).
    At initialisation the block is exactly the identity, so inserting it between COCO-pretrained neck and head does not
    perturb the pretrained features (plain CoordAtt multiplies them by ~0.25 at start); training decides how much
    attention to add. Same TensorRT-native operators as CoordAtt plus one Mul and one Add.
    """

    def __init__(self, c1: int, c2: int | None = None, reduction: int = 32):
        super().__init__(c1, c2, reduction)
        self.gamma = nn.Parameter(torch.zeros(1, c1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return x + gamma * CA(x)."""
        return x + self.gamma * super().forward(x)
