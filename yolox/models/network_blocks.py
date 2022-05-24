#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self,
        in_channels,
        out_channels,
        ksize,
        stride,
        groups=1,
        bias=False,
        act="silu",
        dilation=1,
    ):
        super().__init__()
        # same padding
        pad = dilation * (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            dilation=dilation,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        bottleneck_expansion=0.5,
        depthwise=False,
        dilated=False,
        act="silu",
        attn=None,
    ):
        super().__init__()
        hidden_channels = int(out_channels * bottleneck_expansion)
        # Conv = DWConv if depthwise else BaseConv
        if depthwise:
            if dilated:
                Conv = DDWConv
            else:
                Conv = DWConv
        else:
            if dilated:
                Conv = DConv
            else:
                Conv = BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

        if attn is None:
            self.attn = nn.Identity()
        elif attn == "SE":  # Squeeze & Excitation attention
            self.attn = SELayer(
                in_channels, out_channels, reduction=int(1 / bottleneck_expansion)
            )

    def forward(self, x):
        y = self.attn(self.conv2(self.conv1(x)))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        bottleneck_expansion=1.0,
        depthwise=False,
        dilated=False,
        act="silu",
        attn=None,
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels,
                hidden_channels,
                shortcut,
                bottleneck_expansion,
                depthwise,
                dilated,
                act=act,
                attn=attn,
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Networks (2017)
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf
    Figure 3 참조
    ==============================================================================================================
    channel GAP -> (b, c, 1, 1) -> FC (reduction) -> ReLU -> FC (out_channel) -> Sigmoid (attention)
    채널 간 평균의 relation 반영하는 attention
    """

    def __init__(self, in_channels, out_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // reduction, out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DConv(nn.Module):
    """A Dilated Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self,
        in_channels,
        out_channels,
        ksize,
        stride,
        groups=1,
        bias=False,
        act="silu",
        dilation=2,
    ):
        super().__init__()
        # same padding
        pad = dilation * (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DDWConv(nn.Module):
    """Dilated Depthwise Conv + Conv"""

    def __init__(
        self, in_channels, out_channels, ksize, stride=1, act="silu", dilation=2
    ):
        super().__init__()

        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
            dilation=dilation,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Mobile_Bottleneck(nn.Module):
    """
    Rethinking Bottleneck Structure for Efficient Mobile Network Design (2020)
    https://arxiv.org/pdf/2007.02269.pdf           figure 3 참조
    ===========================================================================
    DW -> PW(reduction) -> PW(expansion) -> DW
    DW를 양 끝에 배치해서 spatial 정보 유지
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        bottleneck_expansion=0.5,
        depthwise=False,
        dilated=False,
        act="silu",
        attn=None,
    ):
        super().__init__()
        hidden_channels = int(out_channels * bottleneck_expansion)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        self.reduction = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, 1, 0),
            nn.BatchNorm2d(hidden_channels),
        )
        self.expansion = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
        )
        if attn is None:
            self.attn = nn.Identity()
        elif attn == "SE":  # Squeeze & Excitation attention
            self.attn = SELayer(
                in_channels, out_channels, reduction=int(1 / bottleneck_expansion)
            )
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.expansion(self.reduction(self.conv1(x))))
        y = self.attn(y)
        if self.use_add:
            y = y + x
        return y


class Mobile_CSPLayer(nn.Module):
    """
    Rethinking Bottleneck Structure for Efficient Mobile Network Design (2020)
    Bottleneck -> Mobile_Bottleneck 사용
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        bottleneck_expansion=1.0,
        depthwise=False,
        dilated=False,
        act="silu",
        attn=None,
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Mobile_Bottleneck(
                hidden_channels,
                hidden_channels,
                shortcut,
                bottleneck_expansion,
                depthwise,
                dilated,
                act=act,
                attn=attn,
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Mobile_Bottleneck_k5x5(nn.Module):
    """
    MnasNet: Platform-Aware Neural Architecture Search for Mobile (2018)
    https://arxiv.org/pdf/1807.11626.pdf           figure 7 참조
    ===========================================================================
    3x3 * 2 은 5x5 와 receptive 필드는 같지만 연산속도에서는 5x5가 빠를수도 있다.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        bottleneck_expansion=0.5,
        depthwise=False,
        dilated=False,
        act="silu",
        attn=None,
    ):
        super().__init__()
        hidden_channels = int(out_channels * bottleneck_expansion)

        self.reduction = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, 1, 0),
            nn.BatchNorm2d(hidden_channels),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                hidden_channels, hidden_channels, 5, 1, 2, groups=hidden_channels
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )
        self.expansion = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
        )

        if attn is None:
            self.attn = nn.Identity()
        elif attn == "SE":  # Squeeze & Excitation attention
            self.attn = SELayer(
                in_channels, out_channels, reduction=int(1 / bottleneck_expansion)
            )
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv1(self.expansion(self.reduction(x)))
        y = self.attn(y)
        if self.use_add:
            y = y + x
        return y


class Mobile_CSPLayer_k5x5(nn.Module):
    """
    MnasNet: Platform-Aware Neural Architecture Search for Mobile (2018)
    Bottleneck -> Mobile_Bottleneck -> Mobile_Bottleneck_k5x5 사용
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        bottleneck_expansion=1.0,
        depthwise=False,
        dilated=False,
        act="silu",
        attn=None,
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Mobile_Bottleneck_k5x5(
                hidden_channels,
                hidden_channels,
                shortcut,
                bottleneck_expansion,
                depthwise,
                dilated,
                act=act,
                attn=attn,
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)
