#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from ..network_blocks import *


class PAFPN_add(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
        fpn_attn=None,
        expansion=0.5,
        bottleneck_expansion=1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
            attn=fpn_attn,
            expansion=expansion,
            bottleneck_expansion=bottleneck_expansion,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
            attn=fpn_attn,
            expansion=expansion,
            bottleneck_expansion=bottleneck_expansion,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
            attn=fpn_attn,
            expansion=expansion,
            bottleneck_expansion=bottleneck_expansion,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
            attn=fpn_attn,
            expansion=expansion,
            bottleneck_expansion=bottleneck_expansion,
        )

        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w3 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w4 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4

    def forward(self, features):

        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        # f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        w1 = nn.ReLU()(self.w1)
        f_out0 = (f_out0 * w1[0] + x1 * w1[1]) / (torch.sum(w1, dim=0) + self.epsilon)
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        # f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        w2 = nn.ReLU()(self.w2)
        f_out1 = (f_out1 * w2[0] + x2 * w2[1]) / (torch.sum(w2, dim=0) + self.epsilon)
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        # p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        w3 = nn.ReLU()(self.w3)
        p_out1 = (p_out1 * w3[0] + fpn_out1 * w3[1]) / (
            torch.sum(w3, dim=0) + self.epsilon
        )
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        # p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        w4 = nn.ReLU()(self.w4)
        p_out0 = (p_out0 * w4[0] + fpn_out0 * w4[1]) / (
            torch.sum(w4, dim=0) + self.epsilon
        )
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs


class YOLOPAFPN_add(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        backbone_dilated=False,
        act="silu",
        backbone_attn=None,
        fpn_attn=None,
        expansion=0.5,
        bottleneck_expansion=1.0,
    ):
        super().__init__()
        self.backbone = CSPDarknet(
            depth,
            width,
            depthwise=depthwise,
            dilated=backbone_dilated,
            act=act,
            attn=backbone_attn,
            bottleneck_expansion=bottleneck_expansion,
        )
        self.in_features = in_features
        self.in_channels = in_channels

        self.PAFPN_0 = PAFPN_add(
            depth=depth,
            width=width,
            in_features=in_features,
            in_channels=in_channels,
            depthwise=depthwise,
            act="silu",
            fpn_attn=fpn_attn,
            expansion=expansion,
            bottleneck_expansion=bottleneck_expansion,
        )
        self.PAFPN_1 = PAFPN_add(
            depth=depth,
            width=width,
            in_features=in_features,
            in_channels=in_channels,
            depthwise=depthwise,
            act="silu",
            fpn_attn=fpn_attn,
            expansion=expansion,
            bottleneck_expansion=bottleneck_expansion,
        )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]

        outputs = self.PAFPN_0(features)
        outputs = self.PAFPN_1(features)
        return outputs
