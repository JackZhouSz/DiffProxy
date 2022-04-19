# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d

import torch.nn as nn
import training.vgg as vgg

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, real_img, real_c, gen_z): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D

        # l1 loss
        self.l1_criterion = nn.L1Loss()

        # VGG loss
        self.vgg19 = vgg.get_vgg19()
        self.style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
        self.style_weights = [1 / n ** 2 for n in [64, 128, 256, 512, 512]]
        self.feature_layers = ['r22', 'r32', 'r42']
        self.feature_weights = [1e-3, 1e-3, 1e-3]
        self.criterion_feature = vgg.WeightedLoss(self.feature_weights, metric='l1')
        self.criterion_style = vgg.WeightedLoss(self.style_weights, metric='l1')
        self.feat_w = 10.0
        self.style_w = 1.0

    def run_G(self, z, c, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        img = self.G.synthesis(ws, update_emas=update_emas)
        return img, ws

    def accumulate_gradients(self, real_img, real_c, gen_z):
        with torch.autograd.profiler.record_function('Gmain_forward'):
            gen_img, _gen_ws = self.run_G(gen_z, real_c)

            # l1 loss
            loss_l1 = self.l1_criterion(gen_img, real_img)
            training_stats.report('Loss/L1/loss', loss_l1)

            # VGG loss
            real_feat, real_style = self.vgg19.extract_features(real_img, self.feature_layers, self.style_layers,
                                                                detach_features=True, detach_styles=True)
            recon_feat, recon_style = self.vgg19.extract_features(gen_img, self.feature_layers, self.style_layers)
            feature_loss = self.criterion_feature(real_feat, recon_feat) * self.feat_w
            style_loss = self.criterion_style(real_style, recon_style) * self.style_w
            training_stats.report('Loss/Feat/loss', feature_loss)
            training_stats.report('Loss/Style/loss', style_loss)

            loss = loss_l1 + feature_loss + style_loss
            training_stats.report('Loss/G/loss', loss)

        with torch.autograd.profiler.record_function('Gmain_backward'):
            loss.backward()

#----------------------------------------------------------------------------
