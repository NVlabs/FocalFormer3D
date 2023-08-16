# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FocalFormer3D/blob/main/LICENSE

import torch
from mmcv.cnn import build_conv_layer
from torch import nn
from mmdet3d.models.builder import NECKS
from projects.mmdet3d_plugin.models.utils.encoder_utils import *
from projects.mmdet3d_plugin.models.utils.time_utils import T
import torchvision.models.resnet as resnet

class FocalEncoderLayer(nn.Module):
    def __init__(self, hidden_channel, iterbev='bevfusion', max_points_height=5, iterbev_wo_img=False, 
            multiscale_outputs=False, layer_id=None, iter_bev_cam=None, need_projbev=True):
        super(FocalEncoderLayer, self ).__init__()

        self.iterbev = iterbev
        self.iterbev_wo_img = iterbev_wo_img
        self.multiscale_outputs = multiscale_outputs
        self.layer_id = layer_id
        self.iter_bev_cam = iter_bev_cam

        self.need_projbev = need_projbev
        if self.iterbev in ['bevfusion', 'bevfusionmb2']:
            if need_projbev and (not self.iter_bev_cam or self.layer_id == 0):
                if not self.iterbev_wo_img:
                    self.I2P_block = I2P(hidden_channel, hidden_channel, 0.1, max_points_height=max_points_height)
            else:
                self.I2P_block = None

        if self.iterbev == 'bevfusionmb2':
            import torchvision.models.mobilenetv2 as mobilenetv2
            self.P_IML = mobilenetv2.InvertedResidual(hidden_channel, hidden_channel, stride=1, expand_ratio=2, norm_layer=nn.BatchNorm2d)
            self.P_out_proj = mobilenetv2.InvertedResidual(2*hidden_channel, hidden_channel, stride=1, expand_ratio=1, norm_layer=nn.BatchNorm2d)
            self.P_integration = mobilenetv2.InvertedResidual(2*hidden_channel, hidden_channel, stride=1, expand_ratio=1, norm_layer=nn.BatchNorm2d)
        elif self.iterbev == 'bevfusion':
            self.P_IML = LocalContextAttentionBlock(hidden_channel, hidden_channel, 9)
            self.P_out_proj = ConvBNReLU(2 * hidden_channel, hidden_channel, kernel_size = 1, norm_layer=nn.BatchNorm2d, activation_layer=None)
            self.P_integration = ConvBNReLU(2 * hidden_channel, hidden_channel, kernel_size = 1, norm_layer=nn.BatchNorm2d, activation_layer=None)
        else:
            self.iterbev_conv = ConvBNReLU(hidden_channel, hidden_channel, kernel_size = 3, norm_layer=nn.BatchNorm2d, activation_layer=None)

        if self.iterbev_wo_img:
            self.iterimg_conv = None
        else:
            self.iterimg_conv = nn.Sequential(
                resnet.BasicBlock(hidden_channel, hidden_channel, norm_layer=nn.BatchNorm2d),
            )
        
    def forward(self, img_feat, lidar_feat, img_metas, extra_args=None):
        batch_size = lidar_feat.shape[0]
        if not self.iterbev_wo_img:
            BN, I_C, I_H, I_W = img_feat.shape
        
        if self.iterbev in ['bevfusion', 'bevfusionmb2']:
            if not self.iterbev_wo_img:
                if self.iter_bev_cam:
                    if self.layer_id == 0 and self.need_projbev:
                        I2P_feat = self.I2P_block(lidar_feat, img_feat.view(batch_size, -1, I_C, I_H, I_W), img_metas) # same shape as lidar_feat
                        img_feat = I2P_feat
                    else:
                        I2P_feat = img_feat
                else:
                    I2P_feat = self.I2P_block(lidar_feat, img_feat.view(batch_size, -1, I_C, I_H, I_W), img_metas) # same shape as lidar_feat
            else:
                I2P_feat = lidar_feat

        if self.iterbev == 'bevfusion':
            P2P_feat = self.P_IML(lidar_feat, lidar_feat)
            P_Aug_feat = self.P_out_proj(torch.cat((I2P_feat, P2P_feat),dim=1))
            new_lidar_feat = self.P_integration(torch.cat((P_Aug_feat, lidar_feat),dim=1))
        elif self.iterbev in 'bevfusionmb2':
            P2P_feat = self.P_IML(lidar_feat)
            P_Aug_feat = self.P_out_proj(torch.cat((I2P_feat, P2P_feat),dim=1))
            new_lidar_feat = self.P_integration(torch.cat((P_Aug_feat, lidar_feat),dim=1))
        else:
            new_lidar_feat = self.iterbev_conv(lidar_feat)

        if not self.iterimg_conv:
            new_img_feat = None
        else:
            new_img_feat = self.iterimg_conv(img_feat)
        
        return new_img_feat, new_lidar_feat

@NECKS.register_module()
class FocalEncoder(nn.Module):
    def __init__(self,
                num_layers=2,
                in_channels_img=64,
                in_channels_pts=128 * 3,
                hidden_channel=128,
                bn_momentum=0.1,
                bias='auto',
                iterbev='bevfusion',
                max_points_height=5,
                multistage_heatmap=False,
                input_img=True,
                input_pts=True,
                iterbev_wo_img=False,
                extra_feat=False,
                iter_bev_cam=False,
                cam_lss=False,
                newbevpool=False,
                pc_range=None,
                img_scale=None,
                ):
        super(FocalEncoder, self).__init__()

        self.iterbev_wo_img = iterbev_wo_img
        self.iterbev = iterbev
        self.iter_bev_cam = iter_bev_cam

        self.multistage_heatmap = multistage_heatmap
        self.input_pts = input_pts
        if self.input_pts:
            self.shared_conv_pts = build_conv_layer(
                dict(type='Conv2d'),
                in_channels_pts,
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
            )

        self.input_img = input_img
        self.cam_proj_type = cam_lss
        if self.input_img:
            if cam_lss:
                from .lss import LiftSplatShoot
                self.cam_lss = LiftSplatShoot(grid=0.6, inputC=256, outputC=hidden_channel, camC=64, 
                    pc_range=pc_range, img_scale=img_scale, downsample=4, newbevpool=newbevpool)
            else:
                self.cam_lss = None
                self.shared_conv_img = build_conv_layer(
                    dict(type='Conv2d'),
                    in_channels_img,
                    hidden_channel,
                    kernel_size=3,
                    padding=1,
                    bias=bias,
                )

        self.num_layers = num_layers if num_layers else 0
        self.fusion_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.fusion_blocks.append(
                FocalEncoderLayer(hidden_channel, iterbev=iterbev, 
                    max_points_height=max_points_height, iterbev_wo_img=self.iterbev_wo_img, 
                    multiscale_outputs=False, layer_id=i, iter_bev_cam=iter_bev_cam, need_projbev=not cam_lss),
            )

        self.extra_feat = extra_feat
        if self.extra_feat:
            self.extra_output = ConvBNReLU(hidden_channel, hidden_channel, kernel_size = 3, norm_layer=nn.BatchNorm2d, activation_layer=None)

        self.bn_momentum = bn_momentum
        self.init_weights()

    def init_weights(self):
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def forward(self, img_feats, pts_feats, img_metas):
        batch_size = len(img_metas)
        if self.input_img:
            if self.cam_proj_type:
                if self.cam_proj_type == 'proj':
                    new_img_feat = self.cam_lss(img_feats.view(batch_size, -1, *img_feats.shape[-3:]), img_metas=img_metas)
                else:
                    rots = []
                    trans = []
                    for sample_idx in range(batch_size):
                        rot_list = []
                        trans_list = []
                        for mat in img_metas[sample_idx]['lidar2img']:  
                            mat = torch.Tensor(mat).cuda()
                            inverse_mat = mat.inverse()
                            rot_list.append(inverse_mat[:3, :3])
                            trans_list.append(inverse_mat[:3, 3].view(-1))
                        rot_list = torch.stack(rot_list, dim=0)
                        trans_list = torch.stack(trans_list, dim=0)
                        rots.append(rot_list)
                        trans.append(trans_list)
                    rots = torch.stack(rots)
                    trans = torch.stack(trans)
                    new_img_feat, depth = self.cam_lss(img_feats.view(batch_size, -1, *img_feats.shape[-3:]), rots=rots, trans=trans, img_metas=img_metas)

                if not self.input_pts and not self.multistage_heatmap:
                    return None, [new_img_feat, new_img_feat]
            else:
                new_img_feat = self.shared_conv_img(img_feats)
        else:
            new_img_feat = None
        
        if self.input_pts:
            new_pts_feat = self.shared_conv_pts(pts_feats)
        else:
            new_pts_feat = torch.zeros((batch_size, 128, 180, 180), device='cuda') # set empty for train image only (nuscenes bev feature size)
        pts_feat_conv = new_pts_feat.clone()

        if self.input_img or self.iterbev_wo_img:
            multistage_ptsfeats = []
            extra_args = {}
            for i in range(self.num_layers):
                new_img_feat, new_pts_feat = self.fusion_blocks[i](new_img_feat, new_pts_feat, img_metas, extra_args)
                if self.multistage_heatmap:
                    multistage_ptsfeats.append(new_pts_feat)
            if self.multistage_heatmap:
                new_pts_feat = multistage_ptsfeats
                if self.extra_feat:
                    new_pts_feat.append( self.extra_output(new_pts_feat[-1]) )
            return new_img_feat, [pts_feat_conv, new_pts_feat]
        else:
            return None, [new_pts_feat, None]
