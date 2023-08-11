# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FocalFormer3D/blob/main/LICENSE

import mmcv
import torch
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
from os import path as osp
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result, show_result)
from mmdet3d.ops import Voxelization
from mmdet.core import multi_apply
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
import pdb

from projects.mmdet3d_plugin.models.utils.time_utils import T
from projects.mmdet3d_plugin.core.post_processing.merge_augs import merge_aug_bboxes_3d

@DETECTORS.register_module()
class FocalFormer3D(MVXTwoStageDetector):
    def __init__(self,
                 freeze_img=False,
                 freeze_img_level=None,
                 freeze_camlss=False,
                 freeze_pts=False,
                 trainneck_ms=False,
                 train_middle_encoder=False,
                 pts_pillar_layer=None,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 imgpts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 input_img=True,
                 use_grid_mask=False,
                 input_pts=True,
                 init_cfg=None):
        super(FocalFormer3D, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained, init_cfg)
        if pts_pillar_layer:
            self.pts_pillar_layer = Voxelization(**pts_pillar_layer)
        self.freeze_img_level = freeze_img_level
        self.freeze_camlss = freeze_camlss
        self.imgpts_neck = builder.build_neck(imgpts_neck)
        
        self.freeze_img = freeze_img
        self.freeze_pts = freeze_pts
        self.trainneck_ms = trainneck_ms
        self.train_middle_encoder = train_middle_encoder

        self.input_img = input_img
        self.input_pts = input_pts

        self.use_grid_mask = use_grid_mask
        if self.use_grid_mask:
            from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        
        self.apply_dynamic_voxelize = 'Dynamic' in pts_voxel_encoder['type']
        
    def init_weights(self):
        """Initialize model weights."""
        super(FocalFormer3D, self).init_weights()
        if self.input_img and self.freeze_img:
            if self.with_img_backbone:
                if self.freeze_img_level:
                    param_levels = [['conv1', 'bn1'], ['layer1'], ['layer2'], ['layer3'], ['layer4']]
                    for i in range(self.freeze_img_level):
                        for pn in param_levels[i]:
                            print(f'freezing image {pn}')
                            for param in self.img_backbone.get_submodule(pn).parameters():
                                param.requires_grad = False
                else:
                    for param in self.img_backbone.parameters():
                        param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False
            if self.freeze_camlss and hasattr(self.imgpts_neck, 'cam_lss'):
                for param in self.imgpts_neck.cam_lss.parameters():
                    param.requires_grad = False

        if self.freeze_pts:
            for name, param in self.named_parameters():
                if 'pts' in name and 'pts_bbox_head' not in name and 'imgpts_neck' not in name:
                    if self.trainneck_ms:
                        if 'pts_backbone' in name: continue
                        if 'pts_neck' in name: continue
                    if self.train_middle_encoder:
                        if 'pts' in name: continue
                    param.requires_grad = False
            def fix_bn(m):
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False
            if not self.train_middle_encoder:
                self.pts_voxel_layer.apply(fix_bn)
                self.pts_voxel_encoder.apply(fix_bn)
                self.pts_middle_encoder.apply(fix_bn)
            if not self.trainneck_ms:
                self.pts_backbone.apply(fix_bn)
                if self.with_pts_neck:
                    self.pts_neck.apply(fix_bn)

        if not self.input_pts:
            self.voxelize=None
            self.pts_voxel_encoder=None
            self.pts_middle_encoder=None
            self.pts_backbone=None
            if self.with_pts_neck:
                self.pts_neck=None

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask and self.training:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img.float())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    def extract_pts_feat(self, pts, img_feats=None, img_metas=None):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        if self.apply_dynamic_voxelize:
            voxels, coors = self.dynamic_voxelize(pts)
            voxel_features, feature_coors = self.pts_voxel_encoder(voxels, coors)
            batch_size = coors[-1, 0] + 1
            coors = feature_coors # update
        else:
            voxels, num_points, coors = self.voxelize(pts,voxel_type='voxel')
            voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
            batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        else:
            x = [x]

        return x
    
    def extract_feat(self, points, img, img_metas):
        if self.input_img:
            img_feats = self.extract_img_feat(img, img_metas)
        else:
            img_feats = [None,]
        if self.input_pts:
            pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        else:
            pts_feats = [None,]
        new_img_feat, new_pts_feat = self.imgpts_neck(img_feats[0], pts_feats[0], img_metas)
        return (new_img_feat, new_pts_feat)

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points, voxel_type='voxel'):
        assert voxel_type=='voxel' or voxel_type=='pillar'
        voxels, coors, num_points = [], [], []
        for res in points:
            if voxel_type == 'voxel':
                res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            elif voxel_type == 'pillar':
                res_voxels, res_coors, res_num_points = self.pts_pillar_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    @torch.no_grad()
    @force_fp32()
    def dynamic_voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.pts_voxel_layer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(pts_feats, img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_feats, img_metas, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def simple_test_pts(self, x, x_img, img_metas, rescale=False, gt_bboxes_3d=None, gt_labels_3d=None, **kwargs):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, x_img, img_metas, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, **kwargs)
        if True:
            bbox_list = self.pts_bbox_head.get_bboxes(
                outs, img_metas, rescale=rescale)
        else:
            bbox_list = self.pts_bbox_head.get_heatmap_bboxes(
                outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, points, img_metas, img=None, rescale=False, gt_bboxes_3d=None, gt_labels_3d=None):
        """Test function without augmentaiton."""
        # with T('time', enable=True, sync=True, record=True):
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            pts_feats, img_feats, img_metas, rescale=rescale, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d)

        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        precompute=False

        if not precompute:
            print('Precomputing aug_test ...')
            img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)

            bbox_list = dict()
            if pts_feats and self.with_pts_bbox:
                bbox_pts = self.aug_test_pts(pts_feats, img_feats, img_metas, rescale=rescale)
                bbox_list.update(pts_bbox=bbox_pts)
        else:
            print('Using precomputed results ...')
            bbox_list = dict()
            bbox_pts = self.aug_test_pts(None, None, img_metas, rescale=rescale)
            bbox_list.update(pts_bbox=bbox_pts)
        return [bbox_list]

    def aug_test_pts(self, xs, x_imgs, img_metas, rescale=False):
        """Test function of point cloud branch with augmentaiton."""
        if xs is not None:
            # only support aug_test for one sample
            aug_bboxes = []
            for x, x_img, img_meta in zip(xs, x_imgs, img_metas):
                outs = self.pts_bbox_head(x, x_img, img_meta)
                bbox_list = self.pts_bbox_head.get_bboxes(
                    outs, img_meta, rescale=rescale)
                bbox_list = [
                    dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                    for bboxes, scores, labels in bbox_list
                ]
                aug_bboxes.append(bbox_list[0])
            # after merging, bboxes will be rescaled to the original image size
            merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                                self.pts_bbox_head.test_cfg)
        else:
            merged_bboxes = merge_aug_bboxes_3d(None, img_metas,
                                                self.pts_bbox_head.test_cfg)

        return merged_bboxes
