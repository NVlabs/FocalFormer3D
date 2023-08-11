plugin=True
plugin_dir='projects/mmdet3d_plugin/'

point_cloud_range = [-76.8, -76.8, -2, 76.8, 76.8, 4]
class_names = ['Car', 'Pedestrian', 'Cyclist']
voxel_size = [0.1, 0.1, 0.15]
out_size_factor = 8
evaluation = dict(interval=11)
dataset_type = 'WaymoDataset'
data_root = 'data/waymo/kitti_format'
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
img_scale = (800, 448)
num_views = 6
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

multistage_heatmap = 2
inter_channel = 128
extra_feat = True

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=6, use_dim=5),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample',
         db_sampler=dict(
             data_root=data_root,
             info_path=data_root + '/waymo_dbinfos_train_14split.pkl',
             rate=1.0,
             prepare=dict(
                 filter_by_difficulty=[-1],
                 filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
             classes=class_names,
             sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10),
             points_loader=dict(
                 type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=[0, 1, 2, 3, 4]))),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
    ),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=6, use_dim=5),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(800, 1333),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=6,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            load_interval=1,
            ann_file=data_root + '/waymo_infos_train_14split.pkl',
            split='training',
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/waymo_infos_val.pkl',
        split='training',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/waymo_infos_val.pkl',
        split='training',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))
model = dict(
    type='FocalFormer3D',
    freeze_img=True,
    freeze_pts=True,
    input_img=False,
    # img_backbone=dict(
    #     type='ResNet',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     frozen_stages=1,
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     norm_eval=True,
    #     style='pytorch'),
    # img_neck=dict(
    #     type='FPN',
    #     in_channels=[256, 512, 1024, 2048],
    #     out_channels=256,
    #     num_outs=5),
    pts_voxel_layer=dict(
        max_num_points=5,
        voxel_size=voxel_size,
        max_voxels=150000,
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(
        type='HardVFE',
        in_channels=5,
        # num_features=5,
        feat_channels=[64],
        with_distance=False,
        with_cluster_center=False,
        with_voxel_center=False,
        voxel_size=voxel_size,
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        point_cloud_range=point_cloud_range,
    ),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=64,
        sparse_shape=[41, 1536, 1536],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    imgpts_neck=dict(
        type='FocalEncoder',
        num_layers=multistage_heatmap,
        in_channels_img=256,
        in_channels_pts=sum([256, 256]),
        hidden_channel=inter_channel,
        bn_momentum=0.1,
        max_points_height=10,
        bias='auto',
        iterbev='bevfusionmb2',
        input_img=False,
        iterbev_wo_img=True,
        multistage_heatmap=multistage_heatmap,
        extra_feat=extra_feat,
    ),
    pts_bbox_head=dict(
        type='FocalDecoder',
        reuse_first_heatmap=True,
        extra_feat=extra_feat,
        roi_feats=14,
        roi_dropout_rate=0.1,
        roi_based_reg=True,
        roi_expand_ratio=1.2,
        heatmap_box=False,
        thin_heatmap_box=False,
        multiscale=True,
        multistage_heatmap=multistage_heatmap,
        mask_heatmap_mode='poscls',
        input_img=False,
        iterbev_wo_img=True,
        add_gt_groups=3,
        add_gt_groups_noise='box,1',
        add_gt_groups_noise_box='gtnoise',
        add_gt_pos_thresh=10.,
        add_gt_pos_boxnoise_thresh=0.75,
        gt_center_limit=5,
        bevpos=True,
        loss_weight_heatmap=1.,
        loss_weight_separate_heatmap=0.,
        loss_weight_separate_bbox=0.3,
        num_proposals=200,
        hidden_channel=inter_channel,
        num_classes=len(class_names),
        num_decoder_layers=2,
        num_heads=8,
        initialize_by_heatmap=True,
        nms_kernel_size=3,
        bn_momentum=0.1,
        activation='relu',
        classaware_reg=True,
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
            out_size_factor=out_size_factor,
            post_center_range=[-80, -80, -10.0, 80, 80, 10.0],
            score_threshold=0.0,
            code_size=8,
        ),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=2.0),
        loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        decoder_cfg=dict(
            type='DeformableDetrTransformerDecoder',
            num_layers=3,
            return_intermediate=False,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=inter_channel,
                        num_heads=8,
                        dropout=0.1),
                    dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=inter_channel,
                        num_levels=3,
                        num_points=4,
                        num_heads=8,)
                ],
                feedforward_channels=1024,
                ffn_dropout=0.1,
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=inter_channel,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                ),
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                    'ffn', 'norm')))
    ),
    train_cfg=dict(
        pts=dict(
            dataset='Waymo',
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(type='FocalLossCost', gamma=2, alpha=0.25, weight=0.6),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=2.0),
                iou_cost=dict(type='IoU3DCost', weight=2.0)
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[1536, 1536, 40],  # [x_len, y_len, 1]
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(
            dataset='Waymo',
            grid_size=[1536, 1536, 40],
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range[0:2],
            voxel_size=voxel_size[:2],
            nms_type=None,
        )))
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
total_epochs = 11
checkpoint_config = dict(interval=1, max_keep_ckpts=7)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = './work_dirs/DeformFormer3D_Waymo15_L/epoch_36.pth'
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 8)
find_unused_parameters = True

custom_hooks = [dict(type='Fading', fade_epoch=5)]
