import torch.nn as nn

from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
import torch
import numpy as np
import spconv.pytorch as spconv
import torch.nn.functional as F
from opencood.models.fuse_modules.self_attn import AttFusion


class PointPillarBaseMultiScaleTeacher(nn.Module):
    def __init__(self, args):
        super(PointPillarBaseMultiScaleTeacher, self).__init__()

        self.is_train = args['is_train']
        self.max_cav = args['max_cav']
        # Pillar VFE
        self.pillar_vfe_teacher = PillarVFE(args['pillar_vfe'],
                                    num_point_features=5,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter_teacher = PointPillarScatter(args['point_pillar_scatter'])
        # self.backbone_teacher = BaseBEVBackbone(args['base_bev_backbone'], 64)
        self.backbone_teacher = ResNetBEVBackbone(args['base_bev_backbone'], 64)

        # Used to down-sample the feature map for efficient computation
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv_teacher = DownsampleConv(args['shrink_header'])
        else:
            self.shrink_flag = False

        if args['compression']:
            self.compression = True
            self.naive_compressor_teacher = NaiveCompressor(256, args['compression'])
        else:
            self.compression = False

        self.multi_scale = args['fusion']['multi_scale']
        if self.multi_scale:
            layer_nums = args['base_bev_backbone']['layer_nums']
            num_filters = args['base_bev_backbone']['num_filters']
            self.num_levels = len(layer_nums)
            self.fuse_modules_teacher = nn.ModuleList()
            for idx in range(self.num_levels):
                fuse_network = AttFusion(num_filters[idx])
                self.fuse_modules_teacher.append(fuse_network)
        else:
            self.fuse_modules_teacher = AttFusion(args['fusion']['in_channels'])


        self.cls_head_teacher = nn.Conv2d(args['head_dim'], args['anchor_number'], kernel_size=1)
        self.reg_head_teacher = nn.Conv2d(args['head_dim'], 7 * args['anchor_number'], kernel_size=1)

        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay.
        """

        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False
            
    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, data_dict):
        record_len = data_dict['record_len']


        voxel_features_paint = data_dict['processed_lidar_paint']['voxel_features']
        voxel_coords_paint = data_dict['processed_lidar_paint']['voxel_coords']
        voxel_num_points_paint = data_dict['processed_lidar_paint']['voxel_num_points']

        batch_dict_teacher = {'voxel_features': voxel_features_paint,
                              'voxel_coords': voxel_coords_paint,
                              'voxel_num_points': voxel_num_points_paint,
                              'record_len': record_len}

        batch_dict_teacher = self.pillar_vfe_teacher(batch_dict_teacher)
        # n, c -> N, C, H, W
        batch_dict_teacher = self.scatter_teacher(batch_dict_teacher)

        double_supervise_feature = batch_dict_teacher['spatial_features']

        double_record_len = torch.tensor([i * 2 for i in record_len])
        split_x = self.regroup(double_supervise_feature, double_record_len)
        object_feature_list = []
        supervise_feature_list = []
        for i in range(len(record_len)):
            supervise_feature_list.append(split_x[i][:record_len[i], :, :, :])
            object_feature_list.append(split_x[i][record_len[i]:, :, :, :])
        object_feature = torch.cat(object_feature_list, dim=0)
        supervise_feature = torch.cat(supervise_feature_list, dim=0)

        batch_dict_teacher['spatial_features'] = supervise_feature

        spatial_features_teacher = batch_dict_teacher['spatial_features']
        if self.compression:
            spatial_features_teacher = self.naive_compressor_teacher(spatial_features_teacher)

        # multiscale fusion
        feature_list_teacher = self.backbone_teacher.get_multiscale_feature(spatial_features_teacher)

        fused_feature_list_teacher = []
        for i, fuse_module in enumerate(self.fuse_modules_teacher):
            fused_feature_list_teacher.append(fuse_module(feature_list_teacher[i], record_len))
        fused_feature_teacher = self.backbone_teacher.decode_multiscale_feature(fused_feature_list_teacher)

        if self.shrink_flag:
            fused_feature_teacher = self.shrink_conv_teacher(fused_feature_teacher)

        psm_teacher = self.cls_head_teacher(fused_feature_teacher)
        rm_teacher = self.reg_head_teacher(fused_feature_teacher)

        output_dict = {'psm': psm_teacher, 'rm': rm_teacher, }
        
        # exit()


        return output_dict
