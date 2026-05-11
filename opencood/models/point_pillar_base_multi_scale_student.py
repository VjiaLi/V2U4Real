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
from opencood.models.fuse_modules.commucation import Communication
from opencood.models.sub_modules.pcr import PCR

import matplotlib.pyplot as plt
def show_heatmaps(matrices,path=None, figsize=(5, 5),
                  cmap='Blues'):
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)

    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
                pcm = ax.imshow(matrix, cmap=cmap)
    plt.savefig(path,dpi=1300)


class PointPillarBaseMultiScaleStudent(nn.Module):
    def __init__(self, args):
        super(PointPillarBaseMultiScaleStudent, self).__init__()

        self.is_train = args['is_train']
        self.max_cav = args['max_cav']
        # Pillar VFE
        ##########################################student#########################################
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)

        # Used to down-sample the feature map for efficient computation
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        else:
            self.shrink_flag = False

        if args['compression']:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])
        else:
            self.compression = False

        self.multi_scale = args['fusion']['multi_scale']
        if self.multi_scale:
            layer_nums = args['base_bev_backbone']['layer_nums']
            num_filters = args['base_bev_backbone']['num_filters']
            self.num_levels = len(layer_nums)
            self.fuse_modules = nn.ModuleList()
            for idx in range(self.num_levels):
                fuse_network = AttFusion(num_filters[idx])
                self.fuse_modules.append(fuse_network)
        else:
            self.fuse_modules = AttFusion(args['fusion']['in_channels'])


        self.cls_head = nn.Conv2d(args['head_dim'], args['anchor_number'], kernel_size=1)
        self.reg_head = nn.Conv2d(args['head_dim'], 7 * args['anchor_number'], kernel_size=1)

        self.pillar_vfe_teacher = PillarVFE(args['pillar_vfe'],
                                            num_point_features=5,
                                            voxel_size=args['voxel_size'],
                                            point_cloud_range=args['lidar_range'])
        self.scatter_teacher = PointPillarScatter(args['point_pillar_scatter'])
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

        self.object_mask = Communication(0.501)
        self.backbone_fix_ours()
        self.pcr = PCR(256)
        
        self.voxel_size = args['voxel_size']
        self.lidar_range = args['lidar_range']
        

    def mask_offset_loss(self, gen_offset, gen_mask, gt, grid):

        # grid 
        gt_mask = gt.sum(1) != 0
        count_pos = gt_mask.sum()
        count_neg = (~gt_mask).sum()
        beta = count_neg/count_pos
        loss = F.binary_cross_entropy_with_logits(gen_mask[:,0],gt_mask.float(),pos_weight= beta)

        grid = grid * gt_mask[:,None]
        gt = gt[:,:3] - grid
        gt_ind = gt != 0
        com_loss = F.l1_loss(gen_offset[gt_ind],gt[gt_ind])
        return loss, com_loss 

    def backbone_fix_ours(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe_teacher.parameters():
            p.requires_grad = False

        for p in self.backbone_teacher.parameters():
            p.requires_grad = False

        for p in self.fuse_modules_teacher.parameters():
            p.requires_grad = False

        for p in self.shrink_conv_teacher.parameters():
            p.requires_grad = False

        for p in self.cls_head_teacher.parameters():
            p.requires_grad = False

        for p in self.reg_head_teacher.parameters():
            p.requires_grad = False

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
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)

        spatial_features = batch_dict['spatial_features']

        

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

        object_mask = self.object_mask(object_feature)

        batch_dict_teacher['spatial_features'] = supervise_feature

        spatial_features_teacher = batch_dict_teacher['spatial_features']
        

        if self.compression:
            spatial_features = self.naive_compressor(spatial_features)

        # multiscale fusion
        feature_list = self.backbone.get_multiscale_feature(spatial_features)

        if self.compression:
            spatial_features_teacher = self.naive_compressor_teacher(spatial_features_teacher)

        # multiscale fusion
        feature_list_teacher = self.backbone_teacher.get_multiscale_feature(spatial_features_teacher)

        fused_feature_list = []
        fused_feature_list_teacher = []
        for i, fuse_module in enumerate(self.fuse_modules):
            fused_feature_list.append(fuse_module(feature_list[i], record_len))
        fused_feature = self.backbone.decode_multiscale_feature(fused_feature_list)

        rec_loss = None
        for i, fuse_module_teacher in enumerate(self.fuse_modules_teacher):
            fused_feature_list_teacher.append(fuse_module_teacher(feature_list_teacher[i], record_len))
            
            B,C,H,W = feature_list_teacher[i].shape
            deformable_object_mask = F.interpolate(object_mask, size=(H, W),mode='bilinear', align_corners=False)  

            masked_feature = deformable_object_mask*feature_list[i]
            masked_feature_teacher = deformable_object_mask*feature_list_teacher[i]
            
            if rec_loss is None:
                rec_loss = F.mse_loss(masked_feature, masked_feature_teacher)
            else:
                rec_loss = rec_loss + F.mse_loss(masked_feature, masked_feature_teacher)
                

        fused_feature_teacher = self.backbone_teacher.decode_multiscale_feature(fused_feature_list_teacher)

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        if self.shrink_flag:
            fused_feature_teacher = self.shrink_conv_teacher(fused_feature_teacher)


        B, C, H, W = fused_feature_teacher.shape
        object_mask = F.interpolate(object_mask, size=(H, W), mode='bilinear', align_corners=False)
        split_object_mask = self.regroup(object_mask, record_len)
        out = []
        for xx in split_object_mask:
            xx = torch.any(xx, dim=0).unsqueeze(0)
            out.append(xx)
        object_mask = torch.vstack(out)
        
        masked_fused_feature = object_mask * fused_feature


        masked_fused_feature_teacher = object_mask * fused_feature_teacher
        fused_loss = F.mse_loss(masked_fused_feature, masked_fused_feature_teacher)


        reconstruction_voxels = data_dict['early_fusion_processed_lidar']['voxel_features']
        reconstruction_coordinates = data_dict['early_fusion_processed_lidar']['voxel_coords']
        reconstruction_voxel_num_points = data_dict['early_fusion_processed_lidar']['voxel_num_points']

        sparse_shape = np.array([10,400,504]).astype('int64')
        coors = reconstruction_coordinates.int()
        input_feature = (reconstruction_voxels[:, :, : 10].sum( dim=1, keepdim=False
                    ) /reconstruction_voxel_num_points.view(-1, 1)).contiguous()

        reconstruction_gt = spconv.SparseConvTensor(input_feature, coors, sparse_shape, len(record_len))
        reconstruction_gt = reconstruction_gt.dense() 
        gen_offset,gen_mask = self.pcr(fused_feature)

        N,_,D,H,W = gen_offset.shape
            
        zs, ys, xs = torch.meshgrid([torch.arange(0,D),torch.arange(0, H), torch.arange(0, W)])   
        xs = xs * self.voxel_size[0] + (self.voxel_size[0]/2 + self.lidar_range[0])
        ys = ys * self.voxel_size[1] + (self.voxel_size[1]/2 + self.lidar_range[1])
        zs = zs * self.voxel_size[2] + (self.voxel_size[2]/2 + self.lidar_range[2])
            
        grid = torch.cat([xs[None],ys[None],zs[None]],0)[None].repeat(N,1,1,1,1).to(gen_offset)
        
        mask_loss, offset_loss = self.mask_offset_loss(gen_offset, gen_mask, reconstruction_gt, grid)

        psm_teacher = self.cls_head_teacher(fused_feature_teacher)
        rm_teacher = self.reg_head_teacher(fused_feature_teacher)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        kl_loss_mean = nn.KLDivLoss(size_average=True, reduce=True)
        N, C, H, W = rm_teacher.shape
        teacher_rm = rm_teacher.permute(0, 2, 3, 1).reshape(N * H * W, C)
        student_rm = rm.permute(0, 2, 3, 1).reshape(N * H * W, C)
        kd_loss_rm = kl_loss_mean(F.log_softmax(student_rm, dim=1), F.softmax(teacher_rm, dim=1))


        N, C, H, W = psm_teacher.shape
        teacher_psm = psm_teacher.permute(0, 2, 3, 1).reshape(N * H * W, C)
        student_psm = psm.permute(0, 2, 3, 1).reshape(N * H * W, C)
        kd_loss_psm = kl_loss_mean(F.log_softmax(student_psm, dim=1), F.softmax(teacher_psm, dim=1))

        kd_loss = kd_loss_rm + kd_loss_psm

        late_loss = rec_loss + fused_loss + kd_loss + mask_loss + offset_loss

        
        print("REC Loss: %.4f || Fused Loss: %.4f || KD Loss: %.4f|| mask Loss: %.4f|| off Loss: %.4f||" %
              (rec_loss.item(), fused_loss.item(), kd_loss.item(),mask_loss.item(),offset_loss.item()))
        
        output_dict = {'psm': psm, 'rm': rm, }
        
        if self.is_train:
            output_dict.update({     
                'late_loss': late_loss
            })


        return output_dict
