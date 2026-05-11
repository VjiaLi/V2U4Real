import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.models.fuse_modules.self_attn import ScaledDotProductAttention

def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x

def normalize_pairwise_tfm(pairwise_t_matrix, H, W, discrete_ratio, downsample_rate=1):
    """
    normalize the pairwise transformation matrix to affine matrix need by torch.nn.functional.affine_grid()

    Parameters
    ----------
        pairwise_t_matrix: torch.tensor
            [B, L, L, 4, 4], B batchsize, L max_cav
        H: num.
            Feature map height
        W: num.
            Feature map width
        discrete_ratio * downsample_rate: num.
            One pixel on the feature map corresponds to the actual physical distance

    Returns
    -------
        normalized_affine_matrix: torch.tensor
            [B, L, L, 2, 3], B batchsize, L max_cav
    """
    pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
    pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
    pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
    pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (downsample_rate * discrete_ratio * W) * 2
    pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (downsample_rate * discrete_ratio * H) * 2

    normalized_affine_matrix = pairwise_t_matrix

    return normalized_affine_matrix

def warp_affine_simple(
        src, M, dsize,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False):
    """
    
    """
    B, C, H, W = src.size()
    grid = F.affine_grid(M, [B, C, dsize[0], dsize[1]],
                         align_corners=align_corners).to(src)
    return F.grid_sample(src, grid, align_corners=align_corners)

class Att_w_Warp(nn.Module):
    def __init__(self, feature_dims):
        super(Att_w_Warp, self).__init__()
        self.att = ScaledDotProductAttention(feature_dims)

    def forward(self, xx, record_len, normalized_affine_matrix, data_dict=None):
        _, C, H, W = xx.shape
        B, L = normalized_affine_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            t_matrix = normalized_affine_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            x = warp_affine_simple(batch_node_features[b], t_matrix[i, :, :, :], (H, W))
            cav_num = x.shape[0]
            x = x.view(cav_num, C, -1).permute(2, 0, 1) #  (H*W, cav_num, C), perform self attention on each pixel.
            h = self.att(x, x, x)
            h = h.permute(1, 2, 0).view(cav_num, C, H, W)[0, ...]  # C, W, H before
            out.append(h)

        out = torch.stack(out)
        return out
    
    def forward_debug(self, xx, record_len, normalized_affine_matrix, data_dict=None):
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        _, C, H, W = xx.shape
        B, L = normalized_affine_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch

        scenario_names = []
        timestamps = []
        
        if data_dict is not None:
            if 'scenario_name' in data_dict:
                scenario_names = data_dict['scenario_name'] if isinstance(data_dict['scenario_name'], list) else [data_dict['scenario_name']] * B
            if 'timestamp' in data_dict:
                timestamps = data_dict['timestamp'] if isinstance(data_dict['timestamp'], list) else [data_dict['timestamp']] * B
        
        if not scenario_names:
            scenario_names = [f"scenario_{b}" for b in range(B)]
        if not timestamps:
            timestamps = [f"timestamp_{b}" for b in range(B)]
        
        save_dir = "opencood/logs/pj_later_debug"
        os.makedirs(save_dir, exist_ok=True)
        
        for b in range(B):
            scenario_name = scenario_names[b] if b < len(scenario_names) else f"scenario_{b}"
            timestamp = timestamps[b] if b < len(timestamps) else f"timestamp_{b}"
            N = record_len[b]
            t_matrix = normalized_affine_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            x = warp_affine_simple(batch_node_features[b], t_matrix[i, :, :, :], (H, W))
            cav_num = x.shape[0]
            # debug
            x_np = x.detach().cpu().numpy()
            for j in range(cav_num):
                plt.imshow(x_np[j].sum(axis=0))
                h, w = x_np[j].sum(axis=0).shape
                filename = f"{scenario_name}_{timestamp}_H{h}_W{w}_batch{b}_cav{j}.png"
                plt.savefig(os.path.join(save_dir, filename), dpi=400)
                plt.close()
            
            x = x.view(cav_num, C, -1).permute(2, 0, 1) #  (H*W, cav_num, C), perform self attention on each pixel.
            h = self.att(x, x, x)
            h = h.permute(1, 2, 0).view(cav_num, C, H, W)[0, ...]  # C, W, H before
            h_np = h.detach().cpu().numpy()   # (C, H, W)
            fusion_map = h_np.sum(axis=0)     # (H, W)
            plt.imshow(fusion_map)
            fh, fw = fusion_map.shape

            fusion_filename = f"{scenario_name}_{timestamp}_FUSION_H{fh}_W{fw}_batch{b}.png"
            plt.savefig(os.path.join(save_dir, fusion_filename), dpi=400)
            plt.close()
            out.append(h)

        out = torch.stack(out)
        return out