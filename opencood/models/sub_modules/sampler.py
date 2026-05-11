import torch
from torch import nn
import random
import matplotlib.pyplot as plt
import numpy as np

def show_heatmaps(matrices,path=None, figsize=(5, 5),
                  cmap='Reds'):
    """显示矩阵热图"""
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)

    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
                pcm = ax.imshow(matrix, cmap=cmap)
    # fig.colorbar(pcm, ax=axes, shrink=0.6)
    # fig.canvas.set_window_title(titles)
    plt.savefig(path,dpi=300)
    # plt.show()

class SortSampler(nn.Module):

    def __init__(self, topk_ratio, input_dim, score_pred_net='2layer-fc-256'):
        super().__init__()
        self.topk_ratio = topk_ratio
        # self.topk_ratio = random.uniform(0.05,0.25)
        if score_pred_net == '2layer-fc-256':
            self.score_pred_net = nn.Sequential(nn.Conv2d(input_dim, input_dim, 1),
                                                 nn.ReLU(),
                                                 nn.Conv2d(input_dim, 1, 1))
        elif score_pred_net == '2layer-fc-32':
            self.score_pred_net = nn.Sequential(nn.Conv2d(input_dim, 32, 1),
                                                 nn.ReLU(),
                                                 nn.Conv2d(32, 1, 1))
        elif score_pred_net == '1layer-fc':
            self.score_pred_net = nn.Conv2d(input_dim, 1, 1)
        else:
            raise ValueError

        self.norm_feature = nn.LayerNorm(input_dim,elementwise_affine=False)
        self.v_proj = nn.Linear(input_dim, input_dim)

    def forward(self, src, pos_embed, sample_ratio,dis_priority):

        bs,c ,h, w  = src.shape
        #各位置的分数
        src_dis = dis_priority*src.permute(1,0,2,3)
        src_dis = src_dis.permute(1,0,2,3).float() 
        sample_weight = self.score_pred_net(src_dis).sigmoid().view(bs,-1)
        # sample_weight[mask] = sample_weight[mask].clone() * 0.
        # sample_weight.data[mask] = 0.
        sample_weight_clone = sample_weight.clone().detach()

        if sample_ratio==None:
            sample_ratio = self.topk_ratio
        ##max sample number:
        sample_lens = torch.tensor(h * w * sample_ratio).repeat(bs,1).int()
        max_sample_num = sample_lens.max()
        
        min_sample_num = sample_lens.min()
        sort_order = sample_weight_clone.sort(descending=True,dim=1)[1]
        sort_confidence_topk = sort_order[:,:max_sample_num]
        sort_confidence_topk_remaining = sort_order[:,min_sample_num:]
        ## flatten for gathering
        src = src.flatten(2).permute(2, 0, 1)
        src = self.norm_feature(src)

        sample_reg_loss = sample_weight.gather(1,sort_confidence_topk).mean()
        src_sampled = src.gather(0,sort_confidence_topk.permute(1,0)[...,None].expand(-1,-1,c)) *sample_weight.gather(1,sort_confidence_topk).permute(1,0).unsqueeze(-1)
        # pos_embed_sampled = pos_embed.gather(0,sort_confidence_topk.permute(1,0)[...,None].expand(-1,-1,c))
        pos_embed_sampled = pos_embed.gather(0, sort_confidence_topk.permute(1, 0)[..., None])

        return src_sampled, sample_reg_loss, sort_confidence_topk, pos_embed_sampled
