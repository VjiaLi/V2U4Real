

import torch
import torch.nn as nn
import numpy as np

class Communication(nn.Module):
    def __init__(self, thre):
        super(Communication, self).__init__()
        
        self.thre = thre

    def forward(self, batch_confidence_maps):

        B, _, H, W = batch_confidence_maps.shape
        ori_communication_maps = batch_confidence_maps.sigmoid().max(dim=1)[0].unsqueeze(1) # dim1=2 represents the confidence of two anchors
        communication_maps = ori_communication_maps

        ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
        zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
        communication_mask = torch.where(communication_maps>self.thre, ones_mask, zeros_mask)

        return communication_mask