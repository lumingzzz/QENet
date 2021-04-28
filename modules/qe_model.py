import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.common import *

class VQE(nn.Module):
    def __init__(self, 
                 num_in_ch=3, 
                 num_out_ch=3, 
                 num_feat=128, 
                 num_frame=7, 
                 num_extract_block=8, 
                 num_deformable_group=8, 
                 num_reconstruct_block=16,
                 center_frame_idx=3):
        super(VQE, self).__init__()
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx

        # extract features for each frame
        self.first_conv = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.fea_ext = make_layer(ResidualBlock, num_extract_block, num_feat=num_feat)

        # align and tsa module
        self.alignment = Alignment(num_feat=num_feat, num_extract_block=num_extract_block//2, num_deformable_group=num_deformable_group)
        self.aggregation = Aggregation(num_feat=num_feat, num_frame=num_frame, center_frame_idx=self.center_frame_idx)

        # reconstruction
        self.fea_recon = make_layer(ResidualBlock, num_reconstruct_block, num_feat=num_feat)

        self.conv_out = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, train=True):
        B, N, C, H, W = x.size()
        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        # extract features for each frame
        feat = self.lrelu(self.first_conv(x.view(-1, C, H, W)))
        feat = self.fea_ext(feat)
        feat = feat.view(B, N, -1, H, W)

        # alignment
        cur_feat = feat[:, self.center_frame_idx, :, :, :].clone()
        aligned_feat_l = []
        for i in range(N):
            nbr_feat = feat[:, i, :, :, :].clone()
            aligned_feat_l.append(self.alignment(nbr_feat, cur_feat))

        aligned_feat = torch.stack(aligned_feat_l, dim=1)  # (B, N, C, H, W)

        aligned_feat = self.aggregation(aligned_feat)
        aligned_feat = self.fea_recon(aligned_feat) + aligned_feat

        res = self.conv_out(aligned_feat)
        
        out = res + x_center
        return out