import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.dcn import ModulatedDeformConvPack, modulated_deform_conv


class ResidualBlock(nn.Module):
    def __init__(self, num_feat=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.lrelu(self.conv1(x)))
        return identity + out


def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class DCNv2Pack(ModulatedDeformConvPack):
    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(
                f'Offset abs mean is {offset_absmean}, larger than 50.')

        return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
                                     self.stride, self.padding, self.dilation,
                                     self.groups, self.deformable_groups)


class Alignment(nn.Module):
    def __init__(self, num_feat=64, num_extract_block=4, num_deformable_group=8):
        super(Alignment, self).__init__()
        self.offset_conv1_1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.resblock1_1 = make_layer(ResidualBlock, num_basic_block=num_extract_block, num_feat=num_feat)
        self.offset_conv2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.resblock2_1 = make_layer(ResidualBlock, num_basic_block=num_extract_block, num_feat=num_feat)
        self.offset_conv3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.resblock3_1 = make_layer(ResidualBlock, num_basic_block=num_extract_block, num_feat=num_feat)

        self.resblock2_2 = make_layer(ResidualBlock, num_basic_block=num_extract_block, num_feat=num_feat)
        self.resblock1_2 = make_layer(ResidualBlock, num_basic_block=num_extract_block, num_feat=num_feat)

        self.dcn = DCNv2Pack(num_feat, num_feat, 3, padding=1, deformable_groups=num_deformable_group)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat, ref_feat):
        # step 1
        feat_cat = torch.cat([nbr_feat, ref_feat], dim=1)
        offset1_1 = self.lrelu(self.offset_conv1_1(feat_cat))
        offset1_1 = self.resblock1_1(offset1_1)
        offset2_1 = self.lrelu(self.offset_conv2_1(offset1_1))
        offset2_1 = self.resblock2_1(offset2_1)
        offset3_1 = self.lrelu(self.offset_conv3_1(offset2_1))
        offset3_1 = self.resblock3_1(offset3_1)

        offset2_2 = self.upsample(offset3_1)
        offset2_2 = self.resblock2_2(offset2_1+offset2_2)
        offset1_2 = self.upsample(offset2_2)
        offset = self.resblock1_2(offset1_1+offset1_2)
        nbr_feat = self.lrelu(self.dcn(nbr_feat, offset))

        # step 2
        feat_cat = torch.cat([nbr_feat, ref_feat], dim=1)
        offset1_1 = self.lrelu(self.offset_conv1_1(feat_cat))
        offset1_1 = self.resblock1_1(offset1_1)
        offset2_1 = self.lrelu(self.offset_conv2_1(offset1_1))
        offset2_1 = self.resblock2_1(offset2_1)
        offset3_1 = self.lrelu(self.offset_conv3_1(offset2_1))
        offset3_1 = self.resblock3_1(offset3_1)

        offset2_2 = self.upsample(offset3_1)
        offset2_2 = self.resblock2_2(offset2_1+offset2_2)
        offset1_2 = self.upsample(offset2_2)
        offset = self.resblock1_2(offset1_1+offset1_2)
        feat = self.lrelu(self.dcn(nbr_feat, offset))

        return feat


class Aggregation(nn.Module):
    def __init__(self, num_feat=64, num_frame=5, center_frame_idx=2):
        super(Aggregation, self).__init__()
        self.center_frame_idx = center_frame_idx
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = nn.Conv2d(num_frame * num_feat, num_feat, 1)
        self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, aligned_feat):
        B, N, C, H, W = aligned_feat.size()
        # temporal attention
        embedding_ref = self.temporal_attn1(aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        embedding = self.temporal_attn2(aligned_feat.view(-1, C, H, W))
        embedding = embedding.view(B, N, -1, H, W)  # (B, N, C, H, W)

        corr_l = []  # correlation list
        for i in range(N):
            emb_neighbor = embedding[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (B, H, W)
            corr_l.append(corr.unsqueeze(1))  # (B, 1, H, W)
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (B, N, H, W)
        corr_prob = corr_prob.unsqueeze(2).expand(B, N, C, H, W)
        corr_prob = corr_prob.contiguous().view(B, -1, H, W)  # (B, N*C, H, W)
        aligned_feat = aligned_feat.view(B, -1, H, W) * corr_prob

        # fusion
        feat = self.lrelu(self.feat_fusion(aligned_feat))

        # spatial attention
        attn = self.lrelu(self.spatial_attn1(aligned_feat))
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.lrelu(
            self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
        # pyramid levels
        attn_level = self.lrelu(self.spatial_attn_l1(attn))
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.lrelu(
            self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
        attn_level = self.upsample(attn_level)

        attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
        attn = self.lrelu(self.spatial_attn4(attn))
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(
            self.lrelu(self.spatial_attn_add1(attn)))
        attn = torch.sigmoid(attn)

        # after initialization, * 2 makes (attn * 2) to be close to 1.
        feat = feat * attn * 2 + attn_add
        return feat