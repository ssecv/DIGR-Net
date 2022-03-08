# _*_ coding: utf-8 _*_
# @Time : 2020/12/23 11:17
# @Author : xiaolong
# @File : two_stages_RGBD_A_mul.py
# @desc : 这个版本将所有的模块名称改为了论文里面的模块名称，增加了可读性

import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import torchvision.models as models
from ResNet import ResNet50

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class cross_modality_interaction_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(cross_modality_interaction_block, self).__init__()

        self.theta_rgb = nn.Conv2d(in_channels, in_channels, 1)
        self.phi_rgb = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax_rgb = nn.Softmax(dim=1)
        self.g_rgb = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_tpg_rgb = nn.Conv2d(in_channels, in_channels, 1)

        self.theta_dep = nn.Conv2d(in_channels, in_channels, 1)
        self.phi_dep = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax_dep = nn.Softmax(dim=1)
        self.g_dep = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_tpg_dep = nn.Conv2d(in_channels, in_channels, 1)

        self.squeeze = nn.Conv2d(in_channels*2, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, rgb_feat, dep_feat, score):
        #rgb side
        rgb_theta = self.theta_rgb(rgb_feat)
        rgb_phi = self.phi_rgb(rgb_feat)
        rgb_tp = torch.matmul(rgb_theta, rgb_phi)

        rgb_tq_softmax = self.softmax_rgb(rgb_tp)
        rgb_g = self.g_rgb(rgb_feat)

        #depth side
        dep_theta = self.theta_dep(dep_feat)
        dep_phi = self.phi_dep(dep_feat)
        dep_tq = torch.matmul(dep_theta, dep_phi)

        dep_tq_softmax = self.softmax_dep(dep_tq)
        dep_g = self.g_dep(dep_feat)

        #cross matmul for RGB and depth
        rgb_tqg = rgb_tq_softmax * dep_g
        rgb_tqg = self.conv_tpg_rgb(rgb_tqg)

        dep_tqg = dep_tq_softmax * rgb_g
        dep_tqg = self.conv_tpg_dep(dep_tqg)

        rgb_out = rgb_tqg + rgb_feat
        dep_out = dep_tqg + dep_feat
        return rgb_out, dep_out, self.squeeze(torch.cat([rgb_tqg + rgb_feat*(1-score), dep_tqg + dep_feat*score], dim=1))

class cross_modality_interaction_block_r(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(cross_modality_interaction_block_r, self).__init__()
        self.theta_rgb = nn.Conv2d(in_channels, in_channels, 1)
        self.phi_rgb = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax_rgb = nn.Softmax(dim=1)
        self.g_rgb = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_tpg_rgb = nn.Conv2d(in_channels, in_channels, 1)

        self.theta_dep = nn.Conv2d(in_channels, in_channels, 1)
        self.phi_dep = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax_dep = nn.Softmax(dim=1)
        self.g_dep = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_tpg_dep = nn.Conv2d(in_channels, in_channels, 1)
        self.squeeze = nn.Conv2d(in_channels*2, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, rgb_feat, dep_feat, score):
        #rgb side
        rgb_theta = self.theta_rgb(rgb_feat)
        rgb_phi = self.phi_rgb(rgb_feat)
        rgb_tp = torch.matmul(rgb_theta, rgb_phi)

        rgb_tq_softmax = self.softmax_rgb(rgb_tp)
        rgb_g = self.g_rgb(rgb_feat)

        #depth side
        dep_theta = self.theta_dep(dep_feat)
        dep_phi = self.phi_dep(dep_feat)
        dep_tq = torch.matmul(dep_theta, dep_phi)

        dep_tq_softmax = self.softmax_dep(dep_tq)
        dep_g = self.g_dep(dep_feat)

        #cross matmul for RGB and depth
        rgb_tqg = rgb_tq_softmax * dep_g
        rgb_tqg = self.conv_tpg_rgb(rgb_tqg)

        dep_tqg = dep_tq_softmax * rgb_g
        dep_tqg = self.conv_tpg_dep(dep_tqg)

        return self.squeeze(torch.cat([rgb_tqg + rgb_feat*(1-score), dep_tqg + dep_feat*score], dim=1))
#

class mutually_guided_cross_level_fusion(nn.Module):
    def __init__(self, channels):
        super(mutually_guided_cross_level_fusion, self).__init__()
        self.left1 = ConvBNReLU(channels//2, channels//2)
        self.left2 = ConvBNReLU(channels//2, channels//2, 3, 2, 1)
        self.right1 = ConvBNReLU(channels, channels//2)
        self.right2 = ConvBNReLU(channels, channels//2)
        self.left = nn.Sigmoid()

        self.right = nn.Sigmoid()

        # self.sp_att_high = SpatialAttention()
        # self.sp_att_low = SpatialAttention()

    def forward(self, high_level, low_level):
        right1 = F.interpolate(self.right1(high_level), size=low_level.size()[2:], mode='bilinear', align_corners=True)
        right2 = self.right2(high_level)
        left1 = self.left1(low_level)
        left2 = self.left2(low_level)
        # left = (left1 + right1) * self.sp_att_high(right1)
        left = (left1 + right1) * self.left(right1)
        right = self.right(left2) * (left2 + right2)
        right = F.interpolate(right, size=left.size()[2:], mode='bilinear', align_corners=True)
        out = left + right
        return out

class cross_level_fusion(nn.Module):
    def __init__(self, high_channels):
        super(cross_level_fusion, self).__init__()
        self.squeeze = ConvBNReLU(high_channels, high_channels//2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, high_level, low_level):
        high_level_feat = self.squeeze(high_level)
        high_level_feat = self.upsample(high_level_feat)
        fusion_feat = high_level_feat + low_level
        return fusion_feat

#attention mechanism
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.ch_atten = ChannelAttention(in_channels)
        self.sp_atten = SpatialAttention()

    def forward(self, fusion):
        fusion_atten = self.ch_atten(fusion)
        fusion_ch = torch.mul(fusion_atten, fusion)
        fusions = self.sp_atten(fusion_ch) * fusion_ch
        return fusions

class DAC(nn.Module):

    def __init__(self, channels):
        super(DAC, self).__init__()
        self.conv11 = nn.Conv2d(channels, channels, kernel_size=3, dilation=1, padding=1)

        self.conv21 = nn.Conv2d(channels, channels, kernel_size=3, dilation=3, padding=3)
        self.conv22 = nn.Conv2d(channels, channels, kernel_size=1, dilation=1, padding=0)

        self.conv31 = nn.Conv2d(channels, channels, kernel_size=3, dilation=1, padding=1)
        self.conv32 = nn.Conv2d(channels, channels, kernel_size=3, dilation=3, padding=3)
        self.conv33 = nn.Conv2d(channels, channels, kernel_size=1, dilation=1, padding=0)

        self.conv41 = nn.Conv2d(channels, channels, kernel_size=3, dilation=1, padding=1)
        self.conv42 = nn.Conv2d(channels, channels, kernel_size=3, dilation=3, padding=3)
        self.conv43 = nn.Conv2d(channels, channels, kernel_size=3, dilation=5, padding=5)
        self.conv44 = nn.Conv2d(channels, channels, kernel_size=1, dilation=1, padding=0)

    def forward(self, x):
        c1 = self.conv11(x)

        c2 = self.conv21(x)
        c2 = self.conv22(c2)

        c3 = self.conv31(x)
        c3 = self.conv32(c3)
        c3 = self.conv33(c3)

        c4 = self.conv41(x)
        c4 = self.conv42(c4)
        c4 = self.conv43(c4)
        c4 = self.conv44(c4)

        c = torch.cat((c1, c2, c3, c4), dim=1)
        return c


class RMP(nn.Module):
    def __init__(self, channels):
        super(RMP, self).__init__()

        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)

        self.max2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)

        self.max3 = nn.MaxPool2d(kernel_size=5, stride=5)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=1)

        self.max4 = nn.MaxPool2d(kernel_size=6)
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):

        m1 = self.max1(x)
        m1 = F.interpolate(self.conv1(m1), size=x.size()[2:], mode='bilinear', align_corners=True)

        m2 = self.max2(x)
        m2 = F.interpolate(self.conv2(m2), size=x.size()[2:], mode='bilinear', align_corners=True)

        m3 = self.max3(x)
        m3 = F.interpolate(self.conv3(m3), size=x.size()[2:], mode='bilinear', align_corners=True)

        m4 = self.max4(x)
        m4 = F.interpolate(self.conv4(m4), size=x.size()[2:], mode='bilinear', align_corners=True)

        m = torch.cat([m1, m2, m3, m4], dim=1)
        return m

class PDC(nn.Module):
    def __init__(self, channels):
        super(PDC, self).__init__()
        self.conv_input = nn.Conv2d(channels, channels // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.dac = DAC(channels//4)
        self.rmp = RMP(channels//4)

        self.squeeze_feats = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x_0 = self.conv_input(x)
        x_dac = self.dac(x_0)
        x_rmp = self.rmp(x_0)
        out = self.squeeze_feats(x_dac + x_rmp + x)
        return out

#空洞卷积，多尺度,改进空间---》增加横向互动
class CPFE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CPFE, self).__init__()
        self.dil_rates = [3, 5, 7]
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Define layers
        self.conv_1_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, bias=False)
        self.conv_dil_3 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[0], padding=self.dil_rates[0], bias=False)
        self.conv_dil_5 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[1], padding=self.dil_rates[1], bias=False)
        self.conv_dil_7 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[2], padding=self.dil_rates[2], bias=False)
        self.bn = nn.BatchNorm2d(self.out_channels*4)

    def forward(self, input_):
        # Extract features
        conv_1_1_feats = self.conv_1_1(input_)
        conv_dil_3_feats = self.conv_dil_3(input_)
        conv_dil_5_feats = self.conv_dil_5(input_)
        conv_dil_7_feats = self.conv_dil_7(input_)

        # Aggregate features
        concat_feats = torch.cat((conv_1_1_feats, conv_dil_3_feats, conv_dil_5_feats, conv_dil_7_feats), dim=1)
        bn_feats = F.relu(self.bn(concat_feats))

        return bn_feats

class generate_coarse_sal(nn.Module):
    def __init__(self):
        super(generate_coarse_sal, self).__init__()
        # cross level fusion
        self.pdc = PDC(1024)

        self.mgcf4_3 = mutually_guided_cross_level_fusion(1024)
        self.mgcf43_2 = mutually_guided_cross_level_fusion(512)
        self.mgcf432_1 = mutually_guided_cross_level_fusion(256)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.mgcf4321_0 = mutually_guided_cross_level_fusion(128)
        self.cbam_43210 = CBAM(64)

    def forward(self, fusion4, fusion3, fusion2, fusion1, fusion0):
        #1024 chanels and 16x
        fusion_4 = self.pdc(fusion4)
        fusion_0 = self.upsample(fusion0)
        fusion_43 = self.mgcf4_3(fusion_4, fusion3)
        fusion_432 = self.mgcf43_2(fusion_43, fusion2)
        fusion_4321 = self.mgcf432_1(fusion_432, fusion1)
        fusion_43210 = self.mgcf4321_0(fusion_4321, fusion_0)
        sal_feats = self.cbam_43210(fusion_43210)
        return sal_feats

#feed back refine mechanism
class feature_enhanced_operation(nn.Module):
    def __init__(self):
        super(feature_enhanced_operation, self).__init__()

    def forward(self, fb_feat, feat_4, feat_3, feat_2, feat_1, feat_0):
        refine4 = F.interpolate(fb_feat, size=feat_4.size()[2:], mode='bilinear', align_corners=True)
        refine3 = F.interpolate(fb_feat, size=feat_3.size()[2:], mode='bilinear', align_corners=True)
        refine2 = F.interpolate(fb_feat, size=feat_2.size()[2:], mode='bilinear', align_corners=True)
        refine1 = F.interpolate(fb_feat, size=feat_1.size()[2:], mode='bilinear', align_corners=True)
        refine0 = F.interpolate(fb_feat, size=feat_0.size()[2:], mode='bilinear', align_corners=True)

        feat_4r = feat_4 + torch.mul(feat_4, refine4)
        feat_3r = feat_3 + torch.mul(feat_3, refine3)
        feat_2r = feat_2 + torch.mul(feat_2, refine2)
        feat_1r = feat_1 + torch.mul(feat_1, refine1)
        feat_0r = feat_0 + torch.mul(feat_0, refine0)

        return feat_4r, feat_3r, feat_2r, feat_1r, feat_0r


class IRFF_output(nn.Module):
    def __init__(self, in_channels, mid_channels=64, out_channels=1):
        super(IRFF_output, self).__init__()
        self.ch_atten = ChannelAttention(in_channels)
        self.conv_mid = ConvBNReLU(in_channels, mid_channels, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, rgb, dep):
        cat_feats = torch.cat([rgb, dep],dim=1)
        cat_ch_atten = self.ch_atten(cat_feats)
        cat_att_feat = torch.mul(cat_feats, cat_ch_atten)
        cat_mid = self.conv_mid(cat_att_feat)
        sal_map = self.conv_out(cat_mid)

        cat_w_sum = torch.sum(cat_ch_atten, dim=1, keepdim=True)
        dep_w_sum = torch.sum(cat_ch_atten[:, rgb.size()[1]:],dim=1, keepdim=True)
        dep_score = dep_w_sum/cat_w_sum
        return sal_map, dep_score

#IRFFNet
class IRFF(nn.Module):
    def __init__(self):
        super(IRFF, self).__init__()

        #use resnet50 as backbone
        self.resnet_rgb = ResNet50('rgb')
        self.resnet_depth = ResNet50('depth')

        # fuse the cross modal features
        self.cmib_0 = cross_modality_interaction_block(64, 64)
        self.cmib_1 = cross_modality_interaction_block(256, 128)
        self.cmib_2 = cross_modality_interaction_block(512, 256)
        self.cmib_3 = cross_modality_interaction_block(1024, 512)
        self.cmib_4 = cross_modality_interaction_block(2048, 1024)

        #generate a coarse sal map
        self.GCS = generate_coarse_sal()
        self.dim_reuce_sal_init = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

        #refine procedure
        self.refine_fusion_feat = feature_enhanced_operation()

        self.cmib_0r = cross_modality_interaction_block_r(64, 64)
        self.cmib_1r = cross_modality_interaction_block_r(128, 128)
        self.cmib_2r = cross_modality_interaction_block_r(256, 256)
        self.cmib_3r = cross_modality_interaction_block_r(512, 512)
        self.cmib_4r = cross_modality_interaction_block_r(1024, 1024)

        self.ID_D4 = IRFF_output(in_channels=2048, mid_channels=32, out_channels=1)
        self.ID_D3 = IRFF_output(in_channels=1024, mid_channels=32, out_channels=1)
        self.ID_D2 = IRFF_output(in_channels=512, mid_channels=32, out_channels=1)
        self.ID_D1 = IRFF_output(in_channels=256, mid_channels=32, out_channels=1)
        self.ID_D0 = IRFF_output(in_channels=128, mid_channels=32, out_channels=1)

        self.upsample32x = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.upsample16x = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample8x = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        #generate a coarse sal map by refined features
        self.GCS_r = generate_coarse_sal()
        self.dim_reuce_sal_r = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

        if self.training:
            self.initialize_weights()

    def forward(self, rgb, depth, id1):
        #layer_0 feature maps-> 64 channels and 4x
        rgb = self.resnet_rgb.relu(self.resnet_rgb.bn1(self.resnet_rgb.conv1(rgb)))
        rgb_0 = self.resnet_rgb.maxpool(rgb)
        depth = self.resnet_depth.relu(self.resnet_depth.bn1(self.resnet_depth.conv1(depth)))
        depth_0 = self.resnet_depth.maxpool(depth)
        # fuse cross modal features
        rgb_0, depth_0, fusion_0 = self.cmib_0(rgb_0, depth_0, id1)

        # layer_1 feature maps-> 256 channels and 4x
        rgb_1 = self.resnet_rgb.layer1(rgb_0)
        depth_1 = self.resnet_depth.layer1(depth_0)
        rgb_1, depth_1, fusion_1 = self.cmib_1(rgb_1, depth_1, id1)

        # layer_2 feature maps-> 512 channels and 8x
        rgb_2 = self.resnet_rgb.layer2(rgb_1)
        depth_2 = self.resnet_depth.layer2(depth_1)
        rgb_2, depth_2, fusion_2 = self.cmib_2(rgb_2, depth_2, id1)

        # layer_3 feature maps-> 1024 channels and 16x
        rgb_3 = self.resnet_rgb.layer3_1(rgb_2)
        depth_3 = self.resnet_depth.layer3_1(depth_2)
        rgb_3, depth_3, fusion_3 = self.cmib_3(rgb_3, depth_3, id1)

        # layer_4 feature maps-> 2048 channels and 32x
        rgb_4 = self.resnet_rgb.layer4_1(rgb_3)
        depth_4 = self.resnet_depth.layer4_1(depth_3)
        rgb_4, depth_4, fusion_4 = self.cmib_4(rgb_4, depth_4, id1)

        #progresively integrate multi-level features and generate the initial sal map
        fusion_43210 = self.GCS(fusion_4, fusion_3, fusion_2, fusion_1, fusion_0)
        coarse_sal_feat = self.dim_reuce_sal_init(fusion_43210)

        #use the coarse salient map to refine fusion feat
        fusion_4r, fusion_3r, fusion_2r, fusion_1r, fusion_0r = self.refine_fusion_feat(coarse_sal_feat, fusion_4, fusion_3,
                                                                        fusion_2, fusion_1, fusion_0)

        sal_branch_r_0, id2_0 = self.ID_D0(fusion_0, fusion_0r)
        fusion_0rc = self.cmib_0r(fusion_0, fusion_0r, id2_0)

        sal_branch_r_1, id2_1 = self.ID_D1(fusion_1, fusion_1r)
        fusion_1rc = self.cmib_1r(fusion_1, fusion_1r, id2_1)

        sal_branch_r_2, id2_2 = self.ID_D2(fusion_2, fusion_2r)
        fusion_2rc = self.cmib_2r(fusion_2, fusion_2r, id2_2)

        sal_branch_r_3, id2_3 = self.ID_D3(fusion_3, fusion_3r)
        fusion_3rc = self.cmib_3r(fusion_3, fusion_3r, id2_3)

        sal_branch_r_4, id2_4 = self.ID_D4(fusion_4, fusion_4r)
        fusion_4rc = self.cmib_4r(fusion_4, fusion_4r, id2_4)

        # 64 channels and 4x
        fusion_43210r = self.GCS_r(fusion_4rc, fusion_3rc, fusion_2rc, fusion_1rc, fusion_0rc)
        sal_feat_r = fusion_43210r
        refin_feat_r = self.dim_reuce_sal_r(sal_feat_r)

        sal_map_coarse = self.upsample2x(coarse_sal_feat)
        sal_map_r = self.upsample2x(refin_feat_r)
        return sal_map_coarse,self.upsample32x(sal_branch_r_4),self.upsample16x(sal_branch_r_3),self.upsample8x(sal_branch_r_2),self.upsample4x(sal_branch_r_1), self.upsample4x(sal_branch_r_0), sal_map_r

    # initialize the weights
    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet_rgb.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet_rgb.state_dict().keys())
        self.resnet_rgb.load_state_dict(all_params)

        all_params = {}
        for k, v in self.resnet_depth.state_dict().items():
            if k == 'conv1.weight':
                all_params[k] = torch.nn.init.normal_(v, mean=0, std=1)
            elif k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet_depth.state_dict().keys())
        self.resnet_depth.load_state_dict(all_params)

if __name__ == '__main__':
    net = IRFF()
    net.cuda()
    net.eval()
    rgb = np.random.randint(0, 255.0, size=[2, 3, 256, 256])
    rgb = torch.from_numpy(rgb)
    rgb = rgb.float().cuda()

    dep = np.random.randint(0, 255.0, size=[2, 1, 256, 256])
    dep = torch.from_numpy(dep)
    dep = dep.float().cuda()

    score = np.random.randint(0, 255.0, size=[2, 1, 1, 1])
    score = torch.from_numpy(score)
    score = score.float().cuda()
    # print(rgb)
    out = net(rgb, dep, score)
    print(out[0].size(), out[1].size(), out[2].size(), out[3].size(), out[4].size(), out[5].size(), out[6].size())
