import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

from vgg16 import VGG_16
#对cmib进行了改动，对各层的通道数也进行了不同程度的改动

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

        # self.squeeze_0 = ConvBNReLU(in_channels*2, in_channels)
        # self.squeeze = ConvBNReLU(in_channels, in_channels//2)
        # if not self.level_0:
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

class mutually_guided_cross_level_fusion(nn.Module):
    def __init__(self, channels):
        super(mutually_guided_cross_level_fusion, self).__init__()
        # self.cpfe_h = CPFE(channels, channels//4)
        # self.cpfe_l = CPFE(channels//2, channels//8)
        self.left1 = ConvBNReLU(channels//2, channels//2)
        self.left2 = ConvBNReLU(channels//2, channels//2, 3, 2, 1)
        self.right1 = ConvBNReLU(channels, channels//2)
        self.right2 = ConvBNReLU(channels, channels//2)

        self.sp_att_high = SpatialAttention()
        self.sp_att_low = SpatialAttention()

        # self.conv = ConvBNReLU(channels//2, channels//2)
        # self.conv = nn.Conv2d(channels//2, channels//2, kernel_size=1, stride=1, padding=0)

    def forward(self, high_level, low_level):
        # high_level = self.cpfe_h(high_level)
        # low_level = self.cpfe_l(low_level)

        right1 = F.interpolate(self.right1(high_level), size=low_level.size()[2:], mode='bilinear', align_corners=True)
        right2 = self.right2(high_level)
        left1 = self.left1(low_level)
        left2 = self.left2(low_level)
        left = (left1 + right1) * self.sp_att_high(right1)
        right = self.sp_att_low(left2) * (left2 + right2)
        right = F.interpolate(right, size=left.size()[2:], mode='bilinear', align_corners=True)
        out = left + right
        return out



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

        # self.squeeze_cat_feat = nn.Sequential(
        #     nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(channels),
        #     nn.ReLU(),
        #     )

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

        # self.squeeze_cat_feat = nn.Sequential(
        #     nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(channels),
        #     nn.ReLU(),
        # )

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
        # m = self.squeeze_cat_feat(m)
        return m

class PDC(nn.Module):
    def __init__(self, channels):
        super(PDC, self).__init__()
        self.conv_input = nn.Conv2d(channels, channels // 4, kernel_size=1, stride=1, padding=0, bias=False)
        # self.conv_inputD = nn.Conv2d(channels, channels // 4, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv_inputR = nn.Conv2d(channels, channels // 4, kernel_size=3, stride=1, padding=1, bias=False)

        self.dac = DAC(channels//4)
        self.rmp = RMP(channels//4)

        self.squeeze_feats = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

    def forward(self, x):
        # x_0 = self.conv_inputD(x)
        # x_1 = self.conv_inputR(x)
        x_0 = self.conv_input(x)
        x_dac = self.dac(x_0)
        x_rmp = self.rmp(x_0)
        out = self.squeeze_feats(x_dac + x_rmp + x)
        return out


class generate_coarse_sal(nn.Module):
    def __init__(self):
        super(generate_coarse_sal, self).__init__()
        # cross level fusion
        self.dr = PDC(512)

        self.mgcf4_3 = mutually_guided_cross_level_fusion(512)
        self.mgcf43_2 = mutually_guided_cross_level_fusion(256)
        self.mgcf432_1 = mutually_guided_cross_level_fusion(128)
        self.mgcf4321_0 = mutually_guided_cross_level_fusion(64)
        self.cbam_43210 = CBAM(32)

    def forward(self, fusion4, fusion3, fusion2, fusion1, fusion0):
        # 1024 chanels and 16x
        fusion_4 = self.dr(fusion4)

        fusion_43 = self.mgcf4_3(fusion_4, fusion3)
        fusion_432 = self.mgcf43_2(fusion_43, fusion2)
        fusion_4321 = self.mgcf432_1(fusion_432, fusion1)
        fusion_43210 = self.mgcf4321_0(fusion_4321, fusion0)
        sal_feats = self.cbam_43210(fusion_43210)
        return sal_feats

#feed back refine mechanism
class feature_enhanced_operation(nn.Module):
    def __init__(self):
        super(feature_enhanced_operation, self).__init__()

    def forward(self, fb_feat, feat_4, feat_3, feat_2, feat_1, feat_0):
        #fb_feat就是fusion_all的降维
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

class DIGR_output(nn.Module):
    def __init__(self, in_channels, mid_channels=64, out_channels=1):
        super(DIGR_output, self).__init__()
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

class DIGR(nn.Module):
    def __init__(self):
        super(DIGR, self).__init__()
        self.vgg_rgb = VGG_16()
        self.vgg_depth=VGG_16()

        # fuse the cross modal features
        self.cmib_0 = cross_modality_interaction_block(64, 32)
        self.cmib_1 = cross_modality_interaction_block(128, 64)
        self.cmib_2 = cross_modality_interaction_block(256, 128)
        self.cmib_3 = cross_modality_interaction_block(512, 256)
        self.cmib_4 = cross_modality_interaction_block(512, 512)

        # generate a coarse sal map
        self.GCS_i = generate_coarse_sal()
        self.dim_reduce_sal_init = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)

        # refine procedure
        self.refine_fused_feat = feature_enhanced_operation()

        self.cmib_0r = cross_modality_interaction_block_r(32, 32)
        self.cmib_1r = cross_modality_interaction_block_r(64, 64)
        self.cmib_2r = cross_modality_interaction_block_r(128, 128)
        self.cmib_3r = cross_modality_interaction_block_r(256, 256)
        self.cmib_4r = cross_modality_interaction_block_r(512, 512)

        self.conv_out4 = DIGR_output(in_channels=512 * 2, mid_channels=16, out_channels=1)
        self.conv_out3 = DIGR_output(in_channels=256 * 2, mid_channels=16, out_channels=1)
        self.conv_out2 = DIGR_output(in_channels=128 * 2, mid_channels=16, out_channels=1)
        self.conv_out1 = DIGR_output(in_channels=64 * 2, mid_channels=16, out_channels=1)
        self.conv_out0 = DIGR_output(in_channels=32 * 2, mid_channels=16, out_channels=1)

        # generate a coarse sal map by refined features
        self.GCS_r = generate_coarse_sal()
        self.dim_reduce_sal_r = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)

        self.up_2x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8x = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_16x = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

    def forward(self, rgb, depth, score):
        temp = torch.cat((depth, depth), 1)
        depths = torch.cat((temp, depth), 1)

        #64 channels and 1x
        rgb_0 = self.vgg_rgb.conv1(rgb)
        depth_0 = self.vgg_depth.conv1(depths)
        rgb_0, depth_0,fusion_0 = self.cmib_0(rgb_0, depth_0, score)

        #128 channels and 2x
        rgb_1 = self.vgg_rgb.conv2(rgb_0)
        depth_1 = self.vgg_depth.conv2(depth_0)
        rgb_1, depth_1, fusion_1 = self.cmib_1(rgb_1, depth_1, score)

        #256 channels and 4x
        rgb_2 = self.vgg_rgb.conv3(rgb_1)
        depth_2 = self.vgg_depth.conv3(depth_1)
        rgb_2, depth_2, fusion_2 = self.cmib_2(rgb_2, depth_2, score)

        #512 channels and 8x
        rgb_3 = self.vgg_rgb.conv4_1(rgb_2)
        depth_3 = self.vgg_depth.conv4_1(depth_2)
        rgb_3, depth_3, fusion_3 = self.cmib_3(rgb_3, depth_3, score)

        #512 channels and 16x
        rgb_4 = self.vgg_rgb.conv5_1(rgb_3)
        depth_4 = self.vgg_depth.conv5_1(depth_3)
        rgb_4, depth_4, fusion_4 = self.cmib_4(rgb_4, depth_4, score)



        sal_feats_init = self.GCS_i(fusion_4, fusion_3, fusion_2, fusion_1, fusion_0)
        sal_map_init = self.dim_reduce_sal_init(sal_feats_init)

        # refine fused features respectively
        fusion_4r, fusion_3r, fusion_2r, fusion_1r, fusion_0r = self.refine_fused_feat(sal_map_init, fusion_4, fusion_3,
                                                                            fusion_2, fusion_1, fusion_0)


        cat_sal_0, score_0 = self.conv_out0(fusion_0, fusion_0r)
        fusion_0rc = self.cmib_0r(fusion_0, fusion_0r, score_0)

        cat_sal_1, score_1 = self.conv_out1(fusion_1, fusion_1r)
        fusion_1rc = self.cmib_1r(fusion_1, fusion_1r, score_1)

        cat_sal_2, score_2 = self.conv_out2(fusion_2, fusion_2r)
        fusion_2rc = self.cmib_2r(fusion_2, fusion_2r, score_2)

        cat_sal_3, score_3 = self.conv_out3(fusion_3, fusion_3r)
        fusion_3rc = self.cmib_3r(fusion_3, fusion_3r, score_3)

        cat_sal_4, score_4 = self.conv_out4(fusion_4, fusion_4r)
        fusion_4rc = self.cmib_4r(fusion_4, fusion_4r, score_4)


        # 64 channels and 4x
        sal_feat_r = self.GCS_r(fusion_4rc, fusion_3rc, fusion_2rc, fusion_1rc, fusion_0rc)
        sal_map_r = self.dim_reduce_sal_r(sal_feat_r)

        sal_cat_4 = self.up_16x(cat_sal_4)
        sal_cat_3 = self.up_8x(cat_sal_3)
        sal_cat_2 = self.up_4x(cat_sal_2)
        sal_cat_1 = self.up_2x(cat_sal_1)

        # return sal_map_i, sal_cat_4, sal_cat_3, sal_cat_2, sal_cat_1, cat_sal_0, sal_map_r
        return sal_map_init, sal_cat_4, sal_cat_3, sal_cat_2, sal_cat_1,cat_sal_0,  sal_map_r

if __name__ == '__main__':
    net = DIGR()
    net.cuda()
    net.eval()
    rgb = np.random.randint(0, 255.0, size=[2, 3, 128, 128])
    rgb = torch.from_numpy(rgb)
    rgb = rgb.float().cuda()

    dep = np.random.randint(0, 255.0, size=[2, 1, 128, 128])
    dep = torch.from_numpy(dep)
    dep = dep.float().cuda()

    score = np.random.randint(0, 255.0, size=[2, 1, 1, 1])
    score = torch.from_numpy(score)
    score = score.float().cuda()
    out = net(rgb, dep, score)

    print(out[0].size(), out[1].size(), out[2].size(), out[3].size(), out[4].size(), out[5].size(), out[6].size())
