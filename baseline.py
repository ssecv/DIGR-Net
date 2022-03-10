
import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import torchvision.models as models
from ResNet import  ResNet50

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

class cross_modal_fusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(cross_modal_fusion, self).__init__()
        self.squeeze = nn.Conv2d(in_channels*2, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, rgb_feat, dep_feat):
        return self.squeeze(torch.cat([rgb_feat, dep_feat], dim=1))

class cross_level_fusion(nn.Module):
    def __init__(self, high_channels):
        super(cross_level_fusion, self).__init__()
        self.squeeze = ConvBNReLU(high_channels, high_channels//2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.squeeze_out = nn.Conv2d(high_channels, high_channels//2, kernel_size=1, stride=1, padding=0)

    def forward(self, high_level, low_level):
        high_level_feat = self.squeeze(high_level)
        high_level_feat = self.upsample(high_level_feat)
        fusion_feat = self.squeeze_out(torch.cat([high_level_feat, low_level], dim=1))
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

class generate_coarse_sal(nn.Module):
    def __init__(self):
        super(generate_coarse_sal, self).__init__()
        self.clf4_3 = cross_level_fusion(1024)
        self.clf43_2 = cross_level_fusion(512)
        self.clf432_1 = cross_level_fusion(256)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.clf4321_0 = cross_level_fusion(128)

    def forward(self, fusion4, fusion3, fusion2, fusion1, fusion0):
        #1024 chanels and 16x
        # fusion_4 = self.cpfe_4(fusion4)
        # fusion_3 = self.cpfe_3(fusion3)
        # fusion_2 = self.cpfe_2(fusion2)
        # fusion_1 = self.cpfe_1(fusion1)
        fusion_0 = self.upsample(fusion0)
        fusion_43 = self.clf4_3(fusion4, fusion3)
        fusion_432 = self.clf43_2(fusion_43, fusion2)
        fusion_4321 = self.clf432_1(fusion_432, fusion1)
        fusion_43210 = self.clf4321_0(fusion_4321, fusion_0)
        return fusion_4321, fusion_43210

#IRFFNet
class IRFF(nn.Module):
    def __init__(self):
        super(IRFF, self).__init__()

        #use resnet50 as backbone
        self.resnet_rgb = ResNet50('rgb')
        self.resnet_depth = ResNet50('depth')

        # fuse the cross modal features
        self.cmf_0 = cross_modal_fusion(64, 64)
        self.cmf_1 = cross_modal_fusion(256, 128)
        self.cmf_2 = cross_modal_fusion(512, 256)
        self.cmf_3 = cross_modal_fusion(1024, 512)
        self.cmf_4 = cross_modal_fusion(2048, 1024)

        #generate a coarse sal map
        self.GCS = generate_coarse_sal()
        self.dim_reuce_sal_init = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.upsample_final_init = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        if self.training:
            self.initialize_weights()

    def forward(self, rgb, depth):
        #layer_0 feature maps-> 64 channels and 4x
        rgb = self.resnet_rgb.relu(self.resnet_rgb.bn1(self.resnet_rgb.conv1(rgb)))
        rgb_0 = self.resnet_rgb.maxpool(rgb)
        depth = self.resnet_depth.relu(self.resnet_depth.bn1(self.resnet_depth.conv1(depth)))
        depth_0 = self.resnet_depth.maxpool(depth)
        # fuse cross modal features
        fusion_0 = self.cmf_0(rgb_0, depth_0)

        # layer_1 feature maps-> 256 channels and 4x
        rgb_1 = self.resnet_rgb.layer1(rgb_0)
        depth_1 = self.resnet_depth.layer1(depth_0)
        fusion_1 = self.cmf_1(rgb_1, depth_1)

        # layer_2 feature maps-> 512 channels and 8x
        rgb_2 = self.resnet_rgb.layer2(rgb_1)
        depth_2 = self.resnet_depth.layer2(depth_1)
        fusion_2 = self.cmf_2(rgb_2, depth_2)

        # layer_3 feature maps-> 1024 channels and 16x
        rgb_3 = self.resnet_rgb.layer3_1(rgb_2)
        depth_3 = self.resnet_depth.layer3_1(depth_2)
        fusion_3 = self.cmf_3(rgb_3, depth_3)

        # layer_4 feature maps-> 2048 channels and 32x
        rgb_4 = self.resnet_rgb.layer4_1(rgb_3)
        depth_4 = self.resnet_depth.layer4_1(depth_3)
        fusion_4 = self.cmf_4(rgb_4, depth_4)

        #progresively integrate multi-level features and generate the initial sal map
        fusion_43, fusion_43210 = self.GCS(fusion_4, fusion_3, fusion_2, fusion_1, fusion_0)
        refin_feat_init = self.dim_reuce_sal_init(fusion_43210)

        sal_map_i = self.upsample_final_init(refin_feat_init)

        avgout = torch.mean(depth_4, dim=1, keepdim=True)
        # max_out, _ = torch.max(depth_4, dim=1, keepdim=True)
        # return depth_0[:,0:1,:,:], sal_map_i
        return avgout[:,0:1,:,:], sal_map_i

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
    rgb = np.random.randint(0, 255.0, size=[2, 3, 128, 128])
    rgb = torch.from_numpy(rgb)
    rgb = rgb.float().cuda()

    dep = np.random.randint(0, 255.0, size=[2, 1, 128, 128])
    dep = torch.from_numpy(dep)
    dep = dep.float().cuda()

    out = net(rgb, dep)
    print(out.size())
