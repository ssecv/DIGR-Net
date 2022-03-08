import torch
import torch.nn as nn
from torchvision import models

class VGG_16(nn.Module):
    # VGG16 with two branches
    # pooling layer at the front of block
    def __init__(self):
        super(VGG_16, self).__init__()
        conv1 = nn.Sequential()
        conv1.add_module('conv1_1', nn.Conv2d(3, 64, 3, 1, 1))
        conv1.add_module('relu1_1', nn.ReLU(inplace=True))
        conv1.add_module('conv1_2', nn.Conv2d(64, 64, 3, 1, 1))
        conv1.add_module('relu1_2', nn.ReLU(inplace=True))

        self.conv1 = conv1
        conv2 = nn.Sequential()
        conv2.add_module('pool1', nn.AvgPool2d(2, stride=2))
        conv2.add_module('conv2_1', nn.Conv2d(64, 128, 3, 1, 1))
        conv2.add_module('relu2_1', nn.ReLU())
        conv2.add_module('conv2_2', nn.Conv2d(128, 128, 3, 1, 1))
        conv2.add_module('relu2_2', nn.ReLU())
        self.conv2 = conv2

        conv3 = nn.Sequential()
        conv3.add_module('pool2', nn.AvgPool2d(2, stride=2))
        conv3.add_module('conv3_1', nn.Conv2d(128, 256, 3, 1, 1))
        conv3.add_module('relu3_1', nn.ReLU())
        conv3.add_module('conv3_2', nn.Conv2d(256, 256, 3, 1, 1))
        conv3.add_module('relu3_2', nn.ReLU())
        conv3.add_module('conv3_3', nn.Conv2d(256, 256, 3, 1, 1))
        conv3.add_module('relu3_3', nn.ReLU())
        self.conv3 = conv3

        conv4_1 = nn.Sequential()
        conv4_1.add_module('pool3_1', nn.AvgPool2d(2, stride=2))
        conv4_1.add_module('conv4_1_1', nn.Conv2d(256, 512, 3, 1, 1))
        conv4_1.add_module('relu4_1_1', nn.ReLU())
        conv4_1.add_module('conv4_2_1', nn.Conv2d(512, 512, 3, 1, 1))
        conv4_1.add_module('relu4_2_1', nn.ReLU())
        conv4_1.add_module('conv4_3_1', nn.Conv2d(512, 512, 3, 1, 1))
        conv4_1.add_module('relu4_3_1', nn.ReLU())
        self.conv4_1 = conv4_1

        conv5_1 = nn.Sequential()
        conv5_1.add_module('pool4_1', nn.AvgPool2d(2, stride=2))
        conv5_1.add_module('conv5_1_1', nn.Conv2d(512, 512, 3, 1, 1))
        conv5_1.add_module('relu5_1_1', nn.ReLU())
        conv5_1.add_module('conv5_2_1', nn.Conv2d(512, 512, 3, 1, 1))
        conv5_1.add_module('relu5_2_1', nn.ReLU())
        conv5_1.add_module('conv5_3_1', nn.Conv2d(512, 512, 3, 1, 1))
        conv5_1.add_module('relu5_3_1', nn.ReLU())
        self.conv5_1 = conv5_1

        # conv4_2 = nn.Sequential()
        # conv4_2.add_module('pool3_2', nn.AvgPool2d(2, stride=2))
        # conv4_2.add_module('conv4_1_2', nn.Conv2d(256, 512, 3, 1, 1))
        # conv4_2.add_module('relu4_1_2', nn.ReLU())
        # conv4_2.add_module('conv4_2_2', nn.Conv2d(512, 512, 3, 1, 1))
        # conv4_2.add_module('relu4_2_2', nn.ReLU())
        # conv4_2.add_module('conv4_3_2', nn.Conv2d(512, 512, 3, 1, 1))
        # conv4_2.add_module('relu4_3_2', nn.ReLU())
        # self.conv4_2 = conv4_2
        #
        # conv5_2 = nn.Sequential()
        # conv5_2.add_module('pool4_2', nn.AvgPool2d(2, stride=2))
        # conv5_2.add_module('conv5_1_2', nn.Conv2d(512, 512, 3, 1, 1))
        # conv5_2.add_module('relu5_1_2', nn.ReLU())
        # conv5_2.add_module('conv5_2_2', nn.Conv2d(512, 512, 3, 1, 1))
        # conv5_2.add_module('relu5_2_2', nn.ReLU())
        # conv5_2.add_module('conv5_3_2', nn.Conv2d(512, 512, 3, 1, 1))
        # conv5_2.add_module('relu5_3_2', nn.ReLU())
        # self.conv5_2 = conv5_2
        # pre_train = torch.load('vgg16-397923af.pth')
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x1 = self.conv4_1(x)
        x1 = self.conv5_1(x1)
        # x2 = self.conv4_2(x)
        # x2 = self.conv5_2(x2)
        return x1

    def _initialize_weights(self):
        vgg_16 = models.vgg16(pretrained=True)
        pre_trained_dict = vgg_16.state_dict()

        keys = pre_trained_dict.keys()
        keys=list(keys)
        self.conv1.conv1_1.weight.data.copy_(pre_trained_dict[keys[0]])
        self.conv1.conv1_2.weight.data.copy_(pre_trained_dict[keys[2]])
        self.conv2.conv2_1.weight.data.copy_(pre_trained_dict[keys[4]])
        self.conv2.conv2_2.weight.data.copy_(pre_trained_dict[keys[6]])
        self.conv3.conv3_1.weight.data.copy_(pre_trained_dict[keys[8]])
        self.conv3.conv3_2.weight.data.copy_(pre_trained_dict[keys[10]])
        self.conv3.conv3_3.weight.data.copy_(pre_trained_dict[keys[12]])
        self.conv4_1.conv4_1_1.weight.data.copy_(pre_trained_dict[keys[14]])
        self.conv4_1.conv4_2_1.weight.data.copy_(pre_trained_dict[keys[16]])
        self.conv4_1.conv4_3_1.weight.data.copy_(pre_trained_dict[keys[18]])
        self.conv5_1.conv5_1_1.weight.data.copy_(pre_trained_dict[keys[20]])
        self.conv5_1.conv5_2_1.weight.data.copy_(pre_trained_dict[keys[22]])
        self.conv5_1.conv5_3_1.weight.data.copy_(pre_trained_dict[keys[24]])
        # self.conv4_2.conv4_1_2.weight.data.copy_(pre_trained_dict[keys[14]])
        # self.conv4_2.conv4_2_2.weight.data.copy_(pre_trained_dict[keys[16]])
        # self.conv4_2.conv4_3_2.weight.data.copy_(pre_trained_dict[keys[18]])
        # self.conv5_2.conv5_1_2.weight.data.copy_(pre_trained_dict[keys[20]])
        # self.conv5_2.conv5_2_2.weight.data.copy_(pre_trained_dict[keys[22]])
        # self.conv5_2.conv5_3_2.weight.data.copy_(pre_trained_dict[keys[24]])

        self.conv1.conv1_1.bias.data.copy_(pre_trained_dict[keys[1]])
        self.conv1.conv1_2.bias.data.copy_(pre_trained_dict[keys[3]])
        self.conv2.conv2_1.bias.data.copy_(pre_trained_dict[keys[5]])
        self.conv2.conv2_2.bias.data.copy_(pre_trained_dict[keys[7]])
        self.conv3.conv3_1.bias.data.copy_(pre_trained_dict[keys[9]])
        self.conv3.conv3_2.bias.data.copy_(pre_trained_dict[keys[11]])
        self.conv3.conv3_3.bias.data.copy_(pre_trained_dict[keys[13]])
        self.conv4_1.conv4_1_1.bias.data.copy_(pre_trained_dict[keys[15]])
        self.conv4_1.conv4_2_1.bias.data.copy_(pre_trained_dict[keys[17]])
        self.conv4_1.conv4_3_1.bias.data.copy_(pre_trained_dict[keys[19]])
        self.conv5_1.conv5_1_1.bias.data.copy_(pre_trained_dict[keys[21]])
        self.conv5_1.conv5_2_1.bias.data.copy_(pre_trained_dict[keys[23]])
        self.conv5_1.conv5_3_1.bias.data.copy_(pre_trained_dict[keys[25]])
        # self.conv4_2.conv4_1_2.bias.data.copy_(pre_trained_dict[keys[15]])
        # self.conv4_2.conv4_2_2.bias.data.copy_(pre_trained_dict[keys[17]])
        # self.conv4_2.conv4_3_2.bias.data.copy_(pre_trained_dict[keys[19]])
        # self.conv5_2.conv5_1_2.bias.data.copy_(pre_trained_dict[keys[21]])
        # self.conv5_2.conv5_2_2.bias.data.copy_(pre_trained_dict[keys[23]])
        # self.conv5_2.conv5_3_2.bias.data.copy_(pre_trained_dict[keys[25]])
