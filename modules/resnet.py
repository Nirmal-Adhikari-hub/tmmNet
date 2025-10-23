import time

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from einops import rearrange

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

from modules.gcn_lib.torch_vertex import Grapher, act_layer
from modules.gcn_lib.temgraph import TemporalGraph


def conv3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1, 3, 3),
        stride=(1, stride, stride),
        padding=(0, 1, 1),
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4  # Output channels are 4× the input channels

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d if hasattr(nn, "BatchNorm3d") else nn.BatchNorm2d

        width = planes
        self.conv1 = nn.Conv3d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)

        self.conv2 = nn.Conv3d(
            width, width, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = norm_layer(width)

        self.conv3 = nn.Conv3d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, use_graph=False):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.use_graph = use_graph

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # graph modules: construct or bypass
        if self.use_graph:
            self.localG = Grapher(in_channels=256, kernel_size=3, dilation=1, conv='edge', #mr
                                act='relu', norm="batch", bias=True, stochastic=False,
                                epsilon=0.0, r=1, n=14 * 14, drop_path=0.0, relative_pos=True)  # kernel_size=2
            self.localG2 = Grapher(in_channels=512, kernel_size=4, dilation=1, conv='edge',
                                act='relu', norm="batch", bias=True, stochastic=False,
                                epsilon=0.0, r=1, n=7 * 7, drop_path=0.0, relative_pos=True)  # kernel_size=2
            self.temporalG = TemporalGraph(k=14 * 14 // 4, in_channels=256, drop_path=0)
            self.temporalG2 = TemporalGraph(k=7 * 7, in_channels=512, drop_path=0)
        else:
            self.localG = self.localG2 = nn.Identity()
            self.temporalG = self.temporalG2 = nn.Identity()

        self.alpha = nn.Parameter(torch.ones(4), requires_grad=True)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
 

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        N, C, T, H, W = x.size()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # (n,c,t,h,w) = (1, 64, 100, 56, 56)
        x = self.layer1(x)  # (1, 64, 100, 56, 56)
        x = self.layer2(x)  # (1, 128, 100, 28, 28)
        x = self.layer3(x)  # (1, 256, 100, 14, 14)
        #
        N, C, T, H, W = x.size()
        x = rearrange(x, 'N C T H W -> (N T) C H W')  # [100x1, 256, 14, 14])
        x = x + (self.localG(x) * self.alpha[0] if self.use_graph else 0)
        x = x + (self.temporalG(x, N) * self.alpha[1] if self.use_graph else 0)
        x = x.view(N, T, C, H, W).permute(0, 2, 1, 3, 4)
        # #
        x = self.layer4(x)  # [1, 512, 100, 7, 7])
        # #
        N, C, T, H, W = x.size()
        x = rearrange(x, 'N C T H W -> (N T) C H W')  # [100x1, 512, 7, 7])
        x = x + (self.localG2(x) * self.alpha[2] if self.use_graph else 0)
        x = x + (self.temporalG2(x, N) * self.alpha[3] if self.use_graph else 0)
        x = x.view(N, T, C, H, W).permute(0, 2, 1, 3, 4)
        #

        x = x.transpose(1, 2).contiguous()  # [1, 100, 512, 7, 7]
        x = x.view((-1,) + x.size()[2:])  # bt,c,h,w  [100x1, 512, 7, 7]
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # bt,c (100x1,512)
        x = self.fc(x)  # bt,c (100x1,1000)

        return x


def resnet18(**kwargs):
    """Constructs a ResNet-18 based model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet18'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name:
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)
    model.load_state_dict(checkpoint, strict=False)
    return model


def resnet34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    # inflate 2D ImageNet weights to 3D (time kernel=1)
    checkpoint = model_zoo.load_url(model_urls['resnet34'])
    layer_names = list(checkpoint.keys())
    for ln in layer_names:
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)  # (out,in,1,kh,kw)
    model.load_state_dict(checkpoint, strict=False)
    return model

def inflate_weight_2d_to_3d(weight_2d, time_dim=3):
    """Replicate 2D weights along temporal dimension and normalize."""
    # weight_2d: (out, in, h, w)
    weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
    weight_3d /= time_dim  # normalize so total energy stays similar
    return weight_3d


def resnet101(**kwargs):
    """
    Build a ResNet-101 (3D version) inflated from 2D ImageNet weights.
    Temporal kernel = 3 for all convs.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet101'])
    inflated_checkpoint = {}

    for k, v in checkpoint.items():
        if 'conv' in k or 'downsample.0.weight' in k:
            # Inflate all 2D convs to 3D with temporal kernel size 3
            inflated_checkpoint[k] = inflate_weight_2d_to_3d(v, time_dim=3)
        else:
            inflated_checkpoint[k] = v

    msg = model.load_state_dict(inflated_checkpoint, strict=False)
    print(f"Loaded inflated resnet101 weights with message: {msg}")
    return model
 
if __name__ == "__main__":
    model = resnet34(use_graph=False)
    input = torch.randn(2, 3, 100, 224, 224)
    outputs = model(input)
    print(outputs.shape)