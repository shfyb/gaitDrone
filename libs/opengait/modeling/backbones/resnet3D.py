from torch.nn import functional as F
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from ..modules import BasicConv2d,BasicConv3d
from typing import Any, Callable, List, Optional, Type, Union
import pdb

block_map = {'BasicBlock': BasicBlock,
             'Bottleneck': Bottleneck}


def trans(x):
    n, c, s, h, w = x.size()
    x = x.transpose(1, 2).reshape(-1, c, h, w)
    return x

def trans_out(x, n, s):
    # pdb.set_trace()
    output_size = x.size()
    return x.reshape(n, s, *output_size[1:]).transpose(1, 2).contiguous()
class BasicResBlock_3D(nn.Module):
    def __init__(self,in_channels, out_channels,  stride=1, downsample=None):
        super(BasicResBlock_3D, self).__init__()
        self.downsample = downsample
        self.conv1 = BasicConv3d(in_channels, out_channels, kernel_size=3, stride=(1,stride,stride), padding=1)
        self.conv2 = BasicConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        if downsample is not None or stride!=1 :
            self.downsample = nn.Sequential(
                BasicConv3d(in_channels, out_channels, kernel_size=1, stride=(1,stride,stride),padding=0),
                nn.BatchNorm3d(out_channels))
        else:
            self.dowmsample = None
    
   
  
    def forward(self, x):

       
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicResBlock_2D(nn.Module):
    def __init__(self,in_channels, out_channels,  stride=1, downsample=None):
        super(BasicResBlock_2D, self).__init__()
        self.downsample = downsample
        self.conv1 = BasicConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = BasicConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        if downsample is not None or stride!=1 :
            self.downsample = nn.Sequential(
                BasicConv2d(in_channels, out_channels, kernel_size=1, stride=stride,padding=0),
                nn.BatchNorm2d(out_channels))
        else:
            self.dowmsample = None
    
   
  
    def forward(self, x):

       
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet9_3D(nn.Module):
    def __init__(self, block, channels=[32, 64, 128, 256], in_channel=1, layers=[1, 2, 2, 1], strides=[1, 2, 2, 1], maxpool=True):
        # if block in block_map.keys():
        #     block = block_map[block]
        # else:
        #     raise ValueError(
        #         "Error type for -block-Cfg-, supported: 'BasicBlock' or 'Bottleneck'.")
        self.maxpool_flag = maxpool
        super(ResNet9_3D, self).__init__()

        # Not used #
        self.fc = None
        ############
        self.inplanes = channels[0]
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = BasicConv2d(in_channel, self.inplanes, 3, 1, 1)

        self.layer1 = BasicResBlock_2D(
             channels[0], channels[0], stride=strides[0])

        self.layer2 = BasicResBlock_2D(
             channels[0], channels[1], stride=strides[1])
        self.layer3 = BasicResBlock_2D(
             channels[1], channels[2], stride=strides[2])
        self.layer4 = BasicResBlock_2D(
             channels[2], channels[3], stride=strides[3],downsample=True)



    def forward(self, x, seqL):
        
        n,c,s,_,_ = x.size()
        
        x = trans(x)
      
        
           
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        if self.maxpool_flag:
            x = self.maxpool(x)
        
        
        x = self.layer1(x)

        # else:
        #     x = trans_out(x,n,s)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if seqL is not None:
            x = trans_out(x,n,seqL)
        else:
            x = trans_out(x,n,s)
        return x



