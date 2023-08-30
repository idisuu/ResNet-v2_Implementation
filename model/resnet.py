# reference from three below three scripts 
# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
# https://github.com/JayPatwardhan/ResNet-PyTorch/blob/master/ResNet/ResNet.py
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py

import torch.nn as nn

from typing import List

class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_layers: List[int],        
    ):
        super(ResNet, self).__init__()

        self.block = block
        self.num_layers = num_layers

        # feature map size 유지를 위해 stride 1로 설정
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 =  nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        # feature map size 유지를 위해 maxpooling 생략
        #self.max_pool = nn.MaxPool2d(kernel_size =3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 1, self.num_layers[0])
        self.layer2 = self._make_layer(64, 128, 2, self.num_layers[1])
        self.layer3 = self._make_layer(128, 256, 2, self.num_layers[2])
        self.layer4 = self._make_layer(256, 512, 2, self.num_layers[3])        
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512*self.block.expansion, 10)

        # weight 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        print_mode = False
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if print_mode:
            print(x.shape)
        x = self.layer1(x)
        if print_mode:
            print(x.shape)
        x = self.layer2(x)
        if print_mode:
            print(x.shape)
        x = self.layer3(x)
        if print_mode:
            print(x.shape)
        x = self.layer4(x)
        if print_mode:
            print(x.shape)

        if self.block.preactivation:            
            x = self.relu(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return x

    def _make_layer(self, in_planes, planes, stride, num_layer):
        layer = []
        if not num_layer:
            raise Exception("num_layer should not be 0")
        # Bottleneck의 경우 layer 1은 256 --> 64가 아니라 64 --> 64이기 때문에 예외처리
        if in_planes == 64 and planes == 64 and self.block.expansion == 4:
            in_planes = int(in_planes/4)
        layer.append(
            self.block(
                in_planes,
                planes,
                stride
            )
        )
        for _ in range(1, num_layer):
            layer.append(
                self.block(
                    planes,
                    planes,
                    stride=1
                )
            )
        return nn.Sequential(*layer)


class Block(nn.Module):
    expansion = 1
    preactivation = False
    def __init__(self, in_planes, planes, stride):
        super(Block, self).__init__()        

        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 =  nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = None
        if (in_planes != planes) or (stride != 1):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes)
            )        

    def forward(self, x):
        identity = x.clone()
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample:
            identity = self.downsample(identity)
            
        x += identity
        x = self.relu(x)        
        return x


class Bottleneck(nn.Module):
    expansion = 4
    preactivation = False
    def __init__(self, in_planes, planes, stride):
        super(Bottleneck, self).__init__()
                
        
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_planes*self.expansion, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)

        self.downsample = None
        if (in_planes != planes) or (stride != 1):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes*self.expansion, planes*self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*self.expansion)
            )

    def forward(self, x):
        print_mode = False
        identity = x
        if print_mode:
            print('1', identity.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample:
            identity = self.downsample(identity)
        if print_mode:
            print('2', identity.shape)
            print('3', x.shape)
            print()
        x += identity
        x = self.relu(x)
        return x


class PreactivationBlock(nn.Module):
    expansion = 1
    preactivation = True
    def __init__(self, in_planes, planes, stride):
        super(PreactivationBlock, self).__init__()        

        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 =  nn.BatchNorm2d(in_planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = None
        if (in_planes != planes) or (stride != 1):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes)
            )        

    def forward(self, x):
        identity = x.clone()
        
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)       

        if self.downsample:
            identity = self.downsample(identity)
            
        x += identity               
        return x


class PreactivationBottleneck(nn.Module):
    expansion = 4
    preactivation = True
    def __init__(self, in_planes, planes, stride):
        super(PreactivationBottleneck, self).__init__()
                
        
        self.relu = nn.ReLU()
        
        self.bn1 = nn.BatchNorm2d(in_planes*self.expansion)
        self.conv1 = nn.Conv2d(in_planes*self.expansion, planes, kernel_size=1, stride=stride, bias=False)
        
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, stride=1, bias=False)        

        self.downsample = None
        if (in_planes != planes) or (stride != 1):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes*self.expansion, planes*self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*self.expansion)
            )

    def forward(self, x):
        identity = x

        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)

        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3(x)        

        if self.downsample:
            identity = self.downsample(identity)
        x += identity
        return x