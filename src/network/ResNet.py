import torch.nn as nn
import torch
from einops.layers.torch import Rearrange

class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 kernel_stride: int = 1,
                 kernel_padding: int = 1,
                 pool: bool=False,
                 act_func: str = 'ReLU',
                 ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, kernel_stride, kernel_padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.act_func = getattr(nn, act_func)()
        self.pool = nn.MaxPool2d(2, 2) if pool else None

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.act_func(x)
        if self.pool:
            x = self.pool(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 kernel_stride: int = 1,
                 kernel_padding: int = 1,
                 act_func: str = 'ReLU',
                 ):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, kernel_stride, kernel_padding, act_func=act_func)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size, kernel_stride, kernel_padding, act_func=act_func)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=kernel_stride),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else None

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        if self.downsample is not None:
            output = torch.add(output, self.downsample(x))
        else:
            output = torch.add(output, x)
        return output

class ResNet11(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 act_func: str = 'ReLU',
                 ):
        super(ResNet11, self).__init__()
        self.prep = ConvBlock(in_channels, 64, kernel_size=3, kernel_stride=1, kernel_padding=1, act_func=act_func)
        self.layer1 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, kernel_stride=1, kernel_padding=1, pool=True, act_func=act_func),
            BasicBlock(128, 128, act_func=act_func),
        )
        self.layer2 = ConvBlock(128, 256, kernel_size=3, kernel_stride=1, kernel_padding=1, pool=True, act_func=act_func)
        self.layer3 = nn.Sequential(
            ConvBlock(256, 512, kernel_size=3, kernel_stride=1, kernel_padding=1, pool=True, act_func=act_func),
            BasicBlock(512, 512, act_func=act_func),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Rearrange('b c h w -> b (c h w)')
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d((1,1)), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(512, 1024),
                                        nn.Dropout(0.5),
                                        nn.Linear(1024, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out
