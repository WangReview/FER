# -*- coding:utf-8 -*-
"""
作者：wwm
日期：2024年02月19日
"""




import torch
import torch.nn as nn
from torchvision import models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型权重路径
# resnet18_pretrained_path='D:/Code/TestCode/My_Test_Model/networks/resnet18_msceleb.pth'
# resnet18_pretrained_path='D:/Code/TestCode/My_Test_Model/networks/resnet18_ImageNet.pth'


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


# (1) 注意力机制SENet
class SEAttention(nn.Module):

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        # 对空间信息进行压缩
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 经过两次全连接层，学习不同通道的重要性
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 取出batch size和通道数
        b, c, _, _ = x.size()
        # b,c,w,h -> b,c,1,1 -> b,c 压缩与通道信息学习
        y = self.avg_pool(x).view(b, c)
        # b,c->b,c->b,c,1,1
        y = self.fc(y).view(b, c, 1, 1)
        # 激励操作
        return x * y.expand_as(x)


# if __name__ == '__main__':
#     input = torch.randn(50, 512, 7, 7)
#     se = SEAttention(channel=512, reduction=8)
#     output = se(input)
#     print(input.shape)
#     print(output.shape)


# (2) 注意力机制
# 空间注意力模块
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
        # self.act=SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        out = x * out
        return out

# if __name__ == '__main__':
#     input = torch.randn(50, 512, 7, 7)
#     sp = SpatialAttentionModule()
#     output = sp(input)
#     print(input.shape)
#     print(output.shape)


# (3) CBAM注意力机制
class CBAMBlock(nn.Module):
    def __init__(self, in_channel, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        # 通道注意力机制
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=in_channel//reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel//reduction, out_features=in_channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        # 空间注意力机制
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              stride=1, padding=kernel_size//2, bias=False)

    def forward(self, x):
        # 通道注意力机制
        maxout = self.max_pool(x)       # N,C,H,W --> N,C,1,1
        maxout = self.mlp(maxout.view(maxout.size(0), -1))  # N,C,1,1 --> N,C
        avgout = self.avg_pool(x)       # N,C,H,W --> N,C,1,1
        avgout = self.mlp(avgout.view(avgout.size(0), -1))  # N,C,1,1 --> N,C
        channel_out = self.sigmoid(maxout+avgout)       # N,C --> N,C
        channel_out = channel_out.view(x.size(0), x.size(1), 1, 1)  # N,C --> N,C,1,1
        channel_out = channel_out * x       # N,C,1,1 --> N,C,H,W
        # 空间注意力机制
        max_out, _ = torch.max(channel_out, dim=1, keepdim=True)    # N,C,H,W --> N,1,C,H
        mean_out = torch.mean(channel_out, dim=1, keepdim=True)     # N,C,H,W --> N,1,C,H
        out = torch.cat((max_out, mean_out), dim=1)     # N,1,C,H --> N,2,C,H
        out = self.sigmoid(self.conv(out))      # N,2,C,H --> N,1,C,H
        out = out * channel_out         # N,1,C,H --> N,C,H,W
        return out

# if __name__ == '__main__':
#     input = torch.randn(50, 512, 7, 7)  # b,c,h,w
#     cbam = CBAMBlock(in_channel=512, reduction=16, kernel_size=7)
#     output = cbam(input)
#     print(input.shape)
#     print(output.shape)



# pretrained_path = resnet18_pretrained_path
class ImpResnet(nn.Module):
    def __init__(self, pretrained_path, pretrained_data='ImageNet', num_class=7, pretrained=True):
        super(ImpResnet, self).__init__()

        self.in_channel = 256

        # resnet = models.resnet18(weights=None).to(device)    # 本地加载
        resnet = models.resnet18(pretrained=False).to(device)  # 服务器加载

        if pretrained:
            if pretrained_data=='ImageNet':
                # (1) ImageNet数据集上训练好的模型，官方模型
                pretrained_dict = torch.load(pretrained_path, map_location=device)
                resnet.load_state_dict(pretrained_dict)
            if pretrained_data == 'msceleb':
                # (2) 加载在msceleb数据集上已经训练好的模型
                checkpoint = torch.load(pretrained_path, map_location=device)
                resnet.load_state_dict(checkpoint['state_dict'], strict=True)

        # 将ResNet18的特征提取部分迁移过来，去掉resnet18的最后两层。【B,C,H,W】= [B, 256, 14, 14]
        self.features = nn.Sequential(*list(resnet.children())[:-3])    # [:-3] --> [0:-3]

        # 分支一
        self.cbam1 = CBAMBlock(in_channel=256, reduction=4, kernel_size=3)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.cbam2 = CBAMBlock(in_channel=512, reduction=4, kernel_size=3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_1 = nn.Linear(512, num_class)

        # 分支二
        self.conv1 = nn.Conv2d(256, 512, 3, 2)
        self.se1 = SEAttention(channel=512, reduction=8)
        self.conv2 = nn.Conv2d(256, 512, 3, 2)
        self.se2 = SEAttention(channel=512, reduction=8)
        self.conv3 = nn.Conv2d(256, 512, 3, 2)
        self.se3 = SEAttention(channel=512, reduction=8)
        self.conv4 = nn.Conv2d(256, 512, 3, 2)
        self.se4 = SEAttention(channel=512, reduction=8)
        self.sp = SpatialAttentionModule()
        self.avgpool_2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_2 = nn.Linear(512, num_class)


    def forward(self, x):
        x = self.features(x)    # (B,256,14,14)

        # 分支一
        out1 = self.cbam1(x)    # (B,512,7,7)
        out1 = self.layer4(out1)   # (B,512,7,7)
        out1 = self.cbam2(out1)  # (B,512,7,7)
        out1 = self.avgpool(out1)  # (B,512,1,1)
        out1 = torch.flatten(out1, 1)   # (B,512)
        out1 = self.fc_1(out1)  # (B,7)

        # 分支二
        patch_11 = x[:, :, 0:7, 0:7]          # 7*7*256，注意拆分的维度dim=3，后面合并
        patch_12 = x[:, :, 0:7, 7:14]         # 7*7*256
        patch_21 = x[:, :, 7:14, 0:7]         # 7*7*256
        patch_22 = x[:, :, 7:14, 7:14]        # 7*7*256
        branch_1_1 = self.conv1(patch_11)   # 3*3*512
        branch_1_1 = self.se1(branch_1_1)
        branch_1_2 = self.conv2(patch_12)  # 3*3*512
        branch_1_2 = self.se2(branch_1_2)
        branch_2_1 = self.conv3(patch_21)  # 3*3*512
        branch_2_1 = self.se3(branch_2_1)
        branch_2_2 = self.conv4(patch_22)  # 3*3*512
        branch_2_2 = self.se4(branch_2_2)
        branch_1_out = torch.cat([branch_1_1, branch_1_2], dim=3)   # 3*6*512
        branch_2_out = torch.cat([branch_2_1, branch_2_2], dim=3)   # 3*6*512
        branch = torch.cat([branch_1_out, branch_2_out], dim=2)     # 6*6*512
        branch = self.sp(branch)

        out2 = self.avgpool_2(branch)   # (B,512,1,1)
        out2 = torch.flatten(out2, 1)   # (B,512)
        out2 = self.fc_2(out2)  # (B,7)

        return out1, out2


    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel))

        return nn.Sequential(*layers)



# 模型权重路径
# resnet18_pretrained_path='D:/Code/TestCode/My_Test_Model/networks/resnet18_msceleb.pth'
# resnet18_pretrained_path='D:/Code/TestCode/My_Test_Model/networks/resnet18_ImageNet.pth'
# model = ImpResnet(pretrained_path=resnet18_pretrained_path, pretrained_data='ImageNet')
# # print(model)
# from torchsummary import summary
# summary(model, (3,224,224))


























































