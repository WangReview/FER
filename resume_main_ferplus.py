# -*- coding:utf-8 -*-
"""
作者：wwm
日期：2024年03月01日
"""

import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
# from grad_cam import GradCAM, show_cam_on_image, center_crop_img
from cam.grad_2_cam import GradCAM, show_cam_on_image, center_crop_img

import torch.optim as optim
# 加载网络模型和预训练模型
from Result5_2 import ImpResnet
# 导入相关信息设置的类：随机种子，断点续训，预训练模型
from utilities import setup_seed, resume_training


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 服务器权重路径
# resnet18_pretrained_path = '/root/autodl-nas/TestCode/Test1/networks/resnet18_msceleb.pth'
# resnet18_pretrained_path = '/root/autodl-nas/TestCode/Test1/networks/resnet18_ImageNet.pth'
# 本地模型权重路径
resnet18_pretrained_path='D:/Code/TestCode/My_Test_Model/networks/resnet18_msceleb.pth'
# resnet18_pretrained_path='D:/Code/TestCode/My_Test_Model/networks/resnet18_ImageNet.pth'
model = ImpResnet(pretrained_path=resnet18_pretrained_path, pretrained_data='msceleb',
                  num_class=8, pretrained=False).to(device)


batch_size = 32
lr = 0.01
momentum = 0.9
weight_decay = 1e-4     # 0.0001

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
# (1) StepLR(固定步长衰减)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)   # lr=0.01,step_size=10,此效果最好

# 加载网络权重
# 断点续训权重路径
# checkpoint_path = '/root/autodl-nas/TestCode/test5/checkpoints/2024-02-19-21-49/current_best.pth'
checkpoint_path = 'D:/Code/TestCode/My_Test5_Model/my_model5_2/FERPlus/88.8110/current_best.pth'
model, optimizer, scheduler, start_epoch, train_acc, val_acc, recorder = \
    resume_training(checkpoint_path, model, optimizer, scheduler)


'''================ 设置新的学习率 =============='''
# 设置学习率
def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

re_lr = 0.0001
set_lr(optimizer, re_lr)
'''================ 设置新的学习率 =============='''


# optimizer.state_dict()['param_groups'][0]['lr'] = 0.0001
# 打印出来：记录超参数设置情况
current_lr = optimizer.state_dict()['param_groups'][0]['lr']  # 获取当前学习率
print("start_epoch:%d, train_acc:%.6f,   val_acc:%.6f,,  current_lr:%.6f"
      % (start_epoch, train_acc, val_acc, current_lr))






















