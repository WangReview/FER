# -*- coding:utf-8 -*-
"""
author：wwm
data：2024-02-18
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
from Result5_2 import ImpResnet
from utilities import setup_seed, resume_training


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


resnet18_pretrained_path='D:/Code/TestCode/My_Test_Model/networks/resnet18_msceleb.pth'
# resnet18_pretrained_path='D:/Code/TestCode/My_Test_Model/networks/resnet18_ImageNet.pth'
model = ImpResnet(pretrained_path=resnet18_pretrained_path, pretrained_data='msceleb',
                  num_class=8, pretrained=False).to(device)


batch_size = 32
lr = 0.01
momentum = 0.9
weight_decay = 1e-4     # 0.0001

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)   


# checkpoint_path = '/root/autodl-nas/TestCode/test5/checkpoints/2024-02-19-21-49/current_best.pth'
checkpoint_path = 'D:/Code/TestCode/My_Test5_Model/my_model5_2/FERPlus/88.8110/current_best.pth'
model, optimizer, scheduler, start_epoch, train_acc, val_acc, recorder = \
    resume_training(checkpoint_path, model, optimizer, scheduler)


# Set learning rate
def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

re_lr = 0.0001
set_lr(optimizer, re_lr)



# optimizer.state_dict()['param_groups'][0]['lr'] = 0.0001
current_lr = optimizer.state_dict()['param_groups'][0]['lr']  
print("start_epoch:%d, train_acc:%.6f,   val_acc:%.6f,,  current_lr:%.6f"
      % (start_epoch, train_acc, val_acc, current_lr))






















