# -*- coding:utf-8 -*-
"""
作者：wwm
日期：2024年02月18日
"""




'''==================工具函数======================='''

import torch
import random
import numpy as np
import os
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''==============1. 设置随机数种子============================'''
# 设置随机数种子
# def setup_seed(seed=3407):
def setup_seed(seed=666):
    random.seed(seed)   # Python本身的随机因素
    np.random.seed(seed)    # numpy随机因素
    torch.manual_seed(seed)     # pytorch随机因素,cpu随机种子
    torch.cuda.manual_seed(seed)  # pytorch随机因素,GPU随机种子
    torch.cuda.manual_seed_all(seed)    # pytorch随机因素,GPU随机种子

    # 用于保证CUDA 卷积运算的结果确定。
    torch.backends.cudnn.deterministic = True
    # 用于保证数据变化的情况下，减少网络效率的变化。为True的话容易降低网络效率。
    torch.backends.cudnn.benckmark = False
# setup_seed()

'''
# 设置随机数种子
# (1) Pytorch设置
seed = 2
torch.manual_seed(seed)     # 为CPU设置随机种子
torch.cuda.manual_seed(seed)    # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)    # 为所有GPU设置随机种子
# (2) Python & Numpy设置
# 如果读取数据的过程中采用了随机处理(如RandomCrop、RandomHorizontalFlip等)，那么对python、Numpy的随机数生成器也需要设置种子
random.seed(seed)
np.random.seed(seed)
'''


'''==============2. 设置断点续训============================'''
# check_path = resume_check_path
def resume_training(check_path, model, optimizer, scheduler, dataname='rafdb'):
    if os.path.isfile(check_path):
        print("==> loading checkpoint '{}'".format(check_path))
        checkpoint = torch.load(check_path, map_location=device)
        # model.load_state_dict(checkpoint['model_state_dict'])
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_schedule'])

        start_epoch = checkpoint['epoch']
        train_acc = checkpoint['train_acc']
        if dataname=='rafdb':
            test_acc = checkpoint['val_acc']
            # test_acc = checkpoint['test_acc']
        else:
            test_acc = checkpoint['test_acc']
        recorder = checkpoint['recorder']

        print("==> loaded checkpoint success '{}' (epoch {})".format(check_path, start_epoch))

        return model, optimizer, scheduler, start_epoch, train_acc, test_acc, recorder
    else:
        print("==> no checkpoint fond at '{}'".format(check_path))



'''==============3. 方法二：加载预训练模型权重============================'''
# checkpoint_path='networks/resnet18_ImageNet.pth'
# checkpoint_path='networks/resnet18_msceleb.pth'
# resnet18 = pretrained_model(resnet18, checkpoint_path, num_classes=7)
# print(resnet18)
# 实现表情7分类
'''===resnet18加载7分类并冻结层==='''
# def pretrained_model(model, checkpoint_path, pretrained_data='ImageNet', num_classes=7, freeze=True):
def pretrained_model(model, checkpoint_path, pretrained_data='ImageNet', num_classes=7):
    # # 加载预训练模型的state_dict
    # (1) ImageNet数据集上训练好的模型，官方模型 checkpoint_path='networks/resnet18_ImageNet.pth'
    if pretrained_data=='ImageNet':
        pretrained_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(pretrained_dict)
        # print('成功加载预训练模型的state_dict')
        # print(resnet18)

    # (2) 加载在msceleb数据集上已经训练好的模型，num_class也是1000分类
    if pretrained_data == 'msceleb':
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        # print('成功加载预训练模型的state_dict')

    # # 如果冻层
    # if freeze:
    #     # 遍历每个权重参数
    #     for param in model.parameters():
    #         param.requires_grad = False     # 冻结所有层，参数不会更新

    # 修改全连接层的输出数量
    model.fc = nn.Linear(512, num_classes)
    # summary(resnet18, (3,224,224))
    # print(model)

    # # 修改冻结层
    # for param in model.fc.parameters():
    #     param.requires_grad = True  # 最后全连接层取消冻，参数更新

    # # 修改后fc全连接层不再冻结，参数可被更新
    # # 查看模型各层的冻结情况
    # for m, n in model.named_parameters():
    #     print(m, n.requires_grad)

    return model
'''==============3. 方法二：加载预训练模型权重============================'''



'''==============3. 方法一：加载预训练模型权重============================'''
# # from torchvision import models
# # resnet18 = models.resnet18(weights=None)  # 加载官网resnet18模型
# # 实现表情7分类
# '''===官方resnet18加载7分类==='''
# def pretrained_model(resnet18, checkpoint_path, num_classes=7):
#     # 修改全连接层的输出数量
#     resnet18.fc = nn.Linear(512, num_classes)
#     # summary(resnet18, (3,224,224))
#     # # print(resnet18)
#
#     # 加载预训练模型的state_dict
#     # (1) ImageNet数据集上训练好的模型，官方模型 checkpoint_path='networks/resnet18_ImageNet.pth'
#     # (2) 加载在msceleb数据集上已经训练好的模型，num_class也是1000分类, checkpoint_path='networks/resnet18_msceleb.pth'
#     pretrained_dict = torch.load(checkpoint_path, map_location=device)
#
#     model_dict = resnet18.state_dict()
#
#     new_dict = {}
#     '''==============逐层查看代码==================='''
#     for k, v in pretrained_dict.items():
#         # print('输出', k, v.size())
#         if k not in model_dict:
#             ...
#             # print('预训练模型的state_dict中的键{}不在当前模型中'.format(k))
#         elif v.size() != model_dict[k].size():
#             ...
#             # print('键{}的尺寸不匹配，预训练模型的尺寸为{}，而当前模型的尺寸为{}'.format(k, v.size(), model_dict[k].size()))
#         elif (k in model_dict) and (v.size() == model_dict[k].size()):
#             # print('输出', k, v.size())
#             new_dict.update({k:v})
#     '''==============逐层查看代码==================='''
#
#     # 更新当前模型的state_dict
#     model_dict.update(new_dict)
#     resnet18.load_state_dict(model_dict)
#     print('成功加载预训练模型的state_dict')
#     # print(resnet18)
#
#     return resnet18
'''==============3. 方法一：加载预训练模型权重============================'''












































