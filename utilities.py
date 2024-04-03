# -*- coding:utf-8 -*-
"""
author：wwm
data：2024-02-18
"""




'''==================Instrumental function======================='''

import torch
import random
import numpy as np
import os
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''==============1. Set random number seeds============================'''

def setup_seed(seed=666):
    random.seed(seed)   
    np.random.seed(seed)    
    torch.manual_seed(seed)     
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)    

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benckmark = False
# setup_seed()



'''==============2. Set breakpoint continuation training============================'''
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



'''==============3. Load pre trained model weights============================'''
# checkpoint_path='networks/resnet18_ImageNet.pth'
# checkpoint_path='networks/resnet18_msceleb.pth'
# resnet18 = pretrained_model(resnet18, checkpoint_path, num_classes=7)
# print(resnet18)
# def pretrained_model(model, checkpoint_path, pretrained_data='ImageNet', num_classes=7, freeze=True):
def pretrained_model(model, checkpoint_path, pretrained_data='ImageNet', num_classes=7):
    # (1) ImageNet checkpoint_path='networks/resnet18_ImageNet.pth'
    if pretrained_data=='ImageNet':
        pretrained_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(pretrained_dict)
        # print(resnet18)

    # (2) msceleb
    if pretrained_data == 'msceleb':
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=True)

    model.fc = nn.Linear(512, num_classes)


    return model
'''==============3. ============================'''











































