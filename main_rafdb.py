# -*- coding:utf-8 -*-
"""
author：wwm
data：2024-02-19
"""



import os
import torch
import time
import torch.nn.parallel
import datetime
import torch.nn as nn
import torch.optim as optim
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eps = sys.float_info.epsilon

from utilities import setup_seed, resume_training

setup_seed()



from my_model5 import ImpResnet
resnet18_pretrained_path = '/root/autodl-nas/TestCode/Test1/networks/resnet18_msceleb.pth'
# resnet18_pretrained_path = '/root/autodl-nas/TestCode/Test1/networks/resnet18_ImageNet.pth'
# resnet18_pretrained_path='D:/Code/TestCode/My_Test_Model/networks/resnet18_msceleb.pth'
# resnet18_pretrained_path='D:/Code/TestCode/My_Test_Model/networks/resnet18_ImageNet.pth'
model = ImpResnet(pretrained_path=resnet18_pretrained_path, pretrained_data='msceleb',
                  num_class=7, pretrained=True).to(device)


from loops import train, validate


lr = 0.01
# lr = 0.1
momentum = 0.9
weight_decay = 1e-4     # 0.0001

batch_size = 32  

start_epoch = 0
total_epoch = 100

save_freq = 20



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)   
schedule_mode = str(scheduler).split('.')[3].split(' ')[0]

# Recorder
from record import RecorderMeter
recorder = RecorderMeter()


# Breakpoint continuation training
resume_check_path = 'D:/Code/TestCode/My_Test_Model/checkpoints/2024-01-21-00-47/4.pth'
resume = False
# resume = True
if resume:
    resume_file_name = resume_check_path.split('/')[-2]
    resume_file_name += str('_resume')
    model, optimizer, scheduler, start_epoch, train_acc, val_acc, recorder = \
        resume_training(resume_check_path, model, optimizer, scheduler)


# Load Dataset
# (1) RAFDB
from rafdb import get_rafdb_loaders
raf_path = '/root/autodl-nas/datasets/RAFDB'    
data_name = raf_path.split('/')[-1]
train_loader, val_loader = get_rafdb_loaders(data_path=raf_path, batch_size=batch_size)
# (2) FERPlus
# from ferplus import get_ferplus_loaders
# # ferplus_path = '/root/autodl-nas/datasets/FER2013_data'     
# ferplus_path = '/root/autodl-nas/datasets/ferplus'
# data_name = ferplus_path.split('/')[-1]
# train_loader, test_loader = get_ferplus_loaders(data_path=ferplus_path, batch_size=batch_size)


# Create weights and log folders
save_path = os.path.join(os.getcwd(), 'checkpoints')
if not os.path.exists(save_path):
    os.makedirs(save_path)



def main():
    # Record time
    now = datetime.datetime.now()
    print('Training time: ' + now.strftime("%Y-%m-%d %H:%M:%S"))  # Training time: 2024-01-19 00:42:20
    time_str = now.strftime("%Y-%m-%d-%H-%M")    # '2024-01-19-00-49'  年-月-日-时-分
    # Set breakpoint continuation path
    if resume:
        time_str = resume_file_name

    log_path = os.path.join(save_path, time_str)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    log_file_path = os.path.join(log_path, 'log.txt')
    with open(log_file_path, 'a') as f:
        f.write("data_name:%s, lr:%.6f,   momentum:%.2f,  weight_decay:%.6f,  batch_size:%d, schedule_mode:%s, \n"
                % (data_name, lr, momentum, weight_decay, batch_size, schedule_mode))
    print("data_name:%s, lr:%.6f,   momentum:%.2f,  weight_decay:%.6f,  batch_size:%d, schedule_mode:%s"
                % (data_name, lr, momentum, weight_decay, batch_size, schedule_mode))

    best_acc = 0.0
    best_epoch = 0
    best_train_acc = 0.0

    # Start training
    for epoch in range(start_epoch, total_epoch):
        start_time = time.time()
        # current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        # print('Current learning rate: ', current_learning_rate)

        # train for one epoch
        train_acc, train_loss = train(train_loader, model, criterion, optimizer)

        # evaluate on validation set
        val_acc, val_loss = validate(val_loader, model, criterion)

        scheduler.step()

        current_lr = optimizer.state_dict()['param_groups'][0]['lr']


        # recorder：epoch, train_loss, train_acc, val_loss, val_acc
        recorder.update(epoch, train_loss, train_acc, val_loss, val_acc)
        recorder.plot_curve(log_path)


        checkpoint = {'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'lr_schedule': scheduler.state_dict(),
                      'epoch': epoch+1,
                      'train_acc': train_acc,
                      'val_acc': val_acc,
                      'recorder': recorder}

        # Save the model every 10 epochs
        if ((epoch+1) % save_freq) == 0:
            freq_check_path = os.path.join(log_path, str(epoch+1) + '.pth')
            torch.save(checkpoint, freq_check_path)

        # If the accuracy on the current validation set is the highest, save the current model
        if val_acc >= best_acc:
            best_acc = val_acc
            best_epoch = epoch+1
            best_check_path = os.path.join(log_path, 'current_best.pth')
            if train_acc >= best_train_acc:
                best_train_acc = train_acc
                torch.save(checkpoint, best_check_path)


        # Record information related to each epoch
        # log_file_path = os.path.join(log_path, 'log.txt')
        template = ('Epoch:{:2d},   train_acc:{:.4f},   train_loss:{:.4f},  val_acc:{:.4f}, val_loss:{:.4f},    '
                    'lr:{:.4f}, best_epoch:{:2d},   best_acc:{:.4f}' + '\n')
        with open(log_file_path, 'a') as f:
            f.write(template.format(epoch+1, train_acc, train_loss, val_acc, val_loss,
                                    current_lr, best_epoch, best_acc))


        # Calculate the time consumed for each epoch
        end_time = time.time()
        epoch_time = end_time - start_time
        epoch_minutes = epoch_time / 60

        print("[Epoch %d]   train_acc:%.4f, train_loss:%.4f,    val_acc:%.4f,   val_loss:%.4f,  lr:%f,  "
              "time:%.2f"
              % (epoch+1, train_acc, train_loss, val_acc, val_loss, current_lr, epoch_minutes))




if __name__ == '__main__':
    main()



