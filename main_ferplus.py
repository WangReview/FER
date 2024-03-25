# -*- coding:utf-8 -*-
"""
作者：wwm
日期：2024年02月18日
"""




import os
import torch
import time
import torch.nn.parallel
import datetime
import torch.nn as nn
import torch.optim as optim
# from torchvision import models
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sys.float_info.epsilon：最小可表示的正浮点数，通常为2.220446049250313e-16；
eps = sys.float_info.epsilon

# 导入相关信息设置的类：随机种子，断点续训，预训练模型
from utilities import setup_seed, resume_training, pretrained_model

# 设置随机种子
setup_seed()


# 加载网络模型和预训练模型
from my_model5 import ImpResnet
# resnet18_pretrained_path='/root/autodl-nas/TestCode/Test1/networks/resnet18_msceleb.pth'
resnet18_pretrained_path='D:/Code/TestCode/My_Test1_Model/networks/resnet18_msceleb.pth'
model = ImpResnet(pretrained_path=resnet18_pretrained_path, pretrained_data='msceleb',
                  num_class=8, pretrained=True).to(device)


# 导入训练和验证程序
from loops import train, validate


# 超参数
lr = 0.01
# lr = 0.1
momentum = 0.9
weight_decay = 1e-4     # 0.0001

# 两模型合并，resnet18+vit, bs=8(88.4)和bs=32(88.2)，效果最好
# batch_size = 8  # 本地批次
# batch_size = 64  # 服务器批次
batch_size = 32  # 服务器批次

start_epoch = 0
total_epoch = 100


# 每个20个epoch保存一次模型
save_freq = 20


# 设置损失函数，优化器，学习率调整
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
# (1) StepLR(固定步长衰减)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)   # lr=0.01,step_size=10,此效果最好
# (2) CosineAnnealingLR 余弦退火衰减会使学习率产生周期性的变化，其主要参数有两个，一个表示周期，一个表示学习率的最小值。
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=1e-6)
# (3) CosineAnnealingWarmRestarts 使用余弦退火调度设置每个参数组的学习率
# scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, T_mult=2, eta_min=-1e-5)
# (4) ReduceLROnPlateau(动态衰减学习率) 当指标停止改进时降低学习率。
# 一旦学习停滞，模型通常会受益于将学习率降低 2-10 倍。该调度程序读取一个指标数量，如果“patience”的 epoch 数量没有改善，则学习率会降低。
# (5) MultiStepLR(多步长衰减)
# print(str(scheduler).split('.')[3].split(' ')[0])
schedule_mode = str(scheduler).split('.')[3].split(' ')[0]

# 记录器
from record import RecorderMeter
recorder = RecorderMeter()


# 断点续训
# 断点续训权重路径
resume_check_path = 'D:/Code/TestCode/My_Test_Model/checkpoints/2024-01-21-00-47/4.pth'
resume = False
# resume = True
if resume:
    # 设置断点续训路径
    resume_file_name = resume_check_path.split('/')[-2]
    resume_file_name += str('_resume')
    # 断点续训
    model, optimizer, scheduler, start_epoch, train_acc, test_acc, recorder = \
        resume_training(resume_check_path, model, optimizer, scheduler)


# 加载数据集
# (1) RAFDB数据集
# from rafdb import get_rafdb_loaders
# raf_path = 'D:/Code/TestCode/My_Test_Model/data'  # 本地数据集
# raf_path = '/root/autodl-nas/datasets/RAFDB'    # 服务器数据集
# data_name = raf_path.split('/')[-1]
# train_loader, val_loader = get_rafdb_loaders(data_path=raf_path, batch_size=batch_size)
# (2) FERPlus数据集
from ferplus import get_ferplus_loaders
# ferplus_path = '/root/autodl-nas/datasets/FER2013_data'     # 本地数据集
# 北京A区 FERPlus数据集
ferplus_path = '/root/autodl-nas/datasets/ferplus'
data_name = ferplus_path.split('/')[-1]
train_loader, test_loader = get_ferplus_loaders(data_path=ferplus_path, batch_size=batch_size)


# 创建权重、日志文件夹
save_path = os.path.join(os.getcwd(), 'checkpoints')
if not os.path.exists(save_path):
    os.makedirs(save_path)


def main():
    # 记录时间
    now = datetime.datetime.now()
    print('Training time: ' + now.strftime("%Y-%m-%d %H:%M:%S"))  # Training time: 2024-01-19 00:42:20
    time_str = now.strftime("%Y-%m-%d-%H-%M")    # '2024-01-19-00-49'  年-月-日-时-分
    # 设置断点续训路径
    if resume:
        time_str = resume_file_name

    # 创建当前训练时刻的权重、日志文件夹
    log_path = os.path.join(save_path, time_str)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # 日志记录：记录超参数设置情况
    log_file_path = os.path.join(log_path, 'log.txt')
    with open(log_file_path, 'a') as f:
        f.write("data_name:%s, lr:%.6f,   momentum:%.2f,  weight_decay:%.6f,  batch_size:%d, schedule_mode:%s, \n"
                % (data_name, lr, momentum, weight_decay, batch_size, schedule_mode))
    # 打印出来：记录超参数设置情况
    print("data_name:%s, lr:%.6f,   momentum:%.2f,  weight_decay:%.6f,  batch_size:%d, schedule_mode:%s"
                % (data_name, lr, momentum, weight_decay, batch_size, schedule_mode))

    best_acc = 0.0
    best_epoch = 0
    best_train_acc = 0.0

    # 开始训练
    for epoch in range(start_epoch, total_epoch):
        start_time = time.time()
        # current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        # print('Current learning rate: ', current_learning_rate)

        # train for one epoch
        train_acc, train_loss = train(train_loader, model, criterion, optimizer)

        # evaluate on test dataset
        test_acc, test_loss = validate(test_loader, model, criterion)

        scheduler.step()

        # 获取当前学习率
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']


        # 记录器，记录epoch, train_loss, train_acc, val_loss, val_acc
        recorder.update(epoch, train_loss, train_acc, test_loss, test_acc)
        # 记录器，绘图：绘制损失值曲线和准确率曲线
        recorder.plot_curve(log_path)


        # 将网络训练过程中的网络权重，优化器的权重，epoch保存，便于继续恢复训练
        checkpoint = {'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'lr_schedule': scheduler.state_dict(),
                      'epoch': epoch+1,
                      'train_acc': train_acc,
                      'test_acc': test_acc,
                      'recorder': recorder}

        # 每隔10个epoch保存一次模型
        if ((epoch+1) % save_freq) == 0:
            freq_check_path = os.path.join(log_path, str(epoch+1) + '.pth')
            torch.save(checkpoint, freq_check_path)

        # 如当前验证集上的准确率最高，保存当前模型
        if test_acc >= best_acc:
            best_acc = test_acc
            best_epoch = epoch+1
            best_check_path = os.path.join(log_path, 'current_best.pth')
            if train_acc >= best_train_acc:
                best_train_acc = train_acc
                torch.save(checkpoint, best_check_path)


        # 记录每个epoch相关信息
        # log_file_path = os.path.join(log_path, 'log.txt')
        template = ('Epoch:{:2d},   train_acc:{:.4f},   train_loss:{:.4f},  val_acc:{:.4f}, val_loss:{:.4f},    '
                    'lr:{:.4f}, best_epoch:{:2d},   best_acc:{:.4f}' + '\n')
        with open(log_file_path, 'a') as f:
            f.write(template.format(epoch+1, train_acc, train_loss, test_acc, test_loss,
                                    current_lr, best_epoch, best_acc))


        # 计算每个epoch消耗的时间
        end_time = time.time()
        epoch_time = end_time - start_time
        # 将时间差转换成分钟
        epoch_minutes = epoch_time / 60
        # # 将时间差值转换为小时、分钟和秒
        # hours = int(epoch_time // (60 * 60))
        # minutes = int((epoch_time % (60 * 60)) // 60)
        # seconds = epoch_time % 60
        # print("经过了{}小时 {}分钟 {}秒".format(hours, minutes, seconds))
        # print('An Epoch time: ', epoch_time)

        # 打印每个epoch相关信息，每批次消耗时间
        print("[Epoch %d]   train_acc:%.4f, train_loss:%.4f,    val_acc:%.4f,   val_loss:%.4f,  lr:%f,  "
              "time:%.2f"
              % (epoch+1, train_acc, train_loss, test_acc, test_loss, current_lr, epoch_minutes))















if __name__ == '__main__':
    main()

















