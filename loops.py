# -*- coding:utf-8 -*-
"""
作者：wwm
日期：2024年02月18日
"""



import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

beta = 0.6

# 模型训练
# epoch = 0
def train(train_loader, net, criterion, optimizer):

    # print_freq = 10     # 打印频率

    losses = 0.0
    correct = 0
    total_num = len(train_loader.dataset)
    batch_num = len(train_loader)

    net.train()

    """====取第一个批次的数据出来==="""
    # for images, target in train_loader:
    #     print("Shape of X [N, C, H, W]: ", images.shape)     # torch.Size([8, 3, 224, 224])
    #     print("Shape of y: ", target.shape, target.dtype)     # torch.Size([8]) torch.int64
    #     print(target)   # tensor([3, 4, 3, 3, 4, 3, 3, 3])
    #     break
    """====取第一个批次的数据出来==="""

    for batch_idx, (images, target) in enumerate(train_loader):
        # images.size(): (8, 3, 224, 224)    target.size(): (8)   target.size(0): 8
        images = images.to(device)
        target = target.to(device)

        output1, output2 = net(images)      # output: (8,7)
        output = (beta*output1) + ((1-beta)*output2)

        # loss = criterion(output, target)    # tensor(2.2116, grad_fn=<NllLossBackward0>)
        loss = (beta * criterion(output1,target)) + ((1-beta) * criterion(output2,target))

        # 损失值
        losses += loss.item()
        # print(loss)
        _, predicted = torch.max(output.data, 1)    # 返回所在行(dim=1)的最大值对应的索引
        # 准确率
        # correct += (predicted==target).sum().item()
        correct_num = torch.eq(predicted, target).sum().item()

        batch_acc = 100.0 * correct_num / target.size(0)
        correct += correct_num

        # compute gradient and do SGD step
        optimizer.zero_grad()   # 梯度归零
        loss.backward()      #反向传播
        optimizer.step()    #更新参数

        # # print loss and accuracy, batch_idx=0
        # # 打印损失值和精确度，每个10batch打印一次
        # if batch_idx % print_freq == 0:
        #     print("[Training] batch_idx: %d.  batch_acc: %2.4f%%.  batch_loss: %.4f "
        #           % (batch_idx, batch_acc, loss))

    train_acc = 100 * correct / float(total_num)
    train_loss = losses / batch_num

    return train_acc, train_loss







# 模型测试
def validate(val_loader, net, criterion):

    # print_freq = 5  # 打印频率

    losses = 0.0
    correct = 0
    total_num = len(val_loader.dataset)
    batch_num = len(val_loader)

    net.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # output = net(images)
            # loss = criterion(output, target)

            output1, output2 = net(images)  # output: (8,7)
            output = (beta * output1) + ((1 - beta) * output2)

            # loss = criterion(output, target)    # tensor(2.2116, grad_fn=<NllLossBackward0>)
            loss = (beta * criterion(output1, target)) + ((1 - beta) * criterion(output2, target))

            # 损失值
            losses += loss.item()
            _, predicted = torch.max(output.data, 1)  # 返回所在行(dim=1)的最大值对应的索引
            # 准确率
            # correct += (predicted==target).sum().item()
            correct_num = torch.eq(predicted, target).sum().item()
            # batch_acc = 100* correct_num.float() / float(target.size(0))
            batch_acc = 100.0 * correct_num / target.size(0)
            # print(batch_acc)
            correct += correct_num

            # if i % print_freq == 0:
            #     print("[Test] batch_idx: %d.  batch_acc: %2.4f%%.  batch_loss: %.4f "
            #           % (i, batch_acc, loss))

    test_acc = 100 * correct / float(total_num)
    test_loss = losses / batch_num

    return test_acc, test_loss

































