# -*- coding:utf-8 -*-
"""
author：wwm
data：2024-02-18
"""



import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

beta = 0.6

# model training
# epoch = 0
def train(train_loader, net, criterion, optimizer):

    losses = 0.0
    correct = 0
    total_num = len(train_loader.dataset)
    batch_num = len(train_loader)

    net.train()


    for batch_idx, (images, target) in enumerate(train_loader):
        # images.size(): (8, 3, 224, 224)    target.size(): (8)   target.size(0): 8
        images = images.to(device)
        target = target.to(device)

        output1, output2 = net(images)      # output: (8,7)
        output = (beta*output1) + ((1-beta)*output2)

        # loss = criterion(output, target)    # tensor(2.2116, grad_fn=<NllLossBackward0>)
        loss = (beta * criterion(output1,target)) + ((1-beta) * criterion(output2,target))

        # loss
        losses += loss.item()
        # print(loss)
        _, predicted = torch.max(output.data, 1)    
        # correct += (predicted==target).sum().item()
        correct_num = torch.eq(predicted, target).sum().item()

        batch_acc = 100.0 * correct_num / target.size(0)
        correct += correct_num

        # compute gradient and do SGD step
        optimizer.zero_grad()   
        loss.backward()      
        optimizer.step()    

    train_acc = 100 * correct / float(total_num)
    train_loss = losses / batch_num

    return train_acc, train_loss







# Model testing
def validate(val_loader, net, criterion):
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

            # loss
            losses += loss.item()
            _, predicted = torch.max(output.data, 1)  
            correct_num = torch.eq(predicted, target).sum().item()
            # batch_acc = 100* correct_num.float() / float(target.size(0))
            batch_acc = 100.0 * correct_num / target.size(0)
            # print(batch_acc)
            correct += correct_num

    test_acc = 100 * correct / float(total_num)
    test_loss = losses / batch_num

    return test_acc, test_loss

