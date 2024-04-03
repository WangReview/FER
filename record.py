# -*- coding:utf-8 -*-
"""
author：wwm
data：2024-02-18
"""



import os
import matplotlib.pyplot as plt


# Record the accuracy and loss values on the training and testing sets of each epoch, 
# and draw the accuracy and loss curve on the training and testing sets
class RecorderMeter(object):
    def __init__(self):
        self.train_acc = []
        self.val_acc = []
        self.train_loss = []
        self.val_loss = []
        self.current_epoch = 0

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):
        acc_path = os.path.join(save_path, 'train_test_Acc.jpg')
        loss_path = os.path.join(save_path, 'train_test_Loss.jpg')

        # Draw accuracy curve
        fig_acc = plt.figure(1)
        plt.plot(self.train_acc, 'b', label='Train Acc')
        plt.plot(self.val_acc, 'r', label='Test Acc')
        plt.title('the accuracy curve of train and test')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()
        plt.savefig(acc_path, dpi=300)
        plt.close(fig_acc)

        # Draw loss value curve
        fig_loss = plt.figure(2)
        plt.plot(self.train_loss, 'b', label='Train Loss')
        plt.plot(self.val_loss, 'r', label='Test Loss')
        plt.title('the loss curve of train and val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.savefig(loss_path, dpi=300)
        plt.close(fig_loss)


















