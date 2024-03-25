# -*- coding:utf-8 -*-
"""
作者：wwm
日期：2024年02月18日
"""



import torch.utils.data as data
import pandas as pd
import os
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

# ferplus_path = 'D:/Code/DataSets/FER2013Plus/ferplus'
# phase = 'train'
class FerPlusDataSet(data.Dataset):
    def __init__(self, ferplus_path, phase, transform=None):
        self.phase = phase
        self.transform = transform
        self.ferplus_path = ferplus_path

        if phase == 'train':
            self.data = pd.read_csv(os.path.join(self.ferplus_path, 'labels/ferplus_train_valid.txt'), sep=' ',
                             header=None, names=['name', 'label'])
        else:
            self.data = pd.read_csv(os.path.join(self.ferplus_path, 'labels/ferplus_test.txt'), sep=' ',
                             header=None, names=['name', 'label'])

        # 获取文件名
        # 获取name列的所有数据，返回ndarray(一维)    # ndarray: (28236,)
        file_names = self.data.loc[:, 'name'].values
        # 获取标签
        # 0：neutral，1：happiness，2：surprise，3：sadness，4：anger，5：disgust，6：fear，7：contempt
        self.label = self.data.loc[:, 'label'].values   # ndarray: (28236,)
        # 获取各个标签的次数，array([9913, 8146, 3547, 3370, 2387,  141,  596,  136], dtype=int64)
        _, self.sample_counts = np.unique(self.label, return_counts=True)

        self.file_paths = []
        for f in file_names:
            path = os.path.join(self.ferplus_path, 'images', phase, f)
            self.file_paths.append(path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.label[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# 数据预处理
mu=[0.485, 0.456, 0.406]
st=[0.229, 0.224, 0.225]
# 数据预处理
ferplus_transfrom = {
    "train": transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomApply([
                                     transforms.RandomRotation(20),
                                     transforms.RandomCrop(224, padding=32)], p=0.2),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=mu, std=st),
                                 transforms.RandomErasing(scale=(0.02, 0.25))]),
    "test": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=mu, std=st)])}


# 加载数据
# ferplus_path = 'D:/Code/DataSets/FER2013Plus/ferplus'
# def get_rafdb_loaders(data_path, batch_size=8):
def get_ferplus_loaders(data_path, batch_size):

    train_dataset = FerPlusDataSet(data_path, phase='train', transform=ferplus_transfrom["train"])
    test_dataset = FerPlusDataSet(data_path, phase='test', transform=ferplus_transfrom["test"])

    # 查看训练集和测试集数量
    # print('Whole train set size:', train_dataset.__len__())     # train set size: 12271
    # print('Validation set size:', val_dataset.__len__())        # Validation set size: 3068

    # 查看当前工作数
    workers = min([os.cpu_count(), batch_size if batch_size>1 else 0, 12])  # number of workers
    # print(workers)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, num_workers=workers,
                                   shuffle=True, pin_memory=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, num_workers=workers,
                                  shuffle=False, pin_memory=True)

    return train_loader, test_loader


# # """====取第一个批次的数据出来==="""
# if __name__ == '__main__':
#     # print()
#
#     ferplus_path = 'D:/Code/DataSets/FER2013Plus/ferplus'
#     train_loader, test_loader = get_ferplus_loaders(data_path=ferplus_path, batch_size=8)
#
#     train_num = len(train_loader.dataset)
#     test_num = len(test_loader.dataset)
#     print('train_num: ', train_num, 'test_num: ', test_num)
#
#
#     for images,target in test_loader:
#         print("Shape of X [N, C, H, W]: ", images.shape)     # torch.Size([8, 3, 224, 224])
#         print("Shape of y: ", target.shape, target.dtype)     # torch.Size([8]) torch.int64
#         print(target)   # tensor([3, 4, 3, 0, 3, 4, 3, 4])
#         break
# """====取第一个批次的数据出来==="""







































