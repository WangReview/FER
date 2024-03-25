# -*- coding:utf-8 -*-
"""
作者：wwm
日期：2024年02月19日
"""




import torch.utils.data as data
import pandas as pd
import os
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


# # 设置随机种子
# from utilities import setup_seed
# setup_seed()

# 数据读取
# raf_path = 'D:/Code/DataSets/RAF-DB/basic'
# raf_path = r'D:\Code\DataSets\RAF-DB\basic'
class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform=None):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path

        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ',
                         header=None, names=['name', 'label'])

        if phase == 'train':
            # startswitch()用于检测字符串是否以指定字符串开头。如果是则返回True，否则返回False.
            self.data = df[df['name'].str.startswith('train')]  # 选出训练集pands结构
        else:
            self.data = df[df['name'].str.startswith('test')]  # 选出测试集pands结构

        # 获取name列的所有数据，返回类型为ndarray（一维） # ndaaray:(12271,)
        file_names = self.data.loc[:, 'name'].values
        # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        # 由于数据集标签从1开始到7，为了从0开始到6，对应所有标签减1
        self.label = self.data.loc[:, 'label'].values - 1  # ndaaray:(12271,) 所有标签数值减1

        # numpy.unique(arr, return_index, return_inverse, return_counts)
        # 对于一维数组或者列表，np.unique() 函数 去除其中重复的元素，并按元素由小到大返回一个新的无元素重复的元组或者列表。
        # return_counts：如果为true，返回去重数组中的元素在原数组中的出现次数。
        # sample_counts：array([1290, 281, 717, 4772, 1982, 705, 2524], dtype=int64)
        _, self.sample_counts = np.unique(self.label, return_counts=True)  # sample_counts各个标签出现的次数
        # print(f' distribution of {phase} samples: {self.sample_counts}')

        # use raf-db aligned images for training/testing
        self.file_paths = []
        for f in file_names:
            f = f.split(".")[0]
            f = f + "_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        # 使用convert(‘RGB’)进行通道转换
        image = Image.open(path).convert('RGB')
        label = self.label[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# 数据预处理
mu=[0.485, 0.456, 0.406]
st=[0.229, 0.224, 0.225]
# 数据预处理
rafdb_transfrom = {
    "train": transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomApply([
                                     transforms.RandomRotation(20),
                                     transforms.RandomCrop(224, padding=32)], p=0.2),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=mu, std=st),
                                 transforms.RandomErasing(scale=(0.02, 0.25))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=mu, std=st)])}


# 加载数据
# raf_path = 'D:/Code/DataSets/RAF-DB/basic'
# 测试数据，小型数据
# raf_path = 'D:/Code/TestCode/My_Test_Model/data'
# def get_rafdb_loaders(data_path, batch_size=8):
def get_rafdb_loaders(data_path, batch_size):

    train_dataset = RafDataSet(data_path, phase='train', transform=rafdb_transfrom["train"])
    val_dataset = RafDataSet(data_path, phase='test', transform=rafdb_transfrom["val"])

    # 查看训练集和测试集数量
    # print('Whole train set size:', train_dataset.__len__())     # train set size: 12271
    # print('Validation set size:', val_dataset.__len__())        # Validation set size: 3068

    # 查看当前工作数
    workers = min([os.cpu_count(), batch_size if batch_size>1 else 0, 12])  # number of workers
    # print(workers)

    # train_loader = data.DataLoader(train_dataset, batch_size=batch_size, num_workers=workers,
    #                                shuffle=True, pin_memory=True)
    # train_loader = data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0,
    #                                worker_init_fn=setup_seed(), shuffle=True, pin_memory=True)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size,
                                   num_workers=workers, shuffle=True, pin_memory=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size,
                                 num_workers=workers, shuffle=False, pin_memory=True)

    return train_loader, val_loader


# train_loader, val_loader = get_rafdb_loaders()

"""====取第一个批次的数据出来==="""
# for images, target in train_loader:
#     print("Shape of X [N, C, H, W]: ", images.shape)     # torch.Size([8, 3, 224, 224])
#     print("Shape of y: ", target.shape, target.dtype)     # torch.Size([8]) torch.int64
#     print(target)   # tensor([3, 4, 3, 0, 3, 4, 3, 4])
#     break
"""====取第一个批次的数据出来==="""

# tensor([3, 0, 4, 0, 0, 5, 3, 3])
























