# -*- coding:utf-8 -*-
"""
author：wwm
data：2024-02-19
"""




import torch.utils.data as data
import pandas as pd
import os
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


# data fetch
class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform=None):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path

        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ',
                         header=None, names=['name', 'label'])

        if phase == 'train':
            self.data = df[df['name'].str.startswith('train')]  
        else:
            self.data = df[df['name'].str.startswith('test')]  

        file_names = self.data.loc[:, 'name'].values
        # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        self.label = self.data.loc[:, 'label'].values - 1  # ndaaray:(12271,) 

        # sample_counts：array([1290, 281, 717, 4772, 1982, 705, 2524], dtype=int64)
        _, self.sample_counts = np.unique(self.label, return_counts=True)  # sample_counts
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
        image = Image.open(path).convert('RGB')
        label = self.label[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label



mu=[0.485, 0.456, 0.406]
st=[0.229, 0.224, 0.225]
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


def get_rafdb_loaders(data_path, batch_size):

    train_dataset = RafDataSet(data_path, phase='train', transform=rafdb_transfrom["train"])
    val_dataset = RafDataSet(data_path, phase='test', transform=rafdb_transfrom["val"])

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






















