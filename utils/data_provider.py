import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import h5py
import torchvision
from torchvision import transforms
import random


class HashingDataset(Dataset):
    def __init__(self, data_path, split, training=False):
        h5f = h5py.File(data_path, 'r')
        self.label = torch.tensor(h5f[f"{split}_L"][:]).float()
        # self.total_tag = torch.tensor(h5f[f"tag_{split}"][:])
        index_ = torch.where(self.label.sum(1) > 0)[0]
        self.label = self.label[index_]
        if split == 'dataset':
            self.total_img = h5f[f"data_set"][:]
        else:
            self.total_img = h5f[f"{split}_data"][:]
        self.img_path = data_path
        self.total_img = self.total_img[index_]
        training = split=='train'
        if training:
            self.transform = transforms.Compose([transforms.Resize(256),
            transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        else:
            self.transform = transforms.Compose(
                [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

    def __getitem__(self, index):

        img = Image.fromarray(self.total_img[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.Tensor(self.label[index]).float()


        while label.sum() < 1:
            index = random.randint(0, len(self.total_img) - 1)
            img = Image.fromarray(self.total_img[index]).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            label = torch.Tensor(self.label[index]).float()

        return img, label, index

    def __len__(self):
        return len(self.total_img)

def load_label(filename, data_dir):
    label_filepath = os.path.join(data_dir, filename)
    label = np.loadtxt(label_filepath, dtype=np.int64)
    return torch.from_numpy(label).float()
