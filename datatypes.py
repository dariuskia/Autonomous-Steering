import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import v2

import pandas as pd
from torchvision.io import read_image

class DrivingImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform = None, target_transform = None):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = torch.tensor(self.img_labels.iloc[idx, -1], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def get_steering_data():
    SIZE = (66, 200)
    img_transforms = v2.Compose([
        v2.Resize(SIZE, antialias=True),
        v2.ToDtype(torch.float32)
    ])
    # label_transform = lambda x: x.to(dtype=torch.float32)
    steering_trainset = DrivingImageDataset("Data/labelsB_train.csv", "Data/trainB", transform = img_transforms)
    steering_testset = DrivingImageDataset("Data/labelsB_val.csv", "Data/valB", transform = img_transforms)
    return steering_trainset, steering_testset
