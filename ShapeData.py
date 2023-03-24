import os
import numpy as np 
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision as tv
from sklearn.model_selection import train_test_split

#Medium shape classifier: https://studentsxstudents.com/creating-a-shape-classification-cnn-model-using-pytorch-2cce61077834
class ShapeDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir):
        self.annotation = torch.zeros(7485)
        self.annotation[:3720] = 0 #nonsquares are 0
        self.annotation[3720:7485] = 1 #Squares are 1
        self.annotation = F.one_hot(self.annotation.to(torch.int64), 2)
        self.img_dir = img_dir
        self.transform = transforms.Compose ([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize ([0.5], [0.5]) #0-1 to [-1,1], formula (x-mean)/std
        ])

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, idx):
        label = self.annotation[idx]
        if label.eq(torch.tensor([1, 0], dtype=torch.int64)).all():
            img_path = os.path.join(self.img_dir, 'non-square', f"{idx}.png")

        elif label.eq(torch.tensor([0, 1], dtype=torch.int64)).all():
            img_path = os.path.join(self.img_dir, 'square', f"{idx - 3720}.png")
        image = tv.io.read_image(img_path)
        image = self.transform(image)
        return image, label

#Kaggle image dataset containing around 3700 images per shapre group being circles squares stars and triangles created a new folder called non squares which cureently only holds circle images to make processes run faster
dataset = ShapeDataset('archive (4)/shapes')
train_dataset, valid_dataset = train_test_split(dataset, train_size=0.7, test_size=0.3) #splitting kaggle dataset into the training ang test sets since it did not come inherently in the dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)

