#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:35:10 2023

@author: Julia Dietlmeier
"""
from __future__ import absolute_import

import numpy as np 
import os
import numpy as np
import time
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import os, cv2
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.nn import Linear, ReLU, MSELoss, L1Loss, Sequential, Conv2d, ConvTranspose2d, MaxPool2d, AdaptiveAvgPool2d, Module, BatchNorm2d, Sigmoid, Dropout
from torchvision.utils import make_grid

from typing import Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

#from models.denoisingUNET import build_unet
from models.GNA_UNET import GNA_unet

np.random.seed(42)
torch.manual_seed(42)

"training on the subsampled TrainingSet for MultiCoil ========================="
# 1000 LAX images and 1000 SAX images
path='/home/daa/Desktop/CMRxRecon_challenge_MICCAI23/MultiCoil/Cine/Training_Set_julia/'

class DataLoader_Class:
    def __init__(self, path):
        self.path          = path
        self.train_images  = []        
        self.train_cleaned = []
        
    def read_all_image(self):
        self.train_images_path  = [(self.path  + 'train_sax_renamed_ALL/input_AC/' + f) 
                                   for f in sorted(os.listdir(self.path + 'train_sax_renamed_ALL/input_AC/'))]
        
        self.train_cleaned_path = [(self.path  + 'train_sax_renamed_ALL/target_FS/' + f) 
                                   for f in sorted(os.listdir(self.path + 'train_sax_renamed_ALL/target_FS/'))]
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.5, ), (0.5, )),                
        ])
        
        for path in self.train_images_path:
            img = cv2.imread(path)
            img = np.asarray(img, dtype="uint8")
            img = cv2.resize(img, (512, 512))#448,208
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# 0 to 255 ?
            img = (img-np.min(img))/(np.max(img)-np.min(img))
            print('shape img = ', np.shape(img))
            img = img[:,:,0]
            img=transforms.ToTensor()(img.astype('float32'))
            #img = transform(transforms.ToPILImage()(img))# 0 to 1 ?
            self.train_images.append(img)
        
        for path in self.train_cleaned_path:
            img = cv2.imread(path)
            img = np.asarray(img, dtype="uint8")
            img = cv2.resize(img, (512, 512))#img = cv2.resize(img, (528, 416) )
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = (img-np.min(img))/(np.max(img)-np.min(img))
            img=img[:,:,0]
            img=transforms.ToTensor()(img.astype('float32'))
            #img = transform(transforms.ToPILImage()(img.astype('float32')))

            self.train_cleaned.append(img)
        
                
        #convert data list to tensor 
        self.train_images = torch.stack(self.train_images)
        self.train_cleaned = torch.stack(self.train_cleaned)
        
        print(self.train_images.shape)
        print(self.train_cleaned.shape)

        return self.train_images, self.train_cleaned 
        
    def see_an_image(self, number):
        f, axarr = plt.subplots(1,2, figsize=(50,100))
        axarr[0].imshow(self.train_images[number].permute(1,2,0) ,cmap = "gray")
        axarr[1].imshow(self.train_cleaned[number].permute(1,2,0),cmap = "gray")



data_loader = DataLoader_Class(path)
train_set_x, train_set_y = data_loader.read_all_image()
data_loader.see_an_image(0)
# test-set_y is needed to compute PSNR and SSIM on the test set

batch_size = 2

from torch.utils.data import Dataset

class Dataset_AE(Dataset):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]

training_set = Dataset_AE(train_set_x, train_set_y)
train_loader = torch.utils.data.DataLoader(training_set,
                                           batch_size=batch_size, 
                                           shuffle=True)

"=== MODEL Definition ========================================================="
model = GNA_unet() # through the import
"=============================================================================="

def train(model, num_epochs=5, batch_size=2, learning_rate=0.001):
    criterion = nn.MSELoss() # mean square error loss
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    
    outputs = []
    loss_arr= []
    for epoch in range(num_epochs):
        for data in train_loader:
            img, _ = data

            img, _ = img.cuda(), _.cuda()

            optimizer.zero_grad()
            recon = model(img)

            loss = criterion(recon, _)
            loss.backward()
            optimizer.step()          
            
        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
        outputs.append((epoch, img, recon),)
        loss_arr.append(float(loss))
    return outputs, loss_arr

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

"=== TRAIN ===================================================================="
model = GNA_unet()
device = get_device()
print(device)
model.to(device)

outputs, loss_arr = train(model, num_epochs=300)

torch.save(model.state_dict(), 'MC_GNAUnet_300epochs_sax.pt')

plt.figure(),plt.plot(np.asarray(loss_arr)),plt.title('Training Loss')

total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")




