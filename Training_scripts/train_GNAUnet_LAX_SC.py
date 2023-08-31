#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:35:10 2023

@author: Julia Dietlmeier
email: <julia.dietlmeier@insight-centre.org>
"""
from __future__ import absolute_import

import numpy as np 
import os
import numpy as np

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
from MS_SSIM_L1_LOSS import MS_SSIM_L1_LOSS
import torchvision.transforms.functional as TF
import random
import time

np.random.seed(42)
torch.manual_seed(42)

"training on the full 100% of the TrainingSet for SingleCoil =================="

path='/home/daa/Desktop/CMRxRecon_challenge_MICCAI23/SingleCoil/Cine/Training_Set_julia/'


class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

rotation_transform = MyRotationTransform(angles=[30])#-30, -15, 0, 15, 30

class DataLoader_Class:
    def __init__(self, path):
        self.path          = path
        self.train_images  = []        
        self.train_cleaned = []
        self.test_images   = []
        self.test_cleaned  = []
        
    def read_all_image(self):
        self.train_images_path  = [(self.path  + 'train_lax_renamed_ALL/input_AC/' + f) 
                                   for f in sorted(os.listdir(self.path + 'train_lax_renamed_ALL/input_AC/'))]
        
        self.train_cleaned_path = [(self.path  + 'train_lax_renamed_ALL/target_FS/' + f) 
                                   for f in sorted(os.listdir(self.path + 'train_lax_renamed_ALL/target_FS/'))]
        
        self.test_images_path  = [(self.path  + 'val_lax_renamed_ALL/input_AC/' + f) 
                                   for f in sorted(os.listdir(self.path + 'val_lax_renamed_ALL/input_AC/'))]
        
        self.test_cleaned_path = [(self.path  + 'val_lax_renamed_ALL/target_FS/' + f) 
                                   for f in sorted(os.listdir(self.path + 'val_lax_renamed_ALL/target_FS/'))]
        
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            #rotation_transform(),
            #transforms.Normalize((0.5, ), (0.5, )),                
        ])
        
        for path in self.train_images_path:
            pth_noisy=path.split('/')[-1]
            #print('pth_noisy =',pth_noisy)
            fname=pth_noisy.split('.')[0]
            path_clean=path.replace('input_AC','target_FS')
            print('path_clean =', path_clean)
            
            
            
            img = cv2.imread(path)
            img = np.asarray(img, dtype="uint8")
            img = cv2.resize(img, (512, 512))#448,208
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# 0 to 255 ?
            img = (img-np.min(img))/(np.max(img)-np.min(img))
            #print('shape img = ', np.shape(img))
            img = img[:,:,0]
            
            img_cleaned = cv2.imread(path_clean)
            img_cleaned = np.asarray(img_cleaned, dtype="uint8")
            img_cleaned = cv2.resize(img_cleaned, (512, 512))#448,208
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# 0 to 255 ?
            img_cleaned = (img_cleaned-np.min(img_cleaned))/(np.max(img_cleaned)-np.min(img_cleaned))
            #print('shape img_cleaned = ', np.shape(img_cleaned))
            img_cleaned = img_cleaned[:,:,0]
            
            img=transforms.ToTensor()(img.astype('float32'))
            #state = torch.get_rng_state()
            img_transformed=rotation_transform(img)
            clean_img=transforms.ToTensor()(img_cleaned.astype('float32'))
            #torch.set_rng_state(state)
            clean_img_transformed=rotation_transform(clean_img)
            
            #img = transform(transforms.ToPILImage()(img))# 0 to 1 ?
            self.train_images.append(img)
            #self.train_images.append(img_transformed)
            self.train_cleaned.append(clean_img)
            #self.train_cleaned.append(clean_img_transformed)
        
        #for path in self.train_cleaned_path:
        #    img = cv2.imread(path)
        #    img = np.asarray(img, dtype="uint8")
        #    img = cv2.resize(img, (512, 512))#img = cv2.resize(img, (528, 416) )
        #    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #    img = (img-np.min(img))/(np.max(img)-np.min(img))
        #    img=img[:,:,0]
        #    img=transforms.ToTensor()(img.astype('float32'))
        #    #img = transform(transforms.ToPILImage()(img.astype('float32')))

        #    self.train_cleaned.append(img)
        
        for path in self.test_images_path:
                 img = cv2.imread(path)
                 print('test_images_path = ', path)
                 img = np.asarray(img, dtype="uint8")
                 img = cv2.resize(img, (512, 512))#img = cv2.resize(img, (528, 416) )
                 #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                 img = (img-np.min(img))/(np.max(img)-np.min(img))
                 #img=transform(img)
                 img=img[:,:,0]
                 img=transforms.ToTensor()(img.astype('float32'))
                 #img = transform(transforms.ToPILImage()(img.astype('float32')))

                 self.test_images.append(img)
             
        for path in self.test_cleaned_path:
                 img = cv2.imread(path)
                 img = np.asarray(img, dtype="uint8")
                 img = cv2.resize(img, (512, 512))#img = cv2.resize(img, (528, 416) )
                 #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                 img = (img-np.min(img))/(np.max(img)-np.min(img))
                 #img=transform(img)
                 img=img[:,:,0]
                 img=transforms.ToTensor()(img.astype('float32'))
                 #img = transform(transforms.ToPILImage()(img.astype('float32')))

                 self.test_cleaned.append(img)
                 
        #convert data list to tensor 
        self.train_images = torch.stack(self.train_images)
        self.train_cleaned = torch.stack(self.train_cleaned)
        self.test_images = torch.stack(self.test_images)
        self.test_cleaned = torch.stack(self.test_cleaned)
        
        print(self.train_images.shape)
        print(self.train_cleaned.shape)
        print(self.test_images.shape)
        print(self.test_cleaned.shape)

        return self.train_images, self.train_cleaned, self.test_images, self.test_cleaned  
        
    def see_an_image(self, number):
        f, axarr = plt.subplots(1,2, figsize=(50,100))
        axarr[0].imshow(self.train_images[number].permute(1,2,0) ,cmap = "gray")
        axarr[1].imshow(self.train_cleaned[number].permute(1,2,0),cmap = "gray")



data_loader = DataLoader_Class(path)
train_set_x, train_set_y, test_set_x, test_set_y = data_loader.read_all_image()
data_loader.see_an_image(0)

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
test_set = Dataset_AE(test_set_x, test_set_y)
test_loader  = torch.utils.data.DataLoader(test_set, 
                                           batch_size=batch_size, 
                                           shuffle=False)

"=== MODEL Definition ========================================================="
model = GNA_unet() # through the import
"=============================================================================="
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

EPS=1e-7
def PSNR(input, target):
    return -10*torch.log10(torch.mean((input - target) ** 2, dim=[1, 2, 3])+EPS)
def MSE(input, target):
    return torch.mean((input - target) ** 2, dim=[1, 2, 3])
def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if maxval is None:
        maxval = gt.max()
    return structural_similarity(gt, pred, data_range=maxval)

def train(model, num_epochs=5, batch_size=2, learning_rate=0.001):#was 0.001 and gave better results
    
    criterion = nn.MSELoss() # mean square error loss
    #criterion = MS_SSIM_L1_LOSS()
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)
    
    early_stopper = EarlyStopper(patience=3, min_delta=10)#was min_delta=10
    outputs = []
    loss_arr= []
    val_loss_arr=[]
    
    psnr = []
    mse = []
    SSIM_arr=[]
    
    psnr_all = []
    mse_all = []
    
    val_psnr = []
    val_mse = []
    val_SSIM_arr=[]
    
    val_psnr_all = []
    val_mse_all = []
    
    for epoch in range(num_epochs):
        
        for data in train_loader:
            img, _ = data

            img, _ = img.cuda(), _.cuda()

            optimizer.zero_grad()
            recon = model(img)

            loss = criterion(recon, _)
            loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()   
            
            psnr.extend(PSNR(_.cpu().detach(), recon.cpu().detach()))
            mse.extend(MSE(_.cpu().detach(), recon.cpu().detach()))
        
        psnr_all.append(np.array(psnr).mean())
        mse_all.append(np.array(mse).mean())
#---- VALIDATION part ---------------------------------------------------------
        for data in test_loader:
            with torch.no_grad():          
                img, _ = data
                img, _ = img.cuda(), _.cuda()
                recon_val = model(img)
                loss_val = criterion(recon_val, _)
                
                val_psnr.extend(PSNR(_.cpu().detach(), recon_val.cpu().detach()))
                val_mse.extend(MSE(_.cpu().detach(), recon_val.cpu().detach()))
        
        val_psnr_all.append(np.array(val_psnr).mean())
        val_mse_all.append(np.array(val_mse).mean())
        #scheduler.step()
#-- EARLY STOPPING ------------------------------------------------------------
        if early_stopper.early_stop(loss_val):             
            break
#------------------------------------------------------------------------------            
        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
        outputs.append((epoch, img, recon),)
        loss_arr.append(float(loss))
        val_loss_arr.append(float(loss_val))
        
    return outputs, loss_arr, val_loss_arr, np.array(psnr_all), np.array(mse_all), np.array(val_psnr_all), np.array(val_mse_all)

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

"=== TRAIN ===================================================================="
model = GNA_unet()
#model = cascade()

device = get_device()
print(device)
model.to(device)
start=time.time()
outputs, loss_arr, val_loss_arr, psnr, mse, val_psnr, val_mse = train(model, num_epochs=300)# was 300 for MSE
end=time.time()
print('training time = ',end-start)

torch.save(model.state_dict(), 'SC_GNAUnet_300epochs_lax.pt')

plt.figure(),plt.plot(np.asarray(loss_arr)),plt.plot(np.asarray(val_loss_arr)),plt.title('Training/Validation Loss')
plt.savefig('train_val_loss_withDropout.png')
plt.figure(),plt.plot(np.asarray(psnr),label='Train PSNR'),plt.plot(np.asarray(val_psnr),label='Val PSNR'),plt.title('Training/Validation PSNR')
plt.legend()
plt.savefig('train_val_PSNR_withDropout.png')

total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")




