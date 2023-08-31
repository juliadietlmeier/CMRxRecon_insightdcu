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

from models.denoisingUNET import build_unet

path='/home/daa/Desktop/CMRxRecon_challenge_MICCAI23/SingleCoil/Cine/Training_Set_julia/'

class DataLoader_Class:
    def __init__(self, path):
        self.path          = path
        self.train_images  = []        
        self.train_cleaned = []
        self.test_images   = []
        self.test_cleaned  = []
        
    def read_all_image(self):
        self.train_images_path  = [(self.path  + 'new_train/input_AC/' + f) for f in sorted(os.listdir(self.path + 'new_train/input_AC/'))]
        
        self.train_cleaned_path = [(self.path  + 'new_train/target_FS/' + f) for f in sorted(os.listdir(self.path + 'new_train/target_FS/'))]
        
        self.test_images_path   = [(self.path  + 'test/input_AC/' + f) for f in sorted(os.listdir(self.path + 'test/input_AC/'))]
        
        self.test_cleaned_path   = [(self.path + 'test/target_FS/' + f) for f in sorted(os.listdir(self.path + 'test/target_FS/'))]
        
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
        
        for path in self.test_images_path:
            img = cv2.imread(path)
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
train_set_x, train_set_y , test_set_x, test_set_y = data_loader.read_all_image()
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

test_set = Dataset_AE(test_set_x, test_set_y)
test_loader  = torch.utils.data.DataLoader(test_set, 
                                           batch_size=batch_size, 
                                           shuffle=False)
"=== MODEL Definition ========================================================="
model = build_unet() # through the import
"=============================================================================="

def train(model, num_epochs=5, batch_size=2, learning_rate=0.001):
    criterion = nn.MSELoss() # mean square error loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    
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
model = build_unet()
device = get_device()
print(device)
model.to(device)

outputs, loss_arr = train(model, num_epochs=150)

torch.save(model.state_dict(), 'denoisingUNET_100epochs_v2.pt')

plt.figure(),plt.plot(np.asarray(loss_arr)),plt.title('Training Loss')

"=============================================================================="

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

save_path='/home/daa/Desktop/CMRxRecon_challenge_MICCAI23/UNET_results/'

def eval_model(model, test_loader):
    model.eval()
    psnr = []
    mse = []
    SSIM_arr=[]
    with torch.no_grad():
        j=0
        SSIM=[]
        for noisy_images, targets in test_loader:

            targets = targets.to(device)
            noisy_images = noisy_images.to(device)
            preds = model(noisy_images)
            predictions=preds.cpu().detach().numpy()
            ln=len(predictions)
            print('len_predictions=',ln)
            print('predictions_shape',np.shape((np.moveaxis(predictions,1,-1))))
            for l in range(0,ln):
                p=(np.moveaxis(predictions[l],1,-1))
                #print('p=',np.shape(p[l,:,:]))
                p=np.expand_dims(p,axis=-1)
                p=np.concatenate((p,p,p),axis=-1)
                p=np.squeeze(p,axis=0)
                p=(p-np.min(p))/(np.max(p)-np.min(p))
                print('np.max(p)=',np.max(p))
                print('np_shape(p)',np.shape(p))
                p=np.moveaxis(p,0,1)
                print(save_path+str(j)+'_predict.jpg')
                p=cv2.resize(p, (448, 206))
                
                cv2.imwrite(save_path+str(j)+'_predict.jpg', p*255)
            
            psnr.extend(PSNR(targets.cpu().detach(), preds.cpu().detach()))
            mse.extend(MSE(targets.cpu().detach(), preds.cpu().detach()))
            j=j+1
            
            
            t=targets.cpu().detach().numpy()
            pr=preds.cpu().detach().numpy()
            t=np.squeeze(t,axis=1)
            pr=np.squeeze(pr,axis=1)

            for l in range(0,ln):
                s=ssim(t[l,:,:], pr[l,:,:])
                SSIM.append(s)

        SSIM_arr.append(SSIM)  
            
        print(f"Peak Signal to Noise Ratio:   Mean: {np.array(psnr).mean()} || Std: {np.array(psnr).std()}")
        print(f"Mean Squared Error:   Mean: {np.array(mse).mean()} || Std: {np.array(mse).std()}")
        
        return np.array(psnr).mean(), np.array(mse).mean(), SSIM_arr, predictions
    
    
    
"=== Predictions =============================================================="
recon_model = build_unet()
state_dict = torch.load("denoisingUNET_100epochs_v2.pt")
recon_model.load_state_dict(state_dict)

normal_ae_mse = recon_model.to(device)

psnr = []
with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            noisy_images = images
            noisy_images = noisy_images.to(device)
            targets = targets.to(device)
            preds = recon_model(noisy_images)
            if (i<30):
                
                print(PSNR(targets.cpu().detach(), preds.cpu().detach()))
                
            psnr.extend(PSNR(targets.cpu().detach(), preds.cpu().detach()))

psnr,mse, SSIM_arr, preds=eval_model(model,test_loader)
ss=np.mean(SSIM_arr)




