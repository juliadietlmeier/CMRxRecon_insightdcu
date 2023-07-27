#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 10:26:26 2023

@author: Julia Dietlmeier
"""
from __future__ import absolute_import

import numpy as np
import cv2
import torch
import h5py
from torchvision import datasets, transforms
from models.denoisingUNET import build_unet


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

# reconstruct SingleCoil image
def reconstruct_sc_image(path, accfactor):
    
    if accfactor=='AccFactor04':
        str_key='kspace_single_sub04'
    elif accfactor=='AccFactor08':
        str_key='kspace_single_sub08'
    elif accfactor=='AccFactor10':
        str_key='kspace_single_sub10'
    
    ff=h5py.File(path, 'r')
    arr = (ff[str_key])
    shape=np.shape(arr)#[t,sz,sy,sx], [12,3,204,448]
    sz=shape[1]
    timeframe=shape[0]
    sy=shape[2]
    sx=shape[3]
    print('sx=',sx)
    
    recon_arr=np.zeros((sx,sy,sz,timeframe))
    for t in range(0,timeframe):
        for k in range(0,sz):
        
            slice=arr[t,k,:,:]
            print('slice shape =',np.shape(slice))
            arr_complex=slice['real']+1j*slice['imag']

            x2 = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(arr_complex)))
        
            img=np.abs(x2)/np.max(np.abs(x2))
            # the following normalization needed for feeding the image into UNET as it was trained that way    
            img = cv2.resize(img, (512, 512))
            noisy_image = (img-np.min(img))/(np.max(img)-np.min(img))
            noisy_image = np.expand_dims(noisy_image,axis=-1)
            
            noisy_image=transforms.ToTensor()(noisy_image.astype('float32'))
            noisy_image=torch.unsqueeze(noisy_image, 0)
            
            recon_model = build_unet()
            state_dict = torch.load("/home/daa/Desktop/CMRxRecon_challenge_MICCAI23/denoisingUNET_100epochs.pt")
            recon_model.load_state_dict(state_dict)
            device=get_device()
            recon_model = recon_model.to(device)

            with torch.no_grad():
                
                    noisy_image = noisy_image.to(device)

                    preds = recon_model(noisy_image)
                    preds=preds.cpu().detach().numpy()
                    
                    preds=np.squeeze(preds,axis=0)
                    preds=np.squeeze(preds,axis=0)
                
            preds=cv2.resize(preds,(sy,sx))   
            #preds=np.moveaxis(preds,0,1)# this will maybe need to be swapped sx for sy np.moveaxis(0,1)
            recon_arr[:,:,k,t]=preds # should be now [sx,sy,sz,t]
            
                
    return recon_arr

# test the above function
#src_file='/media/daa/My Passport/CMRxRecon_data_2023/CMR_Recon/ChallengeData/SingleCoil/SingleCoil/Cine/ValidationSet/AccFactor04/P001/cine_lax.mat'
#RA=reconstruct_sc_image(src_file, 'AccFactor04')


