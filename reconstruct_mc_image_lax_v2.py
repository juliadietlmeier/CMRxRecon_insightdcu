#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 10:26:26 2023

@author: Julia Dietlmeier
email: <julia.dietlmeier@insight-centre.org>
"""
from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import h5py
from torchvision import datasets, transforms
#from models.denoisingUNET import build_unet
from models.GNA_UNET import GNA_unet
from utils.coil_combine import rss, rss_complex

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

def reconstruct_mc_image_lax_v2(path, accfactor):
    
    if accfactor=='AccFactor04':
        str_key='kspace_sub04'
    elif accfactor=='AccFactor08':
        str_key='kspace_sub08'
    elif accfactor=='AccFactor10':
        str_key='kspace_sub10'
    
    ff=h5py.File(path, 'r')
    print(ff.keys())
    arr = (ff[str_key])
    
    
    shape=np.shape(arr)#[t,sz, sc, sy,sx], [12,3, 10, 204,448]
    sz=shape[1]
    timeframe=shape[0]
    sc=shape[2]
    sy=shape[3]
    sx=shape[4]
    
    recon_model = GNA_unet()
    state_dict = torch.load("/home/daa/Desktop/CMRxRecon_challenge_MICCAI23/Python_submission/MC_GNAUnet_300epochs_lax.pt")
    recon_model.load_state_dict(state_dict)
    device=get_device()
    recon_model = recon_model.to(device)
           
    recon_arr=np.zeros((sx,sy, sz,timeframe))
    for t in range(0,timeframe):
        for k in range(0,sz):
            slice=arr[t,k,:, :, :]
            
            #for c in range(0,sc):
            #slice=arr[t,k,c, :, :]
                #print('slice shape =',np.shape(slice))
            arr_complex=slice['real']+1j*slice['imag']
            x2 = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(arr_complex)))
            
            x2 = rss_complex(x2, dim=0)
            
            x2_max=np.max(np.abs(x2))# this is needed later to bring the signal to the original intensity
            x2_min=np.min(np.abs(x2))
                #print('x2_max = ', x2_max)
                #print('x2_min = ', x2_min )
            img=np.abs(x2)/np.max(np.abs(x2))
            #print('x2 after rss_complex shape', np.shape(img))
                # the following normalization needed for feeding the image into UNET as it was trained that way    
            img = cv2.resize(img, (512, 512))
            min_img=np.min(img)
            max_img=np.max(img)
            noisy_image = (img-np.min(img))/(np.max(img)-np.min(img))
            noisy_image = np.expand_dims(noisy_image,axis=-1)
            
            noisy_image=transforms.ToTensor()(noisy_image.astype('float32'))
            noisy_image=torch.unsqueeze(noisy_image, 0)

            with torch.no_grad():
                
                    noisy_image = noisy_image.to(device)

                    preds = recon_model(noisy_image)
                    preds=preds.cpu().detach().numpy()
                    
                    preds=np.squeeze(preds,axis=0)
                    preds=np.squeeze(preds,axis=0)
                #print('np.shape(preds)=',np.shape(preds))
            preds=cv2.resize(preds,(sx,sy))
            preds=(preds-np.min(preds))/(np.max(preds)-np.min(preds))
                #preds=preds*(max_img-min_img)+min_img
                #print('original np.max(preds) = ',np.max(preds))
                # back to the original signal intensity
            preds=preds*x2_max
                #print('np.max(preds) = ', np.max(preds))
            preds=np.moveaxis(preds,0,1)# this will maybe need to be swapped sx for sy np.moveaxis(0,1)
            recon_arr[:,:, k, t]=preds # should be now [sx,sy,sz,t]
            
                
    return recon_arr#rss(recon_arr,dim=4)# was dimension=2

# test the above function
#src_file='/media/daa/My Passport/CMRxRecon_data_2023/CMR_Recon/ChallengeData/MultiCoil/Cine/ValidationSet/ValidationSet/AccFactor04/P001/cine_lax.mat'
#RA=reconstruct_mc_image_lax(src_file, 'AccFactor04')
#plt.figure(),plt.imshow(RA[:,:,0,0])# the x and y axis are flipped somehow. Please, double check

