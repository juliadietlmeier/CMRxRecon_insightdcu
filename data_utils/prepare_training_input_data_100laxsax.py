#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 08:37:27 2023

@author: Julia Dietlmeier
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2
import os 

"== preparing input training and validation data for denoising AEs ============"
"== first you need empty folders ...Training_Set_julia/FullSample/ and ...Training_set_julia/AccFactor04/"
"== also create the following empty folders:"
"== ...Training_Set_julia/train/target_FS/ and ...Training_Set_julia/train/input_AC/"
"== ...Training_Set_julia/val/target_FS/ and ...Training_Set_julia/val/input_AC/"
"== ...Training_Set_julia/test/target_FS/ and ...Training_Set_julia/test/input_AC/"

src_target_FS='/media/daa/My Passport/CMRxRecon_data_2023/CMR_Recon/ChallengeData/SingleCoil/SingleCoil/Cine/TrainingSet/FullSample/'
src_input_AC04='/media/daa/My Passport/CMRxRecon_data_2023/CMR_Recon/ChallengeData/SingleCoil/SingleCoil/Cine/TrainingSet/AccFactor04/'
src_input_AC08='/media/daa/My Passport/CMRxRecon_data_2023/CMR_Recon/ChallengeData/SingleCoil/SingleCoil/Cine/TrainingSet/AccFactor08/'
src_input_AC10='/media/daa/My Passport/CMRxRecon_data_2023/CMR_Recon/ChallengeData/SingleCoil/SingleCoil/Cine/TrainingSet/AccFactor10/'


dst_target_FS='/home/daa/Desktop/CMRxRecon_challenge_MICCAI23/SingleCoil/Cine/Training_Set_julia/FullSample/'
dst_input_AC04='/home/daa/Desktop/CMRxRecon_challenge_MICCAI23/SingleCoil/Cine/Training_Set_julia/AccFactor04/'
dst_input_AC08='/home/daa/Desktop/CMRxRecon_challenge_MICCAI23/SingleCoil/Cine/Training_Set_julia/AccFactor08/'
dst_input_AC10='/home/daa/Desktop/CMRxRecon_challenge_MICCAI23/SingleCoil/Cine/Training_Set_julia/AccFactor10/'

"=== first read and ksp reconstruct FullSample target labels =================="
path, dirs, files = next(os.walk(src_target_FS))
for d in dirs:# walk over patient directories
    path_d, dds, files_d = next(os.walk(path+d))
    for fnames in files_d:
        if fnames=='cine_lax.mat':
            src_f=os.path.join(path, d, fnames)
            idt='lax'
        elif fnames=='cine_sax.mat':
            src_f=os.path.join(path, d, fnames)
            idt='sax'
            
        ff=h5py.File(src_f, 'r')
        arr = (ff['kspace_single_full'])
        shape=np.shape(arr)#[12,3,204,448]
        if idt=='lax':
            n_slices=shape[1]
        elif idt=='sax':
            n_slices=3
        for k in range(0,n_slices):
            slice=arr[11,k,:,:]
            arr_complex=slice['real']+1j*slice['imag']

            x2 = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(arr_complex)))
        
            f_name='FS_'+d+'_'+idt+'_'+str(k)+'.jpg'
        
            x2_normalized=np.abs(x2)/np.max(np.abs(x2))
            cv2.imwrite(os.path.join(dst_target_FS,f_name),x2_normalized*255)
            print('computing FullSample',f_name)
#------------------------------------------------------------------------------
"=== then read and ksp reconstruct AccFactor04 input labels =================="
path, dirs, files = next(os.walk(src_input_AC04))
for d in dirs:# walk over patient directories
    path_d, dds, files_d = next(os.walk(path+d))
    for fnames in files_d:
        if fnames=='cine_lax.mat':
            src_f=os.path.join(path, d, fnames)
            idt='lax'
        elif fnames=='cine_sax.mat':
            src_f=os.path.join(path, d, fnames)
            idt='sax'
        ff=h5py.File(src_f, 'r')
        arr = (ff['kspace_single_sub04'])
        shape=np.shape(arr)
        if idt=='lax':
            n_slices=shape[1]
        elif idt=='sax':
            n_slices=3
        for k in range(0,n_slices):
            slice=arr[11,k,:,:]
            arr_complex=slice['real']+1j*slice['imag']

            x2 = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(arr_complex)))
        
            f_name='AC_'+d+'_'+idt+'_'+str(k)+'.jpg'
        
            x2_normalized=np.abs(x2)/np.max(np.abs(x2))
            cv2.imwrite(os.path.join(dst_input_AC04, f_name),x2_normalized*255)
            print('computing AccFactor04',f_name)
#------------------------------------------------------------------------------
"=== then read and ksp reconstruct AccFactor08 input labels =================="
path, dirs, files = next(os.walk(src_input_AC08))
for d in dirs:# walk over patient directories
    path_d, dds, files_d = next(os.walk(path+d))
    for fnames in files_d:
        if fnames=='cine_lax.mat':
            src_f=os.path.join(path, d, fnames)
            idt='lax'
        elif fnames=='cine_sax.mat':
            src_f=os.path.join(path, d, fnames)
            idt='sax'
        ff=h5py.File(src_f, 'r')
        arr = (ff['kspace_single_sub08'])
        shape=np.shape(arr)
        if idt=='lax':
            n_slices=shape[1]
        elif idt=='sax':
            n_slices=3
        for k in range(0,n_slices):
            slice=arr[11,k,:,:]
            arr_complex=slice['real']+1j*slice['imag']

            x2 = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(arr_complex)))
        
            f_name='AC_'+d+'_'+idt+'_'+str(k)+'.jpg'
        
            x2_normalized=np.abs(x2)/np.max(np.abs(x2))
            cv2.imwrite(os.path.join(dst_input_AC08, f_name),x2_normalized*255)
            print('computing AccFactor08',f_name)
#------------------------------------------------------------------------------

"=== then read and ksp reconstruct AccFactor10 input labels =================="
path, dirs, files = next(os.walk(src_input_AC10))
for d in dirs:# walk over patient directories
    path_d, dds, files_d = next(os.walk(path+d))
    for fnames in files_d:
        if fnames=='cine_lax.mat':
            src_f=os.path.join(path, d, fnames)
            idt='lax'
        elif fnames=='cine_sax.mat':
            src_f=os.path.join(path, d, fnames)
            idt='sax'
        ff=h5py.File(src_f, 'r')
        arr = (ff['kspace_single_sub10'])
        shape=np.shape(arr)
        if idt=='lax':
            n_slices=shape[1]
        elif idt=='sax':
            n_slices=3
        for k in range(0,n_slices):
            slice=arr[11,k,:,:]
            arr_complex=slice['real']+1j*slice['imag']

            x2 = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(arr_complex)))
        
            f_name='AC_'+d+'_'+idt+'_'+str(k)+'.jpg'
        
            x2_normalized=np.abs(x2)/np.max(np.abs(x2))
            cv2.imwrite(os.path.join(dst_input_AC10, f_name),x2_normalized*255)
            print('computing AccFactor10',f_name)
#------------------------------------------------------------------------------
"=============================================================================="
"=== now place all AccFactors images into one train2 folder===================="

path, dirs, files = next(os.walk(dst_target_FS))
vec=np.random.permutation(len(files))

train_vec=vec

src_target_FS='/home/daa/Desktop/CMRxRecon_challenge_MICCAI23/SingleCoil/Cine/Training_Set_julia/FullSample/'
src_input_AC04='/home/daa/Desktop/CMRxRecon_challenge_MICCAI23/SingleCoil/Cine/Training_Set_julia/AccFactor04/'
src_input_AC08='/home/daa/Desktop/CMRxRecon_challenge_MICCAI23/SingleCoil/Cine/Training_Set_julia/AccFactor08/'
src_input_AC10='/home/daa/Desktop/CMRxRecon_challenge_MICCAI23/SingleCoil/Cine/Training_Set_julia/AccFactor10/'


dst_train_target_FS='/home/daa/Desktop/CMRxRecon_challenge_MICCAI23/SingleCoil/Cine/Training_Set_julia/train2/target_FS/'
dst_train_input_AC='/home/daa/Desktop/CMRxRecon_challenge_MICCAI23/SingleCoil/Cine/Training_Set_julia/train2/input_AC/'

f_list=os.listdir(src_target_FS)# ['FS_P001_2.jpg', 'FS_P001_0.jpg', 'FS_P001_1.jpg', ....]
#f_list_train=f_list[train_vec]

new_f_list_train=[]
for h in range(0,len(train_vec)):
    new_f_list_train.append(f_list[train_vec[h]])
f_list_train=new_f_list_train

path, dirs, files = next(os.walk(src_target_FS))
for k in range(0,len(f_list_train)):
    img=cv2.imread(path+f_list_train[k])
    cv2.imwrite(dst_train_target_FS+f_list_train[k], img)
    
"== AccF inputs"      
path, dirs, files = next(os.walk(src_input_AC04))
for k in range(0,len(f_list_train)):
    img=cv2.imread(path+f_list_train[k].replace('FS','AC'))
    cv2.imwrite(dst_train_input_AC+f_list_train[k].replace('FS','AC'), img)

path, dirs, files = next(os.walk(src_input_AC08))
for k in range(0,len(f_list_train)):
    img=cv2.imread(path+f_list_train[k].replace('FS','AC'))
    cv2.imwrite(dst_train_input_AC+f_list_train[k].replace('FS','AC'), img)
    
path, dirs, files = next(os.walk(src_input_AC10))
for k in range(0,len(f_list_train)):
    img=cv2.imread(path+f_list_train[k].replace('FS','AC'))
    cv2.imwrite(dst_train_input_AC+f_list_train[k].replace('FS','AC'), img)









