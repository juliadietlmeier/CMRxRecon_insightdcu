#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 12:47:38 2023

@author: Julia Dietlmeier
"""

import os
import cv2

src_train_target_FS='/home/daa/Desktop/CMRxRecon_challenge_MICCAI23/SingleCoil/Cine/Training_Set_julia/train2/target_FS/'
src_train_input_AC='/home/daa/Desktop/CMRxRecon_challenge_MICCAI23/SingleCoil/Cine/Training_Set_julia/train2/input_AC/'

dst_train_target_FS='/home/daa/Desktop/CMRxRecon_challenge_MICCAI23/SingleCoil/Cine/Training_Set_julia/new_train2/target_FS/'
dst_train_input_AC='/home/daa/Desktop/CMRxRecon_challenge_MICCAI23/SingleCoil/Cine/Training_Set_julia/new_train2/input_AC/'


path, dirs, files = next(os.walk(src_train_target_FS))
k=0
for f in files:
    img=cv2.imread(path+f)
    cv2.imwrite(dst_train_target_FS+str(k)+'.png',img)
    img_AC=cv2.imread(src_train_input_AC+f.replace('FS','AC'))
    cv2.imwrite(dst_train_input_AC+str(k)+'.png',img_AC)
    k=k+1