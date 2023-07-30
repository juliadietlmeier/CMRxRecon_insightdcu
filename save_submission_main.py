#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 10:06:44 2023

@author: Julia Dietlmeier
"""
import os
from scipy.io import loadmat, savemat
from reconstruct_sc_image import *
from reconstruct_mc_image import *

"need to save first to the /Submission/ folder and then run mainRun4Ranking to generate /Submission_ranked folder"
#basepath is where your original data is
# before proceeding create manually empty directories 
#/Submission/SingleCoil/Cine/ValidationSet/AccFactor04
#/Submission/SingleCoil/Cine/ValidationSet/AccFactor08
#/Submission/SingleCoil/Cine/ValidationSet/AccFactor10

"=== do SingleCoil first ======================================================"
AccF=['AccFactor04', 'AccFactor08', 'AccFactor10']
basepath='/media/daa/My Passport/CMRxRecon_data_2023/CMR_Recon/ChallengeData/SingleCoil/SingleCoil/Cine/ValidationSet/'
savepath='/media/daa/My Passport/Submission/SingleCoil/Cine/ValidationSet/'
for k in range(0,3):
    src_pathA=basepath+AccF[k]+'/'
    path, dirs,files=next(os.walk(src_pathA))
    for d in dirs: # walk over patient directories
        path_d, dds, files = next(os.walk(path+d))
        for f in files:
            filename=f
            if filename=='cine_lax.mat':
                rec_img=reconstruct_sc_image(path_d+'/'+filename, AccF[k])
                print('np.shape(rec_img) = ', np.shape(rec_img))
                mdic = {"img4ranking": rec_img}
                savemat(savepath+AccF[k]+'/'+d+'/'+filename, mdic)
            elif filename=='cine_sax.mat':
                rec_img=reconstruct_sc_image(path_d+'/'+filename, AccF[k])
                mdic = {"img4ranking": rec_img}
                savemat(savepath+AccF[k]+'/'+d+'/'+filename, mdic)
            print('Patient #',d)    
m            
"=== now do MultiCoil ========================================================="
AccF=['AccFactor04', 'AccFactor08', 'AccFactor10']
basepath='/media/daa/My Passport/CMRxRecon_data_2023/CMR_Recon/ChallengeData/MultiCoil/Cine/ValidationSet/ValidationSet/'
savepath='/media/daa/My Passport/Submission/MultiCoil/Cine/ValidationSet/'
for k in range(0,3):
    src_pathA=basepath+AccF[k]+'/'
    path, dirs,files=next(os.walk(src_pathA))
    for d in dirs: # walk over patient directories
        path_d, dds, files = next(os.walk(path+d))
        for f in files:
            filename=f
            if filename=='cine_lax.mat':
                rec_img=reconstruct_mc_image(path_d+'/'+filename, AccF[k])
                mdic = {"img4ranking": rec_img}
                savemat(savepath+AccF[k]+'/'+d+'/'+filename, mdic)
            elif filename=='cine_sax.mat':
                rec_img=reconstruct_mc_image(path_d+'/'+filename, AccF[k])
                mdic = {"img4ranking": rec_img}
                savemat(savepath+AccF[k]+'/'+d+'/'+filename, mdic)
            print('Patient #',d)    