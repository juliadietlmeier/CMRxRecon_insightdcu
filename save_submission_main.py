#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 10:06:44 2023

@author: Julia Dietlmeier
email: <julia.dietlmeier@insight-centre.org>
"""
import os
import time
from scipy.io import loadmat, savemat
from reconstruct_sc_image_lax import *
from reconstruct_sc_image_sax import *
from reconstruct_mc_image_lax_v2 import *
from reconstruct_mc_image_sax_v2 import *
#from reconstruct_mc_image_lax import *
#from reconstruct_mc_image_sax import *
#from reconstruct_sc_image_lax_DAE import *
#from reconstruct_sc_image_sax_DAE import *

"run this file first"
"need to save first to the /Submission/ folder and then run mainRun4Ranking to generate /Submission_ranked folder"
#basepath is where your original data is
# before proceeding create manually empty directories 
#/Submission/SingleCoil/Cine/ValidationSet/AccFactor04
#/Submission/SingleCoil/Cine/ValidationSet/AccFactor08
#/Submission/SingleCoil/Cine/ValidationSet/AccFactor10
"then run create_empty_patient_dirs.py"# this will create empty P001 to P060 directories for each AccFactor

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
                start=time.time()
                rec_img=reconstruct_sc_image_lax(path_d+'/'+filename, AccF[k])
                end=time.time()
                print('np.shape(rec_img) = ', np.shape(rec_img))
                print('lax time = ', end-start)
                mdic = {"img4ranking": rec_img}
                savemat(savepath+AccF[k]+'/'+d+'/'+filename, mdic)
            elif filename=='cine_sax.mat':
                start=time.time()
                rec_img=reconstruct_sc_image_sax(path_d+'/'+filename, AccF[k])
                end=time.time()
                print('sax time = ', end-start)
                mdic = {"img4ranking": rec_img}
                savemat(savepath+AccF[k]+'/'+d+'/'+filename, mdic)
            print('SC Patient #',d)    
            
"=== now do MultiCoil ========================================================="
#"=== generate only for 5 patients P001 to P005 but for three AccF to be listed on the leaderboard"

AccF=['AccFactor04', 'AccFactor08', 'AccFactor10']
basepath='/media/daa/My Passport/CMRxRecon_data_2023/CMR_Recon/ChallengeData/MultiCoil/Cine/ValidationSet/ValidationSet/'
savepath='/media/daa/My Passport/Submission/MultiCoil/Cine/ValidationSet/'
for k in range(0,3):
    src_pathA=basepath+AccF[k]+'/'
    path, dirs,files=next(os.walk(src_pathA))
    #dirs=['P001','P002','P003','P004','P005']# can be commented if generating for all 60 patients
    for d in dirs: # walk over patient directories
        path_d, dds, files = next(os.walk(path+d))
        for f in files:
            filename=f
            if filename=='cine_lax.mat':
                start=time.time()
                rec_img=reconstruct_mc_image_lax_v2(path_d+'/'+filename, AccF[k])
                end=time.time()
                print('np.shape(rec_img) = ', np.shape(rec_img))
                print('lax time = ', end-start)
                mdic = {"img4ranking": rec_img}
                savemat(savepath+AccF[k]+'/'+d+'/'+filename, mdic)
            elif filename=='cine_sax.mat':
                start=time.time()
                rec_img=reconstruct_mc_image_sax_v2(path_d+'/'+filename, AccF[k])
                end=time.time()
                print('sax time = ', end-start)
                mdic = {"img4ranking": rec_img}
                savemat(savepath+AccF[k]+'/'+d+'/'+filename, mdic)
            print('MC Patient #',d)    