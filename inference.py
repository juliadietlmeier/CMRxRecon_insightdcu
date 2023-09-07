#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 09:00:38 2023

@author: <julia.dietlmeier@insight-centre.org>
"""
# https://github.com/CmrxRecon/CMRxRecon-DockerBuildExample/blob/master/example/inference.py
import sys
import argparse
import os
import time
import torch
from scipy.io import loadmat, savemat
from reconstruct_sc_image_lax import *
from reconstruct_sc_image_sax import *
from reconstruct_mc_image_lax_v2 import *
from reconstruct_mc_image_sax_v2 import *

def my_makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

if __name__ == '__main__':
    
    print(torch.__version__)
    #print(torch.cuda.get_device_name(0))
    print(torch.cuda.is_available())
    
    argv = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, nargs='?', default='/input', help='input directory')
    parser.add_argument('--output', type=str, nargs='?', default='/output', help='output directory')
    
    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output

    print("Input data store in:", input_dir)
    print("Output data store in:", output_dir)

    # TODO Load input file from input_dir and make your prediction,
    #  then write the predictions to output_dir
    AccF=['AccFactor04', 'AccFactor08', 'AccFactor10']
    basepath = input_dir  + '/SingleCoil/Cine/TestSet/'
    savepath = output_dir + '/SingleCoil/Cine/TestSet/'
    for k in range(0,3):
        src_pathA=basepath+AccF[k]+'/'
        path, dirs,files=next(os.walk(src_pathA))
        for d in dirs: # walk over patient directories
            path_d, dds, files = next(os.walk(path+d))
            for f in files:
                filename=f
                if filename=='cine_lax.mat':

                    rec_img=reconstruct_sc_image_lax(path_d+'/'+filename, AccF[k])
                    mdic = {"img4ranking": rec_img}
                    my_makedirs(savepath+AccF[k]+'/'+d+'/')
                    savemat(savepath+AccF[k]+'/'+d+'/'+filename, mdic)
                
                elif filename=='cine_sax.mat':

                    rec_img=reconstruct_sc_image_sax(path_d+'/'+filename, AccF[k])
                    mdic = {"img4ranking": rec_img}
                    my_makedirs(savepath+AccF[k]+'/'+d+'/')
                    savemat(savepath+AccF[k]+'/'+d+'/'+filename, mdic)
                 
                
    #=== now do MultiCoil ========================================================="
    
    AccF=['AccFactor04', 'AccFactor08', 'AccFactor10']
    basepath = input_dir  + '/MultiCoil/Cine/TestSet/'
    savepath = output_dir + '/MultiCoil/Cine/TestSet/'
    for k in range(0,3):
        src_pathA=basepath+AccF[k]+'/'
        path, dirs,files=next(os.walk(src_pathA))
        
        for d in dirs: # walk over patient directories
            path_d, dds, files = next(os.walk(path+d))
            for f in files:
                filename=f
                if filename=='cine_lax.mat':
                    
                    rec_img=reconstruct_mc_image_lax_v2(path_d+'/'+filename, AccF[k])
                    mdic = {"img4ranking": rec_img}
                    my_makedirs(savepath+AccF[k]+'/'+d+'/')
                    savemat(savepath+AccF[k]+'/'+d+'/'+filename, mdic)
                    
                elif filename=='cine_sax.mat':

                    rec_img=reconstruct_mc_image_sax_v2(path_d+'/'+filename, AccF[k])
                    mdic = {"img4ranking": rec_img}
                    my_makedirs(savepath+AccF[k]+'/'+d+'/')
                    savemat(savepath+AccF[k]+'/'+d+'/'+filename, mdic)
   
                
                
                
                
                
                
                