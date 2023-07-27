#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 09:57:14 2023

@author: Julia Dietlmeier
"""
import os
#--- SingleCoil directories ---------------------------------------------------
src_dir04='/media/daa/My Passport/Submission/SingleCoil/Cine/ValidationSet/AccFactor04/'
src_dir08='/media/daa/My Passport/Submission/SingleCoil/Cine/ValidationSet/AccFactor08/'
src_dir10='/media/daa/My Passport/Submission/SingleCoil/Cine/ValidationSet/AccFactor10/'

for k in range(1,61):# patients from 1 to 60
    st=str(k)
    if not os.path.exists(src_dir04+'P'+st.zfill(3)):
        os.mkdir(src_dir04+'P'+st.zfill(3))

for k in range(1,61):# patients from 1 to 60
    st=str(k)
    if not os.path.exists(src_dir08+'P'+st.zfill(3)):
        os.mkdir(src_dir08+'P'+st.zfill(3))
        
for k in range(1,61):# patients from 1 to 60
    st=str(k)
    if not os.path.exists(src_dir10+'P'+st.zfill(3)):
        os.mkdir(src_dir10+'P'+st.zfill(3))

#--- MultiCoil directories ---------------------------------------------------
src_dir04='/media/daa/My Passport/Submission/MultiCoil/Cine/ValidationSet/AccFactor04/'
src_dir08='/media/daa/My Passport/Submission/MultiCoil/Cine/ValidationSet/AccFactor08/'
src_dir10='/media/daa/My Passport/Submission/MultiCoil/Cine/ValidationSet/AccFactor10/'

for k in range(1,61):# patients from 1 to 60
    st=str(k)
    if not os.path.exists(src_dir04+'P'+st.zfill(3)):
        os.mkdir(src_dir04+'P'+st.zfill(3))

for k in range(1,61):# patients from 1 to 60
    st=str(k)
    if not os.path.exists(src_dir08+'P'+st.zfill(3)):
        os.mkdir(src_dir08+'P'+st.zfill(3))
        
for k in range(1,61):# patients from 1 to 60
    st=str(k)
    if not os.path.exists(src_dir10+'P'+st.zfill(3)):
        os.mkdir(src_dir10+'P'+st.zfill(3))
    