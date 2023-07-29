from __future__ import absolute_import
"author: translated from MATLAB by Julia Dietlmeier"
"email: julia.dietlmeier@insight-centre.org"


import numpy as np
from utils.crop_image import *
# to reduce the computing burden and space, we only evaluate the central 2 slices
# For cine: use the first 3 time frames for ranking!
# For mapping: we need all weighting for ranking!
# crop the middle 1/6 of the original image for ranking

# this function helps you to convert your data for ranking
# img: complex images reconstructed with the dimensions (sx,sy,sz,t/w) in
# original size
# -sx: matrix size in x-axis
# -sy: matrix size in y-axis
# -sz: slice number (short axis view); slice group (long axis view)
# -t/w: time frame/weighting

# img4ranking: "single" format images with the dimensions (sx/3,sy/2,2,3)
# -sx/3: 1/3 of the matrix size in x-axis
# -sy/2: half of the matrix size in y-axis
# img4ranking is the data we used for ranking!!!
"=============================================================================="

def run4Ranking(img,filetype):
    shape = np.shape(img)# [sx,sy,sz,t]
    print('shape=',shape)
    sx=shape[0]
    sy=shape[1]
    sz=shape[2]
    t=shape[3]
    
    #if strcmp(filetype,'cine_lax') || strcmp(filetype,'cine_sax'):
    if filetype=='cine_lax' or filetype=='cine_sax':
        if sz < 3:
            reconImg = img[:,:,:,0:3]
        else:
            reconImg = img[:,:,int(np.round(sz/2))-2:int(np.round(sz/2)),0:3]# need to double check the indexing here
        
        # crop the middle 1/6 of the original image for ranking
        #img4ranking = single(crop(abs(reconImg),[round(sx/3),round(sy/2),2,3]));#MATLAB
        #img4ranking = reconImg[int(np.round(sx/6)*2):int(np.round(sx/6)*4),int(np.round(sy/4)*1):int(np.round(sy/4)*3),:,:]
        
        #img4ranking = reconImg[int(np.floor(sx/6*2))+1:int(np.ceil(sx/6*4)),int(np.floor(sy/4*1)):int(np.ceil(sy/4*3)),:,:]
        img4ranking = center_crop(reconImg, (sy/2, sx/3))
    else:
        if sz < 3:
            reconImg = img
        else:
            reconImg = img[:,:,int(np.round(sz/2))-2:int(np.round(sz/2)),:]#need to double check the indexing here
        
        # crop the middle 1/6 of the original image for ranking
        #img4ranking = single(crop(abs(reconImg),[round(sx/3),round(sy/2),2,t]));# MATLAB
        img4ranking = reconImg[int(np.round(sx/6)*2):int(np.round(sx/6)*4),int(np.round(sy/4)*1):int(np.round(sy/4)*3),:,:]#(sx/3,sy/2,2,3)
    
    return img4ranking
