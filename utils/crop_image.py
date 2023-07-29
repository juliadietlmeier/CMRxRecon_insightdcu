#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 19:21:47 2023

@author: daa
"""
import cv2
import numpy as np
#Importing OpenCV library as usual.

def center_crop(img, dim):
  """Returns center cropped image

  Args:
  img: image to be center cropped
  dim: dimensions (width, height) to be cropped from center
  """

#Let’s create a function center_crop() with parameters as image and crop dimensions as a tuple (width, height).

  width, height = img.shape[1], img.shape[0]  #process crop width and height for max available dimension
  crop_width = dim[0] #if dim[0]<img.shape[1] else img.shape[1]# sy
  crop_height = dim[1] #if dim[1]<img.shape[0] else img.shape[0] # sx

#The first line gets the width and height of the original image. Crop dimensions passed in arguments may exceed the original dimension, this results in improper image cropping. The last two lines choose the maximum dimension without exceeding the original image dimension.

  mid_x, mid_y = int(np.floor(width/2)), int(np.floor(height/2))
  
  cw2, ch2 = int(np.ceil((crop_width/2))), int(np.ceil((crop_height/2))) 
  
  crop_img = img[mid_y-ch2:mid_y+ch2-1, mid_x-cw2+1:mid_x+cw2+1]
  
  sh=np.shape(crop_img)
  if sh[1]==124:
      #crop_img=crop_img[:,0:123,:,:]
      crop_img=crop_img[:,1:124,:,:]
  elif sh[1]==82:
      #crop_img=crop_img[:,0:81,:,:]
      crop_img=crop_img[:,1:82,:,:]
  
  return crop_img