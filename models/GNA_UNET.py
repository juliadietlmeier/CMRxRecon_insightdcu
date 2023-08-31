#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 10:19:05 2023

@author: Julia Dietlmeier
email: <julia.dietlmeier@insight-centre.org>
"""
import numpy as np 
import os
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import os, cv2
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.nn import Linear, ReLU, MSELoss, L1Loss, Sequential, Conv2d, ConvTranspose2d, MaxPool2d, AdaptiveAvgPool2d, Module, BatchNorm2d, Sigmoid, Dropout
from torchvision.utils import make_grid

#https://github.com/changzy00/pytorch-attention/blob/master/attention_mechanisms/gate_channel_module.py
class GCT(nn.Module):# Gated Channel Transformation (CVPR 2020)

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):#was False and l2
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):

        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2,3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
            
        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2,3), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            print('Unknown mode!')

        gate = 1. + torch.tanh(embedding * norm + self.beta)


        return x * gate

class conv_block(nn.Module):
    def __init__(self, in_c, out_c,ng):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.GCT1 = GCT(out_c)

        self.bn1 = nn.GroupNorm(ng, out_c)        
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.GCT2 = GCT(out_c)

        self.bn2 = nn.GroupNorm(ng, out_c)         
        self.relu = nn.ReLU()

        
    def forward(self, inputs):

        x = self.conv1(inputs)
        x = self.GCT1(x)
        x = self.bn1(x)
        x = self.relu(x)        
        x = self.conv2(x)
        x = self.GCT2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
    
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c,ng):
        
        super().__init__()
        self.conv = conv_block(in_c, out_c, ng)
        self.dropout = nn.Dropout(0.25)
        self.pool = nn.MaxPool2d((2, 2))     
        
    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.dropout(x)
        p = self.pool(x)
        return x, p
    
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, ng):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.GCT=GCT(out_c)
        self.conv = conv_block(out_c+out_c, out_c, ng)     
        
    def forward(self, inputs, skip):
        x = self.up(inputs)
        skip = self.GCT(skip)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x
    
class GNA_unet(nn.Module):
    def __init__(self):
        super().__init__()
#         """ Encoder """
        self.e1 = encoder_block(1, 64, 8)
        self.e2 = encoder_block(64, 128,8)
        self.e3 = encoder_block(128, 256,8)
        self.e4 = encoder_block(256, 512,8)
        self.e5 = encoder_block(512, 1024,8)
        
        # """ Bottleneck """
        self.b = conv_block(1024, 2048, 8)
        #""" Decoder """
        self.d0 = decoder_block(2048, 1024,8)
        self.d1 = decoder_block(1024, 512,8)
        self.d2 = decoder_block(512, 256,8)
        self.d3 = decoder_block(256, 128,8)
        self.d4 = decoder_block(128, 64,8)
        # """ Classifier """

        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)   
        
    def forward(self, inputs):
#        """ Encoder """
        s1, p1 = self.e1(inputs)
        
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3) 
        s5, p5 = self.e5(p4)
        # """ Bottleneck """
        b = self.b(p5)
        #""" Decoder """
        d0 = self.d0(b, s5)
        d1 = self.d1(d0, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1) 
        #""" Classifier """
        outputs = self.outputs(d4)        
        return outputs
    
    
    
    
    
    
    
    