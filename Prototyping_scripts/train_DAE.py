#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:35:10 2023

@author: Julia Dietlmeier
email: julia.dietlmeier@insight-centre.org
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

from typing import Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

RES=64# set this resolution to 128, 256 and 512 for th eexperiments
RES=int(RES)

path='/home/daa/Desktop/CMRxRecon_challenge_MICCAI23/SingleCoil/Cine/Training_Set_julia/'
class DataLoader_Class:
    def __init__(self, path):
        self.path          = path
        self.train_images  = []        
        self.train_cleaned = []
        self.test_images   = []
        self.test_cleaned  = []
        self.val_images   = []
        self.val_cleaned  = []
        
    def read_all_image(self):
        self.train_images_path  = [(self.path  + 'new_train/input_AC/' + f) for f in sorted(os.listdir(self.path + 'new_train/input_AC/'))]
        
        self.train_cleaned_path = [(self.path  + 'new_train/target_FS/' + f) for f in sorted(os.listdir(self.path + 'new_train/target_FS/'))]
        
        self.test_images_path   = [(self.path  + 'test/input_AC/' + f) for f in sorted(os.listdir(self.path + 'test/input_AC/'))]
        
        self.test_cleaned_path   = [(self.path + 'test/target_FS/' + f) for f in sorted(os.listdir(self.path + 'test/target_FS/'))]
        
        self.val_images_path   = [(self.path  + 'val/input_AC/' + f) for f in sorted(os.listdir(self.path + 'val/input_AC/'))]
        
        self.val_cleaned_path   = [(self.path + 'val/target_FS/' + f) for f in sorted(os.listdir(self.path + 'val/target_FS/'))]
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.5, ), (0.5, )),                
        ])
        
        for path in self.train_images_path:
            img = cv2.imread(path)
            img = np.asarray(img, dtype="uint8")
            img = cv2.resize(img, (RES, RES))#448,208
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# 0 to 255 ?
            img = (img-np.min(img))/(np.max(img)-np.min(img))
            print('shape img = ', np.shape(img))
            img = img[:,:,0]
            img=transforms.ToTensor()(img.astype('float32'))
            #img = transform(transforms.ToPILImage()(img))# 0 to 1 ?
            self.train_images.append(img)
        
        for path in self.train_cleaned_path:
            img = cv2.imread(path)
            img = np.asarray(img, dtype="uint8")
            img = cv2.resize(img, (RES, RES))#img = cv2.resize(img, (528, 416) )
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = (img-np.min(img))/(np.max(img)-np.min(img))
            img=img[:,:,0]
            img=transforms.ToTensor()(img.astype('float32'))
            #img = transform(transforms.ToPILImage()(img.astype('float32')))

            self.train_cleaned.append(img)
        
        for path in self.test_images_path:
            img = cv2.imread(path)
            img = np.asarray(img, dtype="uint8")
            img = cv2.resize(img, (RES, RES))#img = cv2.resize(img, (528, 416) )
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = (img-np.min(img))/(np.max(img)-np.min(img))
            #img=transform(img)
            img=img[:,:,0]
            img=transforms.ToTensor()(img.astype('float32'))
            #img = transform(transforms.ToPILImage()(img.astype('float32')))

            self.test_images.append(img)
        
        for path in self.test_cleaned_path:
            img = cv2.imread(path)
            img = np.asarray(img, dtype="uint8")
            img = cv2.resize(img, (RES, RES))#img = cv2.resize(img, (528, 416) )
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = (img-np.min(img))/(np.max(img)-np.min(img))
            #img=transform(img)
            img=img[:,:,0]
            img=transforms.ToTensor()(img.astype('float32'))
            #img = transform(transforms.ToPILImage()(img.astype('float32')))

            self.test_cleaned.append(img)
            
        for path in self.val_images_path:
            img = cv2.imread(path)
            img = np.asarray(img, dtype="uint8")
            img = cv2.resize(img, (RES, RES))#img = cv2.resize(img, (528, 416) )
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = (img-np.min(img))/(np.max(img)-np.min(img))
            #img=transform(img)
            img=img[:,:,0]
            img=transforms.ToTensor()(img.astype('float32'))
            #img = transform(transforms.ToPILImage()(img.astype('float32')))

            self.val_images.append(img)
        
        for path in self.val_cleaned_path:
            img = cv2.imread(path)
            img = np.asarray(img, dtype="uint8")
            img = cv2.resize(img, (RES, RES))#img = cv2.resize(img, (528, 416) )
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = (img-np.min(img))/(np.max(img)-np.min(img))
            #img=transform(img)
            img=img[:,:,0]
            img=transforms.ToTensor()(img.astype('float32'))
            #img = transform(transforms.ToPILImage()(img.astype('float32')))

            self.val_cleaned.append(img)
        
        #convert data list to tensor 
        self.train_images = torch.stack(self.train_images)
        self.train_cleaned = torch.stack(self.train_cleaned)
        self.test_images = torch.stack(self.test_images)
        self.test_cleaned = torch.stack(self.test_cleaned)
        self.val_images = torch.stack(self.val_images)
        self.val_cleaned = torch.stack(self.val_cleaned)

        print(self.train_images.shape)
        print(self.train_cleaned.shape)
        print(self.test_images.shape)
        print(self.test_cleaned.shape)
        print(self.val_images.shape)
        print(self.val_cleaned.shape)
        
        return self.train_images, self.train_cleaned, self.test_images, self.test_cleaned, self.val_images, self.val_cleaned 
        
    def see_an_image(self, number):
        f, axarr = plt.subplots(1,2, figsize=(50,100))
        axarr[0].imshow(self.train_images[number].permute(1,2,0) ,cmap = "gray")
        axarr[1].imshow(self.train_cleaned[number].permute(1,2,0),cmap = "gray")



data_loader = DataLoader_Class(path)
train_set_x, train_set_y , test_set_x, test_set_y, val_set_x, val_set_y = data_loader.read_all_image()
data_loader.see_an_image(0)
# test-set_y is needed to compute PSNR and SSIM on the test set

batch_size = 2

from torch.utils.data import Dataset

class Dataset_AE(Dataset):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]

training_set = Dataset_AE(train_set_x, train_set_y)
train_loader = torch.utils.data.DataLoader(training_set,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_set = Dataset_AE(test_set_x, test_set_y)
test_loader  = torch.utils.data.DataLoader(test_set, 
                                           batch_size=batch_size, 
                                           shuffle=False)

val_set = Dataset_AE(val_set_x, val_set_y)
val_loader  = torch.utils.data.DataLoader(val_set, 
                                           batch_size=batch_size, 
                                           shuffle=False)

import torch.nn.functional as F

RESQ=int(RES/4)
"===  MODEL definition ========================================================"
#  defining encoder
class Encoder(nn.Module):
  def __init__(self, in_channels=1, out_channels=16, latent_dim=1000, act_fn=nn.ReLU()):
    super().__init__()
    self.in_channels = in_channels

    self.net = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), # (32, 32)
        act_fn,
        nn.Conv2d(out_channels, out_channels, 3, padding=1), 
        act_fn,
        nn.Conv2d(out_channels, 2*out_channels, 3, padding=1, stride=2), # (16, 16)
        act_fn,
        nn.Conv2d(2*out_channels, 2*out_channels, 3, padding=1),
        act_fn,
        nn.Conv2d(2*out_channels, 4*out_channels, 3, padding=1, stride=2), # (8, 8)
        act_fn,
        nn.Conv2d(4*out_channels, 4*out_channels, 3, padding=1),
        act_fn,
        nn.Flatten(),
        nn.Linear(4*out_channels*RESQ*RESQ, latent_dim),#32,32 for 128x128 #8,8
        act_fn
    )

  def forward(self, x):
    x1 = x.view(-1, self.in_channels, RES, RES)#128,128 for 128x128 32,32
    output = self.net(x1)
    #output = torch.cat([x, self.net(x)], 1)
    return output


#  defining decoder
class Decoder(nn.Module):
  def __init__(self, in_channels=1, out_channels=16, latent_dim=1000, act_fn=nn.ReLU()):
    super().__init__()

    self.out_channels = out_channels

    self.linear = nn.Sequential(
        nn.Linear(latent_dim, 4*out_channels*RESQ*RESQ),#32,32 for 128x128 #8,8
        act_fn
    )

    self.conv = nn.Sequential(
        nn.ConvTranspose2d(4*out_channels, 4*out_channels, 3, padding=1), # (8, 8)
        act_fn,
        nn.ConvTranspose2d(4*out_channels, 2*out_channels, 3, padding=1, 
                           stride=2, output_padding=1), # (16, 16)
        act_fn,
        nn.ConvTranspose2d(2*out_channels, 2*out_channels, 3, padding=1),
        act_fn,
        nn.ConvTranspose2d(2*out_channels, out_channels, 3, padding=1, 
                           stride=2, output_padding=1), # (32, 32)
        act_fn,
        nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1),
        act_fn,
        nn.ConvTranspose2d(out_channels, in_channels, 3, padding=1)
    )

  def forward(self, x):
    output = self.linear(x)
    output = output.view(-1, 4*self.out_channels, RESQ, RESQ)#32,32 for 128x128
    output = self.conv(output)
    #output = torch.cat([output, self.conv(output)], 1)
    return output

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
device=get_device()

#  defining autoencoder
class Autoencoder3(nn.Module):
  def __init__(self, encoder, decoder):
    super().__init__()
    self.encoder = encoder
    self.encoder.to(device)

    self.decoder = decoder
    self.decoder.to(device)

  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded   


EPS=1e-7
def PSNR(input, target):
    return -10*torch.log10(torch.mean((input - target) ** 2, dim=[1, 2, 3])+EPS)
def MSE(input, target):
    return torch.mean((input - target) ** 2, dim=[1, 2, 3])

def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if maxval is None:
        maxval = gt.max()
    return structural_similarity(gt, pred, data_range=maxval)


#  defining class
class ConvolutionalAutoencoder():
    
  def __init__(self, autoencoder):
    self.network = autoencoder
    self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)

  def train(self, loss_function, epochs, batch_size, 
            #training_set, validation_set, test_set,
            image_channels=1):
    
    #  creating log
    log_dict = {
        'training_loss_per_batch': [],
        'validation_loss_per_batch': [],
        'visualizations': []
    } 

    #  defining weight initialization function
    def init_weights(module):
      if isinstance(module, nn.Conv2d):
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)
      elif isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)

    #  initializing network weights
    self.network.apply(init_weights)

    #  creating dataloaders
    train_loader = torch.utils.data.DataLoader(training_set,
                                               batch_size=2, 
                                               shuffle=True)#DataLoader(training_set, batch_size)
    #val_loader = DataLoader(validation_set, batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, 
                                               batch_size=2, 
                                               shuffle=False)#DataLoader(test_set, 10)
    
    val_loader = torch.utils.data.DataLoader(val_set, 
                                               batch_size=2, 
                                               shuffle=False)

    #  setting convnet to training mode
    self.network.train()
    self.network.to(device)

    for epoch in range(epochs):
      print(f'Epoch {epoch+1}/{epochs}')
      train_losses = []

      #------------
      #  TRAINING
      #------------
      print('training...')
      for images, targets in tqdm(train_loader):
        #  zeroing gradients
        self.optimizer.zero_grad()
        #  sending images and targets to device
        images = images.to(device).type(torch.cuda.FloatTensor)
        targets = targets.to(device).type(torch.cuda.FloatTensor)
        #  reconstructing images
        output = self.network(images)
        #  computing loss
        loss = loss_function(output, targets)
        loss = loss#.type(torch.cuda.FloatTensor)
        #  calculating gradients
        loss.backward()
        #  optimizing weights
        self.optimizer.step()

        #--------------
        # LOGGING
        #--------------
        log_dict['training_loss_per_batch'].append(loss.item())

      #--------------
      # VALIDATION
      #--------------
      print('validating...')
      for val_images, val_targets in tqdm(val_loader):
        with torch.no_grad():
          #  sending validation images and targets to device
          val_images = val_images.to(device).type(torch.cuda.FloatTensor)
          val_targets = val_targets.to(device).type(torch.cuda.FloatTensor)
          #  reconstructing images
          output = self.network(val_images)
          #  computing validation loss
          val_loss = loss_function(output, val_targets)

        #--------------
        # LOGGING
        #--------------
        log_dict['validation_loss_per_batch'].append(val_loss.item())


      #--------------
      # VISUALISATION
      #--------------
      #print(f'training_loss: {round(loss.item(), 4)} validation_loss: {round(val_loss.item(), 4)}')
      k=0
      SSIM_arr=[]
      for test_images, test_targets in test_loader:
        #  sending test images to device
        test_images = test_images.to(device).type(torch.cuda.FloatTensor)
        psnr = []
        mse = []
        SSIM =[]
        with torch.no_grad():
          #  reconstructing test images
          reconstructed_imgs = self.network(test_images)
          
          predictions=reconstructed_imgs.cpu().detach().numpy()
          ln=len(predictions)
          print('len_predictions=',ln)
          print('predictions_shape',np.shape((np.moveaxis(predictions,1,-1))))
          for l in range(0,ln):
              p=(np.moveaxis(predictions[l],1,-1))
              #print('p=',np.shape(p[l,:,:]))
              p=np.expand_dims(p,axis=-1)
              p=np.concatenate((p,p,p),axis=-1)
              p=np.squeeze(p,axis=0)
              p=(p-np.min(p))/(np.max(p)-np.min(p))
              print('np.max(p)=',np.max(p))
              print('np_shape(p)',np.shape(p))
              p=np.moveaxis(p,0,1)
              save_path='/home/daa/Desktop/CMRxRecon_challenge_MICCAI23/AE_results/'
              print(save_path+str(k)+'_predict.jpg')
              p=cv2.resize(p, (448, 208))
              
              cv2.imwrite(save_path+str(k)+'_predict.jpg', p*255)
          
          psnr.extend(PSNR(targets.cpu().detach(), reconstructed_imgs.cpu().detach()))
          mse.extend(MSE(targets.cpu().detach(), reconstructed_imgs.cpu().detach()))
          
          t=targets.cpu().detach().numpy()
          pr=reconstructed_imgs.cpu().detach().numpy()
          t=np.squeeze(t,axis=1)
          pr=np.squeeze(pr,axis=1)

          for l in range(0,ln):
              s=ssim(t[l,:,:], pr[l,:,:])
              SSIM.append(s)

        SSIM_arr.append(SSIM)  
        #  sending reconstructed and images to cpu to allow for visualization
        reconstructed_imgs = reconstructed_imgs.cpu()
        test_images = test_images.cpu()

        #  visualisation
        imgs = torch.stack([test_images.view(-1, image_channels, RES, RES), reconstructed_imgs], 
                          dim=1).flatten(0,1)
        grid = make_grid(imgs, nrow=10, normalize=True, padding=1)
        grid = grid.permute(1, 2, 0)
        plt.figure(dpi=170)
        plt.title('Original/Reconstructed')
        plt.imshow(grid)
        log_dict['visualizations'].append(grid)
        plt.axis('off')
        plt.show()
        save_path='/home/daa/Desktop/CMRxRecon_challenge_MICCAI23/AE_results/'
        plt.savefig(save_path+str(k)+'.png')
        k=k+1
        plt.close()
        
    return log_dict, np.array(psnr).mean(), np.array(mse).mean(), SSIM_arr

  def autoencode(self, x):
    return self.network(x)

  def encode(self, x):
    encoder = self.network.encoder
    return encoder(x)
  
  def decode(self, x):
    decoder = self.network.decoder
    return decoder(x)

#  training model
model = ConvolutionalAutoencoder(Autoencoder3(Encoder(in_channels=1),
                                               Decoder(in_channels=1)))

NB_EPOCHS=6
log_dict, psnr, mse, SSIM = model.train(nn.MSELoss(), epochs=NB_EPOCHS, batch_size=2, #L1Loss())
                       #training_set=training_data, validation_set=validation_data,
                       #test_set=test_data, 
                       image_channels=1)

ss=np.array(np.reshape(SSIM,60)).mean()
#torch.save(model.state_dict(), 'DAE_15epochs.pt')
#normal_ae_mse = ConvolutionalAutoencoder()
#state_dict = torch.load("normal_ae_mse_30epochs.pt")
#normal_ae_mse.load_state_dict(state_dict)

#normal_ae_mse = normal_ae_mse.to(device)

"=============================================================================="

train_loss_arr=log_dict['training_loss_per_batch']
val_loss_arr=log_dict['validation_loss_per_batch']

tr_range=int(np.shape(train_loss_arr)[0]/NB_EPOCHS)
counter=1
tr_loss_arr=[]
for k in range(0,int(np.shape(train_loss_arr)[0])):
    if k==tr_range*counter:
        tr_loss=np.mean(train_loss_arr[tr_range*(counter-1):tr_range*counter])
        counter=counter+1
        tr_loss_arr.append(tr_loss)
        
tr_range=int(np.shape(val_loss_arr)[0]/NB_EPOCHS)
counter=1
val_loss_a=[]
for k in range(0,int(np.shape(val_loss_arr)[0])):
    if k==tr_range*counter:
        tr_loss=np.mean(val_loss_arr[tr_range*(counter-1):tr_range*counter])
        counter=counter+1
        val_loss_a.append(tr_loss)
        
plt.figure(),plt.plot(np.asarray(tr_loss_arr),'-r'),plt.plot(np.asarray(val_loss_a),'-b'),plt.title('Training and Validation Losses')
save_path='/home/daa/Desktop/CMRxRecon_challenge_MICCAI23/'
plt.savefig(save_path+'DAE_losses'+'.png')
