import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from skimage import io, transform
import matplotlib.image as mpimg
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import scipy
import random
import pickle
import scipy.io as sio
import torch.nn.init as I

import warnings
warnings.filterwarnings("ignore")
# %matplotlib inline
plt.ion()

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from skimage import io
import torch
from torchvision import transforms
from skimage import color
from torch.optim import lr_scheduler
import nibabel

from PIL import ImageEnhance
from skimage import data, exposure, img_as_float
from skimage.filters import gaussian

import re
def get_trailing_number_n_img_name(s):
    m = re.search(r'\d+$', s)
    img_name = re.sub(r'\d+$', '', s)
    return int(m.group()), img_name

class BrainImages(Dataset):
    def __init__(self, image_dir, label_dir,available_segments,rest_available, train_data = False, flipping = False,
                 rotation =True, translation = True,coord = False):
        
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.flipping = flipping
        self.rotation = rotation
        self.translation = translation
        self.train_data = train_data
        self.coord = coord
        self.available_segments = available_segments
        self.rest_available = rest_available

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self,idx):
        image_path = self.image_dir[idx]
        aseg_path = self.label_dir[idx]
        current_idx = self.label_dir[idx]
        if self.train_data:
            slice_num, aseg_name = get_trailing_number_n_img_name(current_idx[:-4])
            plus_one = slice_num + 1
            minus_one = slice_num - 1
            new_p1_path = aseg_name + str(plus_one) + '.mat'
            new_m1_path = aseg_name+str(minus_one)+'.mat'
            aseg_path_p1 = new_p1_path
            aseg_path_m1 = new_m1_path
        
        image = sio.loadmat(image_path)['img']
        aseg_img = sio.loadmat(aseg_path)['img']
        if self.train_data:
            try:
                aseg_img_p1 = sio.loadmat(aseg_path_p1)['img']
            except:
                aseg_img_p1 = np.zeros_like(aseg_img)
            try:
                aseg_img_m1 = sio.loadmat(aseg_path_m1)['img']
            except:
                aseg_img_m1 = np.zeros_like(aseg_img)
        if image.shape[2]>276:
            image = image[:,:,20:276]
            aseg_img = aseg_img[:,:,20:276]
            if self.train_data:
                aseg_img_p1 = aseg_img_p1[:,:,20:276]
                aseg_img_m1 = aseg_img_m1[:,:,20:276]
        else:
            image = image[:,:,:256]
            aseg_img = aseg_img[:,:,:256]
            if self.train_data:
                aseg_img_p1 = aseg_img_p1[:,:,:256]
                aseg_img_m1 = aseg_img_m1[:,:,:256]
        flip = random.random() > 0.5
        angle_list = [0,90.0,-90.0]
        angle = random.choice(angle_list)
        dx = np.round(random.uniform(-15,15))
        dy = np.round(random.uniform(-15,15))
        
        im = Image.fromarray(image[0])
        target = Image.fromarray(aseg_img[0])
        label_rot = 0
        if self.train_data:
            target_p1 = Image.fromarray(aseg_img_p1[0])
            target_m1 = Image.fromarray(aseg_img_m1[0])
            if self.flipping and flip:
                im = im.transpose(0)
                target = target.transpose(0)
                target_p1 = target_p1.transpose(0)
                target_m1 = target_m1.transpose(0)
            if self.rotation:
                im = im.rotate(angle)
                target = target.rotate(angle)
                target_p1 = target_p1.rotate(angle)
                target_m1 = target_m1.rotate(angle)
                label_rot = angle_list.index(angle)
            if self.translation:
                im = im.transform((256,256),0, (1,0,dx,0,1,dy))
                target = target.transform((256,256),0,(1,0,dx,0,1,dy))
                target_p1 = target_p1.transform((256,256),0,(1,0,dx,0,1,dy))
                target_m1 = target_m1.transform((256,256),0,(1,0,dx,0,1,dy))
                
                
        guassian_flag = random.random() > 0.5
        
        im = np.array(im, np.float64, copy=False)
        min_im = np.min(im)
        max_im = np.max(im)
        im = (im - min_im)/(max_im - min_im + 1e-4)
        if self.train_data and guassian_flag:
            sigma_rand = random.uniform(0.65,1.0)
            im_sigma = gaussian(im, sigma = sigma_rand)
            gamma_rand = random.uniform(1.6,2.4)
            im_sigma_gamma = exposure.adjust_gamma(im_sigma, gamma_rand)
            im = (im_sigma_gamma - np.min(im_sigma_gamma))/(np.max(im_sigma_gamma)-np.min(im_sigma_gamma)+1e-4)
        
        
        if self.coord:
            im = np.array([im, x_coordinate, y_coordinate], np.float64, copy=False)
            im = torch.from_numpy(im).type(torch.FloatTensor)
        else:
            im = torch.from_numpy(im).type(torch.FloatTensor).unsqueeze(0)

        target = np.array(target, np.float64, copy=False)
        if self.train_data:
            target_p1 = np.array(target_p1, np.float64, copy=False)
            target_m1 = np.array(target_m1, np.float64, copy=False)
            
            
        target_label = np.zeros((len(self.rest_available)+1,256,256))
        for i,a in enumerate(self.available_segments):
            temp = (target==a).astype(int)
            if a in self.rest_available:
                target_label[self.rest_available.index(a),:,:] = temp
            else:
                target_label[len(self.rest_available),:,:] = target_label[len(self.rest_available),:,:] + temp
        target_label[len(self.rest_available),:,:] = (target_label[len(self.rest_available),:,:]>=1).astype(int)
        target_label[self.rest_available.index(43),:,:] = np.logical_and(target>=100,target%2!=0)
        target_label[self.rest_available.index(42),:,:] = np.logical_and(target>=100,target%2==0)
        
        if self.train_data:
            target_p1_label = np.zeros((len(self.rest_available)+1,256,256))
            for i,a in enumerate(self.available_segments):
                temp = (target_p1==a).astype(int)
                if a in self.rest_available:
                    target_p1_label[self.rest_available.index(a),:,:] = temp
                else:
                    target_p1_label[len(self.rest_available),:,:] = target_p1_label[len(self.rest_available),:,:] + temp
            target_p1_label[len(self.rest_available),:,:] = (target_p1_label[len(self.rest_available),:,:]>=1).astype(int)
            target_p1_label[self.rest_available.index(43),:,:] = np.logical_and(target_p1>=100,target_p1%2!=0)
            target_p1_label[self.rest_available.index(42),:,:] = np.logical_and(target_p1>=100,target_p1%2==0)
            
            target_m1_label = np.zeros((len(self.rest_available)+1,256,256))
            for i,a in enumerate(self.available_segments):
                temp = (target_m1==a).astype(int)
                if a in self.rest_available:
                    target_m1_label[self.rest_available.index(a),:,:] = temp
                else:
                    target_m1_label[len(self.rest_available),:,:] = target_m1_label[len(self.rest_available),:,:] + temp
            target_m1_label[len(self.rest_available),:,:] = (target_m1_label[len(self.rest_available),:,:]>=1).astype(int)
            target_m1_label[self.rest_available.index(43),:,:] = np.logical_and(target_m1>=100,target_m1%2!=0)
            target_m1_label[self.rest_available.index(42),:,:] = np.logical_and(target_m1>=100,target_m1%2==0)
        else:
            target_p1_label = np.zeros_like(target_label)
            target_m1_label = np.zeros_like(target_label)
            
        target_label = torch.from_numpy(target_label).type(torch.FloatTensor)
        target_m1_label = torch.from_numpy(target_m1_label).type(torch.FloatTensor)
        target_p1_label = torch.from_numpy(target_p1_label).type(torch.FloatTensor)
        label_rot = torch.tensor(label_rot).long()
        sample = {'x':im,'y':(target_label,target_m1_label, target_p1_label),'label_rot': label_rot}  
        return sample
    
    