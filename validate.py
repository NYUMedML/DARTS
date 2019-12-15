import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
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
import torchvision
from skimage import color
from torch.optim import lr_scheduler
from torchvision.utils import make_grid
import nibabel

from PIL import ImageEnhance
from skimage import data, exposure, img_as_float
from skimage.filters import gaussian
from DARTS import Unet,Downsample_block,Upsample_block
from train import train_model_self_sup, train_model_non_self_sup
from loss_func import *
from dataloader import BrainImages
from DARTS import Single_level_densenet,Down_sample,Upsample_n_Concat,Dense_Unet

from PIL import ImageEnhance
from skimage import data, exposure, img_as_float
from skimage.filters import gaussian
from matplotlib.pyplot import savefig

import pickle
def pickling(file,path):
    pickle.dump(file,open(path,'wb'))
def unpickling(path):
    file_return=pickle.load(open(path,'rb'))
    return file_return

import argparse


parser = argparse.ArgumentParser()


parser.add_argument('--num_subs', type=int, default=20,
                    help='Number of subjects')

parser.add_argument('--model_path', type=str, default='./saved_models/',
                    help='Path to the directory for model to be saved')

parser.add_argument('--model_name', type=str, required=True,
                    help='Name of model to be saved')

parser.add_argument('--score_path', type=str, required=True,
                    help='Name of model to be saved')

parser.add_argument('--binary', type=int, default=0,
                    help='Target class should be binary or not')

parser.add_argument('--view_idx', type=int, default=2,
                    help='Select a view to train the model for. 0=left_right,2=top_bottom,4=back_front. default=top_bottom')

args = parser.parse_args()

def visualize1(image,num_seg):
    p = F.softmax(image,dim = 1)
    p_maxim = (torch.max(p, dim=1)[1]).cpu().data.numpy()
    img = []
    for seg in range(num_seg):
        masked = np.expand_dims((p_maxim==seg).astype(float),axis = 1)
        img.append(masked)
    return np.concatenate(img,axis = 1)

def dice_score(pred,gt, ep= 1e-4):
    N,C,sh1,sh2 = pred.shape
#     print(pred.shape)
#     print(gt.shape)
    score_list = []
    for i in range(C):
        num = 2*(np.sum(pred[:,i,:,:]*gt[:,i,:,:])) + ep
        denom = np.sum(pred[:,i,:,:] + gt[:,i,:,:]) + ep
        score = num/denom
        score_list.append(score)
    count = np.sum(np.transpose(gt,axes = (1,0,2,3)).reshape(C,-1),axis = 1)
    return score_list,count

def run_subs():
    dc_score_list = []
    count_list = []
    time_list = []
    dice_score_overall_list = []
    for i in range(args.num_subs):
        print(i)
        start = time.time()
        cd = False
        transformed_dataset = BrainImages(np.array(full_val_raw)[i*256:256*(i+1)],np.array(full_val_seg)[i*256:256*(i+1)],\
                                                   available_segments, rest_available,binary=args.binary, coord = cd, aparc = True)
        bs = 1
        dataloader = DataLoader(transformed_dataset, batch_size=bs, shuffle=False, num_workers=0)
        data_sizes =len(transformed_dataset)
                        
        output_1 = []
        true_1 = []
        for data in dataloader:
            model_1.train(False)
            x = Variable(data['x']).cuda()
            y = data['y'][0]
            y = y.numpy()
            out = model_1(x)[0]
            segs = out.size()[1]
            pred = visualize1(out,segs)
            output_1.append(pred[:,:-1,:,:])
            true_1.append(y[:,:-1,:,:])
            
        output = np.concatenate(output_1,axis = 0)
        
        true = np.concatenate(true_1,axis = 0)
    
        dc,count_seg = dice_score(output,true)
        dc_score_list.append(dc)
        count_list.append(count_seg)
        end = time.time()
        total_time = end - start
        time_list.append(total_time)
        
    return dc_score_list,count_list,time_list
    
def plot_box():
    fig, ax1 = plt.subplots(figsize=(20,10))
    fs = 10
    pickle.dump(mean_dc_score,open(args.score_path +args.model_name+"_dicescores",'wb'))
    ax1.boxplot(dc_score_list_np, labels = np.array(rest_available),showfliers = False,whis = 1.0)
    ax1.plot(np.arange(1,num_seg),[np.mean(mean_dc_score)]*len(rest_available),'-r',label = 'Mean dice score')
    overall_dc_score = np.mean(mean_dc_score)
    plt.text(30,overall_dc_score+0.01,'Average Dice Score=%.3f' % overall_dc_score,fontsize=fs)
    ax1.set_xlabel('Segment #')
    ax1.set_ylabel('Dice Score')
#     ax2 = ax1.twinx()
#     ax2.plot(np.arange(1,46),np.log10(count_list_np), label = 'Log(Voxel Count)')
#     ax2.set_ylabel('Log(Voxels)')
    fig.savefig(args.model_name+".png")
    plt.legend()
    plt.show()
    

if __name__=='__main__':
    available_segments = [0,2,4,5,7,8,10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 30, 31, 41,
    43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62, 63, 77, 80, 85, 251, 252, 253,
    254, 255, 1000, 1001, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011,
    1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024,
    1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 2000, 2001,
    2002, 2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
    2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028,
    2029, 2030, 2031, 2032, 2033, 2034, 2035]
    
    rest_available = [2,4,5,7,8,10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 30, 31, 41,
     43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62, 63, 77, 80, 85, 251, 252, 253,
     254, 255, 1000, 1001, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011,
     1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024,
     1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 2000, 2001,
     2002, 2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
     2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028,
     2029, 2030, 2031, 2032, 2033, 2034, 2035]
    
    if args.binary:
        num_seg = 2
    else:
        num_seg = len(rest_available)+1

    model_1 = torch.load(args.model_path+args.model_name)
    file_names = pd.read_csv("./../../brain_segmentation/complete_path_aparc.csv")
    val_subjects = unpickling("./../../brain_segmentation/val_sub_index_aparc")

    full_val_raw = list(file_names.iloc[val_subjects,args.view_idx])
    full_val_seg = list(file_names.iloc[val_subjects,args.view_idx+1])
    
    
    dc_score_list,count_list,time_list = run_subs()
    
#     dc_score_list_np = np.vstack(dc_score_list)
#     mean_dc_score = np.mean(dc_score_list_np,axis = 0)
    count_list_all_np = np.vstack(np.array(count_list))
    dc_score_list_np = np.vstack(dc_score_list)
    
    overall_dc_score = np.sum(count_list_all_np * dc_score_list_np)/np.sum(count_list_all_np)

    mean_dc_score = overall_dc_score
    
    pickle.dump(dc_score_list_np,open(args.score_path + args.model_name+"_complete_dicescores",'wb'))
    
    plot_box()
    
    
    
    
