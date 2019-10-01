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
from models.unet import Unet,Downsample_block,Upsample_block
from train import train_model_self_sup, train_model_non_self_sup
from loss_func import *
from dataloader import BrainImages
from models.dense_unet_model.py import Single_level_densenet,Down_sample,Upsample_n_Concat,Dense_Unet


import pickle
def pickling(file,path):
    pickle.dump(file,open(path,'wb'))
def unpickling(path):
    file_return=pickle.load(open(path,'rb'))
    return file_return

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--bs', type=int, default=12,
                    help='Batch size (default=12)')

parser.add_argument('--num_epochs', type=int, default=50,
                    help='Number of epochs (default=50)')

parser.add_argument('--model_path', type=str, default='./saved_models_self_sup/',
                    help="Path to the directory for model to be saved (default='./saved_models_self_sup/')")

parser.add_argument('--model_name', type=str, required=True,
                    help='Name of model to be saved')

parser.add_argument('--pickle_path',type=str,default="./saved_loss_scores_self_sup/",
                    help="Name of the directory to pickle loss and score history (default='./saved_loss_scores_self_sup/')")

parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate (default=1e-4)')

parser.add_argument('--model_load', type=str, default=None,
                    help='model to load for continued training (default=None)')

parser.add_argument('--loss', type=str, default="dice_n_cel",
                    help='loss function to use. dice loss if only loss="dice", else dice and CEL (default="dice_n_cel")')

parser.add_argument('--view_idx', type=int, default=2,
                    help='Select a view to train the model for. 0=left_right,2=top_bottom,4=back_front. default=top_bottom')

parser.add_argument('--grad_clip', type=bool, default = False,
                    help='Is gradient clipping required? (default = False)')

parser.add_argument('--grad_clip_value', type=float, default = 1e10,
                    help='grad clipping value (default = 1e10)')

parser.add_argument('--numb_of_train_slices',type = int, default = 5000,
                    help = 'default value = 5000')

parser.add_argument('--numb_of_val_slices',type = int, default = 1000,
                    help = 'default value = 1000')

parser.add_argument('--numb_of_test_slices',type = int, default = 1000,
                    help = 'default value = 1000')

parser.add_argument('--dense_net',type = bool, default = False,
                    help = 'use dense net style U-Net? (default = False)')

parser.add_argument('--num_filters',type = int, default = 64,
                    help = 'Number of filters for dense-net (default = 64)')

parser.add_argument('--num_conv_dense',type = int, default = 4,
                    help = 'Number of conv in a dense block (default = 4)')

parser.add_argument('--not_self_sup',type = bool, default = False,
                    help = 'Not self supervised? (default = False)')

args = parser.parse_args()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        I.xavier_normal(m.weight.data)


if __name__=='__main__':
    print(args)
    print("device count",torch.cuda.device_count())
    
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
    
    num_seg = len(rest_available)+1

    print("begin")
    
    file_names = pd.read_csv("./../../brain_segmentation/complete_path_aparc.csv")
    train_subjects = unpickling("./../../brain_segmentation/train_sub_index_aparc")
    val_subjects = unpickling("./../../brain_segmentation/val_sub_index_aparc")
    test_subjects = unpickling("./../../brain_segmentation/test_sub_index_aparc")
    
    
    full_train_raw = list(file_names.iloc[train_subjects,args.view_idx])
    full_train_seg = list(file_names.iloc[train_subjects,args.view_idx+1])

    full_val_raw = list(file_names.iloc[val_subjects,args.view_idx])
    full_val_seg = list(file_names.iloc[val_subjects,args.view_idx+1])

    full_test_raw = list(file_names.iloc[test_subjects,args.view_idx])
    full_test_seg = list(file_names.iloc[test_subjects,args.view_idx+1])
    
    
    rand1 = np.arange(len(full_train_raw))
    np.random.shuffle(rand1)
    rand1 = rand1[:args.numb_of_train_slices]

    rand2 = np.arange(len(val_subjects))
    np.random.shuffle(rand2)
    rand2 = rand2[:args.numb_of_val_slices]

    rand3 = np.arange(len(test_subjects))
    np.random.shuffle(rand3)
    rand3 = rand3[:args.numb_of_test_slices]
    
    cd = False
    print("creating data loaders")
    transformed_dataset = {'train': BrainImages(np.array(full_train_raw)[rand1],np.array(full_train_seg)[rand1],
                                                available_segments, rest_available,train_data=True,binary=False,
                                                flipping=False, coord = cd, aparc = True),
                       'validate': BrainImages(np.array(full_val_raw)[rand2],np.array(full_val_seg)[rand2], available_segments,
                                               rest_available,binary=False, coord = cd, aparc = True),
                       'test': BrainImages(np.array(full_test_raw)[rand3],np.array(full_test_seg)[rand3], available_segments,
                                           rest_available, binary=False,coord = cd, aparc = True)
                                               }
    bs = args.bs
    dataloader = {x: DataLoader(transformed_dataset[x], batch_size=bs,
                        shuffle=True, num_workers=0) for x in ['train', 'validate','test']}
    data_sizes ={x: len(transformed_dataset[x]) for x in ['train', 'validate','test']}
    
    print("loading class weights")
    
    class_wts = unpickling('./../../brain_segmentation/new_class_wts_113_seg')
    
    wts_torch = Variable(torch.from_numpy(class_wts)).cuda()
    
    print("model creation")
    
    if not args.model_load:
        if args.dense_net:
            model = Dense_Unet(in_chan = 1, out_chan = num_seg, filters = args.num_filters,num_conv = args.num_conv_dense).cuda()
        else:
            model = Unet(in_chan = 1,out_chan = num_seg).cuda()
        model.apply(weights_init)
    else:
        model = torch.load(args.model_path+args.model_load).module
        
    model = nn.DataParallel(model)
    print(count_parameters(model))
    
    if args.loss=="dice":
        criterion = dice_loss_1
    else:
        criterion = dice_loss_2
    #scheduler = lr_scheduler.StepLR(optimizer,step_size = 15)
    
    optimizer = optim.Adam(model.parameters(),lr = args.lr)
    
    if not args.not_self_sup:
        model, loss_hist, dice_hist = train_model_self_sup(wts_torch, model, optimizer, dataloader, criterion,
                                              args.model_path+args.model_name, num_seg = num_seg,
                                              num_epochs =args.num_epochs, every = 1, print_all_ds = True, 
                                              clipping = args.grad_clip, clip_value = args.grad_clip_value)
    else:
        model, loss_hist, dice_hist = train_model_non_self_sup(wts_torch, model, optimizer, dataloader, criterion,
                                              args.model_path+args.model_name, num_seg = num_seg,
                                              num_epochs =args.num_epochs, every = 1, print_all_ds = True, 
                                              clipping = args.grad_clip, clip_value = args.grad_clip_value)
    
    with open(args.pickle_path+args.model_name+"_history",'wb') as f:
        pickle.dump(loss_hist,f)
        pickle.dump(dice_hist,f)


