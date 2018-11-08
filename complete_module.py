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
from define_model import Unet,Downsample_block,Upsample_block
from define_training import train_model
from define_loss_score import dice_loss_2, dice_score,dice_loss_1
from define_data_loader_hcp import BrainImages

import pickle
def pickling(file,path):
    pickle.dump(file,open(path,'wb'))
def unpickling(path):
    file_return=pickle.load(open(path,'rb'))
    return file_return

import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--bs', type=int, default=12,
                    help='Batch size')

parser.add_argument('--num_epochs', type=int, default=50,
                    help='Number of epochs')

parser.add_argument('--model_name', type=str, default='model_exp1',
                    help='Name of model to be saved')


parser.add_argument('--pickle_name',type=str,default="model_exp1_loss_dice",
                    help="name of the file to pickle loss and score history")

parser.add_argument('--lr', type=int, default=1e-4,
                    help='Learning rate')

parser.add_argument('--model_load', type=str, default=None,
                    help='model to load for continued training')

parser.add_argument('--loss', type=str, default="dice",
                    help='loss function to use. if only dice=dice, else dice and CEL')

args = parser.parse_args()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        I.xavier_normal(m.weight.data)


if __name__=='__main__':
    print(torch.cuda.device_count())
    available_segments = [  0,   2,   3,   4,   5,   7,   8,  10,  11,  12,  13,  14,  15,
        16,  17,  18,  24,  26,  28,  30,  31,  41,  42,  43,  44,  46,
        47,  49,  50,  51,  52,  53,  54,  58,  60,  62,  63,  72,  77,
        80,  85, 251, 252, 253, 254, 255]
    
    rest_available = [ 2,   3,   4,   5,   7,   8,  10,  11,  12,  13,  14,  15,
        16,  17,  18,  24,  26,  28,  30,  31,  41,  42,  43,  44,  46,
        47,  49,  50,  51,  52,  53,  54,  58,  60,  62,  63,  72,  77,
        80,  85, 251, 252, 253, 254, 255]
    
    num_seg = len(rest_available)+1

    print("begin")
    
    file_names = pd.read_csv("all_complete_path.csv")
    train_subjects = unpickling("train_subject_index")
    val_subjects = unpickling("val_subject_index")
    test_subjects = unpickling("test_subject_index")
    
    
    full_train_raw = list(file_names.iloc[train_subjects,2])
    full_train_seg = list(file_names.iloc[train_subjects,3])

    full_val_raw = list(file_names.iloc[val_subjects,2])
    full_val_seg = list(file_names.iloc[val_subjects,3])

    full_test_raw = list(file_names.iloc[test_subjects,2])
    full_test_seg = list(file_names.iloc[test_subjects,3])
    
    
    rand1 = np.arange(len(full_train_raw))
    np.random.shuffle(rand1)
    rand1 = rand1[:5000]

    rand2 = np.arange(len(val_subjects))
    np.random.shuffle(rand2)
    rand2 = rand2[:1000]

    rand3 = np.arange(len(test_subjects))
    np.random.shuffle(rand3)
    rand3 = rand3[:3]
    
    cd = False
    print("creating data loaders")
    transformed_dataset = {'train': BrainImages(np.array(full_train_raw)[rand1],np.array(full_train_seg)[rand1],
                                                available_segments, rest_available,train_data=True,
                                                flipping=False, coord = cd),
                       'validate': BrainImages(np.array(full_val_raw)[rand2],np.array(full_val_seg)[rand2], available_segments,
                                               rest_available, coord = cd),
                       'test': BrainImages(np.array(full_test_raw)[rand3],np.array(full_test_seg)[rand3], available_segments,
                                           rest_available, coord = cd)
                                               }
    bs = args.bs
    dataloader = {x: DataLoader(transformed_dataset[x], batch_size=bs,
                        shuffle=True, num_workers=0) for x in ['train', 'validate','test']}
    data_sizes ={x: len(transformed_dataset[x]) for x in ['train', 'validate','test']}
    
    print("loading class weights")
    
    class_wts = unpickling('new_class_wts_46_seg')
    
    wts_torch = Variable(torch.from_numpy(class_wts)).cuda()
    
    print("model creation")
    
    if not args.model_load:
        model = Unet(in_chan = 1,out_chan = num_seg).cuda()
        model.apply(weights_init)
    else:
        model = torch.load(args.model_load).module
        
    model = nn.DataParallel(model)
    if args.loss=="dice":
        criterion = dice_loss_1
    else:
        criterion = dice_loss_2
    #scheduler = lr_scheduler.StepLR(optimizer,step_size = 15)
    
    optimizer = optim.Adam(model.parameters(),lr = args.lr)
    
    model, loss_hist, dice_hist = train_model(wts_torch, model, optimizer,dataloader,args.model_name,num_seg = num_seg,
                                              num_epochs =args.num_epochs, every = 1, print_all_ds = True)
    
    with open(args.pickle_name,'wb') as f:
        pickle.dump(loss_hist,f)
        pickle.dump(dice_hist,f)


