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
from loss_func import dice_loss_2, dice_score

from PIL import ImageEnhance
from skimage import data, exposure, img_as_float
from skimage.filters import gaussian
from torch.nn.utils import clip_grad_value_



import time
def train_model_self_sup(wts_torch,model, optimizer,dataloader, criteria, name,num_seg = 3, num_epochs = 100, verbose = False, every = 1,\
                print_all_ds = True, clipping = False, clip_value = None, output_clamp_val = 10000, not_self_sup = False, rot = False):
    since = time.time()
    cel = nn.CrossEntropyLoss()
    best_loss = np.inf
    best_score = 0
    loss_hist = {'train':[],'validate':[]}
    dice_scores_of_all_class = [{'train':[],'validate':[]} for i in range(num_seg)]    
    for i in range(num_epochs):
        for phase in ['train', 'validate']:
            running_loss = 0
            run_class_scores = [0]*num_seg
            running_total = 0
            if phase == 'train':
                model.train(True)
            else:
                model.eval()
    
            for data in dataloader[phase]:
                optimizer.zero_grad()
                x = data['x']
                y,ym1,yp1 = data['y']
                label_rot = data['label_rot']
                x = Variable(x).cuda()
                y = Variable(y).cuda()
                ym1 = Variable(ym1).cuda()
                yp1 = Variable(yp1).cuda()
                if rot:
                    label_rot = label_rot.cuda()
                N, C, sh1, sh2 = y.size()
                rand_num = np.random.rand()
                if rand_num < 0.90:
                    y_mask = (torch.sum(y[:,:(C-1),:,:].contiguous().view(N,-1),dim = 1) != 0)
                    y = y[y_mask]
                    x = x[y_mask]
                    ym1 = ym1[y_mask]
                    yp1 = yp1[y_mask]
                    if rot:
                        label_rot = label_rot[y_mask]
                try:
                    if rot:
                        output, output_m1, output_p1, output_rot = model(x)
                    else:
                        output, output_m1, output_p1 = model(x)
                except:
                    continue
#                 output, output_m1, output_p1 = model(x)
                output = torch.clamp(output,-output_clamp_val,output_clamp_val)
                output_m1 = torch.clamp(output_m1,-output_clamp_val,output_clamp_val)
                output_p1 = torch.clamp(output_p1,-output_clamp_val,output_clamp_val)
                loss = criteria(y, output,wts_torch)
                lossm1 = criteria(ym1, output_m1,wts_torch)
                lossp1 = criteria(yp1, output_p1,wts_torch)
                if rot:
                    loss_rot = cel(output_rot,label_rot)
                else:
                    loss_rot = 0
                if not_self_sup:
                    lossm1 = 0.0
                    lossp1 = 0.0
                if phase == 'train':
                    loss_total = loss + lossm1 + lossp1 + loss_rot
                    loss_total.backward()
                    if clipping:
                        clip_grad_value_(model.parameters(),clip_value)
                        optimizer.step()
    #                     for p in model.parameters():
    #                         p.data.add_(-lr, p.grad.data)
                    else:
                        optimizer.step()
                running_loss += loss.data.item() * N
                running_total += N
                dice_score_batch = dice_score(y,output)

                for j in range(num_seg):
                    run_class_scores[j] += dice_score_batch[j] * N
            epoch_loss = running_loss/running_total
            loss_hist[phase].append(epoch_loss)
            epoch_score = 0
            for j in range(num_seg):
                score = run_class_scores[j]/running_total
                dice_scores_of_all_class[j][phase].append(score.item())
                if j < num_seg - 1:
                    epoch_score += score.item()
            epoch_score_av = epoch_score/ (num_seg - 1)
            if verbose or i%every == 0:
                print('Epoch: {}, Phase: {}, epoch loss: {:.4f}, Av. Dice Score: {:.4f}'\
                      .format(i,phase,epoch_loss,epoch_score_av))
                if print_all_ds:
                    for j in range(num_seg - 1):
                        print('Class {} (DS): {:.4f}'.format(j,dice_scores_of_all_class[j][phase][-1]))
                print('-'*10)
            
        if phase == 'validate' and epoch_score_av > best_score:
            best_loss = epoch_loss
            best_score = epoch_score_av
            best_model_wts = model.state_dict()
            torch.save(model,name)
    print('-'*50)    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val dice loss: {:4f}, Best Average DS: {:4f}'.format(best_loss,best_score))
    
    model.load_state_dict(best_model_wts)
    
    return model, loss_hist, dice_scores_of_all_class

def train_model_non_self_sup(wts_torch,model, optimizer,dataloader, criteria, name,num_seg = 3, num_epochs = 100, verbose = False, every = 1,\
                print_all_ds = True, clipping = False, clip_value = None, output_clamp_val = 10000):
    since = time.time()
    best_loss = np.inf
    best_score = 0
    loss_hist = {'train':[],'validate':[]}
    dice_scores_of_all_class = [{'train':[],'validate':[]} for i in range(num_seg)]    
    for i in range(num_epochs):
        for phase in ['train', 'validate']:
            running_loss = 0
            run_class_scores = [0]*num_seg
            running_total = 0
            if phase == 'train':
                model.train(True)
            else:
                model.eval()
    
            for data in dataloader[phase]:
                optimizer.zero_grad()
                x = data['x']
                y = data['y'][0]
                x = Variable(x).cuda()
                y = Variable(y).cuda()
                N, C, sh1, sh2 = y.size()
                rand_num = np.random.rand()
                if rand_num < 0.90:
                    y_mask = (torch.sum(y[:,:(C-1),:,:].contiguous().view(N,-1),dim = 1) != 0)
                    y = y[y_mask]
                    x = x[y_mask]
                try:
                    output = model(x)[0]
                except:
                    continue
#                 output, output_m1, output_p1 = model(x)
                output = torch.clamp(output,-output_clamp_val,output_clamp_val)
                loss = criteria(y, output,wts_torch)
                if phase == 'train':
                    loss.backward()
                    if clipping:
                        clip_grad_value_(model.parameters(),clip_value)
                        optimizer.step()
    #                     for p in model.parameters():
    #                         p.data.add_(-lr, p.grad.data)
                    else:
                        optimizer.step()
                running_loss += loss.data.item() * N
                running_total += N
                dice_score_batch = dice_score(y,output)

                for j in range(num_seg):
                    run_class_scores[j] += dice_score_batch[j] * N
            epoch_loss = running_loss/running_total
            loss_hist[phase].append(epoch_loss)
            epoch_score = 0
            for j in range(num_seg):
                score = run_class_scores[j]/running_total
                dice_scores_of_all_class[j][phase].append(score.item())
                if j < num_seg - 1:
                    epoch_score += score.item()
            epoch_score_av = epoch_score/ (num_seg - 1)
            if verbose or i%every == 0:
                print('Epoch: {}, Phase: {}, epoch loss: {:.4f}, Av. Dice Score: {:.4f}'\
                      .format(i,phase,epoch_loss,epoch_score_av))
                if print_all_ds:
                    for j in range(num_seg - 1):
                        print('Class {} (DS): {:.4f}'.format(j,dice_scores_of_all_class[j][phase][-1]))
                print('-'*10)
            
        if phase == 'validate' and epoch_score_av > best_score:
            best_loss = epoch_loss
            best_score = epoch_score_av
            best_model_wts = model.state_dict()
            torch.save(model,name)
    print('-'*50)    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val dice loss: {:4f}, Best Average DS: {:4f}'.format(best_loss,best_score))
    
    model.load_state_dict(best_model_wts)
    
    return model, loss_hist, dice_scores_of_all_class






