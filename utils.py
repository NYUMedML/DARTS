import pickle
import numpy as np
import torch
import nibabel
import torch
import torch.nn as nn
import torch.nn.functional as F


def pickling(file,path):
    pickle.dump(file,open(path,'wb'))
def unpickling(path):
    file_return=pickle.load(open(path,'rb'))
    return file_return

def dice_score(pred,gt, ep= 1e-4):
    sh1,sh2,sh2,C = pred.shape
#     print(pred.shape)
#     print(gt.shape)
    score_list = []
    for i in range(C-1):
        num = 2*(np.sum(pred[:,:,:,i]*gt[:,:,:,i])) + ep
        denom = np.sum(pred[:,:,:,i] + gt[:,:,:,i]) + ep
        score = num/denom
        score_list.append(score)
    count = np.sum(np.transpose(gt,axes = (1,0,2,3)).reshape(C,-1),axis = 1)
    return score_list,count

def load_data(path, mgz = False):
    if mgz:
        t1_img_nii = nibabel.MGHImage.from_filename(path)
    else:
        t1_img_nii = nibabel.load(path)
    affine_map = t1_img_nii.affine
    t1_img, orientation = orient_correctly(t1_img_nii)
    sh1, sh2, sh3 = t1_img.shape
    
    
    dir1_pad = (256 - sh1)//2
    if dir1_pad < 0:
        t1_img = t1_img[-dir1_pad:dir1_pad+ sh1%2,:,:]
    else:
        t1_img = np.pad(t1_img,((dir1_pad,dir1_pad + sh1%2),(0,0),(0,0)),mode = 'constant',constant_values = 0.0)
        
        
    dir2_pad = (256 - sh2)//2
    if dir2_pad < 0:
        t1_img = t1_img[:,-dir2_pad:dir2_pad+ sh2%2,:]
    else:
        t1_img = np.pad(t1_img,((0,0),(dir2_pad,dir2_pad + sh2%2),(0,0)),mode = 'constant',constant_values = 0.0)
        
     
    dir3_pad = (256 - sh3)//2
    if dir3_pad < 0:
        t1_img = t1_img[:,:,-dir3_pad:dir3_pad+ sh3%2]
    else:
        t1_img = np.pad(t1_img,((0,0),(0,0),(dir3_pad,dir3_pad + sh3%2)),mode = 'constant',constant_values = 0.0)
        
#     t1_img = np.pad(t1_img,((dir1_pad,dir1_pad + sh1%2),(dir2_pad,dir2_pad + sh2%2),(dir3_pad,dir3_pad + sh3%2)), \
#                     mode = 'constant',constant_values = 0.0)
    return t1_img,orientation, dir1_pad,sh1,dir2_pad,sh2, dir3_pad, sh3, affine_map

def orient_correctly(img_nii):
    orientation = nibabel.io_orientation(img_nii.affine)
    try:
        img_new = nibabel.as_closest_canonical(img_nii, True).get_data().astype(float)
    except:
        img_new = nibabel.as_closest_canonical(img_nii, False).get_data().astype(float)
    img_trans = np.transpose(img_new, (0,2,1))
    img_flip = np.flip(img_trans,0)
    img_flip = np.flip(img_flip,1)
    return img_flip, orientation

def orient_to_ras(image):
    img_flip = np.flip(image,0)
    img_flip = np.flip(img_flip,1)
    if image.ndim > 3:
        img_trans = np.transpose(img_flip,(0,2,1,3))
    else:
        img_trans = np.transpose(img_flip,(0,2,1))
    return img_trans

def back_to_original_4_pred(image,orientation, dir1_pad,sh1, dir2_pad,sh2, dir3_pad,sh3):
    if dir1_pad < 0:
        image = np.pad(image,((-dir1_pad,-dir1_pad- sh1%2),(0,0),(0,0)),mode = 'constant',constant_values = 0.0)
    else:
        image = image[dir1_pad:dir1_pad+sh1,:,:]
        
    if dir2_pad < 0:
        image = np.pad(image,((0,0),(-dir2_pad,-dir2_pad - sh2%2),(0,0)),mode = 'constant',constant_values = 0.0)
    else:
        image = image[:,dir2_pad:dir2_pad+sh2,:]
        
    if dir3_pad < 0:
        image = np.pad(image,((0,0),(0,0),(-dir3_pad,-dir3_pad- sh3%2)),mode = 'constant',constant_values = 0.0)
    else:
        image = image[:,:,dir3_pad:dir3_pad+sh3]
    
#     img_unpadded = image[dir1_pad:dir1_pad+sh1,dir2_pad:dir2_pad+sh2,dir3_pad:dir3_pad+sh3]
    img_ras = orient_to_ras(image)
    img_orig_orient = np.transpose(img_ras, orientation[:,0].astype(int))
    for k,i in enumerate(orientation[:,1]):
        if i == -1.0:
            img_orig_orient = np.flip(img_orig_orient,k)
    
    return img_orig_orient

def back_to_original_4_prob(image,orientation, dir1_pad,sh1, dir2_pad,sh2, dir3_pad,sh3):
    if dir1_pad < 0:
        image = np.pad(image,((-dir1_pad,-dir1_pad- sh1%2),(0,0),(0,0),(0,0)),mode = 'constant',constant_values = 0.0)
    else:
        image = image[dir1_pad:dir1_pad+sh1,:,:,:]
        
    if dir2_pad < 0:
        image = np.pad(image,((0,0),(-dir2_pad,-dir2_pad - sh2%2),(0,0),(0,0)),mode = 'constant',constant_values = 0.0)
    else:
        image = image[:,dir2_pad:dir2_pad+sh2,:,:]
        
    if dir3_pad < 0:
        image = np.pad(image,((0,0),(0,0),(-dir3_pad,-dir3_pad- sh3%2),(0,0)),mode = 'constant',constant_values = 0.0)
    else:
        image = image[:,:,dir3_pad:dir3_pad+sh3,:]
#     img_unpadded = image[dir1_pad:dir1_pad+sh1,dir2_pad:dir2_pad+sh2,dir3_pad:dir3_pad+sh3,:]
    img_ras = orient_to_ras(image)
    transpose_axis = orientation[:,0].astype(int)
    transpose_axis = np.append(transpose_axis, 3)
    img_orig_orient = np.transpose(img_ras, transpose_axis)
    for k,i in enumerate(orientation[:,1]):
        if i == -1.0:
            img_orig_orient = np.flip(img_orig_orient,k)
    
    return img_orig_orient

def create_one_hot_seg(image,num_seg):
    p = F.softmax(image,dim = 1)
    p_maxim = (torch.max(p, dim=1)[1]).cpu().data.numpy()
    img = []
    for seg in range(num_seg):
        masked = np.expand_dims((p_maxim==seg).astype(float),axis = 1)
        img.append(masked)
    return np.concatenate(img,axis = 1)
