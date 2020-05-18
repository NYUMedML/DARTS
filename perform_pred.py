import torch
import torch.nn as nn
import numpy as np
import os
import nibabel
import time
from models.dense_unet_model import Single_level_densenet, Down_sample, Upsample_n_Concat, Dense_Unet
# from models.dense_unet_model import *
from utils import load_data, create_one_hot_seg, back_to_original_4_pred, back_to_original_4_prob, pickling
from models.unet import Unet,Downsample_block,Upsample_block

import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument('--input_image_path', type=str,
                    help='Path to input image (can be of .mgz or .nii.gz format)(required)')

parser.add_argument('--segmentation_dir_path', type=str,
                    help='Directory path to save the output segmentation (required)')

parser.add_argument('--file_name', type = str,
                    help= 'Name of the segmentation file (required)')

parser.add_argument('--model_type', type = str,default = "dense-unet",
                    help = 'Model types: "dense-unet", "unet" (default: "dense-unet")')

parser.add_argument('--model_wts_path', type=str, default='./saved_model_wts/dense_unet_back2front_finetuned.pth',
                    help="Path for model wts to be used (default='./saved_model_wts/dense_unet_back2front_finetuned.pth')")

parser.add_argument('--is_mgz', type=str2bool, nargs='?',const=True, default=False,
                    help='Is the image in .mgz format (default=False, default format is .nii.gz)')

parser.add_argument('--save_prob', type=str2bool, nargs='?',const=True, default=False,
                    help='Should the softmax prob values for each voxel be saved ? (default: False)')

parser.add_argument('--use_gpu', type=str2bool, nargs='?',const=True, default=True,
                    help='Use GPU for inference? (default: True)')


args = parser.parse_args()


if __name__=='__main__':
    start_time = time.time()
    if args.use_gpu:
        device = 'cuda'
    else:
        device = 'cpu'
    # load model
    print("Loading model")
    
    if args.model_type == "dense-unet":
        model =  Dense_Unet(in_chan = 1, out_chan = 113, filters = 256,num_conv = 4)
    elif args.model_type == "unet":
        model =  Unet(in_chan = 1,out_chan = 113)
    model.load_state_dict(torch.load(args.model_wts_path))
    model = model.to(device)
    model.eval()
    
    print("Model Loaded, Loading image")
    
    #load image
    input_image, orient, pad_sh1, sh1 , pad_sh2 , sh2, pad_sh3, sh3, affine_map = load_data(args.input_image_path, args.is_mgz)
    input_image[np.isnan(input_image)] = 0
    input_image = np.clip(input_image, a_min=0.0, a_max = np.max(input_image))
    
    print(input_image.shape)
    
    #segments to be removed
    remove_idx_list = [34, 35, 42, 43, 73, 74, 77, 78, 108, 109]
    keep_idx_list = [i for i in range(113) if i not in remove_idx_list]
    
    print("Segmenting the image")
    
    # normalize the image
    sh_2 = input_image.shape[2]
    input_image = torch.from_numpy(input_image).to(device).float()
    for i in range(sh_2):
        max_value = torch.max(input_image[:,:,i])
        min_value = torch.min(input_image[:,:,i])
        input_image[:,:,i] = (input_image[:,:,i] - min_value)/(max_value - min_value + 1e-4)
        
    pred_arg_max = [0]*sh_2 #stores the prediction
    pred_prob = [0]*sh_2 # stores the softmax prob
    pred_oh = [0]*sh_2
    input_data = [0]*sh_2
    for i in range(sh_2):
        out = model(input_image[:,:,i].unsqueeze(0).unsqueeze(1))
        out = out[:,keep_idx_list,:,:]
#         m = nn.Softmax2d()
#         out = m(out)
#         pred_prob[i] = out.data.cpu().numpy()
        pred_arg_max[i] = torch.argmax(out,dim = 1).data.cpu().numpy()
        segs = out.size()[1]
#         pred_oh[i] = create_one_hot_seg(out,segs)
        input_data[i] = input_image[:,:,i].unsqueeze(0).data.cpu().numpy()
    
    pred_arg_max = np.moveaxis(np.concatenate(pred_arg_max,axis = 0),0,-1)
#     pred_prob = np.moveaxis(np.concatenate(pred_prob,axis = 0),0,-1)
#     pred_oh = np.moveaxis(np.concatenate(pred_oh, axis = 0),0,-1)
    input_data = np.moveaxis(np.concatenate(input_data, axis = 0),0,-1)
    
    pred_orig = back_to_original_4_pred(pred_arg_max, orient, pad_sh1, sh1 , pad_sh2 , sh2, pad_sh3, sh3)
#     pred_prob_orig = back_to_original_4_prob(pred_prob, orient, pad_sh1, sh1 , pad_sh2 , sh2, pad_sh3, sh3)
    
    print("Segmentation Completed, Saving the predictions")
    
    if args.is_mgz:
        pred_orig_nib = nibabel.freesurfer.mghformat.MGHImage(pred_orig.astype(np.float32), None)
        nibabel.save(pred_orig_nib, os.path.join(args.segmentation_dir_path,args.file_name+'_seg.mgz'))
    else:
        pred_orig_nib = nibabel.Nifti1Image(pred_orig, None)
        nibabel.save(pred_orig_nib, os.path.join(args.segmentation_dir_path,args.file_name+'_seg.nii.gz'))
        
    print("Predictions saved at ",args.segmentation_dir_path)
    end_time = time.time()
    print("Time taken ",(end_time - start_time), " secs")
    
    if args.is_mgz:
        seg = nibabel.freesurfer.MGHImage.from_filename(os.path.join(args.segmentation_dir_path,args.file_name+'_seg.mgz'))
        if np.sum(seg.affine[:,:3] != affine_map[:,:3]) != 0:
            pred_orig_nib = nibabel.freesurfer.mghformat.MGHImage(seg.get_data().astype(np.float32), affine_map)
            nibabel.save(pred_orig_nib, os.path.join(args.segmentation_dir_path,args.file_name+'_seg.mgz'))
    else:
        seg = nibabel.load(os.path.join(args.segmentation_dir_path,args.file_name+'_seg.nii.gz'))
        if np.sum(seg.affine[:,:3] != affine_map[:,:3]) != 0:
            pred_orig_nib = nibabel.Nifti1Image(seg.get_data(),affine_map)
            nibabel.save(pred_orig_nib, os.path.join(args.segmentation_dir_path,args.file_name+'_seg.nii.gz'))
        
#     if args.save_prob:
#         print('Saving prob')
#         pickling(pred_prob_orig,os.path.join(args.segmentation_dir_path,args.file_name+'_prob.nii.gz'))
#         print('Prob saved')
    
    
    
    
    
    