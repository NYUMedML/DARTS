import torch
import torch.nn as nn
import numpy as np
import os
import nibabel
import re

from DARTS.models.dense_unet_model import Single_level_densenet, Down_sample, Upsample_n_Concat, Dense_Unet
from DARTS.utils import load_data, create_one_hot_seg, back_to_original_4_pred, back_to_original_4_prob, pickling
from DARTS.models.unet import Unet, Downsample_block, Upsample_block


class Segmentation(nn.Module):
    def __init__(self, model_wts_path, model_type="dense-unet", use_gpu=True):
        """
        Initialize the Segmentation object
        Parameters
        ----------
        model_wts_path : "path to pre-trained model"
        model_type : Type of model = ['dense-unet','unet']
        use_gpu : [True, False]
        """
        super(Segmentation, self).__init__()

        if use_gpu:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        if model_type == "dense-unet":
            self.model = Dense_Unet(in_chan=1, out_chan=113, filters=256, num_conv=4)
        elif model_type == "unet":
            self.model = Unet(in_chan=1, out_chan=113)

        self.model.load_state_dict(torch.load(model_wts_path))
        self.model = self.model.to(self.device)
        self.model.eval()
        self.input_image = None

    def read_image(self, input_image_path, is_mgz=None):
        """
        Read the MRI specified and output normalized MRI as an np.array if shape [256, 256, depth]
        Parameters
        ----------
        input_image_path : Path to the image
        is_mgz : if .mgz=True, .nii.gz = False (optional)

        Returns
        -------
        Normalized input MRI as np.array
        """
        if is_mgz:
            self.is_mgz = is_mgz
        else:
            if ".mgz" in str.lower(input_image_path):
                self.is_mgz = True
            else:
                self.is_mgz = False

        input_image, orient, pad_sh1, sh1, pad_sh2, sh2, pad_sh3, sh3, affine_map = load_data(input_image_path, is_mgz)
        input_image[np.isnan(input_image)] = 0
        input_image = np.clip(input_image, a_min=0.0, a_max=np.max(input_image))

        remove_idx_list = [34, 35, 42, 43, 73, 74, 77, 78, 108, 109]
        self.keep_idx_list = [i for i in range(113) if i not in remove_idx_list]

        self.sh2 = input_image.shape[2]
        input_image = torch.from_numpy(input_image).to(self.device).float()
        for i in range(self.sh2):
            max_value = torch.max(input_image[:, :, i])
            min_value = torch.min(input_image[:, :, i])
            input_image[:, :, i] = (input_image[:, :, i] - min_value) / (max_value - min_value + 1e-4)

        self.input_image = input_image
        self.orient, self.pad_sh1, self.sh1, self.pad_sh2, self.sh2, self.pad_sh3, self.sh3, self.affine_map = orient, pad_sh1, sh1, \
                                                                                              pad_sh2, sh2, pad_sh3, sh3, affine_map

        return self.input_image

    def predict(self, inputs, save_directory=None, save_proba=None, is_mgz=None):
        """
        Segment the input image
        Parameters
        ----------
        inputs : path to MRI image to segment or list of paths
        save_directory : Where the output has to be saved. If None, the output wont be saved
        save_proba : Save the output probability? [True, False, None]
        is_mgz : [True, False , None]. If none, the type if automatically determined

        Returns
        -------
        output segment mask(np.array) or list of output segment mask
        """
        if type(inputs) == str:
            self.pred_orig, self.pred_prob_orig = self.predict_single(inputs, save_directory, save_proba, is_mgz)

        else:
            self.pred_orig = []
            self.pred_prob_orig = []
            for file_name in inputs:
                out1, out2 = self.predict_single(file_name, save_directory, save_proba, is_mgz)
                self.pred_orig.append(out1)
                self.pred_prob_orig.append(out2)

        return self.pred_orig, self.pred_prob_orig

    def predict_single(self, input_image_path, save_output_dir="./", save_proba_file=None, is_mgz=None):
        _ = self.read_image(input_image_path, is_mgz)
        print("Starting Segmentation")
        sh2 = self.sh2
        keep_idx_list = self.keep_idx_list

        pred_arg_max = [0] * sh2  # stores the prediction
        pred_prob = [0] * sh2  # stores the softmax prob
        pred_oh = [0] * sh2
        input_data = [0] * sh2
        for i in range(sh2):
            out = self.model(self.input_image[:, :, i].unsqueeze(0).unsqueeze(1))
            out = out[:, keep_idx_list, :, :]
            m = nn.Softmax2d()
            out = m(out)
            pred_prob[i] = out.data.cpu().numpy()
            pred_arg_max[i] = torch.argmax(out, dim=1).data.cpu().numpy()
            segs = out.size()[1]
            pred_oh[i] = create_one_hot_seg(out, segs)
            input_data[i] = self.input_image[:, :, i].unsqueeze(0).data.cpu().numpy()

        pred_arg_max = np.moveaxis(np.concatenate(pred_arg_max, axis=0), 0, -1)
        pred_prob = np.moveaxis(np.concatenate(pred_prob, axis=0), 0, -1)
        pred_oh = np.moveaxis(np.concatenate(pred_oh, axis=0), 0, -1)
        input_data = np.moveaxis(np.concatenate(input_data, axis=0), 0, -1)

        pred_orig = back_to_original_4_pred(pred_arg_max, self.orient, self.pad_sh1, self.sh1, self.pad_sh2, \
                                            self.sh2, self.pad_sh3, self.sh3)
        pred_prob_orig = back_to_original_4_prob(pred_prob, self.orient, self.pad_sh1, self.sh1, self.pad_sh2, \
                                                 self.sh2, self.pad_sh3, self.sh3)

        print("segmentation complete")

        if save_output_dir:
            self.save_output_one(pred_orig, pred_prob_orig, save_output_dir, input_image_path.split('/')[-1],
                                 save_proba_file)
        return pred_orig, pred_prob_orig

    def save_output(self, pred_orig, pred_prob_orig, segmentation_dir, file_names, save_prob):
        """
        Save the output of the model
        Parameters
        ----------
        pred_orig : Predicted segment mask of type np.array of shape [256, 256, d] where d is depth
        pred_prob_orig : Output probability np.array of size
        segmentation_dir : Directory where the output has to be saved
        file_names : Output file names
        save_prob : [True, False]

        Returns
        -------
        None
        """
        if type(file_names) == str:
            pred_orig = [pred_orig]
            pred_prob_orig = [pred_prob_orig]
            file_names = [file_names]
        for i in range(len(pred_orig)):
            self.save_output_one(pred_orig[i], pred_prob_orig[i], segmentation_dir, file_names[i], save_prob)

    def save_output_one(self, pred_orig, pred_prob_orig=None, segmentation_dir="./", file_name="temp_out",
                        save_prob=False):
        file_name = re.sub("[^0-9a-zA-Z]", "", file_name)
        if self.is_mgz == True:
            pred_orig_nib = nibabel.freesurfer.mghformat.MGHImage(pred_orig.astype(np.float32), None)
            nibabel.save(pred_orig_nib, os.path.join(segmentation_dir, file_name + '_seg.mgz'))
        else:
            pred_orig_nib = nibabel.Nifti1Image(pred_orig, None)
            nibabel.save(pred_orig_nib, os.path.join(segmentation_dir, file_name + '_seg.nii.gz'))

        print("Predictions saved at ", segmentation_dir)

        if save_prob == True:
            pickling(pred_prob_orig, os.path.join(segmentation_dir, file_name + '_prob.nii.gz'))
