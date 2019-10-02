# BrainSeg
Brain Segmentation Project

## Paper Associated with the project
Here is the paper describing the project and experiments in detail (Link to the paper to be updated).

## Deep learning models for brain MR segmentation
We pretrain our Dense Unet model using the Freesurfer segmentations of 1113 subjects available in the [Human Connectome Project](https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release) dataset and fine-tuned the model using 101 manually labeled brain scans from [Mindboggle](https://mindboggle.info/data.html) dataset.

The model is able to perform the segmentation of complete brain **within a minute** (on a machine with single GPU). The model labels 102 regions in the brain making it the first model to segment more than 100 brain regions within a minute. The details of 102 regions can be found below.



## Results on the Mindboggle held out data
The box plot compares the dice scores of different ROIs for Dense U-Net and U-Net. The Dense U-Net consistently outperforms U-Net and achieves good dice scores for most of the ROIs.

![Dice-plot-1](https://github.com/NYUMedML/BrainSeg/edit/master/plots/compare_dice_plot_aparc_manual_fd_part_1_dn_v_unet.png)

![Dice-plot-2](https://github.com/NYUMedML/BrainSeg/edit/master/plots/compare_dice_plot_aparc_manual_fd_part_2_dn_v_unet.png)


## Using Pretrained models for performing complete brain segmentation
The users can use the pre-trained models to perform a complete brain MR segmentation. For using the **coronally** pre-trained models, the user will have to execute the [`perform_pred.py`](https://github.com/NYUMedML/BrainSeg/blob/master/perform_pred.py) script. An illustration can be seen in [`predicting_segmentation_illustration.ipynb`](https://github.com/NYUMedML/BrainSeg/blob/master/predicting_segmentation_illustration.ipynb) notebook.

The following code block could be used to perform the prediction:
```
usage: perform_pred.py [-h] [--input_image_path INPUT_IMAGE_PATH]
                       [--segmentation_dir_path SEGMENTATION_DIR_PATH]
                       [--file_name FILE_NAME] [--model_type MODEL_TYPE]
                       [--model_wts_path MODEL_WTS_PATH] [--is_mgz [IS_MGZ]]
                       [--save_prob [SAVE_PROB]] [--use_gpu [USE_GPU]]

optional arguments:
  -h, --help            show this help message and exit
  --input_image_path INPUT_IMAGE_PATH
                        Path to input image (can be of .mgz or .nii.gz
                        format)(required)
  --segmentation_dir_path SEGMENTATION_DIR_PATH
                        Directory path to save the output segmentation
                        (required)
  --file_name FILE_NAME
                        Name of the segmentation file (required)
  --model_type MODEL_TYPE
                        Model types: "dense-unet", "unet" (default: "dense-
                        unet")
  --model_wts_path MODEL_WTS_PATH
                        Path for model wts to be used (default='./saved_model_
                        wts/dense_unet_back2front_finetuned.pth')
  --is_mgz [IS_MGZ]     Is the image in .mgz format (default=False, default
                        format is .nii.gz)
  --save_prob [SAVE_PROB]
                        Should the softmax prob values for each voxel be saved
                        ? (default: False)
  --use_gpu [USE_GPU]   Use GPU for inference? (default: True)

```
An example could look something like this:
```
python3 perform_pred.py --input_image_path './../../../data_orig/199251/mri/T1.mgz' \
--segmentation_dir_path './sample_pred/' \
--file_name '199251' \
--is_mgz True \
--model_wts_path './saved_model_wts/dense_unet_back2front_non_finetuned.pth' \
--save_prob False \
--use_gpu True \
--save_prob False
```

## Pretrained model wts
Pretrained model wts can be downloaded from [here](https://drive.google.com/file/d/1-reUDvwBhSOUqOa48W9Vgh_LN3F5ZRjQ/view?usp=sharing). 

There are two model architectures: Dense U-Net and U-Net. Each of the model is trained using 2D slices extracted coronally, sagittally,or axially. The name of the model will contain the orientation and model architecture information. 

## Output segmentation
The output segmentation has 103 labeled segments with the last one being the **None** class. The labels of the segmentation closely resembles the aseg+aparc segmentation protocol of Freesurfer. 

We exclude 4 brain regions that are not common to a normal brain: White matter and non-white matter hypointentisites, left and right frontal and temporal poles. We also excluded left and right 'unknown' segments. We also exclude left and right bankssts as there is no common definition for these segments that is widely accepted by the neuroradiology community.


The complete list of class number and the corresponding segment name can be found [here](https://github.com/NYUMedML/BrainSeg/blob/master/name_class_mapping.p).

## Sample Predicitons
### Insula
Here we can clearly see that Freesurfer (FS) incorrectly predicts the right insula segment, the model trained only using FS segmentations also learns a wrong prediction. Our proposed model which is finetuned on manually annotated dataset correctly captures the region. Moreover, the segment looks biologically natural unlike FS's segmentation which is grainy, non-smooth boundaries and noisy.
![](https://github.com/NYUMedML/BrainSeg/edit/master/plots/rt_insula_aparc_with_man_3.png)


Some sample predictions for [Putamen](https://github.com/NYUMedML/BrainSeg/blob/master/plots/Left-Putamen_627549_143_0_1_2.pdf), [Caudate](https://github.com/NYUMedML/BrainSeg/blob/master/plots/Right-Caudate_194443_137_0_1_2.pdf), [Hippocampus](https://github.com/NYUMedML/BrainSeg/blob/master/plots/Right-Hippocampus_894774_108_0_1_2.pdf) and [Insula](https://github.com/NYUMedML/BrainSeg/blob/master/plots/ctx-lh-insula_147030_138_0_1_2.pdf) can be seen here. In all the images, prediction 1 = Freesurfer, Prediction 2 = Non-Finetuned Dense Unet, Prediction 3 = Finetuned Dense Unet. 

It could be seen that Freesurfer often make errors in determining the accurate boundaries whereas the deep learning based models have natural looking ROIs with more accurate boundaries.

## Contact
If you have any questions regarding the code, please contact ark576[at]nyu.edu or raise an issue on the github repo.
