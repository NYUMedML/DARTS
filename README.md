# BrainSeg
Brain Segmentation Project

## Paper Associated with the project
Here is the paper describing the project and experiments in detail (Link to the paper to be updated).

## Deep learning models for brain MR segmentation
We pretrain our Dense Unet model using the Freesurfer segmentations of 1113 subjects available in the [Human Connectome Project](https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release) dataset and fine-tuned the model using 101 manually labeled brain scans from [Mindboggle](https://mindboggle.info/data.html) dataset.

The model is able to perform the segmentation of complete brain **within a minute** (on a machine with single GPU). The model labels 102 regions in the brain making it the first model to segment more than 100 brain regions within a minute. The details of 102 regions can be found below.

## Results on the Mindboggle held out data
The box plot compares the dice scores of different ROIs for Dense U-Net and U-Net. The Dense U-Net consistently outperforms U-Net and achieves good dice scores for most of the ROIs.

![](https://github.com/NYUMedML/BrainSeg/edit/master/plots/compare_dice_plot_aparc_manual_fd_part_1_dn_v_unet.png)

![](https://github.com/NYUMedML/BrainSeg/edit/master/plots/compare_dice_plot_aparc_manual_fd_part_2_dn_v_unet.png)


## Using Pretrained models for performing complete brain segmentation
The users can use the pre-trained models to perform a complete brain MR segmentation. For using the pre-trained models, the user will have to execute the `perform_pred.py` script. An illustration can be seen in `predicting_segmentation_illustration.ipynb` notebook.

## Pretrained model wts
Pretrained model wts can be downloaded from [here](https://drive.google.com/file/d/1m5SSiTFykQc7Bu4UUqE3bX-cotW5-oNK/view?usp=sharing).

## Output segmentation
The output segmentation has 103 labeled segments with the last one being the **None** class. The labels of the segmentation closely resembles the aseg+aparc segmentation protocol of Freesurfer. 

We exclude 4 brain regions that are not common to a normal brain: White matter and non-white matter hypointentisites, left and right frontal and temporal poles. We also excluded left and right 'unknown' segments. We also exclude left and right bankssts as there is no common definition for these segments that is widely accepted by the neuroradiology community.


The complete list of class number and the corresponding segment name is given below:


## Sample Predicitons
<object data="http://github.com/NYUMedML/BrainSeg/edit/master/plots/Right-Hippocampus_894774_108_0_1_2.pdf" type="application/pdf" width="400px" height="300px">
    <embed src="http://github.com/NYUMedML/BrainSeg/edit/master/plots/Right-Hippocampus_894774_108_0_1_2.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="http://github.com/NYUMedML/BrainSeg/edit/master/plots/Right-Hippocampus_894774_108_0_1_2.pdf">Download PDF</a>.</p>
    </embed>
</object>





