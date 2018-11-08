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
from PIL import ImageEnhance
from skimage.util import random_noise
#import warnings
#warnings.filterwarnings("ignore")
#%matplotlib inline
#plt.ion()

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

file_names = pd.read_csv("all_complete_path.csv")

df = pd.DataFrame(columns=np.arange(1000))
for i,f in enumerate(file_names["top_to_bottom_segmented"]):
    print(i)
    img_orig = nibabel.freesurfer.mghformat.MGHImage.from_filename(f)
    image = img_orig.get_data().astype(np.float64)
    x, y = np.unique(image, return_counts=True)
    for j in range(len(x)):
        df.loc[i,x[j]] = y[j]
        
        #print(df.iloc[i])
    

df["file_names"] = file_names["top_to_bottom_segmented"]

df.to_csv("top_bottom_segment_count1.csv",index=False)
