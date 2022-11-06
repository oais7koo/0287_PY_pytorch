# -*- coding: utf-8 -*-
# author : oaiskoo
# date : 2022.


# ################################################################################
# Library
# ################################################################################
import os
import shutil
import time
import pandas as pd
from glob import glob
from os.path import basename
import pickle
import numpy as np

import torch
import torchvision
import torchvision.transforms as tf
import matplotlib.pyplot as plt
import matplotlib.image as image
import numpy as np
from PIL import Image
import cv2
import torchvision.models.segmentation

start = time.time()

# ################################################################################
# Setting
# ################################################################################
prefix = 'ps3121'
workname = 'infer'
print(prefix + '_' + workname)

# output setting
output_dir = 'psdata/' + prefix + '/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ################################################################################
# IO
# ################################################################################
modelPath = "psdata/ps3125/3000.torch"  # Path to trained model
imagePath = "test.jpg"  # Test image
height=width=900


# ################################################################################
# Process
# ################################################################################
transformImg = tf.Compose([tf.ToPILImage(),
                           tf.Resize((height, width)),
                           tf.ToTensor(),tf.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
                           # tf.Resize((300,600)),tf.RandomRotation(145)])#

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Check if there is GPU if not set trainning to CPU (very slow)

# ################################################################################
# Load Model
# ################################################################################
Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)  # Load net
Net.classifier[4] = torch.nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))  # Change final layer to 3 classes
Net = Net.to(device)  # Set net to GPU or CPU
Net.load_state_dict(torch.load(modelPath)) # Load trained model
Net.eval() # Set to evaluation mode

# ################################################################################
# Eval
# ################################################################################
Img = cv2.imread(imagePath) # load test image
height_orgin , widh_orgin ,d = Img.shape # Get image original size
plt.imshow(Img[:,:,::-1])  # Show image
plt.show()
Img = transformImg(Img)  # Transform to pytorch
Img = torch.autograd.Variable(Img, requires_grad=False).to(device).unsqueeze(0)
with torch.no_grad():
    Prd = Net(Img)['out']  # Run net
Prd = tf.Resize((height_orgin,widh_orgin))(Prd[0]) # Resize to origninal size
seg = torch.argmax(Prd, 0).cpu().detach().numpy()  # Get  prediction classes
plt.imshow(seg)  # display image
plt.show()

# ################################################################################
# Finish
# ################################################################################
print("time :", time.time() - start)
print('')
