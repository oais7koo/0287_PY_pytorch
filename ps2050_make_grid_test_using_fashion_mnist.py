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

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.image as image
import numpy as np
import torch.utils.data.dataset as Dataset

start = time.time()


# ################################################################################
# Parameter
# ################################################################################
prefix = 'ps2050'
workname = 'FashionMNIST를 이용한 make_grid 테스트'
print(prefix + '_' + workname)

# ################################################################################
# IO
# ################################################################################
# output setting
output_dir = 'psdata/' + prefix + '/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ################################################################################
# Process
# ################################################################################
tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5,))
])

trainset = torchvision.datasets.FashionMNIST(output_dir, download=True,
                                             transform=tf)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=5, shuffle=True)

for data in trainloader:
    img, label = data
    img_out = torchvision.utils.make_grid(img, Normalize=True).permute(1,2,0)
    print(img_out.shape)

    plt.imshow(img_out)
    plt.show()
    break


# ################################################################################
# Save Result
# ################################################################################


# ################################################################################
# Finish
# ################################################################################
print("time :", time.time() - start)
print('')
