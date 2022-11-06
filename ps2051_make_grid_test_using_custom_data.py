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
prefix = 'ps2051'
workname = '임의 데이터를 이용한 make_grid 테스트'
print(prefix + '_' + workname)

# ################################################################################
# IO
# ################################################################################
# output setting
input_path = 'psdata/ps1010_some_img'
output_dir = 'psdata/' + prefix + '/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ################################################################################
# Process
# ################################################################################
tf = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5,))
])

img_train_folder = torchvision.datasets.ImageFolder(input_path, transform=tf)
trainloader = torch.utils.data.DataLoader(img_train_folder, batch_size=5, shuffle=True)

for data in trainloader:
    img, label = data
    #img_out = torchvision.utils.make_grid(img, Normalize=True).permute(1,2,0)
    # 위는 배치 사이즈(5)만큼 한줄로 출력

    # img_out = torchvision.utils.make_grid(img, nrow=2, Normalize=True).permute(1,2,0)
    # nrow=2는 한줄에 2개씩 출력

    #img_out = torchvision.utils.make_grid(img, nrow=2, padding=20, Normalize=True).permute(1,2,0)
    # padding=20은 각 이미지 사이의 간격을 20으로 설정

    img_out = torchvision.utils.make_grid(img, nrow=2, padding=20, pad_val=0.1, Normalize=True).permute(1,2,0)
    # pad_val=0.1은 배경을 0.1로 설정


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
