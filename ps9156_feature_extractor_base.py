# -*- coding: utf-8 -*-
# author : oaiskoo
# date : 2022.
# goal :

# ################################################################################
# Library
# ################################################################################
import os
import shutil
import time
import re
import pandas as pd
import numpy as np

from os.path import basename
import pickle

if os.path.exists('../python/oaislib_org.py'):
    shutil.copy('../python/oaislib_org.py', 'oaislib.py')
import oaislib
start = time.time()

import copy
import glob
import cv2
import shutil


import torch
import torchvision
# 컴퓨터 비전 용도의 패키지

import torchvision.transforms as transforms
# 데어티 전처리를 위해 사용되는 패키지

import torchvision.models as models
# 다양한 파이토치 네트워크를 사용할 수 있도록 하는 패키지

import torch.nn as nn
import torch.optim as potim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# ################################################################################
# Parameter
# ################################################################################
prefix = 'ps9156'
workname = '특성추출'
print(prefix + '_' + workname)


# ################################################################################
# IO
# ################################################################################
# output setting
output_dir = 'psdata/' + prefix + '/'
oaislib.fn_output_dir_gen(output_dir)

#################################################################################
# Process
# ################################################################################

#################################################################################
# 코드 5-13
# ################################################################################
# 이미지 데이터가 위치한 경로
data_path = 'psdata/ps9100_book_code/chap05/data/catanddog/train/'

# torchvision.transforms.Compose()를 이용하여 이미지를 모델의 입력으로 사용할 수 있도록 변환

transform = transforms.Compose(
    [
        transforms.Resize([256, 256]),
        transforms.RandomResizedCrop(224),
        # 이미지를 224x224 크기로 자르되, 랜덤하게 자르도록 설정
        # 그러면 모든 이미지의 크기가 224 * 224인가?
        # 아래 train_dataset[0][0].shape 가 torch.Size([3, 224, 224])
        # 따라서 224 * 224 임

        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

# 데이터셋은 원데이터를 transfomr을 이용하여 변환한 데이터셋
train_dataset = torchvision.datasets.ImageFolder(
    data_path,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    num_workers=8,
    shuffle=True
)

print(len(train_dataset))

#################################################################################
# Save Result
# ################################################################################

#################################################################################
# Finish
# ################################################################################
print("time :", time.time() - start)
print('')
