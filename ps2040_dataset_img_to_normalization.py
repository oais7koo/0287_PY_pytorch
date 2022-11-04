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
from torchvision import datasets

start = time.time()

# ################################################################################
# Parameter
# ################################################################################
prefix = 'ps2040'
workname = '데이터셋 이미지 로딩 후 정규화'
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

# ################################################################################
# 이미지 로딩
# ################################################################################
data_transformer = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.STL10(output_dir, split='train', download=True,
                          transform=data_transformer)

print(train_ds.data.shape)

# ################################################################################
# 평균과 표준편차 계산
# ################################################################################
meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in train_ds]
# numpy의 2차원 연산 참고
# 결론적으로는 3개의 채널에 대한 각 이미지의 전체 픽셀의 평균이 남음

stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in train_ds]

meanR = np.mean([m[0] for m in meanRGB])
# 위에서 meanRGB의 m이 0000 ~ 의 각 배열에 해당되고
#그 중 m[0]이면 첫번째 요소 0.519.. 가 됨

meanG = np.mean([m[1] for m in meanRGB])
meanB = np.mean([m[2] for m in meanRGB])

stdR = np.mean([s[0] for s in stdRGB])
stdG = np.mean([s[1] for s in stdRGB])
stdB = np.mean([s[2] for s in stdRGB])

print(meanR, meanG, meanB)
print(stdR, stdG, stdB)
# ################################################################################
# Normalize ftransformation을 적용
# ################################################################################
train_transformer = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([meanR, meanG, meanB], [stdR, stdG, stdB])
])
train_ds.transform = train_transformer

# ################################################################################
# transformation이 적용된 이미지 확인
# ################################################################################
np.random.seed(0)
torch.manual_seed(0)

grid_size = 4
rnd_inds = np.random.randint(0, len(train_ds), grid_size)
print('image indices: ', rnd_inds)

x_grid = [train_ds[i][0] for i in rnd_inds]
y_grid = [train_ds[i][1] for i in rnd_inds]

x_grid = torchvision.utils.make_grid(x_grid, Normalize= True, nrow=4, padding=2).permute(1,2,0)
print(x_grid.shape)

plt.figure()
torchvision.show(x_grid, y_grid)

# ################################################################################
# Save Result
# ################################################################################

# ################################################################################
# Finish
# ################################################################################
print("time :", time.time() - start)
print('')
