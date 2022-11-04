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
import numpy as np
from PIL import Image

start = time.time()

# ################################################################################
# Parameter
# ################################################################################
prefix = 'ps2030'
workname = 'CIFAR-10 데이터셋을 불러와서 계층구조로 저장'
print(prefix + '_' + workname)

# ################################################################################
# IO
# ################################################################################
# output setting
output_dir = 'psdata/' + prefix + '/'
if os.path.exists(output_dir):
    pass
else:
    os.makedirs(output_dir)

# ################################################################################
# Process
# ################################################################################

# ################################################################################
# CIFAR-10 불러오기
# ################################################################################
transform_train = transforms.Compose([
    transforms.ToTensor(),
])
train_dataset = torchvision.datasets.CIFAR10(root=output_dir,
                                             download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                           shuffle=True, num_workers=4)

# ################################################################################
# 각 이미지 별로 각 레이블의 이미지가 몇번 등장했는지 확인
# ################################################################################
num_classes = 10
number_per_class = {}

for i in range(num_classes):
    number_per_class[i] = 0

# ################################################################################
# 이미지 텐서와 레이블 정수값 저장 함수 작성
# ################################################################################
def custom_imsave(img, label):
    path = output_dir + str(label) + '/'
    if not os.path.exists(path):
        os.makedirs(path)

    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))
    img.imsave(path + str(number_per_class[label]) + '.png', img)
    number_per_class[label] += 1

# ################################################################################
# 데이터셋을 배치로 읽고 파일로 폴더에 저장
# ################################################################################
for batch_idx, (inputs, targets) in enumerate(train_loader):
    print("[ Current Batch index: " + str(batch_idx) + " ]")
    for i in range(inputs.size(0)):
        custom_imsave(inputs[i], targets[i].item())

# ################################################################################
# 0번 레이블의 첫번째 이미지를 출력
# ################################################################################
img = Image.open(output_dir + '0/0.jpg')
plt.imshow(np.asarray(img))


# ################################################################################
# Finish
# ################################################################################
print("time :", time.time() - start)
print('')
