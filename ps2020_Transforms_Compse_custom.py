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


import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

# ################################################################################
# Parameter
# ################################################################################
prefix = 'ps2020'
workname = 'Transforms_Compse_gen'
print(prefix + '_' + workname)

# ################################################################################
# IO
# ################################################################################
# output setting
output_dir = 'psdata/' + prefix + '/'
oaislib.fn_output_dir_gen(output_dir)


# ################################################################################
# Cuda 확인 및 GPU 할당
# ################################################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# ################################################################################
# transform 커스터마이징
# ################################################################################
# 기본적으로 돌아가는 trochvision.transforms.Compose를 커스터마이징

# ################################################################################
# transform 정의
# ################################################################################
#
transform1 = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root=output_dir, train=True,
                                        download=True, transform=transform1)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                          shuffle = True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root=output_dir, train=False,
                                       download=True, transform=transform1)

testloader = torch.utils.data.DataLoader(testset, batch_size=512,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ################################################################################
# trainset을 불러와 transpose 실행
# ################################################################################
#
sample = trainset[1][0].numpy()
sample = np.transpose(sample, (1, 2, 0))
plt.imshow(sample)
type(sample)


#################################################################################
# Finish
# ################################################################################
print("time :", time.time() - start)
print('')
