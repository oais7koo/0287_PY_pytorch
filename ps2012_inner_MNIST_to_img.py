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
import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


start = time.time()

# ################################################################################
# Setting
# ################################################################################
prefix = 'ps2012'
workname = '내장 MNIST to img'
print(prefix + '_' + workname)

# output setting
output_dir = 'psdata/' + prefix + '/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ################################################################################
# IO
# ################################################################################
train_set = datasets.MNIST(output_dir, train=True, download=True)
test_set = datasets.MNIST(output_dir, train=False, download=True)

trainset_np = train_set.data.numpy()
testset_np = test_set.data.numpy()

trainset_label_np = train_set.targets.numpy()
testset_label_np = test_set.targets.numpy()


# ################################################################################
# make img
# ################################################################################
train_cnt = trainset_np.shape[0]
test_cnt = testset_np.shape[0]

# label
train_label_df = pd.DataFrame(index=range(train_cnt), columns=['file','label']) 
test_label_df = pd.DataFrame(index=range(test_cnt), columns=['file','label']) 

train_img_filepath = output_dir + 'train_img'
if not os.path.exists(train_img_filepath):
    os.makedirs(train_img_filepath)

test_img_filepath = output_dir + 'test_img'
if not os.path.exists(test_img_filepath):
    os.makedirs(test_img_filepath)

for i in range(train_cnt):
    if not (i % 100): print(i)
    idx = 10000 + i
    img = Image.fromarray(trainset_np[i])
    # img.show()
    img_filepath = train_img_filepath + '/train' + str(idx) + '.png'
    img_label = trainset_label_np[i]

    train_label_df.loc[i, 'file'] = os.path.basename(img_filepath)
    train_label_df.loc[i, 'label'] = img_label
    
    img.save(img_filepath)

for i in range(test_cnt):
    if not (i % 100): print(i)
    idx = 10000 + i
    img = Image.fromarray(testset_np[i])
    #img.show()
    img_filepath = test_img_filepath + '/test' + str(idx) + '.png'
    img_label = testset_label_np[i]

    test_label_df.loc[i, 'file'] = os.path.basename(img_filepath)
    test_label_df.loc[i, 'label'] = img_label

    img.save(img_filepath)

train_label_df.to_excel(output_dir + 'train_label.xlsx', index=False)
test_label_df.to_excel(output_dir + 'test_label.xlsx', index=False)
    
# ################################################################################
# Finish
# ################################################################################
print("time :", time.time() - start)
print('')
