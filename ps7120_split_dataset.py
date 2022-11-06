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
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


start = time.time()

# ################################################################################
# Setting
# ################################################################################
prefix = 'ps7120'
workname = '데이터셋을 학습과 테스트 이미지로 분리'
print(prefix + '_' + workname)

# output setting
output_dir = 'psdata/' + prefix + '/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ################################################################################
# IO
# ################################################################################
img_filepath = 'psdata/ps7010_tunnel/img'
mask_filepath = 'psdata/ps7010_tunnel/mask'

img_list=os.listdir(img_filepath)
mask_list=os.listdir(mask_filepath)

testlist_df = pd.read_excel('psdata/ps7120/testdata.xlsx',
                            sheet_name = 'test',
                            header=None,
                            names= ['nm']
)

# ################################################################################
# Process
# ################################################################################
img_cnt = len(img_list)


# ################################################################################
# 테스트 이미지 리스트를 리스트화 해서 in 사용해서 분류함
# ################################################################################
# 이미지 파일 읽기

for i in range(img_cnt):
    img_nm = img_list[i]
    mask_nm = mask_list[i]
    if img_nm in testlist_df['nm'].values:
        print(img_nm, 'test')
        shutil.copy(img_filepath + '/' + img_nm, output_dir + 'test/img/' + img_nm)
        shutil.copy(mask_filepath + '/' + mask_nm, output_dir + 'test/mask/' + mask_nm)
    else:
        print(img_nm, 'train')
        shutil.copy(img_filepath + '/' + img_nm, output_dir + 'train/img/' + img_nm)
        shutil.copy(mask_filepath + '/' + mask_nm, output_dir + 'train/mask/' + mask_nm)

# ################################################################################
# Finish
# ################################################################################
print("time :", time.time() - start)
print('')
