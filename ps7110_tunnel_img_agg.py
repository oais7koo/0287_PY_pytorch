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
import tqdm

import torch
import torchvision
import torchvision.transforms as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pr0287

start = time.time()

# ################################################################################
# Setting
# ################################################################################
prefix = 'ps7110'
workname = '이노팸 입력데이터 전처리 및 집계 '
print(prefix + '_' + workname)

# output setting
output_dir = 'psdata/' + prefix + '/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# output 파일 생성

if not os.path.exists(output_dir + 'train'):
    os.makedirs(output_dir + 'train')

if not os.path.exists(output_dir + 'mask'):
    os.makedirs(output_dir + 'mask')

# ################################################################################
# IO
# ################################################################################
train_filepath = 'psdata/ps7010_tunnel/img'
mask_filepath = 'psdata/ps7010_tunnel/mask'

train_filelist = glob(train_filepath + '/*.png')
mask_filelist = glob(mask_filepath + '/*.png')

# ################################################################################
# Process
# ################################################################################
# 이미지 별로 매스크 파일이 다 있는지 확인후 데이터 수 집계
# 매스크 파일 별로 라벨링 수 확인
# ? 이미지와 매스크가 매칭된 데이터셋
#   ○ img
#   ○ mask
# ? 각 이미지 별 집계
#   ○ 전체 픽셀
#   ○ 크랙 라벨링 픽셀
#   ○ 배경 픽셀
#     기타

# 집계 파일 생성

train_cnt = len(train_filelist)
mask_cnt = len(mask_filelist)

agg_df = pd.DataFrame(index=range(train_cnt),
                      columns = ['train_file','mask_yn','t_height','t_width',
                                 'm_height','m_width','t_all_px', 'back_px',
                                 'crack_px','crack_rate'])
# train 파일명 리스트 생성
train_filenames = []
for filepath in train_filelist:
    filename = os.path.basename(filepath)
    train_filenames.append(filename)

# mask 파일명 리스트 생성
mask_filenames = []
for filepath in mask_filelist:
    filename = os.path.basename(filepath)
    mask_filenames.append(filename)

# db 집계
for i in range(train_cnt):
    # mask 파일이 있는지 확인
    t_nm = train_filenames[i]
    t_path = train_filepath + '/' + t_nm
    t_img = Image.open(t_path)

    agg_df.loc[i, 'train_file'] = t_nm
    agg_df.loc[i, 't_height'] = t_img.height
    agg_df.loc[i, 't_width'] = t_img.width
    agg_df.loc[i, 't_all_px'] = t_img.height * t_img.width

    if t_nm in mask_filenames:
        agg_df.loc[i, 'mask_yn'] = 1
        m_path  = mask_filepath+  '/' + t_nm
        m_img = Image.open(m_path)
        m_np = np.array(m_img)
        # print(list(np.unique(m_np)))
        agg_df.loc[i, 'm_height'] = m_img.height
        agg_df.loc[i, 'm_width'] = m_img.width
        agg_df.loc[i, 'back_px'] = pr0287.count_val_in_2darray(m_np, 0)
        agg_df.loc[i, 'crack_px'] = pr0287.count_val_in_2darray(m_np, 1)
        agg_df.loc[i, 'crack_rate'] = round(agg_df.loc[i, 'crack_px'] / \
                                      (agg_df.loc[i, 'back_px'] + agg_df.loc[i, 'crack_px']), 4)

    else:
        agg_df.loc[i, 'mask_yn'] = 0

# ################################################################################
# Save Result
# ################################################################################
agg_df.to_excel(output_dir + '균열데이터 집계.xlsx', index=False)

# ################################################################################
# Finish
# ################################################################################
print("time :", time.time() - start)
print('')
