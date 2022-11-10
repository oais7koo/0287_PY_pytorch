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
import glob
from os.path import basename
import pickle
import tqdm
import cv2

from PIL import Image
if os.path.exists('../python/oaislib_org.py'):
    shutil.copy('../python/oaislib_org.py', 'oaislib.py')
import oaislib
start = time.time()

# ################################################################################
# Parameter
# ################################################################################
prefix = 'ps7140'
workname = '매스크 두께 확인'
print(prefix + '_' + workname)

# ################################################################################
# IO
# ################################################################################

input_filepath = 'psdata/ps7011_mask_thick_check/0054_5_13.png'
#_filepath = ''

img = Image.open(input_filepath)
arr = np.array(img)

# output setting
output_dir = 'psdata/' + prefix + '/'
oaislib.fn_output_dir_gen(output_dir)

# ################################################################################
# Process
# ################################################################################
# ndarray인 arr을 pandas 의 df로 변환
arr_df = pd.DataFrame(arr)

# ################################################################################
# Save Result
# ################################################################################
arr_df.to_excel(output_dir + 'output.xlsx', index=False)

# ################################################################################
# dilation 적용
# ################################################################################
# 테스트 이미지에 대한 diation 적용
# 전체 mask에 대한 dilation 적용
# 균열 이미지에 대한 diation 적용

# ################################################################################
# 테스트 이미지에 대한 dilation 적용
# ################################################################################
input_filepath = 'psdata/ps7011_mask_thick_check/0054_5_13.png'
img = cv2.imread(input_filepath)

kernel = np.ones((3, 3), np.uint8)

img_dilation = cv2.dilate(img, kernel, iterations=1)
arr_dilation_df = pd.DataFrame(img_dilation[:, :, 0])
arr_dilation_df.to_excel(output_dir + 'output_dilation.xlsx', index=False)

# ################################################################################
# 전체 mask에 대한 dilation 적용
# ################################################################################
mask_dir = 'psdata/ps7010_tunnel/mask/*.png'
mask_list = glob.glob(mask_dir)
mask_cnt = len(mask_list)

for i in range(mask_cnt):
    mask = cv2.imread(mask_list[i])
    mask_d = cv2.dilate(mask, kernel, iterations=1)
    cv2.imwrite(output_dir + 'mask_d_all/' + basename(mask_list[i]), mask_d)

# 검증은 여기서는 안되고, 균열 이미지에 대한 dilation 적용에서 확인해야 함

# ################################################################################
# 균열이미지에 대한 dilation 적용
# ################################################################################
# 테스트 균열 이미지에 대한 dilation 적용

mask_te_dir = 'psdata/ps7130/test/mask/*.png'
mask_tr_dir = 'psdata/ps7130/train/mask/*.png'

mask_te_list = glob.glob(mask_te_dir)
mask_tr_list = glob.glob(mask_tr_dir)

mask_te_cnt = len(mask_te_list)
mask_tr_cnt = len(mask_tr_list)

for i in range(mask_te_cnt):
    mask = cv2.imread(mask_te_list[i])
    mask_d = cv2.dilate(mask, kernel, iterations=1)
    cv2.imwrite(output_dir + 'test/mask/' + basename(mask_te_list[i]), mask_d)

for i in range(mask_tr_cnt):
    mask = cv2.imread(mask_tr_list[i])
    mask_d = cv2.dilate(mask, kernel, iterations=1)
    cv2.imwrite(output_dir + 'train/mask/' + basename(mask_tr_list[i]), mask_d)


# ################################################################################
# 균열이미지에 대한 dilation 적용 검증
# ################################################################################
input_filepath = 'psdata/ps7140/train/mask/0053_15_10.png'
img = cv2.imread(input_filepath)

test_df = pd.DataFrame(img[:, :, 0])
test_df.to_excel(output_dir + 'dilation_test.xlsx', index=False)


# ################################################################################
# Finish
# ################################################################################
print("time :", time.time() - start)
print('')
