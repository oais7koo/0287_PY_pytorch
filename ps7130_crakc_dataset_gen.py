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

if os.path.exists('../python/oaislib_org.py'):
    shutil.copy('../python/oaislib_org.py', 'oaislib.py')
import oaislib
start = time.time()

# ################################################################################
# Parameter
# ################################################################################
prefix = 'ps7130'
workname = '균열이 있는 영상만으로 데이터셋 구축'
print(prefix + '_' + workname)

# ################################################################################
# IO
# ################################################################################

img_dir = 'psdata/ps7010_tunnel/img/'
mask_dir = 'psdata/ps7010_tunnel/mask/'

input_filepath = 'psdata/ps7130/testdata.xlsx'

# input setting
train_df = pd.read_excel(input_filepath,
                         sheet_name = 'train',
                         # header = 0, #0, None 워크시트의 특정행을 열이름으로 지정, 데이터는 이후 행부터 읽어옮,
                         # names 옵션없이 header=None를 지정하면 열명 자동
                         # usecols = ['그룹','시간기록여부'], # none는 모든열, (A:F), (A,E:F), ['사람', '동물'], [0,1]
                         #names = ['일시','평균','a','b','c'], #엑셀시트에 열이름이 없어서 지정하는 경우 header=None,
                                                # 열이름을 변경하고 싶은 경우 header=0,
                                                #### usecols와 동시에 사용안됨
                         # index_col = None, # 특정열을 index로 사용, 지정하지 않으면 index 자동 생성
                         #dtype = {'일시':str, '평균':float, '최저':float, '최고':float},
                         #skiprows = 32,
                         # nrows = 28, # 읽어들이고 싶은 행 수 지정
                         #na_values = '', # nan으로 처리하고 싶은 값 지정, 공백은 기본적으로 Nan으로 처리
                         # thousands = ',' # 천단위 기호 제외하고 싶을때
                         comment = '#' # #로 시작하면 그 행 뒤는 전부 무시
                         )

# input setting
test_df = pd.read_excel(input_filepath,
                        sheet_name = 'test',
                        header = None, #0, None 워크시트의 특정행을 열이름으로 지정, 데이터는 이후 행부터 읽어옮,
                         # names 옵션없이 header=None를 지정하면 열명 자동
                         # usecols = ['그룹','시간기록여부'], # none는 모든열, (A:F), (A,E:F), ['사람', '동물'], [0,1]
                         names = ['file'], #엑셀시트에 열이름이 없어서 지정하는 경우 header=None,
                                                # 열이름을 변경하고 싶은 경우 header=0,
                                                #### usecols와 동시에 사용안됨
                         # index_col = None, # 특정열을 index로 사용, 지정하지 않으면 index 자동 생성
                         #dtype = {'일시':str, '평균':float, '최저':float, '최고':float},
                         #skiprows = 32,
                         # nrows = 28, # 읽어들이고 싶은 행 수 지정
                         #na_values = '', # nan으로 처리하고 싶은 값 지정, 공백은 기본적으로 Nan으로 처리
                         # thousands = ',' # 천단위 기호 제외하고 싶을때
                         comment = '#' # #로 시작하면 그 행 뒤는 전부 무시
                         )

#_df.replace(np.nan, 0, regex=True, inplace=True)

img_filelist = glob.glob(img_dir + '*.png')
mask_filelist = glob.glob(mask_dir + '*.png')

img_cnt = len(img_filelist)

# output setting
output_dir = 'psdata/' + prefix + '/'
oaislib.fn_output_dir_gen(output_dir)

# ################################################################################
# Process
# ################################################################################
# train_df에서 crack_px > 0 인 리스트 추출
train_df = train_df[train_df['crack_px'] > 0]


# train_df의 train_file를 리스트로 추출
train_list = train_df['train_file'].tolist()

test_list = test_df['file'].tolist()

for i in range(img_cnt):
    img_nm = img_filelist[i].split('\\')[-1]
    mask_filepath = mask_dir + img_nm
    if img_nm in train_list:
        if img_nm in test_list:
            shutil.copy(img_filelist[i], output_dir + 'test/img/' + img_nm)
            shutil.copy(mask_filepath, output_dir + 'test/mask/' + img_nm)
        else:
            shutil.copy(img_filelist[i], output_dir + 'train/img/' + img_nm)
            shutil.copy(mask_filepath, output_dir + 'train/mask/' + img_nm)

# ################################################################################
# Save Result
# ################################################################################
#output_df.to_excel(output_dir + 'output.xlsx', index=False)
#output_df.to_pickle(output_dir + 'output.pkl')

# ################################################################################
# Finish
# ################################################################################
print("time :", time.time() - start)
print('')
