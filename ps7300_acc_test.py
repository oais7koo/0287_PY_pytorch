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
import glob
from os.path import basename
import pickle
import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pr0287
import cv2

start = time.time()

# ################################################################################
# Setting
# ################################################################################
prefix = 'ps7300'
workname = '50_line crack baseline 에 크랙만 + 웨이트 + dilation 사용 모델 검증'
print(prefix + '_' + workname)

# output setting
output_dir = 'psdata/' + prefix + '/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ################################################################################
# Parameter
# ################################################################################
width = height = 512  # image width and height

# ################################################################################
# IO
# ################################################################################

modelPath = "psdata/ps7215/crack_model_1200.pth"  # Path to trained model

img_filepath = 'psdata/ps7130/test/img/*.png'
mask_dir = 'psdata/ps7140/test/mask/'

img_list = glob.glob(img_filepath)
mask_list = pr0287.mask_list_load(img_list, mask_dir)

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')

# ################################################################################
# transform
# ################################################################################
img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # 전학습된 모델을 사용해서인걸로
])

mask_transform = transforms.Compose([
    transforms.ToPILImage(),  # 원본에서는 매스크 파일이 배열이라서 넣어 둔 것임
    transforms.ToTensor()
])

# ################################################################################
# read image and transform to tensor using fransform
# ################################################################################

def read_images(img_list, mask_list, height, width):
    img_cnt = len(img_list)
    imgset = torch.zeros(img_cnt, 3, height, width)
    maskset = torch.zeros(img_cnt, height, width)

    for i in range(len(img_list)):
        img = cv2.imread(os.path.join(img_list[i]))
        mask = cv2.imread(os.path.join(mask_list[i]))[:, :, 0]
        # 채널 1개

        img_t = img_transform(img)
        mask_t = mask_transform(mask)
        # print(torch.unique(mask_t))
        imgset[i] = img_t
        maskset[i] = mask_t

    maskset = torch.where(maskset == 0, maskset, torch.tensor(1))
    print(torch.unique(maskset))
    return imgset, maskset


imgset, maskset = read_images(img_list, mask_list, height, width)

# ################################################################################
# Load Model
# ################################################################################
Net = torchvision.models.segmentation.deeplabv3_resnet50(
    pretrained=True)  # Load net
Net.classifier[4] = torch.nn.Conv2d(
    256, 2, kernel_size=(3, 3), stride=(1, 1), padding=1)
# Change final layer to 2 classes
Net = Net.to(device)
# Set net to GPU or CPU

Net.load_state_dict(torch.load(modelPath))  # Load trained model

Net.eval()  # Set to evaluation mode

# ################################################################################
# Eval
# ################################################################################
with torch.no_grad():
    Pred = Net(imgset)['out']  # Run net

segs = torch.argmax(Pred, 1).cpu().detach()
segs = segs.type(torch.IntTensor)
anns = maskset.type(torch.IntTensor)

# ################################################################################
# 정확도 검증
# ################################################################################
img_cnt = len(img_list)
acc, precision, recall, TPr, TNr, FPr, FNr = pr0287.seg_acc(
    segs, anns, img_cnt)
miou = pr0287.seg_miou(img_cnt, anns, segs, 2)

print(' model name: ', modelPath,
      ' Model Acc: ', acc,
      ' Precision: ', precision,
      ' Recall: ', recall,
      ' TPr: ', TPr,
      ' TNr: ', TNr,
      ' FPr: ', FPr,
      ' FNr: ', FNr,
      ' mIoU: ', miou)

# 결과파일이 없다면
if os.path.isfile(output_dir + '00_model_acc.xlsx') == True:
    model_all_df = pd.read_excel(output_dir + '00_model_acc.xlsx')
else:
    model_all_df = pd.DataFrame(columns=['model', 'acc', 'precision', 'recall',
                                         'TPr', 'TNr', 'FPr', 'FNr', 'miou'])

model_df = pd.DataFrame(index=range(1), columns=['model', 'itr', 'acc', 'precision', 'recall',
                                                 'TPr', 'TNr', 'FPr', 'FNr', 'miou'])
model_df.loc[0, 'model'] = modelPath
model_df.loc[0, 'acc'] = acc
model_df.loc[0, 'precision'] = precision
model_df.loc[0, 'recall'] = recall
model_df.loc[0, 'TPr'] = TPr
model_df.loc[0, 'TNr'] = TNr
model_df.loc[0, 'FPr'] = FPr
model_df.loc[0, 'FNr'] = FNr
model_df.loc[0, 'miou'] = miou

model_all_df = pd.concat([model_all_df, model_df], axis=0)
model_all_df.to_excel(output_dir + '00_model_acc.xlsx', index=False)

# ################################################################################
# 개별 이미지 예측 결과 저장
# ################################################################################
for i in range(img_cnt):
    img = cv2.imread(img_list[i])
    seg = segs[i].numpy()
    ann = anns[i].numpy()
    ann = np.where(ann == 0, 255, 0)

    imgR = img[:, :, 0]
    imgG = img[:, :, 1]
    imgB = img[:, :, 2]

    imgR_s = np.where(seg == 1, 0, imgR)
    imgG_s = np.where(seg == 1, 0, imgG)
    imgB_s = np.where(seg == 1, 0, imgB)

    imgR_m = np.concatenate((imgR, ann, imgR_s), axis=1)
    imgG_m = np.concatenate((imgG, ann, imgG_s), axis=1)
    imgB_m = np.concatenate((imgB, ann, imgB_s), axis=1)

    img_m = np.dstack((imgR_m, imgG_m, imgB_m))

    cv2.imwrite(output_dir + 'seg_' + basename(img_list[i]), img_m)

# ################################################################################
# Finish
# ################################################################################
print("time :", time.time() - start)
print('')
