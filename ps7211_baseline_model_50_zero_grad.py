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
prefix = 'ps7211'
workname = '50_line crack baseline model 모델 수정 zero_grad'
print(prefix + '_' + workname)

# output setting
output_dir = 'psdata/' + prefix + '/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ################################################################################
# Parameter
# ################################################################################
learning_rate=1e-5
width=height=512 # image width and height
batch_size=5

in_channels = 256
out_channels = 2

# ################################################################################
# IO
# ################################################################################
train_img_filepath = 'psdata/ps7120/train/img'
train_mask_filepath = 'psdata/ps7120/train/mask'

test_img_filepath = 'psdata/ps7120/test/img'
test_mask_filepath = 'psdata/ps7120/test/mask'

train_img_list=os.listdir(train_img_filepath)
train_mask_list=os.listdir(train_mask_filepath)

test_img_list = os.listdir(test_img_filepath)
test_mask_list = os.listdir(test_mask_filepath)

# ################################################################################
# transform
# ################################################################################
img_transform = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # 전학습된 모델을 사용해서인걸로
    ])

mask_transform = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.ToTensor()
    ])

# ################################################################################
# read image and transform to tensor using fransform
# ################################################################################
def read_images(img_list, mask_list,height, width):
    img_cnt = len(img_list)
    imgset = torch.zeros(img_cnt, 3, height, width)
    maskset = torch.zeros(img_cnt, height, width)

    for i in range(len(img_list)):
        img = Image.open(os.path.join(train_img_filepath, img_list[i]))
        mask = Image.open(os.path.join(train_mask_filepath, mask_list[i]))
        img_t = img_transform(img)
        mask_t = mask_transform(mask)
        imgset[i] = img_t
        maskset[i] = mask_t
    return imgset, maskset

imgset, maskset = read_images(train_img_list, train_mask_list, height, width)

# ################################################################################
# batch form imgset and maskset
# ################################################################################
def load_batch(batch_size, imgset, maskset):
    cnt = len(imgset)
    idxs = np.random.choice(cnt, batch_size)
    return imgset[idxs], maskset[idxs]

# ################################################################################
# Load and set Net and Optimizer
# ################################################################################
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')

Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
Net.classifier[4] = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1))
## out_channels의 의미는 필터임

Net = Net.to(device)

#Create adma optimizer
optimizer = torch.optim.Adam(params=Net.parameters(), lr=learning_rate)

# ################################################################################
# Model Acc 계산
# ################################################################################
def cal_acc(segs, anns,batch_size):

    TP_all = 0
    TN_all = 0
    FP_all = 0
    FN_all = 0

    for i in range(batch_size):

        seg = segs[i]
        ann = anns[i]

        TP_px = torch.sum((seg == ann)*(ann == 1)).item()
        TN_px = torch.sum((seg == ann)*(ann == 0)).item()
        FP_px = torch.sum((seg != ann)*(ann == 0)).item()
        FN_px = torch.sum((seg != ann)*(ann == 1)).item()

        TP_all += TP_px
        TN_all += TN_px
        FP_all += FP_px
        FN_all += FN_px

    # 전체 픽셀에서 맞춘 픽셀 수
    acc = round((TP_all + TN_all) / (TP_all + TN_all + FP_all + FN_all),5)

    # 검출된 것의 정확도
    precision = round(TP_all / (TP_all + FP_all + 1),5)

    # 전체 크랙에서 몇퍼센트를 맞추었는지
    recall = round(TP_all / (TP_all + FN_all + 1),5)

    return acc, precision, recall


# ################################################################################
# Train
# ################################################################################
# 모델 정확도 저장할 df 작성
model_df = pd.DataFrame(columns=['itr','loss_val','acc','precision','recall'])

for itr in range(1000):
    imgs, anns = load_batch(batch_size, imgset, maskset)

    imgs = torch.autograd.Variable(imgs, requires_grad=False).to(device)
    # torch.Size([5, 512, 512]) (배치 5기준)
    anns = torch.autograd.Variable(anns, requires_grad=False).to(device)
    Pred = Net(imgs)['out']
    # torch.Size([5, 2, 512, 512]) 배치 5에 클래스 2

    #Net.zero_grad()
    criterion = torch.nn.CrossEntropyLoss()
    Loss = criterion(Pred, anns.long())
    Loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(itr,") Loss=",Loss.data.cpu().numpy())
    # 모델 정확도 확인 및 저장
    if itr % 100 == 0:
        # 예측 결과
        segs = torch.argmax(Pred, 1).cpu().detach().numpy()

        # 정확도 계산
        acc, precision, recall = cal_acc(segs, anns, batch_size)

        loss_val = round(Loss.data.numpy().item(),5)

        print('itr: ', itr, ' Loss: ', Loss.item(),
            ' Model Acc: ', acc, ' Precision: ', precision, ' Recall: ', recall)

        # 모델 정확도 저장
        model_df = model_df.append({'itr': itr, 'loss_val': loss_val, 'acc': acc,
                                    'precision': precision, 'recall': recall},
                                   ignore_index=True)
        model_df.to_excel(output_dir + '00_model_acc.xlsx', index=False)

        # 모델 저장
        torch.save(Net.state_dict(), output_dir + 'crack_model_' + str(itr) + '.pth')

# ################################################################################
# Finish
# ################################################################################
print("time :", time.time() - start)
print('')
