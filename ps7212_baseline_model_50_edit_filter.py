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
import pr0287

start = time.time()

# ################################################################################
# Setting
# ################################################################################
prefix = 'ps7212'
workname = '50_line crack baseline model 필터 수정'
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
batch_size=10

in_channels = 256
out_channels = 2

# ################################################################################
# IO
# ################################################################################
train_img_filepath = 'psdata/ps7130/train/img'
train_mask_filepath = 'psdata/ps7130/train/mask/'

train_img_list=os.listdir(train_img_filepath)
train_mask_list=pr0287.mask_list_load(train_img_list, train_mask_filepath)

# ################################################################################
# transform
# ################################################################################
img_transform = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.ToTensor(),
    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
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
        mask = Image.open(mask_list[i])
        img_t = img_transform(img)
        mask_t = mask_transform(mask)
        imgset[i] = img_t
        maskset[i] = mask_t
    maskset = torch.where(maskset == 0, maskset, torch.tensor(1))
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
Net.classifier[4] = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(3,3),
                                    stride=(1, 1), padding=1)
## out_channels의 의미는 필터임

Net = Net.to(device)

#Create adma optimizer
optimizer = torch.optim.Adam(params=Net.parameters(), lr=learning_rate)

# ################################################################################
# Train
# ################################################################################
# 모델 정확도 저장할 df 작성

for itr in range(100000):
    imgs, anns = load_batch(batch_size, imgset, maskset)

    imgs = torch.autograd.Variable(imgs, requires_grad=False).to(device)
    # torch.Size([5, 512, 512]) (배치 5기준)
    anns = torch.autograd.Variable(anns, requires_grad=False).to(device)
    Pred = Net(imgs)['out']
    # torch.Size([5, 2, 512, 512]) 배치 5에 클래스 2

    Net.zero_grad()
    criterion = torch.nn.CrossEntropyLoss()
    Loss = criterion(Pred, anns.long())
    Loss.backward()
    optimizer.step()

    print(itr,") Loss=", Loss.data.cpu().numpy())

    # 모델 정확도 확인 및 저장
    if itr % 100 == 0:
        # 예측 결과
        segs = torch.argmax(Pred, 1).cpu().detach()

        # 정확도 계산
        segs = segs.type(torch.IntTensor)
        anns = anns.type(torch.IntTensor)
        acc, precision, recall, TPr, TNr, FPr, FNr = pr0287.seg_acc(segs, anns, batch_size)
        loss_val = round(Loss.data.numpy().item(),5)

        miou = pr0287.seg_miou(batch_size, anns, segs, 2)

        print('itr: ', itr,
              ' Loss: ', Loss.item(),
              ' Model Acc: ', acc,
              ' Precision: ', precision,
              ' Recall: ', recall,
              ' TPr: ', TPr,
              ' TNr: ', TNr,
              ' FPr: ', FPr,
              ' FNr: ', FNr,
              ' mIoU: ', miou)

        # 모델 정확도 저장
        if itr == 0:
            model_all_df = pd.DataFrame(columns=['itr', 'loss_val', 'acc', 'precision', 'recall',
                                                 'TPr', 'TNr', 'FPr', 'FNr','miou'])
        model_df = pd.DataFrame(index=range(1), columns=['itr', 'loss_val', 'acc', 'precision', 'recall',
                                                         'TPr', 'TNr', 'FPr', 'FNr','miou'])
        model_df.loc[0, 'itr'] = itr
        model_df.loc[0, 'loss_val'] = loss_val
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

        # 모델 저장
        torch.save(Net.state_dict(), output_dir + 'crack_model_' + str(itr) + '.pth')

# ################################################################################
# Finish
# ################################################################################
print("time :", time.time() - start)
print('')
