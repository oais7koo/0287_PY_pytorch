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
import torchvision.transforms as tf
import matplotlib.pyplot as plt
import matplotlib.image as image
import numpy as np
from PIL import Image

import cv2
import torchvision.models.segmentation
import pr0287

start = time.time()

# ################################################################################
# Setting
# ################################################################################
prefix = 'ps3126'
workname = '50줄 2분류 물병 sseg 학습 표준화 제외'
print(prefix + '_' + workname)

output_dir = 'psdata/' + prefix + '/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ################################################################################
# Parameter
# ################################################################################
Learning_Rate = 1e-5
width = height = 800  # image width and height
batchSize = 3

# ################################################################################
# IO
# ################################################################################
TrainFolder = "psdata/ps1020_LabPicsV1/Simple/Train/"
ListImages = os.listdir(os.path.join(
    TrainFolder, "Image"))  # Create list of images

# ################################################################################
# Transform image
# ################################################################################
transformImg = tf.Compose([
    tf.ToPILImage(),
    tf.Resize((height, width)),
    tf.ToTensor(),
    #tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transformAnn = tf.Compose([tf.ToPILImage(),
                           tf.Resize((height, width),
                                     tf.InterpolationMode.NEAREST),
                           tf.ToTensor()
                           ])

# ################################################################################
# Read image
# ################################################################################


def ReadRandomImage():  # First lets load random image and  the corresponding annotation
    idx = np.random.randint(0, len(ListImages))  # Select random image
    Img = cv2.imread(os.path.join(TrainFolder, "Image", ListImages[idx]))[
        :, :, 0:3]
    # Filled = cv2.imread(os.path.join(TrainFolder,
    #                                 "Semantic/16_Filled",
    #                                 ListImages[idx].replace("jpg", "png")), 0)
    Vessel = cv2.imread(os.path.join(TrainFolder,
                                     "Semantic/1_Vessel",
                                     ListImages[idx].replace("jpg", "png")), 0)

    AnnMap = np.zeros(Img.shape[0:2], np.float32)
    if Vessel is not None:
        AnnMap[Vessel == 1] = 1
    # if Filled is not None:
    #    AnnMap[Filled == 1] = 2
    Img = transformImg(Img)
    AnnMap = transformAnn(AnnMap)
    return Img, AnnMap

# ################################################################################
# Load batch of images
# ################################################################################
def LoadBatch():  # Load batch of images
    images = torch.zeros([batchSize, 3, height, width])
    ann = torch.zeros([batchSize, height, width])
    for i in range(batchSize):
        images[i], ann[i] = ReadRandomImage()
    return images, ann

# ################################################################################
# Load and set net and optimizer
# ################################################################################
#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
print(device)

Net = torchvision.models.segmentation.deeplabv3_resnet50(
    pretrained=True)  # Load net
Net.classifier[4] = torch.nn.Conv2d(256, 2,
                                    kernel_size=(1, 1),
                                    stride=(1, 1))  # Change final layer to 3 classes
Net = Net.to(device)

# Create adam optimizer
optimizer = torch.optim.Adam(params=Net.parameters(), lr=Learning_Rate)

# ################################################################################
# Train
# ################################################################################

for itr in range(10000):  # Training loop
    images, ann = LoadBatch()  # Load taining batch
    # images 사이즈 : torch.Size([3, 3, 900, 900])
    # 배치로 3을 했기 때문에 3개의 이미지가 들어가 있음

    images = torch.autograd.Variable(images, requires_grad=False).to(device)
    # Load image

    ann = torch.autograd.Variable(ann, requires_grad=False).to(device)
    # Load annotation

    Pred = Net(images)['out']  # make prediction
    # Pred 사이즈 : torch.Size([3, 3, 900, 900])
    # 배치로 3을 했기 때문에 3개의 이미지가 들어가 있음

    Net.zero_grad()
    criterion = torch.nn.CrossEntropyLoss()  # Set loss function
    Loss = criterion(Pred, ann.long())  # Calculate cross entropy loss
    Loss.backward()  # Backpropogate loss
    optimizer.step()  # Apply gradient descent change to weight

    print(itr, ") Loss=", Loss.data.cpu().numpy())

    # 모델 정확도 확인 및 저장
    if itr % 10 == 0:
        # 예측 결과
        segs = torch.argmax(Pred, 1).cpu().detach()

        # 정확도 계산
        acc, precision, recall, TPr, TNr, FPr, FNr = pr0287.seg_acc(
            segs, ann, batchSize)
        loss_val = round(Loss.data.numpy().item(), 5)

        miou = pr0287.seg_miou(batchSize, ann, segs, 2)

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
                                                 'TPr', 'TNr', 'FPr', 'FNr', 'miou'])
        model_df = pd.DataFrame(index=range(1), columns=['itr', 'loss_val', 'acc', 'precision', 'recall',
                                                         'TPr', 'TNr', 'FPr', 'FNr', 'miou'])
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
        torch.save(Net.state_dict(), output_dir + 'model_' + str(itr) + '.pth')


# ################################################################################
# Finish
# ################################################################################
print("time :", time.time() - start)
print('')
