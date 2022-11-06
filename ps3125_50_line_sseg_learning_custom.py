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

start = time.time()

# ################################################################################
# Setting
# ################################################################################
prefix = 'ps3125'
workname = '50줄로 sseg 완성'
print(prefix + '_' + workname)

output_dir = 'psdata/' + prefix + '/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ################################################################################
# Parameter
# ################################################################################
Learning_Rate = 1e-5
width = height = 900  # image width and height
batchSize = 6

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
    tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
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
    Filled = cv2.imread(os.path.join(TrainFolder,
                                     "Semantic/16_Filled",
                                     ListImages[idx].replace("jpg", "png")), 0)
    Vessel = cv2.imread(os.path.join(TrainFolder,
                                     "Semantic/1_Vessel",
                                     ListImages[idx].replace("jpg", "png")), 0)

    AnnMap = np.zeros(Img.shape[0:2], np.float32)
    if Vessel is not None:
        AnnMap[Vessel == 1] = 1
    if Filled is not None:
        AnnMap[Filled == 1] = 2
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
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')

Net = torchvision.models.segmentation.deeplabv3_resnet50(
    pretrained=True)  # Load net
Net.classifier[4] = torch.nn.Conv2d(256, 3,
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

    images = torch.autograd.Variable(
        images, requires_grad=False).to(device)  # Load image
    ann = torch.autograd.Variable(
        ann, requires_grad=False).to(device)  # Load annotation
    Pred = Net(images)['out']  # make prediction
    # Pred 사이즈 : torch.Size([3, 3, 900, 900])
    # 배치로 3을 했기 때문에 3개의 이미지가 들어가 있음

    Net.zero_grad()
    criterion = torch.nn.CrossEntropyLoss()  # Set loss function
    Loss = criterion(Pred, ann.long())  # Calculate cross entropy loss
    Loss.backward()  # Backpropogate loss
    optimizer.step()  # Apply gradient descent change to weight

    # 예측 결과 얻기
    seg = torch.argmax(Pred[0], 0).cpu().detach().numpy()

    # accuracy 계산
    acc = (torch.tensor(seg) == ann[0]).sum().float() / (height * width)
    # 배치로 예측된 결과에 대해서 예측결과 Pred에서 첫번째 것을 가져옴

    print(itr, ") Loss=", Loss.data.cpu().numpy(), 'Acc : ', round(acc.item(),3))
    if itr % 100 == 0:  # Save model weight once every 60k steps permenant file
        print("Saving Model" + str(itr) + ".torch")
        torch.save(Net.state_dict(),   str(itr) + ".torch")

# ################################################################################
# Finish
# ################################################################################
print("time :", time.time() - start)
print('')
