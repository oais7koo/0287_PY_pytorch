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

import numpy as np
import cv2
import torchvision.models.segmentation

start = time.time()

# ################################################################################
# Setting
# ################################################################################
prefix = 'ps3120'
workname = '50 line ssemg learning'
print(prefix + '_' + workname)

output_dir = 'psdata/' + prefix + '/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ################################################################################
# Parameter
# ################################################################################
Learning_Rate=1e-5
width=height=900 # image width and height
batchSize=2

# ################################################################################
# IO
# ################################################################################
TrainFolder="psdata/ps1020_LabPicsV1/Simple/Train/"
ListImages=os.listdir(os.path.join(TrainFolder, "Image")) # Create list of images

# ################################################################################
# Transform image
# ################################################################################
transformImg=tf.Compose([
    tf.ToPILImage(),
    tf.Resize((height,width)),
    tf.ToTensor(),
    tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # 이 숫자에 대해서 확인해봐야 함.
    # pre-trained 모델이 저 숫자로 표준화 되어 있기 때문에 저 숫자를 써야 하는 걸로
    # 일단 이해
])

transformAnn=tf.Compose([tf.ToPILImage(),
                         tf.Resize((height,width), tf.InterpolationMode.NEAREST),
                         tf.ToTensor()
])

# ################################################################################
# Read image
# ################################################################################
def ReadRandomImage(): # First lets load random image and the corresponding annotation
    idx=np.random.randint(0,len(ListImages)) # Select random image

    Img=cv2.imread(os.path.join(TrainFolder, "Image", ListImages[idx]))[:,:,0:3]
    # ListImages 는 TrainFolder/Image 폴더에 있는 파일명들의 리스트이며 1691개의 파일명이 있다.

    # 아래는 어노테이션을 읽어오는 부분인데, 객체 별로 다른 폴더에 정리되어 있음
    # 학습이미지는 jpg파일이고 어노테이션은 png파일이다.
    # 16_Filled 는 물병에 뭔가 들어 있는 부분을 의미
    Filled =  cv2.imread(os.path.join(TrainFolder,
                                      "Semantic/16_Filled",
                                      ListImages[idx].replace("jpg","png")),0)
    # os.path.join는 경로 합침

    # 1_Vessel은 병을 의미
    Vessel =  cv2.imread(os.path.join(TrainFolder,
                                      "Semantic/1_Vessel",
                                      ListImages[idx].replace("jpg","png")),0)

    AnnMap = np.zeros(Img.shape[0:2],np.float32)
    if Vessel is not None:  AnnMap[ Vessel == 1 ] = 1
    if Filled is not None:  AnnMap[ Filled  == 1 ] = 2
    Img=transformImg(Img)
    AnnMap=transformAnn(AnnMap)
    return Img, AnnMap

# ################################################################################
# Load batch of images
# ################################################################################
def LoadBatch(): # Load batch of images
    images = torch.zeros([batchSize,3,height,width])
    ann = torch.zeros([batchSize, height, width])
    for i in range(batchSize):
        images[i],ann[i]=ReadRandomImage()
    return images, ann

# ################################################################################
# Load and set net and optimizer
# ################################################################################
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')

Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True) # Load net
Net.classifier[4] = torch.nn.Conv2d(256, 3,
                                    kernel_size=(1, 1),
                                    stride=(1, 1)) # Change final layer to 3 classes
Net=Net.to(device)

# Create adam optimizer
optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate)

# ################################################################################
# Train
# ################################################################################
for itr in range(10000): # Training loop
    images,ann=LoadBatch() # Load taining batch
    images=torch.autograd.Variable(images,requires_grad=False).to(device) # Load image
    ann = torch.autograd.Variable(ann, requires_grad=False).to(device) # Load annotation
    # ann -> annotation 주석.. 즉 라벨링
    # requires_grad=False <- 이게 학습을 하지 말라는 이야기가 아닌가?

    Pred=Net(images)['out'] # make prediction
    Net.zero_grad()
    criterion = torch.nn.CrossEntropyLoss() # Set loss function
    Loss=criterion(Pred,ann.long()) # Calculate cross entropy loss
    # Pred 차원 : [2, 3, 900, 900] (batchSize, class, height, width)
    # ann 차원 : [2, 900, 900] (batchSize, height, width)

    Loss.backward() # Backpropogate loss
    optimizer.step() # Apply gradient descent change to weight
    seg = torch.argmax(Pred[0], 0).cpu().detach()  # Get  prediction classes
    print(itr,") Loss=",Loss.data.cpu().numpy())


    if itr % 100 == 0: #Save model weight once every 60k steps permenant file
        print("Saving Model" +str(itr) + ".torch")
        torch.save(Net.state_dict(),  output_dir + str(itr) + ".torch")



# ################################################################################
# Finish
# ################################################################################
print("time :", time.time() - start)
print('')
