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
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pr0287
import cv2
import datetime

start = time.time()

# ################################################################################
# Setting
# ################################################################################
prefix = 'ps7400'
workname = '통합베이스모델'
print(prefix + '_' + workname)

# output setting
output_dir = 'psdata/' + prefix + '/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ################################################################################
# Parameter
# ################################################################################
# 선택사항
pc_nm = 'ai'
dataset_type = 'dilation' # base - 기본데이터(전체), crack - 크랩만, dilation - 픽셀1개 확장
batch_size = 50
k_size = 3  # kernel size 1 or 3
normalize_yn = 'y' # y or n
device_val = 'gpu'
acc_interval = 100

weight_yn = 'y'
weight_val = [0.001, 0.999]

para_msg = 'dataset_type::' + dataset_type + '_k_size::' + str(k_size) + '_normal::' + \
    normalize_yn + '_weight::' + weight_yn + '_' + str(weight_val)

# 고정사항
learning_rate = 1e-5
width = height = 512  # image width and height
in_channels = 256
out_channels = 2

# padding size
if k_size == 1:
    p_size = 0
elif k_size == 3:
    p_size = 1

## 지금 시각을 YYMMDD_HHMMSS 형태로 리턴하기
def timetail():
    now = datetime.datetime.now()
    yy = '{0.year:02}'.format(now)[2:4]
    MM = '{0.month:02}'.format(now)
    dd = '{0.day:02}'.format(now)
    hh = '{0.hour:02}'.format(now)
    mm = '{0.minute:02}'.format(now)
    ss = '{0.second:02}'.format(now)
    timetail = yy + MM + dd + '_' + hh + mm + ss
    return timetail

# ################################################################################
# IO
# ################################################################################
if dataset_type == 'base':
    train_folder = 'ps7120'
    mask_folder = 'ps7120'
    test_folder = 'ps7120'

elif dataset_type == 'crack':
    train_folder = 'ps7130'
    mask_folder = 'ps7130'
    test_folder = 'ps7130'

elif dataset_type == 'dilation':
    train_folder = 'ps7130'
    mask_folder = 'ps7140'
    test_folder = 'ps7130'
else:
    sys.exit('dataset error')

train_img_filepath = 'psdata/' + train_folder +'/train/img/*.png'
train_mask_filepath = 'psdata/' + mask_folder + '/train/mask/'

train_img_list = glob.glob(train_img_filepath)
train_mask_list = pr0287.mask_list_load(train_img_list, train_mask_filepath)

# ################################################################################
# transform
# ################################################################################

if normalize_yn == 'y':
    img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # 전학습된 모델을 사용해서인걸로
    ])

    test_img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            # 전학습된 모델을 사용해서인걸로
    ])

else:
    img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    test_img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

mask_transform = transforms.Compose([
    transforms.ToPILImage(),
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
        img = cv2.imread(img_list[i])
        mask = cv2.imread(mask_list[i])
        imgset[i] = test_img_transform(img)
        maskset[i] = mask_transform(mask[:,:,0])
    maskset = torch.where(maskset == 0, maskset, torch.tensor(1))

    return imgset, maskset

# ################################################################################
# read random image
# ################################################################################
def read_random_image():
    idx=np.random.randint(0, len(train_img_list))

    img = cv2.imread(train_img_list[idx])[:,:,0:3]
    mask = cv2.imread(train_mask_list[idx])[:,:,0]
    img = img_transform(img)
    mask = mask_transform(mask)

    return img, mask

# ################################################################################
# batch form imgset and maskset
# ################################################################################
'''
def load_batch(batch_size, imgset, maskset):
    cnt = len(imgset)
    idxs = np.random.choice(cnt, batch_size)
    # print(idxs)
    return imgset[idxs], maskset[idxs]
'''
def load_batch2():
    imgset_sample = torch.zeros([batch_size,3,height,width])
    maskset_sample = torch.zeros([batch_size, height, width])
    for i in range(batch_size):
        imgset_sample[i], maskset_sample[i] = read_random_image()
    return imgset_sample, maskset_sample

# ################################################################################
# Load and set Net and Optimizer
# ################################################################################
if device_val == 'gpu':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
else:
    device = torch.device('cpu')

#if model_v == deeplabv3_resnet50:
Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)


Net.classifier[4] = torch.nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size=k_size,
                                    stride=(1, 1),
                                    padding=p_size)
# out_channels의 의미는 필터임

Net = Net.to(device)

# Create adma optimizer
optimizer = torch.optim.Adam(params=Net.parameters(), lr=learning_rate)

# ################################################################################
# Train
# ################################################################################
# 모델 정확도 저장할 df 작성
for itr in range(100000):
    print(para_msg)
    imgs, anns = load_batch2()
    imgs = torch.autograd.Variable(imgs, requires_grad=False).to(device)
    # torch.Size([5, 512, 512]) (배치 5기준)
    anns = torch.autograd.Variable(anns, requires_grad=False).to(device)
    Pred = Net(imgs)['out']
    # torch.Size([5, 2, 512, 512]) 배치 5에 클래스 2

    Net.zero_grad()

    if weight_yn == 'y':
        weight_loss = torch.FloatTensor(weight_val).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weight_loss)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    Loss = criterion(Pred, anns.long())
    Loss.backward()
    optimizer.step()

    print(itr, ") Loss=", Loss.data.cpu().numpy())

    # 모델 정확도 확인 및 저장
    if itr % acc_interval == 0:

        time_tail = timetail()
        if not os.path.exists(output_dir + time_tail):
            os.makedirs(output_dir + time_tail)

        # 예측 결과
        segs = torch.argmax(Pred, 1).cpu().detach()

        # 정확도 계산
        segs = segs.type(torch.IntTensor)
        anns = anns.type(torch.IntTensor)
        acc, precision, recall, TPr, TNr, FPr, FNr = pr0287.seg_acc(
            segs, anns, batch_size)
        loss_val = round(Loss.data.numpy().item(), 5)

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
        model_acc_df = pd.DataFrame(index=range(1),
                                    columns=['idx','pc_nm','dataset_type','batch_size',
                                                'kernel_size','normalized_yn','weight',
                                                'device','itr', 'acc', 'precision',
                                                'recall','TPr', 'TNr', 'FPr', 'FNr', 'miou'])

        model_acc_df.loc[0, 'pc_nm'] = pc_nm
        model_acc_df.loc[0, 'idx'] = time_tail
        model_acc_df.loc[0, 'dataset_type'] = dataset_type
        model_acc_df.loc[0, 'batch_size'] = batch_size
        model_acc_df.loc[0, 'kernel_size'] = k_size
        model_acc_df.loc[0, 'normalized_yn'] = normalize_yn
        model_acc_df.loc[0, 'device'] = device
        model_acc_df.loc[0, 'itr'] = itr
        model_acc_df.loc[0, 'acc'] = acc
        model_acc_df.loc[0, 'precision'] = precision
        model_acc_df.loc[0, 'recall'] = recall
        model_acc_df.loc[0, 'TPr'] = TPr
        model_acc_df.loc[0, 'TNr'] = TNr
        model_acc_df.loc[0, 'FPr'] = FPr
        model_acc_df.loc[0, 'FNr'] = FNr
        model_acc_df.loc[0, 'miou'] = miou

        if weight_yn == 'y':
            model_acc_df.loc[0, 'weight'] = str(weight_val)
        else:
            model_acc_df.loc[0, 'weight'] = '-'

        model_acc_df.to_excel(output_dir + time_tail + '/_model_acc.xlsx', index=False)

        # 모델 저장
        model_filepath = output_dir + time_tail +  '/_crack_model.pth'
        torch.save(Net.state_dict(),model_filepath )

        # 테스트 파일로 검증
        test_img_filepath = 'psdata/' + test_folder + '/test/img/*.png'
        test_mask_dir = 'psdata/' + test_folder + '/test/mask/'

        test_img_list = glob.glob(test_img_filepath)
        test_mask_list = pr0287.mask_list_load(test_img_list, test_mask_dir)

        test_imgset, test_maskset = read_images(test_img_list, test_mask_list, height, width)

        # load model
        test_Net = torchvision.models.segmentation.deeplabv3_resnet50(
            pretrained=True)  # Load net
        test_Net.classifier[4] = torch.nn.Conv2d(
            256, 2, kernel_size=k_size, stride=(1, 1), padding=p_size)
        # Change final layer to 2 classes
        test_Net = Net.to(device) # Set net to GPU or CPU

        test_Net.load_state_dict(torch.load(model_filepath))  # Load trained model
        test_Net.eval()  # Set to evaluation mode

        # eval
        with torch.no_grad():
            test_pred = test_Net(test_imgset)['out']  # Run net

        test_segs = torch.argmax(test_pred, 1).cpu().detach()
        test_segs = test_segs.type(torch.IntTensor)
        test_anns = test_maskset.type(torch.IntTensor)

        # accuracy test
        test_img_cnt = len(test_img_list)
        acc, precision, recall, TPr, TNr, FPr, FNr = pr0287.seg_acc(
            test_segs, test_anns, test_img_cnt)
        miou = pr0287.seg_miou(test_img_cnt, test_anns, test_segs, 2)

        print(' model name: ', model_filepath,
                ' Model Acc: ', acc,
                ' Precision: ', precision,
                ' Recall: ', recall,
                ' TPr: ', TPr,
                ' TNr: ', TNr,
                ' FPr: ', FPr,
                ' FNr: ', FNr,
                ' mIoU: ', miou)

        # 결과파일이 없다면
        test_acc_df = pd.DataFrame(index=range(1),
                            columns=['idx','pc_nm','dataset_type','batch_size',
                                        'kernel_size','normalized_yn','weight',
                                        'device','itr', 'acc', 'precision',
                                        'recall','TPr', 'TNr', 'FPr', 'FNr', 'miou'])
        test_acc_df.loc[0, 'pc_nm'] = pc_nm
        test_acc_df.loc[0, 'idx'] = time_tail
        test_acc_df.loc[0, 'dataset_type'] = dataset_type
        test_acc_df.loc[0, 'batch_size'] = batch_size
        test_acc_df.loc[0, 'kernel_size'] = k_size
        test_acc_df.loc[0, 'normalized_yn'] = normalize_yn
        test_acc_df.loc[0, 'device'] = device
        test_acc_df.loc[0, 'itr'] = itr
        test_acc_df.loc[0, 'acc'] = acc
        test_acc_df.loc[0, 'precision'] = precision
        test_acc_df.loc[0, 'recall'] = recall
        test_acc_df.loc[0, 'TPr'] = TPr
        test_acc_df.loc[0, 'TNr'] = TNr
        test_acc_df.loc[0, 'FPr'] = FPr
        test_acc_df.loc[0, 'FNr'] = FNr
        test_acc_df.loc[0, 'miou'] = miou

        if weight_yn == 'y':
            test_acc_df.loc[0, 'weight'] = str(weight_val)
        else:
            test_acc_df.loc[0, 'weight'] = '-'

        test_acc_df.to_excel(output_dir + time_tail +  '/_test_acc.xlsx', index=False)


        # 개별 이미지 결과 저장
        for i in range(test_img_cnt):
            test_img = cv2.imread(test_img_list[i])
            test_seg = test_segs[i].numpy()
            test_ann = test_anns[i].numpy()
            test_ann = np.where(test_ann == 0, 255, 0)

            imgR = test_img[:, :, 0]
            imgG = test_img[:, :, 1]
            imgB = test_img[:, :, 2]

            imgR_s = np.where(test_seg == 1, 0, imgR)
            imgG_s = np.where(test_seg == 1, 0, imgG)
            imgB_s = np.where(test_seg == 1, 0, imgB)

            imgR_m = np.concatenate((imgR, test_ann, imgR_s), axis=1)
            imgG_m = np.concatenate((imgG, test_ann, imgG_s), axis=1)
            imgB_m = np.concatenate((imgB, test_ann, imgB_s), axis=1)

            img_m = np.dstack((imgR_m, imgG_m, imgB_m))

            cv2.imwrite(output_dir + time_tail + '/seg_' + basename(test_img_list[i]), img_m)


        # output_dir 아래에 있는 모든 폴더 리스트 가져오기
        try:
            rlt_dirs = glob.glob(output_dir + '*/')
            all_model_acc_df = pd.DataFrame()
            all_test_acc_df = pd.DataFrame()

            # 각 폴더에 포함된 _model_acc_xlsx 파일 리스트 가져오기
            for rlt_dir in rlt_dirs:

                # 폴더 안에 _model_acc_xlsx 파일이 있으면
                if len(glob.glob(rlt_dir + '_model_acc.xlsx')) > 0:
                    # 해당 파일을 불러와서
                    rlt_df = pd.read_excel(rlt_dir + '_model_acc.xlsx')
                    # 해당 파일에 새로운 결과를 추가
                    all_model_acc_df = pd.concat([all_model_acc_df, rlt_df], axis=0)
                    # 결과를 저장
            all_model_acc_df.to_excel(output_dir + 'all_model_acc.xlsx', index=False)

            # 각 폴더에 포함된 _test_acc_xlsx 파일 리스트 가져오기
            for rlt_dir in rlt_dirs:

                # 폴더 안에 _model_acc_xlsx 파일이 있으면
                if len(glob.glob(rlt_dir + '_test_acc.xlsx')) > 0:
                    # 해당 파일을 불러와서
                    rlt_df = pd.read_excel(rlt_dir + '_test_acc.xlsx')
                    # 해당 파일에 새로운 결과를 추가
                    all_test_acc_df = pd.concat([all_test_acc_df, rlt_df], axis=0)
                    # 결과를 저장
            all_test_acc_df.to_excel(output_dir + 'all_test_acc.xlsx', index=False)

        except:
            print('collect result is failed')


# ################################################################################
# Finish
# ################################################################################
print("time :", time.time() - start)
print('')
