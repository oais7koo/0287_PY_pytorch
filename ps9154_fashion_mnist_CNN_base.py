# -*- coding: utf-8 -*-
# author : oaiskoo
# date : 2022.
# goal :

import time
import numpy as np
from glob import glob
from os.path import basename
import pickle

#if os.path.exists('../python/oaislib_org.py'):
#    shutil.copy('../python/oaislib_org.py', 'oaislib.py')
import oaislib
start = time.time()

# ################################################################################
# Parameter
# ################################################################################
prefix = 'ps5011'
workname = 'fashion minist 심층 신경망'
print(prefix + '_' + workname)

# ################################################################################
# 코드 5-1 라이브러리 호출
# ################################################################################

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# ################################################################################
# 코드 5-2 CPU 혹은 GPU 사용 여부 확인
# ################################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# ################################################################################
# 코드 5-3 데이터세트 내려받기
# ################################################################################
train_dataset = torchvision.datasets.FashionMNIST(
    root='psdata/ps5010',
    train = True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]))
test_dataset = torchvision.datasets.FashionMNIST(
    root='psdata/ps5010',
    download=True,
    train=False,
    transform=transforms.Compose([transforms.ToTensor()]))

# ################################################################################
# 코드 5-4 데이터를 메모리에 로딩하기
# ################################################################################
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100)

# ################################################################################
# 코드 5-5 분류에 사용될 클래스 정의
# ################################################################################
labels_map = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress',
              4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

'''
fig = plt.Figure(figsize=(8, 8))
columns = 4
rows = 5
for i in range(1, columns * rows + 1):
    img_xy = np.random.randint(len(train_dataset))
    img = train_dataset[img_xy][0][0, :, :]  # 3차원 배열 생성
    fig.add_subplot(rows, columns, i)
    plt.title(labels_map[train_dataset[img_xy][1]])
    plt.axis('off')
    plt.imshow(img, cmap='gray')
#plt.show()
'''

# ################################################################################
# 코드5-9 합성곱 네트워크 생성
# ################################################################################
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.layer1 = nn.Sequential(
            # __init__()에서 사용할 네트워크 모델을 정의해 줄 뿐 아닐
            # forward()함수에서 구현될 순전파를 계층형태로 좀 더 가독성 높은 코드로 작성
            # nn.Sequential은 계층을 차례로 쌓을 수 있도록 Wx + b와 같은 수식과 활성화 함수를 연결해주는 역할
            # *** 특히 데이터가 각 계층을 순차적으로 지나갈때 사용하면 좋은 방법
            # nn.Sequential은 여러 개의 계층을 하나의 컨테이너에 구현하는 방법

            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            # 합성곱층은 합성곱 연산을 통해서 이미지 특징 추출
            # in_channels는 입력 채널의 수, out_channels는 출력 채널의 수, kernel_size는 필터의 크기
            # 학습과정에서 각 배치 단위별로 데이터가 다양한 분포를 가지더라도 평균과 분산을 이용해서 정규화하는 것을 의미

            nn.BatchNorm2d(32),
            # 배치단위나 계층에 따라 입력 값의 분포가 모두 다르지만, 정규화를 통해 가우시안 형태로 만듦(평균 0, 표준편차 1)

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2)
            # 이미지 크기를 축소하는 용도로 사용
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        # Fully Connected Layer
        # 클래스를 분류하기 위해서 이미지 형태의 데이터를 배열 형태로 변환
        # Conv2d에서 사용된 하이퍼파라미터(패팅과 스트라이드 값)에 따라 출력크기가 달라짐
        # 이렇게 줄어든 출력 크기는 최종적으로 분류를 담당하는 완전연결층으로 전달
        # in_feature는 입력 데이터 크기
        # **in_features의 값은 앞의 layer1,2의 입출력값을 계산해봐야 함

        self.drop = nn.Dropout(0.25)


        self.fc2 = nn.Linear(in_features=600, out_features=120)
        # in, out 숫자를 어떻게 맞추는지?


        self.fc3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

#################################################################################
# 코드 5-10 합성곱 네트워크를 위한 파라미터 정의
# ################################################################################

learning_rate = 0.001
model = FashionCNN()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(model)

#################################################################################
# 코드 5-11. 모델 학습 및 성능 평가
# ################################################################################
num_epochs = 5
count = 0
loss_list = []
iteration_list = []
accuracy_list = []

predictions_list = []
labels_list = []

for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        train = Variable(images.view(100,1,28,28))
        labels = Variable(labels)

        outputs = model(train)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += 1

        if not (count % 50):
            total = 0
            correct = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels_list.append(labels)
                test = Variable(images.view(100,1,28,28))
                outputs = model(test)
                predictions = torch.max(outputs, 1)[1].to(device)
                predictions_list.append(predictions)
                correct += (predictions == labels).sum()
                total += len(labels)

            accuracy = correct * 100 / total
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)

        if not (count % 500):
            print("Iteration: {}  Loss: {}  Accuracy: {} %".format(count, loss.data, accuracy))


#################################################################################
# Save Result
# ################################################################################

#################################################################################
# Finish
# ################################################################################
print("time :", time.time() - start)
print('')
