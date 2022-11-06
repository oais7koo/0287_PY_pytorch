# -*- coding: utf-8 -*-
# author : oaiskoo
# date : 2022.
# goal :


# ################################################################################
# Library
# ################################################################################
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import time
import numpy as np
from glob import glob
from os.path import basename
import pickle

# if os.path.exists('../python/oaislib_org.py'):
#    shutil.copy('../python/oaislib_org.py', 'oaislib.py')
import oaislib
start = time.time()


# ################################################################################
# Parameter
# ################################################################################
prefix = 'ps9152'
workname = 'fashion minist 심층 신경망'
print(prefix + '_' + workname)

# ################################################################################
# Process
# ################################################################################
# fashion_minst dataset은 총 7만장(28*28 픽셀)
# 훈련데이터는 0~255 사이의 값을 갖는 28*28 크기의 넘파이 배열
# 레이블(정답)데이터는 0~9까지의 정수 값을 가지는 배열

# ################################################################################
# 코드 5-2 CPU 혹은 GPU 사용 여부 확인
# ################################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# ################################################################################
# 코드 5-3 데이터세트 내려받기
# ################################################################################
train_dataset = torchvision.datasets.FashionMNIST(
    root='psdata/ps5010', download=True, transform=transforms.Compose([transforms.ToTensor()]))
# train_dataset.data.shape : torch.Size([60000, 28, 28])

test_dataset = torchvision.datasets.FashionMNIST(
    root='psdata/ps5010', download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))
# test_dataset.data.shape  :torch.Size([10000, 28, 28])

# ################################################################################
# 코드 5-4 데이터를 메모리에 로딩하기
# ################################################################################
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100)
# train_loader 클래스
# batch_size : 100
# 데이터 구조 : train_loader.dataset.data.shape 로 torch.Size([60000, 28, 28]) 가 나옴
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100)
# 데이터 구조 : test_loader.dataset.data.shape 로 torch.Size([10000, 28, 28]) 가 나옴

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
# 코드 5-6 심층 신경망 모델생성
# ################################################################################


class FashionDNN(nn.Module):  # nn은 딥러닝 모델(네트워크) 구성에 필요한 모듈이 모여있는 패키지
    # 5-6-1
    def __init__(self):  # 클래스형태의 모델은 항상 torch.nn.Module을 상속받아야 한다.
        # 객체가 갖는 속성 값 초기화, super(FashionDNN) 은 FashionDNN의 부모 클래스(super)의 클래스를 상속받겠다는 것임
        super(FashionDNN, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=256)
        # 단순 선형 회귀모델
        # in_features : 입력층의 노드수 28 * 28 = 784
        # ???out_features : 출력층의 노드수 왜 256인지는 모르겠음

        self.drop = nn.Dropout(0.25)
        # 0.25만큼 텐서의 값이 0이됨, 0이 안되는 갓음 기존값의 1/(1-0.25)만큼 곱해져서 커짐

        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)
        # ??? 여기서 out_features는 10인 이유는 라벨이 10이라서?

    # 5-6-2
    def forward(self, input_data):
        # 순전파 함수. 이름은 반드시 forward로 지정해야함

        out = input_data.view(-1, 784)
        # view는 넘파이의 reshape 역할로 텐서 크기 변경
        # (-1, 784)은 (?, 784)의 크기로 변경(이차원 텐서로)
        # -1은 알아서 계산해라는 의미

        out = F.relu(self.fc1(out))
        out = self.drop(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


# ################################################################################
# 코드 5-7 심층 신경망에서 필요한 파라미터 정의
# ################################################################################
learning_rate = 0.001
model = FashionDNN()
model.to(device)

criterion = nn.CrossEntropyLoss()
# 분류문제에서 사용하는 손실 함수

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 최적화 함수 경사하강법은 Adam 사용

print(model)
'''
FashionDNN(
  (fc1): Linear(in_features=784, out_features=256, bias=True)
  (drop): Dropout(p=0.25, inplace=False)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=10, bias=True)
)
'''

# ################################################################################
# 코드 5-8 심층 신경망을 이용한 모델 학습
# ################################################################################
# 5-8-1
num_epochs = 5
# epoch : 전체 데이터를 한번 학습시키는 것
# epoch가 5이면 학습데이터 5만장 * 5 = 25만장을 학습시킴
count = 0
loss_list = []
iteration_list = []
accuracy_list = []

predictions_list = []
labels_list = []

# 5-8-2
for epoch in range(num_epochs):
    for images, labels in train_loader:  # for를 이용해서 레코드를 하나씩 가지고 옴
        # 5-8-2-1
        images, labels = images.to(device), labels.to(device)

        # images가 100,1,28,28이라서 아래 대로 할 필요가 없고, 에러 미발생
        train = Variable(images)
        # train = Variable(images.view(100, 1, 28, 28)) # 100은 배치 수인듯
        # ??순전파 단계에서 테이브는 수행하는 모든 연산을 저장함(그런데 이게 설명이 맞나)
        # autograd는 Variable을 사용해서 역전파를 위한 미분값을 자동으로 계산해줌
        # 자동미분을 게산하기 위해서는 torch.augograd 패키지 안에 있는 variable를 이용해야 동작함

        # labels = Variable(labels) # 이 행은 없어도 작동한다. 왜냐하면 labels는 이미 로 되어있기 때문이다.

        # 5-8-2-2
        outputs = model(train)
        # 학습데이터를 모델에 적용

        loss = criterion(outputs, labels)
        # 손실함수로 출력값과 라벨을 비교해서 계산

        optimizer.zero_grad()
        # 기울기 초기화
        # 기울기가 검증에서는 초기화 되면 안됨

        loss.backward()
        # 역전파 계산

        optimizer.step()
        # 가중치 업데이트

        count += 1
        #print('count : ', count)
        # 위의 한 루프에서 100장의 이미지 처리
        # 즉 배치사이즈 기준으로 한번의 루프가 끝남

        # 5-8-2-3 테스트 데이터를 통한 정확도 확인
        if not (count % 50):
            # 50으로 나누었을 때 나머지가 0 이면 ( 즉 50 마다)
            # 위의 이터레이션이 50번 즉 이미지가 5000장을 처리했을 때
            # 처음으로 들어옴
            # 모델을 만드는 중간중간에 테스트 데이터를 통해 정확도를 확인

            total = 0
            correct = 0

            # 5-8-2-3-1 예측정확도 확인
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                # 모델이 데이터를 처리하기 위해서는 동일한 device에 있어야 함

                labels_list.append(labels)
                test = Variable(images.view(100, 1, 28, 28))
                # autograd는 자동미분을 수행하는 파이토치 핵심 패키지로 자동 미분에 대한 값을 저장하기 위해서 tape를 사용
                # 순전파 단계에서 테이브는 수행하는 모든 연산을 저장함(그런데 이게 설명이 맞나)
                # autograd는 Variable을 사용해서 역전파를 위한 미분값을 자동으로 계산해줌
                # 자동미분을 게산하기 위해서는 torch.augograd 패키지 안에 있는 variable를 이용해야 동작함

                outputs = model(test) # torch.tensor, torch.Size([100,10])
                '''
                tensor([[-4.3525e+00, -4.9298e+00, -2.8806e+00, -3.9115e+00, -2.1844e+00,
                        3.9496e+00, -3.7661e+00,  3.2566e+00,  1.3057e+00,  6.2313e+00], # 6.23이 가장 높은 점수이고 그래서 prediction에서 9가 됨
                        [-4.5339e-02, -3.2170e+00,  5.3587e+00, -3.5568e+00,  3.9789e+00,
                        ## 중간에 98개 생략 ##
                        [-6.6893e-01, -1.3849e+00,  4.9067e+00, -1.8941e+00,  4.1050e+00, ## 2번이 가장 높음
                        -4.9077e+00,  3.1150e+00, -8.7868e+00, -1.2953e+00, -6.4187e+00]],
                    grad_fn=<AddmmBackward0>)
                '''
                predictions = torch.max(outputs, 1)[1].to(device)
                # outputs는 100,10의 크기를 가지는 텐서로 100개의 레코드에 대한 10개의 클래스에 대한 확률값을 가지고 있음
                # 여기서 확률이 가장 높은 클래스를 예측값으로 사용하기 위해서 torch.max를 사용함
                # [1]은 확률이 가장 높은 클래스의 인덱스를 의미함
                '''
                tensor([9, 2, 1, 1, 6, 1, 4, 4, 7, 7, 4, 9, 7, 3, 4, 1, 2, 6, 8, 0, 0, 7, 7, 7,
                        1, 4, 4, 3, 9, 3, 8, 8, 3, 3, 8, 0, 7, 7, 7, 9, 0, 1, 3, 9, 4, 9, 6, 1,
                        4, 4, 2, 4, 7, 6, 4, 6, 8, 4, 8, 0, 7, 7, 8, 7, 1, 1, 4, 3, 7, 8, 7, 0,
                        6, 0, 4, 3, 1, 2, 8, 4, 1, 8, 5, 9, 5, 0, 3, 4, 0, 6, 5, 3, 4, 7, 1, 8,
                        0, 1, 4, 2])
                '''
                predictions_list.append(predictions)
                correct += (predictions == labels).sum()
                total += len(labels)
                # print(total)

            # 5-8-2-3-2
            accuracy = (correct / total) * 100
            # 백분율로 계산
            # accuracy 데이터 타입 : [tensor(70.92)]
            # accuracy가 텐서인데 사칙연산이 된다.

            loss_list.append(loss.data)
            # 현재로서는 사용처가 없음

            iteration_list.append(count)
            # 현재로서는 사용처가 없음

            accuracy_list.append(accuracy)
            # 현재로서는 사용처가 없음

        # 5-8-2-4
        if not (count % 500):
            # 카운트가 500 번 즉 5000장의 이미지를 처리했을 때
            print("Iteration: {}, Loss: {}, Accuracy: {}".format(
                count, loss.data, accuracy))
            # accuracy는 텐서인데 print({}.format)에서는 텐서를 문자열로 변환해서 출력
            # loss.data는 텐서인데 print({}.format)에서는 텐서를 문자열로 변환해서 출력

#################################################################################
# Save Result
# ################################################################################

#################################################################################
# Finish
# ################################################################################
print("time :", time.time() - start)
print('')
