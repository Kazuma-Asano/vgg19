# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.utils.model_zoo as model_zoo

def define_network(num_class):

    model = VGG19()
    # パラメータの初期化
    model.apply(initialize_weights) # 初期化

    if num_class != 1000:
        num_features = model.fc3.in_features
        model.fc3 = nn.Linear(num_features, num_class)

    return model

def initialize_weights(self):
    for m in self.modules():
        # 畳み込み層 (Conv2)の初期化
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # バッチノーマライゼーションの初期化
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        # 全結合層の初期化
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

################################################################################
class VGG19(nn.Module):
    def __init__(self, num_class=1000):
        super(VGG19, self).__init__()
        ########################################################################

        # inout : w x h x 3ch(RGB)
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=True) # 活性化関数
        # self.relu1_1 = nn.ReLU()

        # 以降，in_channels, out_channelsの表記は省略
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU(inplace=True)
        # self.relu1_2 = nn.ReLU()

        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)

        ########################################################################

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU(inplace=True)
        # self.relu2_1 = nn.ReLU()

        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU(inplace=True)
        # self.relu2_2 = nn.ReLU()

        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)

        ########################################################################

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU(inplace=True)
        # self.relu3_1 = nn.ReLU()

        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU(inplace=True)
        # self.relu3_2 = nn.ReLU()

        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.relu3_3 = nn.ReLU(inplace=True)
        # self.relu3_3 = nn.ReLU()

        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3_4 = nn.BatchNorm2d(256)
        self.relu3_4 = nn.ReLU(inplace=True)
        # self.relu3_4 = nn.ReLU()

        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)

        ########################################################################

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU(inplace=True)
        # self.relu4_1 = nn.ReLU()

        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.ReLU(inplace=True)
        # self.relu4_2 = nn.ReLU()

        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.relu4_3 = nn.ReLU(inplace=True)
        # self.relu4_3 = nn.ReLU()

        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4_4 = nn.BatchNorm2d(512)
        self.relu4_4 = nn.ReLU(inplace=True)
        # self.relu4_4 = nn.ReLU()

        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)

        ########################################################################

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.relu5_1 = nn.ReLU(inplace=True)
        # self.relu5_1 = nn.ReLU()

        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.relu5_2 = nn.ReLU(inplace=True)
        # self.relu5_2 = nn.ReLU()

        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.relu5_3 = nn.ReLU(inplace=True)
        # self.relu5_3 = nn.ReLU()

        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5_4 = nn.BatchNorm2d(512)
        self.relu5_4 = nn.ReLU(inplace=True)
        # self.relu5_4 = nn.ReLU()

        self.max5 = nn.MaxPool2d(kernel_size=2, stride=2)

        ########################################################################

        self.fc1 = nn.Linear(512*7*7, 4096) # forward内の print(out37.size()) でサイズ確認
        self.relu_fc1 = nn.ReLU(inplace=True)
        # self.relu_fc1 = nn.ReLU()
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.relu_fc2 = nn.ReLU(inplace=True)
        # self.relu_fc2 = nn.ReLU()
        self.drop2 = nn.Dropout()
        self.fc3 = nn.Linear(in_features=4096, out_features=num_class)

        # self._initialize_weights() # 初期化

    def forward(self, x):
        out = self.conv1_1(x)
        out = self.bn1_1(out)
        out = self.relu1_1(out)
        out = self.conv1_2(out)
        out = self.bn1_2(out)
        out = self.relu1_2(out)

        out = self.max1(out)

        out = self.conv2_1(out)
        out = self.bn2_1(out)
        out = self.relu2_1(out)
        out = self.conv2_2(out)
        out = self.bn2_2(out)
        out = self.relu2_2(out)

        out = self.max2(out)

        out = self.conv3_1(out)
        out = self.bn3_1(out)
        out = self.relu3_1(out)
        out = self.conv3_2(out)
        out = self.bn3_2(out)
        out = self.relu3_2(out)
        out = self.conv3_3(out)
        out = self.bn3_3(out)
        out = self.relu3_3(out)
        out = self.conv3_4(out)
        out = self.bn3_4(out)
        out = self.relu3_4(out)

        out = self.max3(out)

        out = self.conv4_1(out)
        out = self.bn4_1(out)
        out = self.relu4_1(out)
        out = self.conv4_2(out)
        out = self.bn4_2(out)
        out = self.relu4_2(out)
        out = self.conv4_3(out)
        out = self.bn4_3(out)
        out = self.relu4_3(out)
        out = self.conv4_4(out)
        out = self.bn4_4(out)
        out = self.relu4_4(out)

        out = self.max4(out)

        out = self.conv5_1(out)
        out = self.bn5_1(out)
        out = self.relu5_1(out)
        out = self.conv5_2(out)
        out = self.bn5_2(out)
        out = self.relu5_2(out)
        out = self.conv5_3(out)
        out = self.bn5_3(out)
        out = self.relu5_3(out)
        out = self.conv5_4(out)
        out = self.bn5_4(out)
        out = self.relu5_4(out)

        out = self.max5(out)

        # print(out.size())
        out = out.view(out.size(0), -1) # バッチサイズごとに1次元配列化
        out = self.fc1(out)
        out = self.relu_fc1(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.relu_fc2(out)
        out = self.drop2(out)
        out = self.fc3(out)

        return out


if __name__ == '__main__':

    """
    Testing
    $python network.py
    """

    # モデルの用意
    model = define_network(num_class=10)

    # データの用意
    x = torch.FloatTensor( np.random.random((1, 3, 224, 224))) # (batch_size, channels, width, height)

    # GPUが使えるならば
    if torch.cuda.is_available():
        print(' # on GPU #')
        model.cuda()
        x.cuda()

    out = model(x)
    # print(out)
    # print(out.data)


    softmax = nn.Softmax(dim=1) # 値を0 ~ 1にして確率化
    predicts = softmax(out)
    # print(predicts)
    #
    loss = torch.sum(predicts)
    print(loss)
    loss.backward()
