# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

################################################################################
class VGG19(nn.Module):
    def __init__(self, num_class=10):
        super(VGG19, self).__init__()
        ########################################################################

        # inout : w x h x 3ch(RGB)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True) # conv3-64
        # self.relu1 = nn.ReLU(inplace=True) # 活性化関数
        self.relu1 = nn.ReLU()

        # 以降，in_channels, out_channelsの表記は省略
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True) # conv3-64
        # self.relu2 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU()
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)

        ########################################################################

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True) # conv3-128
        # self.relu3 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True) # conv3-128
        # self.relu4 = nn.ReLU(inplace=True)
        self.relu4 = nn.ReLU()
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)

        ########################################################################

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True) # conv3-256
        # self.relu5 = nn.ReLU(inplace=True)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True) # conv3-256
        # self.relu6 = nn.ReLU(inplace=True)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True) # conv3-256
        # self.relu7 = nn.ReLU(inplace=True)
        self.relu7 = nn.ReLU()

        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True) # conv3-256
        # self.relu8 = nn.ReLU(inplace=True)
        self.relu8 = nn.ReLU()
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)

        ########################################################################

        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True) # conv3-512
        # self.relu9 = nn.ReLU(inplace=True)
        self.relu9 = nn.ReLU()

        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True) # conv3-512
        # self.relu10 = nn.ReLU(inplace=True)
        self.relu10 = nn.ReLU()

        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True) # conv3-512
        # self.relu11 = nn.ReLU(inplace=True)
        self.relu11 = nn.ReLU()

        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True) # conv3-512
        # self.relu12 = nn.ReLU(inplace=True)
        self.relu12 = nn.ReLU()
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)

        ########################################################################

        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True) # conv3-512
        # self.relu13 = nn.ReLU(inplace=True)
        self.relu13 = nn.ReLU()

        self.conv14 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True) # conv3-512
        # self.relu14 = nn.ReLU(inplace=True)
        self.relu14 = nn.ReLU()

        self.conv15 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True) # conv3-512
        # self.relu15 = nn.ReLU(inplace=True)
        self.relu15 = nn.ReLU()

        self.conv16 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True) # conv3-512
        # self.relu16 = nn.ReLU(inplace=True)
        self.relu16 = nn.ReLU()
        self.max5 = nn.MaxPool2d(kernel_size=2, stride=2)

        ########################################################################

        self.fc1 = nn.Linear(512*8*8, 4096) # forward内の print(out37.size()) でサイズ確認
        # self.relu17 = nn.ReLU(inplace=True)
        self.relu17 = nn.ReLU()
        self.drop1 = nn.Dropout()

        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        # self.relu18 = nn.ReLU(inplace=True)
        self.relu18 = nn.ReLU()
        self.drop2 = nn.Dropout()

        self.fc3 = nn.Linear(in_features=4096, out_features=num_class)

        # self._initialize_weights() # 初期化

    def forward(self, x):

        out1 = self.conv1(x)
        out2 = self.relu1(out1)
        out3 = self.conv2(out2)
        out4 = self.relu2(out3)
        out5 = self.max1(out4)

        out6 = self.conv3(out5)
        out7 = self.relu3(out6)
        out8 = self.conv4(out7)
        out9 = self.relu4(out8)
        out10 = self.max2(out9)

        out11 = self.conv5(out10)
        out12 = self.relu5(out11)
        out13 = self.conv6(out12)
        out14 = self.relu6(out13)
        out15 = self.conv7(out14)
        out16 = self.relu7(out15)
        out17 = self.conv8(out16)
        out18 = self.relu8(out17)
        out19 = self.max3(out18)

        out20 = self.conv9(out19)
        out21 = self.relu9(out20)
        out22 = self.conv10(out21)
        out23 = self.relu10(out22)
        out23 = self.relu10(out22)
        out24 = self.conv11(out23)
        out25 = self.relu11(out24)
        out26 = self.conv12(out25)
        out27 = self.relu12(out26)
        out28 = self.max4(out27)

        out29 = self.conv13(out28)
        out30 = self.relu13(out29)
        out31 = self.conv14(out30)
        out32 = self.relu14(out31)
        out33 = self.conv15(out32)
        out34 = self.relu15(out33)
        out35 = self.conv16(out34)
        out36 = self.relu16(out35)
        out37 = self.max5(out36)
        # print(out37.size()) # サイズの確認 今回は (batch_size, 512, 8, 8 )

        x = out37.view(out37.size(0), -1) # batch_size (out37.size(0)) ごとに1次元配列化
                                          # 例) (batch_size, 512, 7, 7) => (batch_size, 512x7x7)
        out38 = self.fc1(x)
        out39 = self.relu17(out38)
        out40 = self.drop1(out39)

        out41 = self.fc2(out40)
        out42 = self.relu18(out41)
        out43 = self.drop2(out42)

        out44 = self.fc3(out43) # この時点では確率になっていない
        # result = self.softmax(out44)

        return out44


if __name__ == '__main__':

    """
    Testing
    $python network.py
    """

    # モデルの用意
    model = VGG19()
    model.apply(initialize_weights) # 初期化
    print(model)

    # データの用意
    x = torch.FloatTensor( np.random.random((1, 3, 256, 256))) # (batch_size, channels, width, height)

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
