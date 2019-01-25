# -*- coding: utf-8 -*-
import os
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image # テスト用

def dataloader(batch_size=4, tbatchSize=4, threads=4):
    # データセットの設定
    ############################################################################
    # 読み込む画像の前処理
    img_size = 256

    train_data_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size), # ランダムな場所をクロップ & リサイズ
            transforms.RandomHorizontalFlip(), # データを増やすため, ランダムに反転
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    test_data_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)), # リサイズ
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    ############################################################################
    # dataloaderを作る
    trainset = datasets.ImageFolder(root='./data/train', transform=train_data_transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=threads)

    testset = datasets.ImageFolder(root='./data/test', transform=test_data_transform)
    testloader = torch.utils.data.DataLoader(testset,
                                                batch_size=tbatchSize,
                                                shuffle=False,
                                                num_workers=threads)
    class_names = trainset.classes

    return trainloader, testloader, class_names


if __name__ == '__main__':
    """
    Test
    $python dataloader.py
    """
    print('==> Preparing data...')
    print('-' * 20)
    
    trainloader, testloader, class_names = dataloader(batch_size=4)

    # どんなデータが読み込まれたか保存してみる
    saveDir = './test/dataloader/'
    os.makedirs(saveDir, exist_ok=True)
    img_name = 'input.png'
    for iteration, batch in enumerate(trainloader, 1):
        img, label = batch[0], batch[1]
        print(label)
        for i in label:
            print(class_names[i])
        save_image(img.data, saveDir + '{}'.format(img_name))
        if(iteration == 1): break
