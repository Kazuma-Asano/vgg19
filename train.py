# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import argparse
import signal

import torch
import torch.nn as nn
import torch.optim as optim

from dataloder import dataloader
from network import VGG19, initialize_weights
from util import progress_bar

def get_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='Pytorch vgg19')
    parser.add_argument('--batchSize', '-b', type=int, default=16, help='training batch size')
    parser.add_argument('--testBatchSize', '-tb', type=int, default=8, help='testing batch size')
    parser.add_argument('--nEpochs', '-e', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
    parser.add_argument('--cuda', action='store_true', help='use cuda?') # GPUを利用するなら $python train.py --cuda
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--resume', '-r', type=int, default=0, help='load trained model epochs')
    option = parser.parse_args()
    print(option)

    return option

####### TRAIN #########
def train(epoch, model, criterion, optimizer, trainloader, scheduler=None):
    model.train() # drop outを適用
    train_loss = 0
    correct = 0
    total = 0

    for iteration, (inputs, labels) in enumerate(trainloader):

        # GPUが使えるなら
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        loss.backward() # Back propagation
        #optimizer.step() # n epoch でlearning rate を m倍する

        train_loss += loss.item()
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

        printLoss = (train_loss/(iteration+1))
        printAcc = 100.*correct/total
        progress_bar(epoch, iteration, len(testloader),
                    ': Loss: {:.4f}, Acc: {:.4f} % ({}/{})'.format(printLoss, printAcc, correct, total) )

######## TEST ########
def test(epoch, model, criterion, testloader):
    model.eval() # drop outを適用しない
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for iteration, (inputs, labels) in enumerate(testloader):

            # GPUが使えるなら
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            printLoss = (train_loss/(iteration+1))
            printAcc = 100.*correct/total
            progress_bar(epoch, iteration, len(testloader),
                        ': Loss: {:.4f}, Acc: {:.4f} % ({}/{})'.format(printLoss, printAcc, correct, total) )

def checkpoint(epoch, model):
    checkpointDir = './checkpoint/'
    os.makedirs(checkpointDir, exist_ok=True)
    model_out_path = './checkpoint/model_epoch{}.pth'.format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print('---Checkpoint saved---\n')


if __name__ == '__main__':
    #####################
    #### dataの読み込み ###
    #####################
    opt = get_parser()

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    gpu_ids = []
    use_gpu = False

    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
        gpu_ids = [0]
        use_gpu = True
    else:
        torch.manual_seed(opt.seed)


    #####################
    ## datasetの読み込み ##
    #####################
    print('==> Loading data...')
    print('-' * 20)
    trainloader, testloader, class_names = dataloader(batch_size=opt.batchSize,
                                                      tbatchSize=opt.testBatchSize,
                                                      threads=opt.threads)
    print('OK')

    #####################
    ### modelの読み込み ###
    #####################
    print('==> Building model...')
    print('-' * 20)
    model = VGG19()
    model.apply(initialize_weights) # 初期化
    print(model)
    print('OK')
    print('-' * 20)

    # 学習したモデルのロード
    if opt.resume > 0:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        model.load_state_dict(torch.load('./checkpoint/model_epoch{}.pth'.format(opt.resume),
                                map_location=lambda storage,
                                loc: storage))
        print('OK')
        print('-' * 20)

    # optimaizerなどの設定
    criterion = nn.CrossEntropyLoss()
    if opt.cuda:
        criterion = criterion.cuda()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    print('==> Start training...')
    try:
        for epoch in range(1, opt.nEpochs+1): #default 100 epoch
            print('Train:')
            train(epoch, model, criterion, optimizer, trainloader)
            print('Test:')
            test(epoch, model, criterion, testloader)
            if epoch%10 == 0:
                checkpoint(epoch, model)
    except KeyboardInterrupt as e: # 強制終了時の処理
        checkpoint(epoch, model)
        print('Ctrl-C Finished')
