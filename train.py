# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn

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

    option = parser.parse_args()
    print(option)

    return option

####### TRAIN #########
def train(model, criterion, optimizer, scheduler=None):
    model.train() #drop outを適用
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, labels) in enumerate(trainloader):

        if use_gpu: #GPUが使えるなら
            inputs = Variable(data.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(data), Variable(labels)

        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        loss.backward() #Back propagation
        #optimizer.step() # n epoch でlearning rate を m倍する

        train_loss += loss.item()
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

        progress_bar(batch_idx, len(trainloader),
                        'Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

######## TEST ########
def test(model):
    model.eval() #drop outを適用しない
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(valloader):

            if use_gpu:
                inputs = Variable(data.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(data), Variable(labels)

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            # statistics
            test_loss += loss.item()
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            progress_bar(batch_idx, len(valloader),
                            'Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    return acc

if __name__ == '__main__':
    #####################
    #### dataの読み込み ###
    #####################
    opt = get_parser()

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    gpu_ids = []
    torch.manual_seed(opt.seed)

    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
        gpu_ids = [0]

    #####################
    ## datasetの読み込み ##
    #####################
    print('==> Loading data...')
    print('-' * 20)
    trainloader, testloader, class_names = dataloader(batch_size=opt.batchSize,
                                                      tbatchSize=opt.tbatchSize,
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

    # optimaizerなどの設定
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    for epoch in range(1, opt.nEpochs):
        train(model, criterion, optimizer)
