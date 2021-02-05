import sys
import os
from os.path import join
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from torchvision import transforms

import matplotlib.pyplot as plt

from model import UNet
from dataloader import DataLoader

import time

# change the learning rate to 0.001


def train_net(net,
              epochs=10,
              data_dir='data/cells/',
              n_classes=2,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              gpu=True):

    loader = DataLoader(data_dir)

    N_train = loader.n_train()

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.99,
                          weight_decay=0.0005)

    start_time = time.time()
    for epoch in range(epochs):
        print('Epoch %d/%d' % (epoch + 1, epochs))
        print('Training...')
        net.train()
        loader.setMode('train')

        epoch_loss = 0

        for i, (img, label) in enumerate(loader):
            shape = img.shape
            # print(shape)
            # print(label.shape)

            # todo: create image tensor: (N,C,H,W) - (batch size=1,channels=1,height,width)
            image = torch.tensor(img)
            label = torch.tensor(label)
            image = image.reshape(1, 1, shape[0], shape[1])
            # print(image.shape)
            image = image.to(torch.float32)
            label = label.to(torch.int)

            # todo: load image tensor to gpu
            if gpu:
                image = image.to('cuda')
                label = label.to('cuda')

            # todo: get prediction and getLoss()
            optimizer.zero_grad()
            outputs = net(image)
            # print(outputs.shape)
            loss = getLoss(outputs, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            print('Training sample %d / %d - Loss: %.6f' %
                  (i + 1, N_train, loss.item()))

            # optimize weights
            # optimizer.step()
        torch.save(net.state_dict(), join(
            data_dir, 'checkpoints') + '/CP%d.pth' % (epoch + 1))
        print('Checkpoint %d saved !' % (epoch + 1))
        print('Epoch %d finished! - Loss: %.6f' % (epoch + 1, epoch_loss / i))
    end_time = time.time()
    total_time = end_time - start_time
    print("Training time: ", total_time)
    print('Testing!')
    # displays test images with original and predicted masks after training
    loader.setMode('test')
    net.eval()
    with torch.no_grad():
        for i, (img, label) in enumerate(loader):
            shape = img.shape
            print(shape)
            print(label.shape)
            img_torch = torch.from_numpy(
                img.reshape(1, 1, shape[0], shape[1])).float()
            if gpu:
                img_torch = img_torch.cuda()
            pred = net(img_torch)
            print(pred.shape)
            pred_sm = softmax(pred)
            _, pred_label = torch.max(pred_sm, 1)
            print(pred_sm.shape)

            plt.subplot(1, 3, 1)
            plt.imshow(img * 255.)
            plt.subplot(1, 3, 2)
            plt.imshow(label * 255.)
            plt.subplot(1, 3, 3)
            plt.imshow(pred_label.cpu().detach().numpy().squeeze() * 255.)
            plt.savefig('./result' + str(i) + '.png')
            plt.show()


def getLoss(pred_label, target_label):
    p = softmax(pred_label)
    return cross_entropy(p, target_label)


def softmax(input):
    # todo: implement softmax function
    # exp(x)/sum(exp(x))
    temp = torch.exp(input)
    # sum of all output over all channel
    sum_temp = torch.sum(temp, 1)
    sum_temp = sum_temp.reshape(temp.shape[0], 1, temp.shape[2], temp.shape[3])
    p = torch.div(temp, sum_temp)
    return p


def cross_entropy(input, targets):
    # todo: implement cross entropy
    # Hint: use the choose function
    channel = choose(input, targets)
    ce = torch.mean(-torch.log(channel))
    return ce

# Workaround to use numpy.choose() with PyTorch


def choose(pred_label, true_labels):
    size = pred_label.size()
    ind = np.empty([size[2] * size[3], 3], dtype=int)
    i = 0
    for x in range(size[2]):
        for y in range(size[3]):
            ind[i, :] = [true_labels[x, y], x, y]
            i += 1

    pred = pred_label[0, ind[:, 0], ind[:, 1],
                      ind[:, 2]].view(size[2], size[3])

    return pred


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs',
                      default=10, type='int', help='number of epochs')
    parser.add_option('-c', '--n-classes', dest='n_classes',
                      default=2, type='int', help='number of classes')
    parser.add_option('-d', '--data-dir', dest='data_dir',
                      default='data/cells/', help='data directory')
    parser.add_option('-g', '--gpu', action='store_true',
                      dest='gpu', default=False, help='use cuda')
    parser.add_option('-l', '--load', dest='load',
                      default=False, help='load file model')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()

    net = UNet(n_classes=args.n_classes)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from %s' % (args.load))

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True

    train_net(net=net,
              epochs=args.epochs,
              n_classes=args.n_classes,
              gpu=args.gpu,
              data_dir=args.data_dir)
