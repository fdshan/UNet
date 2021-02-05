import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        # todo
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        self.down1 = downStep(1, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.down2 = downStep(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.down3 = downStep(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.down4 = downStep(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(512, 1024, 3)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(1024, 1024, 3)
        self.bn2 = nn.BatchNorm2d(1024)
        self.relu2 = nn.ReLU()

        self.up1 = upStep(1024, 512)
        self.up2 = upStep(512, 256)
        self.up3 = upStep(256, 128)
        self.up4 = upStep(128, 64)

        self.conv3 = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        # todo
        # x1, x2, x3, x4 use for skip connection
        x1 = self.down1(x)
        # print('x1 shape is', x1.shape)
        x1_max = self.pool1(x1)
        # print('x1_max shape is', x1_max.shape)
        x2 = self.down2(x1_max)
        # print('x2 shape is', x2.shape)
        x2_max = self.pool2(x2)
        # print('x2_max shape is', x2_max.shape)
        x3 = self.down3(x2_max)
        # print('x3 shape is', x3.shape)
        x3_max = self.pool3(x3)
        # print('x3_max shape is', x3_max.shape)
        x4 = self.down4(x3_max)
        # print('x4 shape is', x4.shape)
        x4_max = self.pool4(x4)
        # print('x4_max shape is', x4_max.shape)

        x5 = self.relu1(self.bn1(self.conv1(x4_max)))
        # print('first bottom shape:', x5.shape)
        x5 = self.relu2(self.bn2(self.conv2(x5)))
        # print('second bottom shape:', x5.shape)

        x6 = self.up1(x5, x4)
        # print('up1 shape is:', x6.shape)
        x7 = self.up2(x6, x3)
        # print('up2 shape is:', x7.shape)
        x8 = self.up3(x7, x2)
        # print('up3 shape is:', x8.shape)
        x9 = self.up4(x8, x1, withReLU=False)
        # print('up4 shape is:', x9.shape)

        x = self.conv3(x9)
        # print('final shape is:', x.shape)

        return x


class downStep(nn.Module):
    def __init__(self, inC, outC):
        super(downStep, self).__init__()
        # todo

        # conv, conv
        self.conv1 = nn.Conv2d(inC, outC, 3)
        self.conv2 = nn.Conv2d(outC, outC, 3)
        #self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(outC)
        self.bn2 = nn.BatchNorm2d(outC)

    def forward(self, x):
        # todo
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        #x_down = x
        #x = self.pool1(x_down)

        return x


class upStep(nn.Module):
    def __init__(self, inC, outC, withReLU=True):
        super(upStep, self).__init__()
        # todo
        # Do not forget to concatenate with respective step in contracting path
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!

        # up-conv; conv; conv
        self.upConv1 = nn.ConvTranspose2d(inC, outC, 2, 2)
        self.conv1 = nn.Conv2d(inC, outC, 3)
        self.conv2 = nn.Conv2d(outC, outC, 3)
        self.bn1 = nn.BatchNorm2d(outC)
        self.bn2 = nn.BatchNorm2d(outC)

    def forward(self, x, x_down, withReLU=True):
        # todo
        x = self.upConv1(x)
        # print('up conv shape:', x.shape)
        # crop x_down
        curr_size_h = x.shape[2]
        curr_size_w = x.shape[3]
        crop_size_h = (x_down.shape[2] - x.shape[2]) // 2
        crop_size_w = (x_down.shape[3] - x.shape[3]) // 2
        x_down = x_down[:, :, crop_size_h:(
            crop_size_h + curr_size_h), crop_size_w:(crop_size_w + curr_size_w)]
        # concatenate
        x = torch.cat((x_down, x), 1)

        if withReLU:
            x = F.relu(self.bn1(self.conv1(x)))
            # print('conv1 shape:', x.shape)
            x = F.relu(self.bn2(self.conv2(x)))
            # print('conv2 shape:', x.shape)
        else:
            x = self.bn1(self.conv1(x))
            # print('conv1 shape:', x.shape)
            x = self.bn2(self.conv2(x))
            # print('conv2 shape:', x.shape)
        return x
