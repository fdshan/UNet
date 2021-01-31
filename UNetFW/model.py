import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        # todo
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        self.down1 = downStep(1, 64)
        self.down2 = downStep(64, 128)
        self.down3 = downStep(128, 256)
        self.down3 = downStep(256, 512)
        self.down4 = downStep(512, 1024)

        self.conv1 = nn.Conv2d(512, 1024, 3)
        self.conv2 = nn.Conv2d(1024, 1024, 3)

        self.up1 = upStep(1024, 512)
        self.up2 = upStep(512, 256)
        self.up3 = upStep(256, 128)
        self.up4 = upStep(128, 64)

        self.conv3 = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        # todo
        x, x_down1 = self.down1(x)
        x, x_down2 = self.down2(x)
        x, x_down3 = self.down3(x)
        x, x_down4 = self.down4(x)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = self.up1(x, x_down4)
        x = self.up2(x, x_down3)
        x = self.up3(x, x_down2)
        x = self.up4(x, x_down1, withReLU=False)

        x = self.conv3(x)

        return x


class downStep(nn.Module):
    def __init__(self, inC, outC):
        super(downStep, self).__init__()
        # todo

        # conv, conv, maxpool
        self.conv1 = nn.Conv2d(inC, outC, 3)
        self.conv2 = nn.Conv2d(inC, outC, 3)
        self.pool1 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # todo
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x_down = x
        x = self.pool1(x)

        return x, x_down


class upStep(nn.Module):
    def __init__(self, inC, outC, withReLU=True):
        super(upStep, self).__init__()
        # todo
        # Do not forget to concatenate with respective step in contracting path
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!

        # up-conv; conv; conv
        self.upConv1 = nn.ConvTranspose2d(inC, outC, 2)
        self.conv2 = nn.Conv2d(inC, outC, 3)
        self.conv3 = nn.Conv2d(inC, outC, 3)

    def forward(self, x, x_down):
        # todo
        x = self.upConv1(x)

        # crop x_down
        curr_size = x.size(2)
        crop_size = (x_down.size(2) - x_size(2)) / 2
        x_down = x_down[:, :, crop_size:(
            crop_size + curr_size), crop_size:(crop_size + curr_size)]

        # concatenate
        x = torch.cat((x_down, x), 1)

        if withReLU == True:
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        else:
            x = self.conv2(x)
            x = self.conv3(x)
        return x
