import os
from os.path import isdir, exists, abspath, join

import random

import numpy as np
from PIL import Image


class DataLoader():
    def __init__(self, root_dir='data', batch_size=2, test_percent=.1):
        self.batch_size = batch_size
        self.test_percent = test_percent

        self.root_dir = abspath(root_dir)
        self.data_dir = join(self.root_dir, 'scans')
        self.labels_dir = join(self.root_dir, 'labels')

        self.files = os.listdir(self.data_dir)

        self.data_files = [join(self.data_dir, f) for f in self.files]
        self.label_files = [join(self.labels_dir, f) for f in self.files]

    def __iter__(self):
        n_train = self.n_train()

        if self.mode == 'train':
            current = 0
            endId = n_train
        elif self.mode == 'test':
            current = n_train
            endId = len(self.data_files)

        while current < endId:
            current += 1

            # todo: load images and labels
            curr_image = Image.open(self.data_files[current])
            curr_label = Image.open(self.label_files[current])

            # ---------------------- Data augmentation ----------------------
            # Flip the image horizontally
            data_image = curr_image.transpose(Image.FILP_LEFT_RIGHT)
            label_image = curr_label.transpose(Image.FILP_LEFT_RIGHT)

            # Zoom images, center of the image
            width1, height1 = data_image.size
            left1 = width1 / 4
            top1 = height1 / 4
            right1 = 3 * width1 / 4
            bottom1 = 3 * height1 / 4

            width2, height2 = label_image.size
            left2 = width2 / 4
            top2 = height2 / 4
            right2 = 3 * width2 / 4
            bottom2 = 3 * height2 / 4

            data_image = data_image.crop((left1, top1, right1, bottom1))
            data_image = data_image.resize((572, 572))
            label_image = label_image.crop((left2, top2, right2, bottom2))
            label_image = label_image.resize((572, 572))

            # Rotate images
            data_image = data_image.transpose(Image.ROTATE_90)
            label_image = label_image.transpose(Image.ROTATE_90)

            # hint: scale images between 0 and 1
            data_image = np.asarray(data_image)
            data_image /= 255.0

            label_image = np.asarray(label_image)
            label_image /= 255.0
            # hint: if training takes too long or memory overflow, reduce image size!

            yield (data_image, label_image)

    def setMode(self, mode):
        self.mode = mode

    def n_train(self):
        data_length = len(self.data_files)
        return np.int_(data_length - np.floor(data_length * self.test_percent))
