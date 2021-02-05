import os
from os.path import isdir, exists, abspath, join

import random

import numpy as np
from PIL import Image, ImageOps


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
            # current += 1

            # todo: load images and labels
            data_image = Image.open(self.data_files[current])
            label_image = Image.open(self.label_files[current])
            # print(data_image.size)
            # print(label_image.size)


            # label_image = label_image.resize((388, 388))
            # hint: if training takes too long or memory overflow, reduce image size!
            data_image = data_image.resize((572, 572))
            label_image = label_image.resize((388, 388))
            # ---------------------- Data augmentation ----------------------
            # Randomly flip the image horizontally or vertically
            rand1 = random.random()
            if rand1 < 0.5:
                data_image = data_image.transpose(Image.FLIP_LEFT_RIGHT)
                label_image = label_image.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                data_image = data_image.transpose(Image.FLIP_TOP_BOTTOM)
                label_image = label_image.transpose(Image.FLIP_TOP_BOTTOM)

            # Zoom images, center of the image
            rand2 = random.randint(3, 9)
            data_shape = data_image.size
            data_image = ImageOps.crop(data_image, data_image.size[1] // rand2)

            data_image = data_image.resize((572, 572))
            label_image = ImageOps.crop(
                label_image, label_image.size[1] // rand2)
            label_image = label_image.resize((388, 388))

            # Rotate images
            rand3 = random.random()
            if rand3 < 0.5:
                data_image = data_image.transpose(Image.ROTATE_90)
                label_image = label_image.transpose(Image.ROTATE_90)
            else:
                data_image = data_image.transpose(Image.ROTATE_270)
                label_image = label_image.transpose(Image.ROTATE_270)

            data_image = np.asarray(data_image, dtype=np.float32)
            data_image = np.divide(data_image, 255.0)

            label_image = np.asarray(label_image, dtype=np.int)
            # print(label_image)
            current += 1

            yield (data_image, label_image)

    def setMode(self, mode):
        self.mode = mode

    def n_train(self):
        data_length = len(self.data_files)
        return np.int_(data_length - np.floor(data_length * self.test_percent))
