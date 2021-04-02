import glob
import math
import os
import random

import cv2
import imgaug
import numpy as np
import tqdm
from imgaug import augmenters as iaa
from imgaug.augmentables.batches import UnnormalizedBatch
from tensorflow import keras

from Utils.print_color import bcolors


class ILSVRC_Generator(keras.utils.Sequence):
    def __init__(self, batch_size, input_shape, num_classes, data_path, augs, is_train=True):
        """

        :param batch_size:
        :param input_shape:
        :param num_classes:
        :param data_path:
        :param augs:
        :param is_train:
        """
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.is_train = is_train

        self.label_list = []
        dirs = glob.glob(data_path + '/train/*')
        for d in dirs:
            self.label_list.append(d.split(os.sep)[-1])
        if is_train:
            self.data = glob.glob(data_path + '/train/*/*.JPEG')
            self.train_list = dict()
            for data in self.data:
                label = data.split(os.sep)[-2]
                self.train_list[data] = self.label_list.index(label)

        else:
            self.data = glob.glob(data_path + '/val/*/*.JPEG')
            self.val_list = dict()
            for data in self.data:
                label = data.split(os.sep)[-2]
                self.val_list[data] = self.label_list.index(label)
        self.augmenter = iaa.Sequential(augs)

        self.indexes = None
        self.step = 0
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, item):
        indexes = self.indexes[item * self.batch_size: (item + 1) * self.batch_size]
        data_list = [self.data[i] for i in indexes]
        x, y = self.__data_gen(data_list)
        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.is_train:
            np.random.shuffle(self.indexes)

    def __data_gen(self, data_list):
        cv2.setNumThreads(0)
        batch_images = np.zeros(shape=(self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                                dtype=np.float32)
        batch_cls = np.zeros(shape=(self.batch_size, self.num_classes), dtype=np.float32)

        imgs = []
        cls = []
        for img_file in data_list:
            img = cv2.imread(img_file)
            img = cv2.resize(img, dsize=(self.input_shape[1], self.input_shape[0]))
            imgs.append(img)
            # cls append
            if self.is_train:
                label = self.train_list[img_file]
            else:
                label = self.val_list[img_file]
            cls.append(label)

        batch = UnnormalizedBatch(images=imgs, data=cls)
        augmented_data = list(self.augmenter.augment_batches(batch, background=False))

        for i in range(len(data_list)):
            img = augmented_data[0].images_aug[i]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float) / 255.
            label = augmented_data[0].data[i]
            batch_images[i] = img
            # print(label, data_list[i])
            label = keras.utils.to_categorical(label, num_classes=self.num_classes)
            batch_cls[i] = label
            # cv2.imshow('test', img)
            # cv2.imshow('unaug', augmented_data[0].images_unaug[i])
            # cv2.waitKey(0)

        return batch_images, batch_cls


class Tiny_imagenet_Generator(keras.utils.Sequence):
    def __init__(self, batch_size, input_shape, data_path, augs, num_classes=200, is_train=True, label_smoothing=False):
        """

        :param batch_size:
        :param input_shape:
        :param output_stride:
        :param num_classes:
        :param data_path:
        :param augs:
        :param is_train:
        """
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.is_train = is_train
        self.label_smoothing = label_smoothing
        self.eps = 0.1
        self.augmenter = iaa.Sequential(augs)

        with open(data_path + '/wnids.txt', 'r') as f:
            self.label_list = f.read().splitlines()
        if is_train:
            self.data = glob.glob(data_path + '/train/*/images/*.JPEG')
            self.train_list = dict()
            for data in self.data:
                label = data.split(os.sep)[-3]
                self.train_list[data] = self.label_list.index(label)

        else:
            self.data = glob.glob(data_path + '/val/images/*.JPEG')
            self.val_list = dict()
            with open(data_path + '/val/val_annotations.txt', 'r') as f:
                val_labels = f.read().splitlines()
                for label in val_labels:
                    f_name, label, _, _, _, _ = label.split('\t')
                    self.val_list[f_name] = self.label_list.index(label)

        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, item):
        indexes = self.indexes[item * self.batch_size: (item + 1) * self.batch_size]
        data_list = [self.data[i] for i in indexes]
        x, y = self.__data_gen(data_list)
        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.is_train:
            np.random.shuffle(self.indexes)

    def __data_gen(self, data_list):
        cv2.setNumThreads(0)
        batch_images = np.zeros(shape=(self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                                dtype=np.float32)

        if self.label_smoothing:
            batch_cls = np.ones(shape=(self.batch_size, self.num_classes), dtype=np.float32) * self.eps / (
                    self.num_classes - 1)
        else:
            batch_cls = np.zeros(shape=(self.batch_size, self.num_classes), dtype=np.float32)

        imgs = []
        cls = []
        for img_file in data_list:
            img = cv2.imread(img_file)
            img = cv2.resize(img, dsize=(self.input_shape[1], self.input_shape[0]))
            imgs.append(img)
            # cls append
            if self.is_train:
                label = self.train_list[img_file]
            else:
                label = self.val_list[os.path.basename(img_file)]
            cls.append(label)

        batch = UnnormalizedBatch(images=imgs, data=cls)
        augmented_data = list(self.augmenter.augment_batches(batch, background=False))

        for i in range(len(data_list)):
            img = augmented_data[0].images_aug[i]
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float) / 255.
            label = augmented_data[0].data[i]
            batch_images[i] = img
            # print(label, data_list[i])
            if self.label_smoothing:
                batch_cls[i, label] = 1.0 - self.eps
            else:
                label = keras.utils.to_categorical(label, num_classes=self.num_classes)
                batch_cls[i] = label
            # cv2.imshow('test', img)
            # cv2.imshow('unaug', augmented_data[0].images_unaug[i])
            # cv2.waitKey(0)

        return batch_images, batch_cls


class ILSVRC_Generator(keras.utils.Sequence):
    def __init__(self, batch_size, input_shape, data_path, augs, is_train=True):
        """

        :param batch_size:
        :param input_shape:
        :param num_classes:
        :param data_path:
        :param augs:
        :param is_train:
        """
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_classes = 1000
        self.is_train = is_train

        self.label_list = []
        dirs = glob.glob(data_path + '/train/*')
        for d in dirs:
            self.label_list.append(d.split(os.sep)[-1])
        if is_train:
            self.data = glob.glob(data_path + '/train/*/*.JPEG')
            self.train_list = dict()
            for data in self.data:
                label = data.split(os.sep)[-2]
                self.train_list[data] = self.label_list.index(label)

        else:
            self.data = glob.glob(data_path + '/val/*/*.JPEG')
            self.val_list = dict()
            for data in self.data:
                label = data.split(os.sep)[-2]
                self.val_list[data] = self.label_list.index(label)
        self.augmenter = iaa.Sequential(augs)

        self.indexes = None
        self.step = 0
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, item):
        indexes = self.indexes[item * self.batch_size: (item + 1) * self.batch_size]
        data_list = [self.data[i] for i in indexes]
        x, y = self.__data_gen(data_list)
        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.is_train:
            np.random.shuffle(self.indexes)

    def __data_gen(self, data_list):
        cv2.setNumThreads(0)
        batch_images = np.zeros(shape=(self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                                dtype=np.float32)
        batch_cls = np.zeros(shape=(self.batch_size, self.num_classes), dtype=np.float32)

        imgs = []
        cls = []
        for img_file in data_list:
            img = cv2.imread(img_file)
            img = cv2.resize(img, dsize=(self.input_shape[1], self.input_shape[0]))
            imgs.append(img)
            # cls append
            if self.is_train:
                label = self.train_list[img_file]
            else:
                label = self.val_list[img_file]
            cls.append(label)

        batch = UnnormalizedBatch(images=imgs, data=cls)
        augmented_data = list(self.augmenter.augment_batches(batch, background=False))

        for i in range(len(data_list)):
            img = augmented_data[0].images_aug[i]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float) / 255.
            label = augmented_data[0].data[i]
            batch_images[i] = img
            # print(label, data_list[i])
            label = keras.utils.to_categorical(label, num_classes=self.num_classes)
            batch_cls[i] = label
            # cv2.imshow('test', img)
            # cv2.imshow('unaug', augmented_data[0].images_unaug[i])
            # cv2.waitKey(0)

        return batch_images, batch_cls


import re


class LPDGenerator(keras.utils.Sequence):
    def __init__(self, batch_size, input_shape, data_path, augs, num_classes=1000, is_train=True,
                 label_smoothing=False):
        """

        :param batch_size:
        :param input_shape:
        :param output_stride:
        :param num_classes:
        :param data_path:
        :param augs:
        :param is_train:
        """
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.is_train = is_train
        self.label_smoothing = label_smoothing
        self.eps = 0.1
        self.augmenter = iaa.Sequential(augs)
        self.data = glob.glob(data_path + "/**/*.jpg", recursive=True)
        self.parsing = re.compile("/\d*/")

        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, item):
        indexes = self.indexes[item * self.batch_size: (item + 1) * self.batch_size]
        data_list = [self.data[i] for i in indexes]
        x, y = self.__data_gen(data_list)
        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.is_train:
            np.random.shuffle(self.indexes)

    def __data_gen(self, data_list):
        cv2.setNumThreads(0)
        batch_image = np.zeros(shape=(self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                               dtype=np.float32)
        if self.label_smoothing:
            batch_label = np.ones(shape=(self.batch_size, self.num_classes), dtype=np.float32) * self.eps / (
                    self.num_classes - 1)
        else:
            batch_label = np.zeros(shape=(self.batch_size, self.num_classes), dtype=np.float32)
        img_list = []
        label_list = []
        for img_path in data_list:
            img_path = img_path.replace("\\", "/")
            label = int(self.parsing.findall(img_path)[0].replace("/", ""))
            img = cv2.imread(img_path)
            img_list.append(img)
            label_list.append(label)
        batch = UnnormalizedBatch(images=img_list, data=label_list)
        augmented_data = list(self.augmenter.augment_batches(batch, background=False))

        for i in range(len(data_list)):
            img = augmented_data[0].images_aug[i]
            label = augmented_data[0].data[i]
            # cv2.imshow("img", img)
            # cv2.waitKey()
            # img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
            img = img.astype(np.float) / 255.
            batch_image[i] = img
            if self.label_smoothing:
                batch_label[i, label] = 1.0 - self.eps
            else:
                batch_label[i, label] = 1
        return batch_image, batch_label


if __name__ == '__main__':
    augments = [iaa.SomeOf((0, 3),
                           [
                               iaa.Identity(),
                               iaa.Rotate((-5, 5)),
                               iaa.Sharpen(),
                               iaa.TranslateX(),
                               iaa.GammaContrast(),
                               iaa.ShearX((-5, 5)),
                               iaa.TranslateY(),
                               iaa.MultiplyHueAndSaturation(mul=(0.8, 1.5)),
                               iaa.MultiplyAndAddToBrightness(),
                               iaa.ShearY((-5, 5))
                           ]),
                iaa.Fliplr(p=0.5),
                iaa.Sometimes(0.3, iaa.Grayscale(alpha=(0.1, 1.0))),
                iaa.Sometimes(0.2, iaa.Dropout(p=(0.01, 0.1))),
                iaa.Sometimes(0.3, iaa.GammaContrast(gamma=(0.3, 1.2))),  # Gamma Distortion
                iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0.0, 1.0))),  # Gaussian Blur
                iaa.SomeOf((0, 3),
                           [iaa.Sometimes(0.5, iaa.CropAndPad(percent=(-0.2, 0.2), keep_size=True)),
                            iaa.Sometimes(0.5, iaa.Affine(scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                                                          order=[0, 1])),
                            iaa.Sometimes(0.5, iaa.PerspectiveTransform(scale=(0.05, 0.2)))])
                ]

    bgen = LPDGenerator(32, (256, 256, 3), 'E:/FSNet2/Datasets/LPD_competition/train', [])
    while True:
        for i in tqdm.tqdm(range(bgen.__len__())):
            bgen.__getitem__(i)
        bgen.on_epoch_end()
