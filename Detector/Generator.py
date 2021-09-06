import glob
import os
import random
import cv2
import imgaug
import numpy as np
from imgaug import augmenters as iaa
from imgaug.augmentables.batches import UnnormalizedBatch
import keras
from AnchorUtil import AnchorComputer, LayerConfigs
from Utils.print_color import bcolors


class AnchorGenerator(keras.utils.Sequence):
    def __init__(self, batch_size, input_shape, num_classes, anchor_util, data_path, augs, is_train=True,
                 multi_scaling=False, scaling_freq=10, scaling_level=4):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.anchor_util = anchor_util

        print(bcolors.GREEN)
        if data_path[-4:] == '.txt':
            with open(data_path, 'r') as f:
                self.data = f.read().splitlines()
            print("[i] Found {} Data in list file".format(len(self.data)) + bcolors.ENDC)
        else:
            self.data = glob.glob(data_path + '/*.jpg')
            print("[i] Found {} Data in path".format(len(self.data)) + bcolors.ENDC)

        self.multi_scaling = multi_scaling
        if self.multi_scaling:
            assert scaling_level % 2 == 0, 'Multi scaling level must be even step.'
            print(bcolors.GREEN + "[i] Generator is running multi scaling mode".format(len(self.data)) +
                  bcolors.ENDC)
            self.scale_idx = 0
            self.input_multi_scales = self.set_multi_scales(scaling_level)
            self.scaling_freq = scaling_freq

        self.augmenter = iaa.Sequential(augs)
        self.is_train = is_train
        self.indexes = None
        self.on_epoch_end()

    def set_multi_scales(self, scaling_level):
        input_scales = [(self.input_shape[0], self.input_shape[1]), ]
        for s in range(1, scaling_level // 2 + 1):
            input_scales.append((int(self.input_shape[0] + 32 * s),
                                 int(self.input_shape[1] + 32 * s),
                                 self.input_shape[2]))
            input_scales.append((int(self.input_shape[0] - 32 * s),
                                 int(self.input_shape[1] - 32 * s),
                                 self.input_shape[2]))
        return input_scales

    def __getitem__(self, iters):
        if self.multi_scaling:
            self.select_random_scale(iters)

        indexes = self.indexes[iters * self.batch_size: (iters + 1) * self.batch_size]
        data_list = [self.data[i] for i in indexes]
        x, y = self.__generate_data(data_list)
        return x, y

    def select_random_scale(self, iters):
        if (iters + 1) % self.scaling_freq == 0:
            scale_idx = self.scale_idx
            while self.scale_idx == scale_idx:
                scale_idx = random.randint(0, len(self.input_multi_scales) - 1)
            self.input_shape = self.input_multi_scales[scale_idx]
            self.anchor_util.calculate_priors(self.input_shape)

    def __generate_data(self, data_list):
        cv2.setNumThreads(0)
        batch_images = np.zeros(shape=(self.batch_size, self.input_shape[0], self.input_shape[1], 3),
                                dtype=np.float32)
        batch_y = np.zeros(shape=(self.batch_size, self.anchor_util.num_anchors, self.num_classes + 4),
                           dtype=np.float32)

        imgs = []
        bounding_boxes = []
        for img_file in data_list:
            img = cv2.imread(img_file)
            img = cv2.resize(img, dsize=(self.input_shape[1], self.input_shape[0]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
            bbox_on_img = self.__read_gt_boxes(img_file.replace('.jpg', '.txt'))
            bounding_boxes.append(bbox_on_img)

        # Data Augment
        batch = UnnormalizedBatch(images=imgs, bounding_boxes=bounding_boxes)
        augmented_data = list(self.augmenter.augment_batches(batch, background=False))

        for i in range(len(data_list)):
            img = augmented_data[0].images_aug[i]
            img = img.astype(np.float) / 255.
            batch_images[i] = img
            bboxes_aug = augmented_data[0].bounding_boxes_aug[i].remove_out_of_image_fraction(0.9).clip_out_of_image_()
            bboxes = bboxes_aug.bounding_boxes
            batch_y[i] = self.anchor_util.assign_anchors(bboxes, self.input_shape)
            # test
            # r = self.anchor_util.postprocess(batch_y[i])
            # for d in r:
            #     x1, y1, x2, y2, cls, conf = d
            #     img = cv2.rectangle(img, (int(x1 * self.input_shape[1]), int(y1 * self.input_shape[1])),
            #                         (int(x2 * self.input_shape[0]), int(y2 * self.input_shape[0])), (255, 255, 255))
            # cv2.imshow('test', img)
            # cv2.waitKey(0)

        # input_anchors = np.expand_dims(self.anchor_util.anchors, axis=0)
        # input_anchors = np.repeat(input_anchors, self.batch_size, axis=0)
        return batch_images, batch_y

    def __read_gt_boxes(self, annotation_file):
        boxes_per_img = []
        if os.path.isfile(annotation_file):
            with open(annotation_file, 'r') as annotation_data:
                annotations = annotation_data.read().splitlines()
                for annot in annotations:
                    annot = annot.split(' ')
                    class_id = int(annot[0])
                    cx = float(annot[1]) * self.input_shape[1]
                    cy = float(annot[2]) * self.input_shape[0]
                    w = float(annot[3]) * self.input_shape[1]
                    h = float(annot[4]) * self.input_shape[0]
                    boundingbox = imgaug.BoundingBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, label=class_id)
                    boxes_per_img.append(boundingbox)

        return imgaug.BoundingBoxesOnImage(boxes_per_img, shape=self.input_shape)

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.is_train:
            np.random.shuffle(self.indexes)


if __name__ == '__main__':
    import AnchorPresets
    augments = [iaa.SomeOf((0, 3),
                           [
                               iaa.Identity(),
                               iaa.Sharpen(),
                               iaa.GammaContrast(),
                               iaa.MultiplyHueAndSaturation(mul=(0.8, 1.5)),
                               iaa.MultiplyAndAddToBrightness(),
                           ]),
                iaa.Fliplr(p=0.5),
                iaa.Sometimes(0.3, iaa.Grayscale(alpha=(0.1, 1.0))),
                iaa.Sometimes(0.2, iaa.Dropout(p=(0.01, 0.1))),
                iaa.Sometimes(0.3, iaa.GammaContrast(gamma=(0.3, 1.2))),  # Gamma Distortion
                iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0.0, 1.0))),  # Gaussian Blur
                iaa.SomeOf((0, 3),
                           [iaa.Sometimes(0.5,
                                          iaa.CropAndPad(percent=(-0.2, 0.2), keep_size=True, pad_mode=imgaug.ALL)),
                            iaa.Sometimes(0.5, iaa.Affine(scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                                                          mode=imgaug.ALL)),
                            iaa.Sometimes(0.5, iaa.PerspectiveTransform(scale=(0.05, 0.05), mode=imgaug.ALL))])
                ]
    c = AnchorPresets.default_config
    anchor = AnchorComputer((320, 320, 3), 20, c)
    gen = AnchorGenerator(batch_size=16, input_shape=(256, 256), num_classes=20, anchor_util=anchor,
                          data_path='E:/FSNet2/Datasets/voc_train.txt', augs=augments)

    for i in range(gen.__len__()):
        print(i)
        gen.__getitem__(i)
