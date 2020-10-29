import glob
import os
import cv2
import imgaug
import numpy as np
from imgaug import augmenters as iaa
from imgaug.augmentables.batches import UnnormalizedBatch
import keras
from Utils.anchor_utils import AnchorUtils
from Utils.print_color import bcolors


class AnchorGenerator(keras.utils.Sequence):
    def __init__(self, batch_size, input_shape, num_classes, anchor_config, data_path, augs, is_train=True):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.anchor_utils = AnchorUtils(self.input_shape, self.num_classes, anchor_config)
        if data_path[-4:] == '.txt':
            with open(data_path, 'r') as f:
                self.data = f.read().splitlines()
            print("[i] Found {} Data in list file".format(len(self.data)) + bcolors.ENDC)
        else:
            self.data = glob.glob(data_path + '/*.jpg')
            print("[i] Found {} Data in path".format(len(self.data)) + bcolors.ENDC)
        self.augmenter = iaa.Sequential(augs)
        self.is_train = is_train
        self.indexes = None
        self.on_epoch_end()

    def __getitem__(self, item):
        indexes = self.indexes[item * self.batch_size: (item + 1) * self.batch_size]
        data_list = [self.data[i] for i in indexes]
        x, y = self.__data_gen(data_list)
        return x, y

    def __data_gen(self, data_list):
        cv2.setNumThreads(0)
        batch_images = np.zeros(shape=(self.batch_size, self.input_shape[0], self.input_shape[1], 3),
                                dtype=np.float32)
        batch_y = np.zeros(
            shape=(self.batch_size, self.anchor_utils.num_anchors, 4 + self.num_classes),
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
            y = self.anchor_utils.assign_anchors(bboxes)
            batch_y[i] = y
            # test
            # r = self.anchor_utils.postprocess_detections(y)
            # for d in r:
            #     x1, y1, x2, y2, cls, conf = d
            #     print(d)
            #     img = cv2.rectangle(img, (int(x1 * self.input_shape[1]), int(y1 * self.input_shape[1])),
            #                         (int(x2 * self.input_shape[0]), int(y2 * self.input_shape[0])), (255, 255, 255))
            # cv2.imshow('test', img)
            # cv2.waitKey(0)

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
    from default_config import cluster_configs

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

    gen = AnchorGenerator(batch_size=1, input_shape=(256, 256), num_classes=3, anchor_config=cluster_configs,
                          data_path='E:/bdd100k/yolo_parsed/day/val.txt', augs=augments)

    for i in range(gen.__len__()):
        gen.__getitem__(i)
