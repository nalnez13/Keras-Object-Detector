import keras
import glob
import cv2
import numpy as np
import tqdm
import os
from models import Backbones, Head
from Utils.anchor_utils import AnchorUtils
from default_config import cluster_configs
from map.detection_map import DetectionMAP
import matplotlib.pyplot as plt

keras.backend.set_learning_phase(0)

input_shape = (256, 256, 3)
NUM_CLASSES = 2

anchor_utils = AnchorUtils(input_shape, NUM_CLASSES, cluster_configs)
model, s1, s2, s3, s4, s5, s6 = Backbones.v4_tiny(input_shape=input_shape, weight_decay=0.0001)
detector = Head.SubNet(model, s4, s6, num_classes=NUM_CLASSES, num_anchors_per_layer=3, weight_decay=0.0001)
detector.load_weights('saved_models/v4_subnet-00735.h5')

with open("valid.txt", 'r') as f:
    img_files = f.read().splitlines()

mAP = DetectionMAP(NUM_CLASSES)

for img_file in tqdm.tqdm(img_files):
    gt_file = img_file.replace('.jpg', '.txt')
    gt_data = open(gt_file, 'r')
    lines = gt_data.read().splitlines()

    gt_cls = []
    gt_bb = []
    for line in lines:
        cid, cx, cy, w, h = list(map(float, line.split(' ')[:5]))
        cid = int(cid)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        gt_cls.append(cid)
        gt_bb.append([x1, y1, x2, y2])
    gt_data.close()
    if len(gt_cls) == 0:
        continue
    img = cv2.imread(img_file)
    img = cv2.resize(img, input_shape[:2])
    img = np.expand_dims(img, 0).astype(np.float32) / 255.
    prediction = detector.predict(img)[0]
    detections = anchor_utils.fast_postprocess(prediction, 0.01)
    pred_cls = []
    pred_bb = []
    pred_conf = []
    for x1, y1, x2, y2, cls, conf in detections:
        pred_bb.append([x1, y1, x2, y2])
        pred_cls.append(cls)
        pred_conf.append(conf)

    frame = (
        np.array(pred_bb), np.array(pred_cls), np.array(pred_conf), np.array(gt_bb), np.array(gt_cls))
    mAP.evaluate(*frame)

ap_by_class, mAP_point = mAP.compute_map()
print(ap_by_class)
print(mAP_point)
mAP.plot()
plt.show()
