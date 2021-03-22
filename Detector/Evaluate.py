import keras
import cv2
import numpy as np
import tqdm
from models import Backbones, Head
from Utils.anchor_utils import AnchorUtils
from Detector.anchor_configs import voc_configs
from map.detection_map import DetectionMAP
import matplotlib.pyplot as plt

keras.backend.set_learning_phase(0)

input_shape = (320, 320, 3)
NUM_CLASSES = 20
names = ['aeroplane',
         'bicycle',
         'bird',
         'boat',
         'bottle',
         'bus',
         'car',
         'cat',
         'chair',
         'cow',
         'diningtable',
         'dog',
         'horse',
         'motorbike',
         'person',
         'pottedplant',
         'sheep',
         'sofa',
         'train',
         'tvmonitor']

anchor_utils = AnchorUtils(input_shape, NUM_CLASSES, voc_configs)
model, s1, s2, s3, s4, s5, s6 = Backbones.GhostNet_CRELU_CSP(input_shape=input_shape, weight_decay=0.0001)
detector = Head.SubNet(model, s3, s4, s5, s6, num_classes=NUM_CLASSES,
                       num_anchors_per_layer=voc_configs['num_anchors_per_layer'], weight_decay=0.0001)
detector.load_weights('saved_models/voc-GhostNet_CRELU_CSP_Normal-00150.h5')

with open("../Datasets/voc_valid.txt", 'r') as f:
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
    img_disp = np.copy(img)
    im_h, im_w, _ = img_disp.shape
    img = cv2.resize(img, input_shape[:2])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
        img_disp = cv2.rectangle(img_disp, (int(x1 * im_w), int(y1 * im_h)), (int(x2 * im_w), int(y2 * im_h)),
                                 (0, 255, 0), 2)
        img_disp = cv2.putText(img_disp, names[int(cls)], (int(x1 * im_w), int(y1 * im_h)), cv2.FONT_HERSHEY_SIMPLEX,
                               0.8, (0, 255, 0),2)
    # cv2.imshow('test', img_disp)
    # cv2.waitKey(0)

    frame = (
        np.array(pred_bb), np.array(pred_cls), np.array(pred_conf), np.array(gt_bb), np.array(gt_cls))
    mAP.evaluate(*frame)

ap_by_class, mAP_point = mAP.compute_map()
print(ap_by_class)
print(mAP_point)
mAP.plot()
plt.show()
