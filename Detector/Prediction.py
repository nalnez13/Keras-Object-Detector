import keras
import glob
import cv2
import numpy as np
from models import Backbones, Head
from Utils.anchor_utils import AnchorUtils

keras.backend.set_learning_phase(0)

input_size = 256
NUM_CLASSES = 2
anchors_per_layer = 3
anchor_utils = AnchorUtils((input_size, input_size, 3), NUM_CLASSES, cluster_configs)
backbone, s1, s2, s3, s4, s5, s6 = Backbones.v4_tiny(input_shape=(input_size, input_size, 3))
model = Head.SubNet(backbone, s4, s6, NUM_CLASSES, anchors_per_layer)
model.load_weights('saved_models/v4_subnet-00735.h5')

# img_files = glob.glob('E:/bdd100k/time_parsed/day/val/image/*.jpg')
vis = glob.glob('//192.168.0.3/videoDrive/_VideoData/MITAC/20201030_day/front/*.MP4')
for v in vis:
    cap = cv2.VideoCapture(v)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # while True:
        #     for img in tqdm.tqdm(img_files):
        #         frame = cv2.imread(img)
        frame = cv2.resize(frame, (1280, 720))
        img_disp = np.copy(frame)
        h, w, c = img_disp.shape
        frame = cv2.resize(frame, (input_size, input_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('img', frame)
        frame = np.expand_dims(frame, 0).astype(np.float32) / 255.

        prediction = model.predict(frame)[0]
        loc_pred = prediction[:, :4]
        conf_pred = prediction[:, 4:]
        r = anchor_utils.fast_postprocess(prediction, .4)
        for d in r:
            x1, y1, x2, y2, cls, conf = d
            # print(conf)
            conf_str = '%.3f' % conf
            if cls == 0:
                img_disp = cv2.putText(img_disp, conf_str, (int(x1 * w), int(y1 * h - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                       0.8,
                                       (0, 255, 0), 1, cv2.LINE_AA)
                img_disp = cv2.rectangle(img_disp, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)),
                                         (0, 255, 0), 2)
            else:
                img_disp = cv2.rectangle(img_disp, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)),
                                         (255, 255, 0), 2)
                img_disp = cv2.putText(img_disp, conf_str, (int(x1 * w), int(y1 * h - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                       0.8,
                                       (255, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('test', img_disp)
        key = cv2.waitKey(10)
        if key == 27:
            break
