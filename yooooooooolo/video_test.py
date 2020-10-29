from yooooooooolo import darknet, util
import cv2
import glob
import numpy as np
from Utils.anchor_utils import AnchorUtils
from default_config import cluster_configs
from models import Backbones, Head

net1, meta1 = util.load_darknet('FSFC_224_A12.cfg', 'FSFC_224_A12_best.weights', 'fsnet_previous.data')
net1_w, net1_h = darknet.network_width(net1), darknet.network_height(net1)
net1_in = darknet.make_image(net1_w, net1_h, 3)

input_size = 256
NUM_CLASSES = 2
anchors_per_layer = 3
anchor_utils = AnchorUtils((input_size, input_size, 3), NUM_CLASSES, cluster_configs)
backbone, s1, s2, s3, s4, s5, s6 = Backbones.v4_tiny(input_shape=(input_size, input_size, 3))
model = Head.SubNet(backbone, s4, s6, NUM_CLASSES, anchors_per_layer)
model.load_weights('../saved_models/v4_subnet-00735.h5')

videos = glob.glob('//192.168.0.3/videoDrive/_VideoData/MITAC/20201013_day/front/*.MP4')

for v in videos:
    cap = cv2.VideoCapture(v)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))
        disp_1 = np.copy(frame)
        disp_2 = np.copy(frame)
        h, w, c = disp_1.shape
        im_h, im_w, _ = frame.shape
        frame_in = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resizeW = 960
        resizeH = 544
        frame_in1 = cv2.resize(frame_in, (resizeW, resizeH))
        crop = frame_in1[109:401, 0:960]
        cv2.imshow("crop", crop)
        crop_height, crop_width, _ = crop.shape
        crop_resized = cv2.resize(crop, (net1_w, net1_h), interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(net1_in, crop_resized.tobytes())

        detections1 = darknet.detect_image(net1, meta1, net1_in)

        frame_in_k_model = cv2.resize(crop, (256, 256))
        frame_in_k_model = np.expand_dims(frame_in_k_model, 0).astype(np.float32) / 255.
        prediction = model.predict(frame_in_k_model)[0]

        loc_pred = prediction[:, :4]
        conf_pred = prediction[:, 4:]
        r = anchor_utils.postprocess_detections(prediction, .4)
        for d in r:
            x1, y1, x2, y2, cls, conf = d
            x1 = x1 * resizeW
            x2 = x2 * resizeW
            y1 = y1 * 296 + 109
            y2 = y2 * 296 + 109

            x1 = int(x1 / 960. * 1280)
            x2 = int(x2 / 960. * 1280)
            y1 = int(y1 / 544. * 720)
            y2 = int(y2 / 544. * 720)
            disp_1 = cv2.rectangle(disp_1, (int(x1), int(y1)), (int(x2), int(y2)),
                                   (0, 255, 0), 3)

        for det in detections1:
            cls = det[0].decode()
            cx, cy, w, h = det[2]
            xmin = int((cx - (w / 2)) * resizeW / net1_w)
            ymin = int((cy - (h / 2)) * 296 / net1_h) + 109
            xmax = int((cx + (w / 2)) * resizeW / net1_w)
            ymax = int((cy + (h / 2)) * 296 / net1_h) + 109

            xmin = int(xmin / 960. * 1280)
            xmax = int(xmax / 960. * 1280)
            ymin = int(ymin / 544. * 720)
            ymax = int(ymax / 544. * 720)

            disp_1 = cv2.rectangle(disp_1, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)

        cv2.imshow('pred1', disp_1)
        # cv2.imshow('pred_k', disp_2)
        key = cv2.waitKey(10)
        if key == 27:
            break
