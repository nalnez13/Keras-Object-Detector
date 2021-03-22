import keras
import tqdm
import numpy as np
import cv2
from map.detection_map import DetectionMAP
import glob
import tensorflow as tf

# TODO: SUPPORT MULTI GPU MODEL
# TODO: ITS NOT WORKING NOW
class mAPCallback(keras.callbacks.Callback):
    def __init__(self, valid_set, num_classes, input_shape, output_stride, freq=1):
        super(mAPCallback, self).__init__()
        if valid_set[-4:] == '.txt':
            with open(valid_set, 'r') as f:
                self.data = f.read().splitlines()
        else:
            self.data = glob.glob(valid_set + '/*.jpg')
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.output_stride = output_stride
        self.freq = freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.freq == 0:
            mAP = DetectionMAP(self.num_classes)
            heat_out = self.model.get_layer(name='cls_pred').output
            size_pred = self.model.get_layer(name='size_pred').output
            offset_pred = self.model.get_layer(name='offset_pred').output
            max_out = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(heat_out)
            heat_model = keras.models.Model(inputs=[self.model.input],
                                            outputs=[heat_out, max_out, size_pred, offset_pred])

            oh = int(self.input_shape[0] / self.output_stride)
            ow = int(self.input_shape[1] / self.output_stride)

            for img_file in tqdm.tqdm(self.data):
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
                # prediction
                img_file = gt_file.replace('.txt', '.jpg')
                img = cv2.imread(img_file)
                img_disp, pred_cls, pred_conf, pred_bb = Predict_img(heat_model, img, self.input_shape,
                                                                     (oh, ow, self.num_classes + 4), None, None,
                                                                     draw_bbox=False)

                frame = (
                    np.array(pred_bb), np.array(pred_cls), np.array(pred_conf), np.array(gt_bb), np.array(gt_cls))
                mAP.evaluate(*frame)

            ap_by_class, mAP_point = mAP.compute_map()
            logs['mAP'] = mAP_point
            print('\nmAP: {}'.format(mAP_point))
