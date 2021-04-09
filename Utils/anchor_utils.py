import numpy as np

"""
DEPRECATED
"""


class AnchorUtils:
    def __init__(self, input_shape, num_classes, config, iou_thresh=0.5, nms_thres=0.6, k=9):
        """

        :param input_shape: (H, W, C) Tuple
        :param config: Anchor Parameter Dictionary
        """
        self.input_shape = input_shape
        self.config = config
        self.iou_thresh = iou_thresh
        self.nms_thres = nms_thres
        self.anchor_config = self.config['anchor_configs']
        self.num_classes = num_classes
        self.variance = [0.1, 0.1, 0.2, 0.2]

        self.anchors, self.num_anchors_per_layers = self.__create_anchor_boxes()
        self.num_anchors = len(self.anchors)
        self.__calc_anchors_area()

        self.prior_width = self.anchors[:, 2] - self.anchors[:, 0]
        self.prior_height = self.anchors[:, 3] - self.anchors[:, 1]
        self.prior_center_x = 0.5 * (self.anchors[:, 2] + self.anchors[:, 0])
        self.prior_center_y = 0.5 * (self.anchors[:, 3] + self.anchors[:, 1])
        self.k = k

    def __create_anchor_boxes(self):
        anchor_params = []
        num_anchors_per_layers = []
        num_anchors = self.config['num_anchors_per_layer']
        for layer_config in self.anchor_config:
            layer_width, layer_height = layer_config['layer_width'], layer_config['layer_height']
            aspect_ratios = layer_config["aspect_ratios"]
            min_size = layer_config["min_size"]
            max_size = layer_config["max_size"]
            step_x = float(self.input_shape[1]) / float(layer_width)
            step_y = float(self.input_shape[0]) / float(layer_height)

            linx = np.linspace(0.5 * step_x, self.input_shape[1] - 0.5 * step_x, layer_width)
            liny = np.linspace(0.5 * step_y, self.input_shape[0] - 0.5 * step_y, layer_height)
            centers_x, centers_y = np.meshgrid(linx, liny)
            centers_x = centers_x.reshape(-1, 1)
            centers_y = centers_y.reshape(-1, 1)
            prior_boxes = np.concatenate((centers_x, centers_y), axis=1)
            prior_boxes = np.tile(prior_boxes, (1, 2 * num_anchors))  # (H*W, num_anchors*4)

            box_widths = []
            box_heights = []
            for ar in aspect_ratios:
                if ar == 1 and len(box_widths) == 0:
                    box_widths.append(min_size)
                    box_heights.append(min_size)
                elif ar == 1 and len(box_widths) > 0:
                    box_widths.append(np.sqrt(min_size * max_size))
                    box_heights.append(np.sqrt(min_size * max_size))
                elif ar != 1:
                    box_widths.append(min_size * np.sqrt(ar))
                    box_heights.append(min_size / np.sqrt(ar))
            box_widths = 0.5 * np.array(box_widths)
            box_heights = 0.5 * np.array(box_heights)

            # Normalize to 0-1
            prior_boxes[:, ::4] -= box_widths
            prior_boxes[:, 1::4] -= box_heights
            prior_boxes[:, 2::4] += box_widths
            prior_boxes[:, 3::4] += box_heights
            prior_boxes[:, ::2] /= self.input_shape[1]
            prior_boxes[:, 1::2] /= self.input_shape[0]
            prior_boxes = prior_boxes.reshape(-1, 4)  # (H*W*num_anchors, x1,y1,x2,y2)
            # clip to 0-1
            prior_boxes = np.minimum(np.maximum(prior_boxes, 0.0), 1.0)
            anchor_params.append(prior_boxes)
            num_anchors_per_layers.append(len(prior_boxes))
        anchors = np.concatenate(anchor_params, axis=0)
        return anchors, num_anchors_per_layers

    def __calc_anchors_area(self):
        self.anchor_area = (self.anchors[:, 2] - self.anchors[:, 0]) * (self.anchors[:, 3] - self.anchors[:, 1])

    def postprocess_detections(self, predictions, confidence_threshold=0.5):
        """
        prediciont decode 함수
        :param predictions: network output
        :param confidence_threshold:
        :return: decoded output
                 results = [batch, num_boxes, 6]
                 results[i,j] = [xmin, ymin, xmax, ymax, cid, conf.]
                 0 ~ 1.0 normalized value.
        """
        mbox_loc = predictions[:, :4]
        mbox_conf = predictions[:, 4:]
        results = []
        print('__decode_boxes')
        decode_box = self.__decode_boxes(mbox_loc)
        for c_idx in range(self.num_classes):
            class_conf = mbox_conf[:, c_idx]
            c_conf_mask = class_conf > confidence_threshold
            if len(class_conf[c_conf_mask]) > 0:
                conf = class_conf[c_conf_mask]
                class_loc = decode_box[c_conf_mask]
                for j in range(len(class_loc)):
                    loc = class_loc[j]
                    conf_obj = conf[j]
                    tmp = np.append(loc, c_idx)
                    tmp = np.append(tmp, conf_obj)
                    results.append(tmp)

        # Non-maximum-suppresion
        print('__non_maximum_suppression', len(results))
        results = self.__non_maximum_suppression(np.array(results))
        print('AFTER', len(results))

        return results

    def fast_postprocess(self, predictions, confidence_threshold=0.5):
        """
        prediciont decode 함수
        :param predictions: network output
        :param confidence_threshold:
        :return: decoded output
                 results = [batch, num_boxes, 6]
                 results[i,j] = [xmin, ymin, xmax, ymax, cid, conf.]
                 0 ~ 1.0 normalized value.
        """
        mbox_loc = predictions[:, :4]
        mbox_conf = predictions[:, 4:]
        results = []
        decode_box = self.__decode_boxes(mbox_loc)
        for c_idx in range(self.num_classes):
            picks = []
            class_conf = mbox_conf[:, c_idx]
            c_conf_mask = class_conf > confidence_threshold
            masked_conf = class_conf[c_conf_mask]
            masked_loc = decode_box[c_conf_mask]

            idxs = np.argsort(masked_conf)
            x1 = masked_loc[:, 0]
            y1 = masked_loc[:, 1]
            x2 = masked_loc[:, 2]
            y2 = masked_loc[:, 3]
            area = (x2 - x1) * (y2 - y1)

            while len(idxs) > 0:
                last = len(idxs) - 1
                i = idxs[last]
                picks.append(i)
                x1_max = np.maximum(x1[i], x1[idxs[:last]])
                y1_max = np.maximum(y1[i], y1[idxs[:last]])
                x2_min = np.minimum(x2[i], x2[idxs[:last]])
                y2_min = np.minimum(y2[i], y2[idxs[:last]])

                w = np.maximum(x2_min - x1_max, 0)
                h = np.maximum(y2_min - y1_max, 0)
                overlap = (w * h) / area[idxs[:last]]
                idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > self.nms_thres)[0])))

            picks = np.array(picks)

            if len(picks) > 0:
                nms_confs = masked_conf[picks]
                nms_locs = masked_loc[picks]
                for i in range(len(picks)):
                    box = np.append(nms_locs[i], c_idx)
                    box = np.append(box, nms_confs[i])
                    results.append(box)

        return results

    def __non_maximum_suppression(self, boxes):
        """
        decode 결과에 nms 적용 함수
        :param decoded_boxes: decoded output. [batch, num_boxes, 6]
                              decoded_boxes[i,j] = [xmin, ymin, xmax, ymax, cid, conf.]
        :return: nms 적용된 output.
                 results = [batch, num_boxes, 6]
                 results[i,j] = [xmin, ymin, xmax, ymax, cid, conf.]
        """
        # confidence sort
        if len(boxes) == 0:
            return np.array([])
        sorted_idx = np.argsort(boxes[:, -1])[::-1]
        boxes = boxes[sorted_idx]
        for j in range(len(boxes)):
            for k in range(j + 1, len(boxes)):
                boxa = boxes[j]
                boxb = boxes[k]
                if boxa[4] != boxb[4]:
                    continue
                if self.calc_iou(boxa, boxb) > self.nms_thres:
                    boxes[k, 5] = 0.0

        nms_mask = np.greater(boxes[:, 5], 0.0)
        boxes = boxes[nms_mask]

        return boxes

    @staticmethod
    def calc_iou(box_a, box_b):
        """
        2개 box의 iou 계산 함수
        :param box_a: box 1
        :param box_b: box 2
        :return: float iou value
        """
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        x1_max = np.maximum(box_a[0], box_b[0])
        y1_max = np.maximum(box_a[1], box_b[1])
        x2_min = np.minimum(box_a[2], box_b[2])
        y2_min = np.minimum(box_a[3], box_b[3])
        inter_w = np.where(x1_max > x2_min, np.zeros_like(x1_max), x2_min - x1_max)
        inter_h = np.where(y1_max > y2_min, np.zeros_like(y1_max), y2_min - y1_max)
        intersection = inter_w * inter_h

        union = area_a + area_b - intersection
        IoU = intersection / (union + 1e-7)

        return IoU

    def __decode_boxes(self, mbox_loc):
        # decode
        decode_bbox_center_x = mbox_loc[:, 0] * self.prior_width * self.variance[0] + self.prior_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * self.prior_height * self.variance[1] + self.prior_center_y
        decode_bbox_width = np.exp(mbox_loc[:, 2] * self.variance[2]) * self.prior_width
        decode_bbox_height = np.exp(mbox_loc[:, 3] * self.variance[3]) * self.prior_height

        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]), axis=-1)
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
        return decode_bbox

    def get_priors(self):
        prior_width = self.anchors[:, 2] - self.anchors[:, 0]
        prior_height = self.anchors[:, 3] - self.anchors[:, 1]
        prior_center_x = 0.5 * (self.anchors[:, 2] + self.anchors[:, 0])
        prior_center_y = 0.5 * (self.anchors[:, 3] + self.anchors[:, 1])
        print(prior_center_x.tolist())
        print(prior_center_y.tolist())
        print(prior_width.tolist())
        print(prior_height.tolist())

    def assign_anchors(self, raw_bboxes):
        y_true = np.zeros((self.num_anchors, 4 + self.num_classes),
                          np.float32)  # box+classes
        boxes = []
        for raw_box in raw_bboxes:
            box = np.zeros((4 + self.num_classes))
            box[0] = raw_box.x1 / self.input_shape[1]
            box[1] = raw_box.y1 / self.input_shape[0]
            box[2] = raw_box.x2 / self.input_shape[1]
            box[3] = raw_box.y2 / self.input_shape[0]
            box[4 + raw_box.label] = 1.
            boxes.append(box)
        boxes = np.array(boxes)
        if len(boxes) == 0:
            return y_true

        encoded_boxes = np.apply_along_axis(self.encode_box, axis=1, arr=boxes[:, :4])
        encoded_boxes = encoded_boxes.reshape(-1, self.num_anchors, 5)

        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]
        assign_num = len(best_iou_idx)
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        y_true[best_iou_mask, :4] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
        y_true[best_iou_mask, 4:] = boxes[best_iou_idx, 4:]
        return y_true

    def encode_box(self, box):
        encoded_box = np.zeros((self.num_anchors, 5))
        IoU = self._calc_IoU_w_anchors(box)
        IoU_maks = IoU > self.iou_thresh

        # threshold 넘는 anchor가 없는 경우 가장 높은 Anchor를 지정
        if not IoU_maks.any():
            IoU_maks[IoU.argmax()] = True
        encoded_box[IoU_maks, -1] = IoU[IoU_maks]

        assinged_anchors = self.anchors[IoU_maks]
        box_center = (box[:2] + box[2:]) / 2.
        box_wh = box[2:] - box[:2]

        anchors_center = (assinged_anchors[:, :2] + assinged_anchors[:, 2:]) / 2.
        anchors_wh = assinged_anchors[:, 2:] - assinged_anchors[:, :2]
        xy_var = np.array((self.variance[0], self.variance[1]))
        wh_var = np.array((self.variance[2], self.variance[3]))
        encoded_box[IoU_maks, :2] = (box_center - anchors_center) / anchors_wh / xy_var  # center offset
        encoded_box[IoU_maks, 2:4] = np.log(box_wh / anchors_wh) / wh_var  # size offset
        # encoded_box[IoU_maks, 2:4] = (box_wh / anchors_wh) / wh_var  # size offset
        return encoded_box.reshape((-1,))

    def _calc_IoU_w_anchors(self, box):
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        x1_max = np.maximum(self.anchors[:, 0], box[0])
        y1_max = np.maximum(self.anchors[:, 1], box[1])
        x2_min = np.minimum(self.anchors[:, 2], box[2])
        y2_min = np.minimum(self.anchors[:, 3], box[3])
        inter_w = np.where(x1_max > x2_min, np.zeros_like(x1_max), x2_min - x1_max)
        inter_h = np.where(y1_max > y2_min, np.zeros_like(y1_max), y2_min - y1_max)
        intersection = inter_w * inter_h

        union = box_area + self.anchor_area - intersection
        IoUs = intersection / (union + 1e-7)
        return IoUs

    def _calc_IoU_w_picked_anchors(self, anchors, box):
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        x1_max = np.maximum(anchors[:, 0], box[0])
        y1_max = np.maximum(anchors[:, 1], box[1])
        x2_min = np.minimum(anchors[:, 2], box[2])
        y2_min = np.minimum(anchors[:, 3], box[3])
        inter_w = np.where(x1_max > x2_min, np.zeros_like(x1_max), x2_min - x1_max)
        inter_h = np.where(y1_max > y2_min, np.zeros_like(y1_max), y2_min - y1_max)
        intersection = inter_w * inter_h

        anchor_area = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
        union = box_area + anchor_area - intersection
        IoUs = intersection / (union + 1e-7)
        return IoUs

    def calc_distance(self, box):
        anchor_cx = (self.anchors[:, 0] + self.anchors[:, 2]) / 2.
        anchor_cy = (self.anchors[:, 1] + self.anchors[:, 3]) / 2.
        box_cx = (box[0] + box[2]) / 2.
        box_cy = (box[1] + box[3]) / 2.
        dist = (anchor_cx - box_cx) ** 2 + (anchor_cy - box_cy) ** 2
        return dist

    def is_inside(self, anchors, box):
        box_cx = (box[0] + box[2]) / 2.
        box_cy = (box[1] + box[3]) / 2.
        x_inside = np.logical_and((anchors[:, 0] < box_cx), (box_cx < anchors[:, 2]))
        y_inside = np.logical_and((anchors[:, 1] < box_cy), (box_cy < anchors[:, 3]))
        return np.logical_and(x_inside, y_inside)

    def encode_box_atss(self, box):
        encoded_box = np.zeros((self.num_anchors, 5))
        # select nearest k boxes
        distances = self.calc_distance(box)
        sorted_idx = np.argsort(distances)
        picked_anchors = self.anchors[sorted_idx[:self.k]]
        IoU = self._calc_IoU_w_picked_anchors(picked_anchors, box)
        iou_mean = np.mean(IoU)
        iou_std = np.std(IoU)
        thresh = iou_mean + iou_std
        IoU_mask = IoU > thresh
        inside_mask = self.is_inside(picked_anchors, box)
        mask = np.logical_and(IoU_mask, inside_mask)
        box_mask = sorted_idx[:self.k][mask]
        encoded_box[box_mask, -1] = IoU[mask]  # [IoU_maks]

        assinged_anchors = self.anchors[box_mask]
        box_center = (box[:2] + box[2:]) / 2.
        box_wh = box[2:] - box[:2]

        anchors_center = (assinged_anchors[:, :2] + assinged_anchors[:, 2:]) / 2.
        anchors_wh = assinged_anchors[:, 2:] - assinged_anchors[:, :2]
        xy_var = np.array((self.variance[0], self.variance[1]))
        wh_var = np.array((self.variance[2], self.variance[3]))
        encoded_box[box_mask, :2] = (box_center - anchors_center) / anchors_wh / xy_var  # center offset
        encoded_box[box_mask, 2:4] = np.log(box_wh / anchors_wh) / wh_var  # size offset
        return encoded_box.reshape((-1,))


if __name__ == '__main__':
    conf = cluster_configs
    anchor_util = AnchorUtils((256, 256, 3), 3, conf)
    anchor_util.get_priors()
    # anchors = anchor_util.anchors
    #
    # anchors *= 224
    #
    # s = 0
    # e = 0
    # for idx, i in enumerate(anchor_util.num_anchors_per_layers):
    #     img = np.zeros((224, 224))
    #     e += i
    #     layer_anchor = anchors[s:e]
    #     for a in layer_anchor:
    #         cv2.rectangle(img, (int(a[0]), int(a[1])), (int(a[2]), int(a[3])), (255, 255, 255))
    #         cv2.imshow('anchor_{}'.format(idx), img)
    #         cv2.waitKey(0)
    #     s = e
