from typing import List, Tuple
import numpy as np
import math


class LayerConfigs:
    def __init__(self, stride: int, size: float, next_layer_size: float, aspect_ratios: List[float]):
        self.stride = stride
        self.size = size
        self.next_layer_size = next_layer_size
        self.aspect_ratios = aspect_ratios
        self.num_anchors = len(self.aspect_ratios)
        assert self.aspect_ratios.count(1.) <= 2, "Num of aspect ratios which are '1.' must be less than 2."

        self.anchor_widths, self.anchor_heights = self.calculate_anchor_aspect_ratio()

    def calculate_anchor_aspect_ratio(self):
        anchor_widths = []
        anchor_heights = []
        for ar in self.aspect_ratios:
            if ar == 1. and len(anchor_widths) == 0:
                anchor_widths.append(self.size)
                anchor_heights.append(self.size)
            elif ar == 1. and len(anchor_widths) > 0:
                anchor_widths.append(np.sqrt(self.size * self.next_layer_size))
                anchor_heights.append(np.sqrt(self.size * self.next_layer_size))
            else:
                anchor_widths.append(self.size * np.sqrt(ar))
                anchor_heights.append(self.size / np.sqrt(ar))

        return np.array(anchor_widths), np.array(anchor_heights)


class AnchorComputer:
    def __init__(self, input_shape, num_classes, config, iou_thresh=0.4, nms_thresh=0.4):
        self.num_classes = num_classes
        self.config = config
        self.calculate_priors(input_shape)
        self.iou_thresh = iou_thresh
        self.nms_thresh = nms_thresh
        # this value is from COCO dataset.
        # See https://github.com/fizyr/keras-retinanet/issues/1273#issuecomment-585828825
        # self.std = np.array([0.2, 0.2, 0.2, 0.2])

    def assign_anchors(self, raw_bboxes, input_shape):
        y_true = np.zeros((self.num_anchors, self.num_classes + 4), np.float32)
        boxes = []
        for raw_box in raw_bboxes:
            box = np.zeros((4 + self.num_classes))
            box[0] = raw_box.x1 / input_shape[1]
            box[1] = raw_box.y1 / input_shape[0]
            box[2] = raw_box.x2 / input_shape[1]
            box[3] = raw_box.y2 / input_shape[0]
            box[4 + raw_box.label] = 1.
            boxes.append(box)
        boxes = np.array(boxes)
        if boxes.shape[0]:
            overlaps = self.compute_overlap_with_gt(self.anchors, boxes[:, :4])
            argmax_overlaps = np.argmax(overlaps, axis=1)
            max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps]
            positives = max_overlaps > self.iou_thresh

            # Assign class label
            y_true[positives, 4:] = boxes[argmax_overlaps[positives], 4:]

            # Assign bbox regression
            gt_boxes = boxes[argmax_overlaps, :4]
            # TODO: Maybe this work is not needed. I will use cIoU Loss for location
            target_dx1 = (gt_boxes[:, 0] - self.anchors[:, 0]) / self.anchor_width
            target_dy1 = (gt_boxes[:, 1] - self.anchors[:, 1]) / self.anchor_height
            target_dx2 = (gt_boxes[:, 2] - self.anchors[:, 2]) / self.anchor_width
            target_dy2 = (gt_boxes[:, 3] - self.anchors[:, 3]) / self.anchor_height

            target = np.stack((target_dx1, target_dy1, target_dx2, target_dy2))
            target = target.T
            y_true[:, :4] = target
        return y_true

    def postprocess(self, y_pred, confidence_thresh=0.5):
        results = []
        cls_pred = y_pred[:, 4:]
        loc_pred = y_pred[:, :4]
        decoded_boxes = self.decode_boxes(loc_pred)
        for class_id in range(self.num_classes):
            picks = []
            class_conf = cls_pred[:, class_id]
            positive_mask = class_conf > confidence_thresh
            masked_class_conf = class_conf[positive_mask]
            masked_loc = decoded_boxes[positive_mask]

            # NMS
            idxs = np.argsort(masked_class_conf)
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
                idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > self.nms_thresh)[0])))

            picks = np.array(picks)

            if len(picks) > 0:
                nms_confs = masked_class_conf[picks]
                nms_locs = masked_loc[picks]
                for i in range(len(picks)):
                    box = np.append(nms_locs[i], class_id)
                    box = np.append(box, nms_confs[i])
                    results.append(box)

        return results

    def decode_boxes(self, loc_pred):
        x1 = loc_pred[:, 0] * self.anchor_width + self.anchors[:, 0]
        y1 = loc_pred[:, 1] * self.anchor_height + self.anchors[:, 1]
        x2 = loc_pred[:, 2] * self.anchor_width + self.anchors[:, 2]
        y2 = loc_pred[:, 3] * self.anchor_height + self.anchors[:, 3]

        boxes = np.concatenate((x1[:, None],
                                y1[:, None],
                                x2[:, None],
                                y2[:, None]), axis=-1)
        return boxes

    def calculate_priors(self, input_shape: Tuple[int, int, int]):
        anchors = []
        input_h, input_w = input_shape[:2]
        for layer_config in self.config:
            num_anchors = layer_config.num_anchors
            stride = layer_config.stride
            layer_h, layer_w = math.ceil(input_h / stride), math.ceil(input_w / stride)

            lin_x = np.linspace(0.5 * stride, input_w - 0.5 * stride, layer_w)
            lin_y = np.linspace(0.5 * stride, input_h - 0.5 * stride, layer_h)
            cx, cy = np.meshgrid(lin_x, lin_y)
            cx, cy = cx.reshape(-1, 1), cy.reshape(-1, 1)
            prior_boxes = np.concatenate((cx, cy), axis=1)
            # normalize to 0 ~ 1
            prior_boxes[:, 0] /= input_w
            prior_boxes[:, 1] /= input_h
            # (H*W, num_anchors * 4)
            prior_boxes = np.tile(prior_boxes, (1, 2 * num_anchors))

            half_w = layer_config.anchor_widths * 0.5
            half_h = layer_config.anchor_heights * 0.5

            # calculate to (x1, y1, x2, y2)
            prior_boxes[:, ::4] -= half_w
            prior_boxes[:, 1::4] -= half_h
            prior_boxes[:, 2::4] += half_w
            prior_boxes[:, 3::4] += half_h

            # reshape to (H*W*num_anchors, 4(x1,y1,x2,y2))
            prior_boxes = prior_boxes.reshape(-1, 4)
            # clip
            prior_boxes = np.minimum(np.maximum(prior_boxes, 0.), 1.)
            anchors.append(prior_boxes)
        self.anchors = np.concatenate(anchors, axis=0)
        self.anchor_width = self.anchors[:, 2] - self.anchors[:, 0]
        self.anchor_height = self.anchors[:, 3] - self.anchors[:, 1]
        self.num_anchors = self.anchors.shape[0]

    @staticmethod
    def compute_overlap_with_gt(anchor_boxes, query_boxes):
        """

        :param anchor_boxes: (N, 4)
        :param query_boxes: (K, 4), G.T annotated bounding boxes, (x1, y1, x2, y2) order.
        :return: (N, K), computed overlap values
        """
        N = anchor_boxes.shape[0]
        K = query_boxes.shape[0]
        overlaps = np.zeros((N, K), np.float32)
        for k in range(K):
            box_area = (
                    (query_boxes[k, 2] - query_boxes[k, 0]) *
                    (query_boxes[k, 3] - query_boxes[k, 1]))
            for n in range(N):
                inter_w = (
                        min(anchor_boxes[n, 2], query_boxes[k, 2]) -
                        max(anchor_boxes[n, 0], query_boxes[k, 0]))
                if inter_w > 0:
                    inter_h = (
                            min(anchor_boxes[n, 3], query_boxes[k, 3]) -
                            max(anchor_boxes[n, 1], query_boxes[k, 1]))
                    if inter_h > 0:
                        union = (
                                (anchor_boxes[n, 2] - anchor_boxes[n, 0]) *
                                (anchor_boxes[n, 3] - anchor_boxes[n, 1]) +
                                box_area -
                                inter_w * inter_h)
                        overlaps[n, k] = inter_w * inter_h / union

        return overlaps


if __name__ == '__main__':
    util = AnchorUtil(1, c)
    util.calculate_priors((128, 128, 3))
