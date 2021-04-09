import tensorflow as tf
import math


class Multibox_Loss:
    def __init__(self, anchors, alpha=0.25, scale=1.0):
        self.anchors = anchors
        self.alpah = alpha
        self.scale = scale

    def compute_loss(self, y_true, y_pred):
        cls_true = y_true[:, :, 4:]
        cls_pred = y_pred[:, :, 4:]
        loc_true = y_true[:, :, :4]
        loc_pred = y_pred[:, :, :4]

        gt_mask = tf.cast(tf.greater(tf.reduce_sum(cls_true, axis=-1), 0.), tf.float32)
        normalizer = tf.reduce_sum(cls_true)
        normalizer = tf.maximum(1., normalizer)
        cls_pred = tf.clip_by_value(cls_pred, 1e-7, 1. - 1e-7)
        alpha_factor = tf.ones_like(cls_true) * self.alpah
        alpha_factor = tf.where(tf.greater(cls_true, 0.5), alpha_factor, 1. - alpha_factor)
        focal_loss = -cls_true * tf.log(cls_pred) * tf.pow(1. - cls_pred, 2) - \
                     (1. - cls_true) * tf.log(1. - cls_pred) * tf.pow(cls_pred, 2)
        focal_loss = focal_loss * alpha_factor
        focal_loss = tf.reduce_sum(focal_loss)

        # loc_loss = cIoULoss().compute(loc_true, loc_pred, self.anchors) * gt_mask
        # loc_loss = tf.reduce_sum(loc_loss)

        # return (focal_loss + loc_loss) / normalizer * self.scale
        return (focal_loss ) / normalizer * self.scale


class cIoULoss:
    def __init__(self, scale=1.0):
        self.scale = scale

    def decode_boxes(self, mbox_loc, anchors):
        anchor_width = anchors[:, :, 2] - anchors[:, :, 0]
        anchor_height = anchors[:, :, 3] - anchors[:, :, 1]

        x1 = mbox_loc[:, :, 0] * anchor_width #+ anchors[:, :, 0]
        y1 = mbox_loc[:, :, 1] * anchor_height #+ anchors[:, :, 1]
        x2 = mbox_loc[:, :, 2] * anchor_width #+ anchors[:, :, 2]
        y2 = mbox_loc[:, :, 3] * anchor_height #+ anchors[:, :, 3]

        decode_bbox = tf.concat((x1[:, :, None],
                                 y1[:, :, None],
                                 x2[:, :, None],
                                 y2[:, :, None]), axis=-1)
        decode_bbox = tf.clip_by_value(decode_bbox, 0., 1.)
        return decode_bbox

    def compute(self, y_true, y_pred, anchors):
        with tf.name_scope('cIoU_Loss'):
            eps = tf.constant(1e-7)
            mbox_true = self.decode_boxes(y_true, anchors)
            mbox_pred = self.decode_boxes(y_pred, anchors)

            # x1_true = mbox_true[:, :, 0]
            # x1_pred = mbox_pred[:, :, 0]
            # y1_true = mbox_true[:, :, 1]
            # y1_pred = mbox_pred[:, :, 1]
            #
            # x2_true = mbox_true[:, :, 2]
            # x2_pred = mbox_pred[:, :, 2]
            # y2_true = mbox_true[:, :, 3]
            # y2_pred = mbox_pred[:, :, 3]
            #
            # w_true = mbox_true[:, :, 2] - mbox_true[:, :, 0]
            # w_pred = mbox_pred[:, :, 2] - mbox_pred[:, :, 0]
            # h_true = mbox_true[:, :, 3] - mbox_true[:, :, 1]
            # h_pred = mbox_pred[:, :, 3] - mbox_pred[:, :, 1]
            #
            # # IoU Calculate
            # x1_max = tf.maximum(x1_true, x1_pred)
            # y1_max = tf.maximum(y1_true, y1_pred)
            # x2_min = tf.minimum(x2_true, x2_pred)
            # y2_min = tf.minimum(y2_true, y2_pred)
            # inter_w = tf.where(x1_max > x2_min, tf.zeros_like(x1_max), x2_min - x1_max)
            # inter_h = tf.where(y1_max > y2_min, tf.zeros_like(y1_max), y2_min - y1_max)
            #
            # intersection = inter_w * inter_h
            # area_true = (x2_true - x1_true) * (y2_true - y1_true)
            # area_pred = (x2_pred - x1_pred) * (y2_pred - y1_pred)
            # iou = intersection / (area_true + area_pred - intersection + eps)
            #
            # # Enclose box Calculate
            # enclose_x1 = tf.minimum(x1_true, x1_pred)
            # enclose_y1 = tf.minimum(y1_true, y1_pred)
            # enclose_x2 = tf.maximum(x2_true, x2_pred)
            # enclose_y2 = tf.maximum(y2_true, y2_pred)
            #
            # # Calculate Penalty term - p2
            # cx_true = (x1_true + x2_true) / 2.
            # cy_true = (y1_true + y2_true) / 2.
            # cx_pred = (x1_pred + x2_pred) / 2.
            # cy_pred = (y1_pred + y2_pred) / 2.
            # p2 = tf.pow(cx_true - cx_pred, 2) + tf.pow(cy_true - cy_pred, 2) + eps
            #
            # # Calculate Penalty term - c2
            # enclose_w = enclose_x2 - enclose_x1
            # enclose_h = enclose_y2 - enclose_y1
            # c2 = tf.pow(enclose_w, 2) + tf.pow(enclose_h, 2) + eps
            #
            # # Calculate Penalty term - v
            # atan_true = tf.atan((w_true + eps) / (h_true + eps))
            # atan_pred = tf.atan((w_pred + eps) / (h_pred + eps))
            # v = tf.constant(4.0 / (math.pi ** 2)) * tf.pow(atan_true - atan_pred, 2) + eps
            # a = v / (1. - iou + v)
            # a = tf.where(iou >= 0.5, a, tf.zeros_like(a))
            # a = tf.stop_gradient(a)
            #
            # ciou_loss = (1.0 - (iou - p2 / c2 - a * v))

        return 1
        # return ciou_loss


class FocalLoss:
    def __init__(self, scale=1.0):
        self.scale = scale

    def compute(self, y_true, y_pred):
        eps = tf.constant(1e-7)
        pos_mask = tf.cast(tf.equal(y_true, 1), tf.float32)
        neg_mask = tf.cast(tf.less(y_true, 1), tf.float32)
        neg_weights = tf.pow(1 - y_true, 8)

        pos_loss = -tf.log(tf.clip_by_value(y_pred, eps, 1. - eps)) * tf.pow(1. - y_pred, 2) * pos_mask
        neg_loss = -tf.log(tf.clip_by_value(1 - y_pred, eps, 1. - eps)) * tf.pow(y_pred, 2) * neg_weights * neg_mask

        pos_loss = tf.reduce_sum(pos_loss)
        neg_loss = tf.reduce_sum(neg_loss)
        cls_loss = pos_loss + neg_loss

        return cls_loss * self.scale


class SmoothL1Loss:
    def __init__(self, scale=1.0):
        self.scale = scale

    def compute(self, y_true, y_pred):
        abs_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred) ** 2
        l1_loss = tf.where(tf.less(abs_loss, 1.0), square_loss, abs_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis=-1)
