from models.Layers import conv2d_bn, depthwiseconv_bn, PartialResidual
import tensorflow as tf
from models import Backbones
import keras
import keras.backend as K
import math


def SubNet(backbone, s3, s4, s5, s6, num_classes, num_anchors_per_layer, weight_decay=0.0001):
    l2_reg = keras.regularizers.l2(weight_decay)

    def PredictionBlock(inputs):
        x1 = conv2d_bn(inputs, 128, (1, 1), (1, 1), 'same', l2_reg)
        x1 = conv2d_bn(x1, 128, (3, 3), (1, 1), 'same', l2_reg)
        x1 = conv2d_bn(x1, 256, (1, 1), (1, 1), 'same', l2_reg)
        x2 = conv2d_bn(inputs, 256, (1, 1), (1, 1), 'same', l2_reg)
        return keras.layers.Add()([x1, x2])

    cls_head = cls_net(num_anchors_per_layer, num_classes, weight_decay=weight_decay)
    reg_head = reg_net(num_anchors_per_layer, weight_decay=weight_decay)

    s3 = conv2d_bn(s3, 256, (1, 1), (1, 1), 'same', l2_reg)
    s4 = conv2d_bn(s4, 256, (1, 1), (1, 1), 'same', l2_reg)
    s5 = conv2d_bn(s5, 256, (1, 1), (1, 1), 'same', l2_reg)

    P4 = keras.layers.Add()([s4, keras.layers.UpSampling2D()(s5)])
    P4 = PredictionBlock(P4)

    P3 = keras.layers.Add()([s3, keras.layers.UpSampling2D()(P4)])
    P3 = PredictionBlock(P3)

    P4 = keras.layers.Add()([conv2d_bn(P3, 256, (3, 3), (2, 2), 'same', l2_reg),
                             P4])
    P4 = PredictionBlock(P4)

    P5 = keras.layers.Add()([conv2d_bn(P4, 256, (3, 3), (2, 2), 'same', l2_reg),
                             s5])
    P5 = PredictionBlock(P5)

    P6 = conv2d_bn(P5, 256, (3, 3), (2, 2), 'same', l2_reg)
    P6 = PredictionBlock(P6)

    P7 = conv2d_bn(P6, 256, (3, 3), (2, 2), 'same', l2_reg)
    P7 = PredictionBlock(P7)

    P3_cls = cls_net(num_anchors_per_layer, num_classes, weight_decay=weight_decay)(P3)
    P3_loc = reg_net(num_anchors_per_layer, weight_decay=weight_decay)(P3)
    P4_cls = cls_net(num_anchors_per_layer, num_classes, weight_decay=weight_decay)(P4)
    P4_loc = reg_net(num_anchors_per_layer, weight_decay=weight_decay)(P4)
    P5_cls = cls_net(num_anchors_per_layer, num_classes, weight_decay=weight_decay)(P5)
    P5_loc = reg_net(num_anchors_per_layer, weight_decay=weight_decay)(P5)
    P6_cls = cls_net(num_anchors_per_layer, num_classes, weight_decay=weight_decay)(P6)
    P6_loc = reg_net(num_anchors_per_layer, weight_decay=weight_decay)(P6)
    P7_cls = cls_net(num_anchors_per_layer, num_classes, weight_decay=weight_decay)(P7)
    P7_loc = reg_net(num_anchors_per_layer, weight_decay=weight_decay)(P7)

    P3_cls = keras.layers.Reshape((-1, num_classes))(P3_cls)
    P4_cls = keras.layers.Reshape((-1, num_classes))(P4_cls)
    P5_cls = keras.layers.Reshape((-1, num_classes))(P5_cls)
    P6_cls = keras.layers.Reshape((-1, num_classes))(P6_cls)
    P7_cls = keras.layers.Reshape((-1, num_classes))(P7_cls)
    cls_pred = keras.layers.Concatenate(axis=1, name='cls_pred')([P3_cls, P4_cls, P5_cls, P6_cls, P7_cls])

    P3_loc = keras.layers.Reshape((-1, 4))(P3_loc)
    P4_loc = keras.layers.Reshape((-1, 4))(P4_loc)
    P5_loc = keras.layers.Reshape((-1, 4))(P5_loc)
    P6_loc = keras.layers.Reshape((-1, 4))(P6_loc)
    P7_loc = keras.layers.Reshape((-1, 4))(P7_loc)
    loc_pred = keras.layers.Concatenate(axis=1, name='loc_pred')([P3_loc, P4_loc, P5_loc, P6_loc, P7_loc])
    prediction = keras.layers.Concatenate(name='prediction')([loc_pred, cls_pred])
    return keras.models.Model(inputs=[backbone.input], outputs=[prediction])


def SharedHeadNet(backbone, s3, s4, s5, s6, num_classes, num_anchors_per_layer, weight_decay=0.0001):
    l2_reg = keras.regularizers.l2(weight_decay)

    def PredictionBlock(inputs):
        x1 = conv2d_bn(inputs, 128, (1, 1), (1, 1), 'same', l2_reg)
        x1 = conv2d_bn(x1, 128, (3, 3), (1, 1), 'same', l2_reg)
        x1 = conv2d_bn(x1, 256, (1, 1), (1, 1), 'same', l2_reg)
        x2 = conv2d_bn(inputs, 256, (1, 1), (1, 1), 'same', l2_reg)
        return keras.layers.Add()([x1, x2])

    anchors = keras.layers.Input((None, 4), name='anchors')
    cls_head = cls_net(num_anchors_per_layer, num_classes, weight_decay=weight_decay)
    reg_head = reg_net(num_anchors_per_layer, weight_decay=weight_decay)

    s3 = conv2d_bn(s3, 256, (1, 1), (1, 1), 'same', l2_reg)
    s4 = conv2d_bn(s4, 256, (1, 1), (1, 1), 'same', l2_reg)
    s5 = conv2d_bn(s5, 256, (1, 1), (1, 1), 'same', l2_reg)

    P4 = keras.layers.Add()([s4, keras.layers.UpSampling2D()(s5)])
    P4 = PredictionBlock(P4)

    P3 = keras.layers.Add()([s3, keras.layers.UpSampling2D()(P4)])
    P3 = PredictionBlock(P3)

    P4 = keras.layers.Add()([conv2d_bn(P3, 256, (3, 3), (2, 2), 'same', l2_reg),
                             P4])
    P4 = PredictionBlock(P4)

    P5 = keras.layers.Add()([conv2d_bn(P4, 256, (3, 3), (2, 2), 'same', l2_reg),
                             s5])
    P5 = PredictionBlock(P5)

    P6 = conv2d_bn(P5, 256, (3, 3), (2, 2), 'same', l2_reg)
    P6 = PredictionBlock(P6)

    P7 = conv2d_bn(P6, 256, (3, 3), (2, 2), 'same', l2_reg)
    P7 = PredictionBlock(P7)

    P3_cls = cls_head(P3)
    P3_loc = reg_head(P3)
    P4_cls = cls_head(P4)
    P4_loc = reg_head(P4)
    P5_cls = cls_head(P5)
    P5_loc = reg_head(P5)
    P6_cls = cls_head(P6)
    P6_loc = reg_head(P6)
    P7_cls = cls_head(P7)
    P7_loc = reg_head(P7)

    P3_cls = keras.layers.Reshape((-1, num_classes))(P3_cls)
    P4_cls = keras.layers.Reshape((-1, num_classes))(P4_cls)
    P5_cls = keras.layers.Reshape((-1, num_classes))(P5_cls)
    P6_cls = keras.layers.Reshape((-1, num_classes))(P6_cls)
    P7_cls = keras.layers.Reshape((-1, num_classes))(P7_cls)
    cls_pred = keras.layers.Concatenate(axis=1, name='cls_pred')([P3_cls, P4_cls, P5_cls, P6_cls, P7_cls])

    P3_loc = keras.layers.Reshape((-1, 4))(P3_loc)
    P4_loc = keras.layers.Reshape((-1, 4))(P4_loc)
    P5_loc = keras.layers.Reshape((-1, 4))(P5_loc)
    P6_loc = keras.layers.Reshape((-1, 4))(P6_loc)
    P7_loc = keras.layers.Reshape((-1, 4))(P7_loc)
    loc_pred = keras.layers.Concatenate(axis=1, name='loc_pred')([P3_loc, P4_loc, P5_loc, P6_loc, P7_loc])
    prediction = keras.layers.Concatenate(name='prediction')([loc_pred, cls_pred])
    return keras.models.Model(inputs=[backbone.input, anchors], outputs=[prediction])


def cls_net(num_anchors_per_layer, num_classes, pyramid_depth=256, weight_decay=0.0001):
    l2_reg = keras.regularizers.l2(weight_decay)
    phi = 0.01
    init_val = -math.log((1 - phi) / phi)
    inputs = keras.layers.Input(shape=(None, None, pyramid_depth))

    x = inputs
    x = depthwiseconv_bn(x, (3, 3), (1, 1), 'same', l2_reg)
    x = conv2d_bn(x, 128, (1, 1), (1, 1), 'same', l2_reg)
    x = depthwiseconv_bn(x, (3, 3), (1, 1), 'same', l2_reg)
    x = conv2d_bn(x, 128, (1, 1), (1, 1), 'same', l2_reg)
    x = depthwiseconv_bn(x, (3, 3), (1, 1), 'same', l2_reg)
    x = conv2d_bn(x, 128, (1, 1), (1, 1), 'same', l2_reg)

    x = keras.layers.Conv2D(num_anchors_per_layer * num_classes,
                            kernel_size=(1, 1),
                            bias_initializer=keras.initializers.constant(init_val),
                            kernel_regularizer=l2_reg,
                            activation='sigmoid')(x)
    return keras.models.Model(inputs=inputs, outputs=x)


def reg_net(num_anchors_per_layer, pyramid_depth=256, weight_decay=0.0001):
    l2_reg = keras.regularizers.l2(weight_decay)
    inputs = keras.layers.Input(shape=(None, None, pyramid_depth))

    x = inputs
    x = depthwiseconv_bn(x, (3, 3), (1, 1), 'same', l2_reg)
    x = conv2d_bn(x, 128, (1, 1), (1, 1), 'same', l2_reg)
    x = depthwiseconv_bn(x, (3, 3), (1, 1), 'same', l2_reg)
    x = conv2d_bn(x, 128, (1, 1), (1, 1), 'same', l2_reg)
    x = depthwiseconv_bn(x, (3, 3), (1, 1), 'same', l2_reg)
    x = conv2d_bn(x, 128, (1, 1), (1, 1), 'same', l2_reg)
    x = keras.layers.Conv2D(num_anchors_per_layer * 4,
                            kernel_size=(1, 1),
                            kernel_regularizer=l2_reg)(x)
    return keras.models.Model(inputs=inputs, outputs=x)


if __name__ == '__main__':
    def get_flops(model):
        run_meta = tf.RunMetadata()
        opts = tf.profiler.ProfileOptionBuilder.float_operation()

        flops = tf.profiler.profile(graph=K.get_session().graph,
                                    run_meta=run_meta, cmd='op', options=opts)

        return flops.total_float_ops  # Prints the "flops" of the model.


    model, s1, s2, s3, s4, s5, s6 = Backbones.GhostNet_CRELU_CSP(input_shape=(320, 320, 3))
    detector = SharedHeadNet(model, s3, s4, s5, s6, 3, 6)
    detector.summary()
    print(get_flops(model))
