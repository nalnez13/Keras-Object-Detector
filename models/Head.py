from models.Layers import conv2d_bn, depthwiseconv_bn, PartialResidual
import tensorflow as tf
from models import Backbones
import keras
import keras.backend as K
import math


def DW_RFB(inputs, weight_decay=0.0001):
    l2_reg = keras.regularizers.l2(weight_decay)
    x1 = conv2d_bn(inputs, 64, (1, 1), (1, 1), 'same', l2_reg)
    x1 = depthwiseconv_bn(x1, (3, 3), (1, 1), 'same', l2_reg, dilation_rate=1)
    x1 = conv2d_bn(x1, 32, (1, 1), (1, 1), 'same', l2_reg)

    x2 = conv2d_bn(inputs, 64, (1, 1), (1, 1), 'same', l2_reg)
    x2 = depthwiseconv_bn(x2, (3, 3), (1, 1), 'same', l2_reg, dilation_rate=2)
    x2 = conv2d_bn(x2, 32, (1, 1), (1, 1), 'same', l2_reg)

    x3 = conv2d_bn(inputs, 64, (1, 1), (1, 1), 'same', l2_reg)
    x3 = depthwiseconv_bn(x3, (3, 3), (1, 1), 'same', l2_reg, dilation_rate=3)
    x3 = conv2d_bn(x3, 32, (1, 1), (1, 1), 'same', l2_reg)

    x = keras.layers.Concatenate()([x1, x2, x3])
    x = conv2d_bn(x, 128, (1, 1), (1, 1), 'same', l2_reg)
    x = PartialResidual()([x, inputs])
    return x


def ContextModule(inputs, pyramid_depth=128, weight_decay=0.0001):
    l2_reg = keras.regularizers.l2(weight_decay)
    x1 = conv2d_bn(inputs, pyramid_depth // 2, (3, 3), (1, 1), 'same', l2_reg)

    x2 = conv2d_bn(x1, pyramid_depth // 4, (3, 3), (1, 1), 'same', l2_reg)
    x2 = conv2d_bn(x2, pyramid_depth // 4, (3, 3), (1, 1), 'same', l2_reg)

    x3 = conv2d_bn(x2, pyramid_depth // 4, (3, 3), (1, 1), 'same', l2_reg)
    x3 = conv2d_bn(x3, pyramid_depth // 4, (3, 3), (1, 1), 'same', l2_reg)

    x = keras.layers.Concatenate()([x1, x2, x3])
    return x


def D_FPN_subnet_fix(backbone, s3, s4, s6, num_classes, num_anchors_per_layer, weight_decay=0.0001):
    l2_reg = keras.regularizers.l2(weight_decay)

    def DownSample(inputs):
        x1 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(inputs)
        x1 = conv2d_bn(x1, 64, (1, 1), (1, 1), 'same', l2_reg)
        x2 = conv2d_bn(inputs, 64, (1, 1), (1, 1), 'same', l2_reg)
        x2 = depthwiseconv_bn(x2, (3, 3), (2, 2), 'same', l2_reg)
        return keras.layers.Concatenate()([x1, x2])

    def UpSample(inputs):
        x = keras.layers.UpSampling2D()(inputs)
        x = depthwiseconv_bn(x, (3, 3), (1, 1), 'same', l2_reg)
        return x

    # s3 = DW_RFB(s3, weight_decay)
    P4 = keras.layers.Concatenate()([DownSample(s3), s4])
    # P4 = DW_RFB(P4, weight_decay)
    P5 = DownSample(P4)
    P6 = DownSample(P5)
    P7 = DownSample(P6)
    P8 = DownSample(P7)
    s6 = conv2d_bn(s6, 128, (1, 1), (1, 1), 'same', l2_reg)
    P8 = keras.layers.Add()([P8, s6])

    P7 = keras.layers.Add()([P7, UpSample(P8)])
    P6 = keras.layers.Add()([P6, UpSample(P7)])
    P5 = keras.layers.Add()([P5, UpSample(P6)])
    P4 = conv2d_bn(P4, 128, (1, 1), (1, 1), 'same', l2_reg)
    P4 = keras.layers.Add()([P4, UpSample(P5)])

    P7 = ContextModule(P7)
    P6 = ContextModule(P6)
    P5 = ContextModule(P5)
    P4 = ContextModule(P4)

    cls_head = cls_net(num_anchors_per_layer, num_classes, weight_decay=weight_decay)
    reg_head = reg_net(num_anchors_per_layer, weight_decay=weight_decay)

    P4_cls = cls_head(P4)
    P4_loc = reg_head(P4)
    P5_cls = cls_head(P5)
    P5_loc = reg_head(P5)
    P6_cls = cls_head(P6)
    P6_loc = reg_head(P6)
    P7_cls = cls_head(P7)
    P7_loc = reg_head(P7)

    P4_cls = keras.layers.Reshape((-1, num_classes))(P4_cls)
    P5_cls = keras.layers.Reshape((-1, num_classes))(P5_cls)
    P6_cls = keras.layers.Reshape((-1, num_classes))(P6_cls)
    P7_cls = keras.layers.Reshape((-1, num_classes))(P7_cls)
    cls_pred = keras.layers.Concatenate(axis=1)([P4_cls, P5_cls, P6_cls, P7_cls])
    cls_pred = keras.layers.Activation('sigmoid', name='cls_pred')(cls_pred)

    P4_loc = keras.layers.Reshape((-1, 4))(P4_loc)
    P5_loc = keras.layers.Reshape((-1, 4))(P5_loc)
    P6_loc = keras.layers.Reshape((-1, 4))(P6_loc)
    P7_loc = keras.layers.Reshape((-1, 4))(P7_loc)
    loc_pred = keras.layers.Concatenate(axis=1, name='loc_pred')([P4_loc, P5_loc, P6_loc, P7_loc])
    prediction = keras.layers.Concatenate(name='prediction')([loc_pred, cls_pred])
    return keras.models.Model(inputs=[backbone.input], outputs=[prediction])


def SubNet(backbone, s4, s6, num_classes, num_anchors_per_layer, weight_decay=0.0001):
    l2_reg = keras.regularizers.l2(weight_decay)

    def DownSample(inputs):
        x1 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(inputs)
        x1 = conv2d_bn(x1, 64, (1, 1), (1, 1), 'same', l2_reg)
        x2 = conv2d_bn(inputs, 64, (1, 1), (1, 1), 'same', l2_reg)
        x2 = conv2d_bn(x2, 64, (3, 3), (2, 2), 'same', l2_reg)
        return keras.layers.Concatenate()([x1, x2])

    def UpSample(inputs):
        x = keras.layers.UpSampling2D()(inputs)
        x = conv2d_bn(x, 128, (3, 3), (1, 1), 'same', l2_reg)
        return x

    P4 = conv2d_bn(s4, 128, (1, 1), (1, 1), 'same', l2_reg)
    P5 = DownSample(P4)
    P6 = DownSample(P5)
    P7 = DownSample(P6)
    P8 = DownSample(P7)
    s6 = conv2d_bn(s6, 128, (1, 1), (1, 1), 'same', l2_reg)
    P8 = keras.layers.Add()([P8, s6])

    P7 = keras.layers.Add()([P7, UpSample(P8)])
    P6 = keras.layers.Add()([P6, UpSample(P7)])
    P5 = keras.layers.Add()([P5, UpSample(P6)])
    P4 = keras.layers.Add()([P4, UpSample(P5)])

    cls_head = cls_net_fix(num_anchors_per_layer, num_classes, weight_decay=weight_decay)
    reg_head = reg_net_fix(num_anchors_per_layer, weight_decay=weight_decay)

    P4_cls = cls_head(P4)
    P4_loc = reg_head(P4)
    P5_cls = cls_head(P5)
    P5_loc = reg_head(P5)
    P6_cls = cls_head(P6)
    P6_loc = reg_head(P6)
    P7_cls = cls_head(P7)
    P7_loc = reg_head(P7)

    P4_cls = keras.layers.Reshape((-1, num_classes))(P4_cls)
    P5_cls = keras.layers.Reshape((-1, num_classes))(P5_cls)
    P6_cls = keras.layers.Reshape((-1, num_classes))(P6_cls)
    P7_cls = keras.layers.Reshape((-1, num_classes))(P7_cls)
    cls_pred = keras.layers.Concatenate(axis=1)([P4_cls, P5_cls, P6_cls, P7_cls])
    cls_pred = keras.layers.Activation('sigmoid', name='cls_pred')(cls_pred)

    P4_loc = keras.layers.Reshape((-1, 4))(P4_loc)
    P5_loc = keras.layers.Reshape((-1, 4))(P5_loc)
    P6_loc = keras.layers.Reshape((-1, 4))(P6_loc)
    P7_loc = keras.layers.Reshape((-1, 4))(P7_loc)
    loc_pred = keras.layers.Concatenate(axis=1, name='loc_pred')([P4_loc, P5_loc, P6_loc, P7_loc])
    prediction = keras.layers.Concatenate(name='prediction')([loc_pred, cls_pred])
    return keras.models.Model(inputs=[backbone.input], outputs=[prediction])


def cls_net(num_anchors_per_layer, num_classes, pyramid_depth=128, weight_decay=0.0001):
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
                            kernel_regularizer=l2_reg)(x)
    return keras.models.Model(inputs=inputs, outputs=x)


def reg_net(num_anchors_per_layer, pyramid_depth=128, weight_decay=0.0001):
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


def cls_net_fix(num_anchors_per_layer, num_classes, pyramid_depth=128, weight_decay=0.0001):
    l2_reg = keras.regularizers.l2(weight_decay)
    phi = 0.01
    init_val = -math.log((1 - phi) / phi)
    inputs = keras.layers.Input(shape=(None, None, pyramid_depth))

    x = inputs
    x = conv2d_bn(x, 128, (1, 1), (1, 1), 'same', l2_reg)
    x = conv2d_bn(x, 64, (3, 3), (1, 1), 'same', l2_reg)
    x = keras.layers.Conv2D(num_anchors_per_layer * num_classes,
                            kernel_size=(1, 1),
                            bias_initializer=keras.initializers.constant(init_val),
                            kernel_regularizer=l2_reg)(x)
    return keras.models.Model(inputs=inputs, outputs=x)


def reg_net_fix(num_anchors_per_layer, pyramid_depth=128, weight_decay=0.0001):
    l2_reg = keras.regularizers.l2(weight_decay)
    inputs = keras.layers.Input(shape=(None, None, pyramid_depth))

    x = inputs
    x = conv2d_bn(x, 128, (1, 1), (1, 1), 'same', l2_reg)
    x = conv2d_bn(x, 64, (3, 3), (1, 1), 'same', l2_reg)
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


    model, s1, s2, s3, s4, s5, s6 = Backbones.GMB_CPRN(input_shape=(224, 224, 3))
    detector = D_FPN_subnet_fix(model, s3, s4, s6, 3, 4)
    detector.summary()
    print(get_flops(model))
    detector.save('1.h5')
