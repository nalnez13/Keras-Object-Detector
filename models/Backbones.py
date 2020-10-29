from models.Layers import conv2d_bn, GMBBlock_linear, depthwiseconv_bn, PartialResidual
import keras
import tensorflow as tf
import keras.backend as K


def CSP_GMB2(input_shape, num_classes=1000, weight_decay=0.0005, tensor=None):
    l2_reg = keras.regularizers.l2(weight_decay)
    if tensor is None:
        input_tensor = keras.layers.Input(input_shape)
    else:
        input_tensor = keras.layers.Input(tensor=tensor, shape=input_shape)

    # 112,112,16
    x = conv2d_bn(input_tensor, 16, (3, 3), (2, 2), 'same', l2_reg)
    stage1 = x
    # 56,56,16
    x, exp = GMBBlock_linear(x, 16, 16, (3, 3), (2, 2), 'same', l2_reg, True, ratio=2)
    x1 = depthwiseconv_bn(exp, (3, 3), (2, 2), 'same', l2_reg)
    x = keras.layers.Concatenate()([x, x1])
    stage2 = x
    # 28,28,24
    x, exp = GMBBlock_linear(x, 36, 24, (3, 3), (2, 2), 'same', l2_reg, False, ratio=2)
    x1 = depthwiseconv_bn(exp, (3, 3), (2, 2), 'same', l2_reg)
    x, _ = GMBBlock_linear(x, 44, 24, (3, 3), (1, 1), 'same', l2_reg, False, ratio=2)
    x = keras.layers.Concatenate()([x, x1])
    stage3 = x
    # 14,14,40
    x, exp = GMBBlock_linear(x, 48, 40, (3, 3), (2, 2), 'same', l2_reg, True, ratio=2)
    x1 = depthwiseconv_bn(exp, (3, 3), (2, 2), 'same', l2_reg)
    x, _ = GMBBlock_linear(x, 120, 40, (3, 3), (1, 1), 'same', l2_reg, True, ratio=2)
    x, _ = GMBBlock_linear(x, 120, 40, (3, 3), (1, 1), 'same', l2_reg, True, ratio=2)
    x, _ = GMBBlock_linear(x, 60, 48, (3, 3), (1, 1), 'same', l2_reg, True, ratio=2)
    x, _ = GMBBlock_linear(x, 72, 48, (3, 3), (1, 1), 'same', l2_reg, True, ratio=2)
    x = keras.layers.Concatenate()([x, x1])
    stage4 = x
    x, exp = GMBBlock_linear(x, 144, 96, (3, 3), (2, 2), 'same', l2_reg, True, ratio=2)
    x1 = depthwiseconv_bn(exp, (3, 3), (2, 2), 'same', l2_reg)

    x, _ = GMBBlock_linear(x, 288, 96, (3, 3), (1, 1), 'same', l2_reg, True, ratio=2)
    x, _ = GMBBlock_linear(x, 288, 96, (3, 3), (1, 1), 'same', l2_reg, True, ratio=2)
    x = keras.layers.Concatenate()([x, x1])
    x = conv2d_bn(x, 576, (1, 1), (1, 1), 'same', l2_reg, False)
    stage5 = x
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Reshape((1, 1, -1))(x)
    x = keras.layers.Dropout(rate=0.2)(x)
    x = keras.layers.Conv2D(filters=1280, kernel_size=(1, 1), strides=(1, 1),
                            kernel_regularizer=l2_reg)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(filters=num_classes, kernel_size=(1, 1), strides=(1, 1),
                            kernel_regularizer=l2_reg, activation='softmax')(x)
    x = keras.layers.Reshape((-1,))(x)

    return keras.models.Model(input_tensor, x), stage1, stage2, stage3, stage4, stage5


def GMB_CPRN(input_shape, num_classes=1000, weight_decay=0.0005, tensor=None):
    l2_reg = keras.regularizers.l2(weight_decay)
    if tensor is None:
        input_tensor = keras.layers.Input(input_shape)
    else:
        input_tensor = keras.layers.Input(tensor=tensor, shape=input_shape)

    # 112,112,16
    x = conv2d_bn(input_tensor, 16, (3, 3), (2, 2), 'same', l2_reg, cRelu=True)
    stage1 = x
    # 56,56,16
    x, exp = GMBBlock_linear(x, 16, 16, (3, 3), (2, 2), 'same', l2_reg, True, ratio=2, cRelu=True)
    x1 = depthwiseconv_bn(exp, (3, 3), (2, 2), 'same', l2_reg)
    x = keras.layers.Concatenate()([x, x1])
    stage2 = x
    # 28,28,24
    x, exp = GMBBlock_linear(x, 36, 24, (3, 3), (2, 2), 'same', l2_reg, True, ratio=2, cRelu=True)
    x_prev = x
    x1 = depthwiseconv_bn(exp, (3, 3), (2, 2), 'same', l2_reg)
    x, _ = GMBBlock_linear(x, 44, 24, (3, 3), (1, 1), 'same', l2_reg, False, ratio=2, cRelu=True)
    x = keras.layers.Concatenate()([x, x1])
    x = PartialResidual()([x, x_prev])
    stage3 = x
    # 14,14,40
    x, exp = GMBBlock_linear(x, 48, 40, (3, 3), (2, 2), 'same', l2_reg, True, ratio=2, cRelu=True)
    x_prev = x
    x1 = depthwiseconv_bn(exp, (3, 3), (2, 2), 'same', l2_reg)
    x, _ = GMBBlock_linear(x, 120, 40, (3, 3), (1, 1), 'same', l2_reg, True, ratio=2, cRelu=True)
    x, _ = GMBBlock_linear(x, 120, 40, (3, 3), (1, 1), 'same', l2_reg, True, ratio=2, cRelu=True)
    x, _ = GMBBlock_linear(x, 60, 48, (3, 3), (1, 1), 'same', l2_reg, True, ratio=2, cRelu=True)
    x, _ = GMBBlock_linear(x, 72, 48, (3, 3), (1, 1), 'same', l2_reg, True, ratio=2, cRelu=True)
    x = keras.layers.Concatenate()([x, x1])
    x = PartialResidual()([x, x_prev])
    stage4 = x
    x, exp = GMBBlock_linear(x, 144, 96, (3, 3), (2, 2), 'same', l2_reg, True, ratio=2)
    x_prev = x
    x1 = depthwiseconv_bn(exp, (3, 3), (2, 2), 'same', l2_reg)

    x, _ = GMBBlock_linear(x, 288, 96, (3, 3), (1, 1), 'same', l2_reg, True, ratio=2)
    x, _ = GMBBlock_linear(x, 288, 96, (3, 3), (1, 1), 'same', l2_reg, True, ratio=2)
    x = keras.layers.Concatenate()([x, x1])
    x = PartialResidual()([x, x_prev])
    x = conv2d_bn(x, 576, (1, 1), (1, 1), 'same', l2_reg, False)
    stage5 = x
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Reshape((1, 1, -1))(x)
    x = keras.layers.Dropout(rate=0.2)(x)
    x = keras.layers.Conv2D(filters=1280, kernel_size=(1, 1), strides=(1, 1),
                            kernel_regularizer=l2_reg)(x)
    x = keras.layers.Activation('relu')(x)
    stage6 = x
    x = keras.layers.Conv2D(filters=num_classes, kernel_size=(1, 1), strides=(1, 1),
                            kernel_regularizer=l2_reg, activation='softmax')(x)
    x = keras.layers.Reshape((-1,))(x)

    return keras.models.Model(input_tensor, x), stage1, stage2, stage3, stage4, stage5, stage6


def v4_tiny(input_shape, num_classes=1000, weight_decay=0.0005, tensor=None):
    l2_reg = keras.regularizers.l2(weight_decay)
    if tensor is None:
        input_tensor = keras.layers.Input(input_shape)
    else:
        input_tensor = keras.layers.Input(tensor=tensor, shape=input_shape)

    x = conv2d_bn(input_tensor, 32, (3, 3), (2, 2), 'same', l2_reg)
    stage1 = x

    x = conv2d_bn(x, 64, (3, 3), (2, 2), 'same', l2_reg)
    stage2 = x

    x = conv2d_bn(x, 64, (3, 3), (1, 1), 'same', l2_reg)
    x_prev1 = x
    x = conv2d_bn(x, 32, (3, 3), (1, 1), 'same', l2_reg)
    x_prev = x
    x = conv2d_bn(x, 32, (3, 3), (1, 1), 'same', l2_reg)
    x = keras.layers.Concatenate()([x, x_prev])
    x = conv2d_bn(x, 64, (1, 1), (1, 1), 'same', l2_reg)
    x = keras.layers.Concatenate()([x, x_prev1])
    x = keras.layers.MaxPooling2D((2, 2), (2, 2), 'same')(x)
    stage3 = x

    x = conv2d_bn(x, 128, (3, 3), (1, 1), 'same', l2_reg)
    x_prev1 = x
    x = conv2d_bn(x, 64, (3, 3), (1, 1), 'same', l2_reg)
    x_prev = x
    x = conv2d_bn(x, 64, (3, 3), (1, 1), 'same', l2_reg)
    x = keras.layers.Concatenate()([x, x_prev])
    x = conv2d_bn(x, 128, (1, 1), (1, 1), 'same', l2_reg)
    x = keras.layers.Concatenate()([x, x_prev1])
    x = keras.layers.MaxPooling2D((2, 2), (2, 2), 'same')(x)
    stage4 = x

    x = conv2d_bn(x, 256, (3, 3), (1, 1), 'same', l2_reg)
    x_prev1 = x
    x = conv2d_bn(x, 128, (3, 3), (1, 1), 'same', l2_reg)
    x_prev = x
    x = conv2d_bn(x, 128, (3, 3), (1, 1), 'same', l2_reg)
    x = keras.layers.Concatenate()([x, x_prev])
    x = conv2d_bn(x, 256, (1, 1), (1, 1), 'same', l2_reg)
    x = keras.layers.Concatenate()([x, x_prev1])
    x = keras.layers.MaxPooling2D((2, 2), (2, 2), 'same')(x)
    stage5 = x
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Reshape((1, 1, -1))(x)
    x = keras.layers.Dropout(rate=0.2)(x)
    x = keras.layers.Conv2D(filters=1280, kernel_size=(1, 1), strides=(1, 1),
                            kernel_regularizer=l2_reg)(x)
    x = keras.layers.Activation('relu')(x)
    stage6 = x
    x = keras.layers.Conv2D(filters=num_classes, kernel_size=(1, 1), strides=(1, 1),
                            kernel_regularizer=l2_reg, activation='softmax')(x)
    x = keras.layers.Reshape((-1,))(x)
    return keras.models.Model(input_tensor, x), stage1, stage2, stage3, stage4, stage5, stage6


if __name__ == '__main__':
    def get_flops(model):
        run_meta = tf.RunMetadata()
        opts = tf.profiler.ProfileOptionBuilder.float_operation()

        flops = tf.profiler.profile(graph=K.get_session().graph,
                                    run_meta=run_meta, cmd='op', options=opts)

        return flops.total_float_ops  # Prints the "flops" of the model.


    model, s1, s2, s3, s4, s5, s6 = v4_tiny((256, 256, 3), tensor=tf.placeholder(tf.float32, (1, 256, 256, 3)))
    model.summary()
    print(get_flops(model))
