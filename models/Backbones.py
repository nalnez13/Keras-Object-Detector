from models.Layers import conv2d_bn, GhostBlock, h_swish, MBBlock_linear, GhostBlock_HSE, GhostBlockRES, \
    PartialResidual, GhostNextBlock, depthwiseconv_bn
import keras
import tensorflow as tf
import keras.backend as K


def MobileNetv3Small(input_shape, num_classes=1000, weight_decay=0.0005, tensor=None):
    l2_reg = keras.regularizers.l2(weight_decay)
    if tensor is None:
        input_tensor = keras.layers.Input(input_shape)
    else:
        input_tensor = keras.layers.Input(tensor=tensor, shape=input_shape)

    # 112,112,16
    x = conv2d_bn(input_tensor, 16, (3, 3), (2, 2), 'same', l2_reg, activation='h-swish')
    stage1 = x
    # 56,56,16
    x = MBBlock_linear(x, 16, 16, (3, 3), (2, 2), 'same', l2_reg, True)
    stage2 = x
    # 28,28,24
    x = MBBlock_linear(x, 72, 24, (3, 3), (2, 2), 'same', l2_reg, False)
    x = MBBlock_linear(x, 88, 24, (3, 3), (1, 1), 'same', l2_reg, False)
    stage3 = x
    # 14,14,40
    x = MBBlock_linear(x, 96, 40, (5, 5), (2, 2), 'same', l2_reg, True, activation='h-swish')
    x = MBBlock_linear(x, 240, 40, (5, 5), (1, 1), 'same', l2_reg, True, activation='h-swish')
    x = MBBlock_linear(x, 240, 40, (5, 5), (1, 1), 'same', l2_reg, True, activation='h-swish')
    x = MBBlock_linear(x, 120, 48, (5, 5), (1, 1), 'same', l2_reg, True, activation='h-swish')
    x = MBBlock_linear(x, 144, 48, (5, 5), (1, 1), 'same', l2_reg, True, activation='h-swish')
    stage4 = x
    x = MBBlock_linear(x, 288, 96, (5, 5), (2, 2), 'same', l2_reg, True, activation='h-swish')

    x = MBBlock_linear(x, 576, 96, (5, 5), (1, 1), 'same', l2_reg, True, activation='h-swish')
    x = MBBlock_linear(x, 576, 96, (5, 5), (1, 1), 'same', l2_reg, True, activation='h-swish')
    x = conv2d_bn(x, 576, (1, 1), (1, 1), 'same', l2_reg, False, activation='h-swish')
    stage5 = x
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Reshape((1, 1, -1))(x)
    x = keras.layers.Conv2D(filters=1280, kernel_size=(1, 1), strides=(1, 1),
                            kernel_regularizer=l2_reg)(x)
    x = keras.layers.Activation(h_swish)(x)
    stage6 = x
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(filters=num_classes, kernel_size=(1, 1), strides=(1, 1),
                            kernel_regularizer=l2_reg, activation='softmax')(x)
    x = keras.layers.Reshape((-1,))(x)

    return keras.models.Model(input_tensor, x), stage1, stage2, stage3, stage4, stage5, stage6


def GhostNet(input_shape, num_classes=1000, weight_decay=0.0005, tensor=None):
    l2_reg = keras.regularizers.l2(weight_decay)
    if tensor is None:
        input_tensor = keras.layers.Input(input_shape)
    else:
        input_tensor = keras.layers.Input(tensor=tensor, shape=input_shape)

    # 112,112,16
    x = conv2d_bn(input_tensor, 16, (3, 3), (2, 2), 'same', l2_reg, activation='h-swish')
    stage1 = x
    # 56,56,16
    x, _ = GhostBlock(x, 16, 16, (3, 3), (2, 2), 'same', l2_reg, True)
    stage2 = x
    # 28,28,24
    x, _ = GhostBlock(x, 72, 24, (3, 3), (2, 2), 'same', l2_reg, False)
    x, _ = GhostBlock(x, 88, 24, (3, 3), (1, 1), 'same', l2_reg, False)
    stage3 = x
    # 14,14,40
    x, _ = GhostBlock(x, 96, 40, (5, 5), (2, 2), 'same', l2_reg, True)
    x, _ = GhostBlock(x, 240, 40, (5, 5), (1, 1), 'same', l2_reg, True)
    x, _ = GhostBlock(x, 240, 40, (5, 5), (1, 1), 'same', l2_reg, True)
    x, _ = GhostBlock(x, 120, 48, (5, 5), (1, 1), 'same', l2_reg, True)
    x, _ = GhostBlock(x, 144, 48, (5, 5), (1, 1), 'same', l2_reg, True)
    stage4 = x
    x, _ = GhostBlock(x, 288, 96, (5, 5), (2, 2), 'same', l2_reg, True)

    x, _ = GhostBlock(x, 576, 96, (5, 5), (1, 1), 'same', l2_reg, True)
    x, _ = GhostBlock(x, 576, 96, (5, 5), (1, 1), 'same', l2_reg, True)
    x = conv2d_bn(x, 576, (1, 1), (1, 1), 'same', l2_reg, False)
    stage5 = x
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Reshape((1, 1, -1))(x)
    x = keras.layers.Conv2D(filters=1280, kernel_size=(1, 1), strides=(1, 1),
                            kernel_regularizer=l2_reg)(x)
    x = keras.layers.Activation('relu')(x)
    stage6 = x
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(filters=num_classes, kernel_size=(1, 1), strides=(1, 1),
                            kernel_regularizer=l2_reg, activation='softmax')(x)
    x = keras.layers.Reshape((-1,))(x)

    return keras.models.Model(input_tensor, x), stage1, stage2, stage3, stage4, stage5, stage6


def GhostNet_CRELU(input_shape, num_classes=1000, weight_decay=0.0005, tensor=None):
    l2_reg = keras.regularizers.l2(weight_decay)
    if tensor is None:
        input_tensor = keras.layers.Input(input_shape)
    else:
        input_tensor = keras.layers.Input(tensor=tensor, shape=input_shape)

    # 112,112,16
    x = conv2d_bn(input_tensor, 32, (3, 3), (2, 2), 'same', l2_reg, cRelu=True)
    stage1 = x
    # 56,56,16
    x, _ = GhostBlock(x, 16, 16, (3, 3), (2, 2), 'same', l2_reg, True)
    stage2 = x
    # 28,28,24
    x, _ = GhostBlock(x, 72, 24, (3, 3), (2, 2), 'same', l2_reg, False)
    x, _ = GhostBlock(x, 88, 24, (3, 3), (1, 1), 'same', l2_reg, False)
    stage3 = x
    # 14,14,40
    x, _ = GhostBlock(x, 96, 40, (5, 5), (2, 2), 'same', l2_reg, True)
    x, _ = GhostBlock(x, 240, 40, (5, 5), (1, 1), 'same', l2_reg, True)
    x, _ = GhostBlock(x, 240, 40, (5, 5), (1, 1), 'same', l2_reg, True)
    x, _ = GhostBlock(x, 120, 48, (5, 5), (1, 1), 'same', l2_reg, True)
    x, _ = GhostBlock(x, 144, 48, (5, 5), (1, 1), 'same', l2_reg, True)
    stage4 = x
    x, _ = GhostBlock(x, 288, 96, (5, 5), (2, 2), 'same', l2_reg, True)

    x, _ = GhostBlock(x, 576, 96, (5, 5), (1, 1), 'same', l2_reg, True)
    x, _ = GhostBlock(x, 576, 96, (5, 5), (1, 1), 'same', l2_reg, True)
    x = conv2d_bn(x, 576, (1, 1), (1, 1), 'same', l2_reg, False)
    stage5 = x
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Reshape((1, 1, -1))(x)
    x = keras.layers.Conv2D(filters=1280, kernel_size=(1, 1), strides=(1, 1),
                            kernel_regularizer=l2_reg)(x)
    x = keras.layers.Activation('relu')(x)
    stage6 = x
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(filters=num_classes, kernel_size=(1, 1), strides=(1, 1),
                            kernel_regularizer=l2_reg, activation='softmax')(x)
    x = keras.layers.Reshape((-1,))(x)

    return keras.models.Model(input_tensor, x), stage1, stage2, stage3, stage4, stage5, stage6


def GhostNet_CRELU_CSP(input_shape, num_classes=1000, weight_decay=0.0005, tensor=None):
    l2_reg = keras.regularizers.l2(weight_decay)
    if tensor is None:
        input_tensor = keras.layers.Input(input_shape)
    else:
        input_tensor = keras.layers.Input(tensor=tensor, shape=input_shape)

    # 112,112,16
    x = conv2d_bn(input_tensor, 32, (3, 3), (2, 2), 'same', l2_reg, cRelu=True)
    stage1 = x
    # 56,56,16
    x_prev = depthwiseconv_bn(x, (3, 3), (2, 2), 'same', l2_reg)
    x, _ = GhostBlock(x, 16, 16, (3, 3), (2, 2), 'same', l2_reg, True)
    x = keras.layers.Concatenate()([x_prev, x])
    stage2 = x
    # 28,28,24
    x_prev = depthwiseconv_bn(x, (3, 3), (2, 2), 'same', l2_reg)
    x, _ = GhostBlock(x, 72, 24, (3, 3), (2, 2), 'same', l2_reg, False)
    x, _ = GhostBlock(x, 88, 24, (3, 3), (1, 1), 'same', l2_reg, False)
    x = keras.layers.Concatenate()([x_prev, x])
    stage3 = x
    # 14,14,40
    x_prev = depthwiseconv_bn(x, (3, 3), (2, 2), 'same', l2_reg)
    x, _ = GhostBlock(x, 96, 40, (5, 5), (2, 2), 'same', l2_reg, True)
    x, _ = GhostBlock(x, 240, 40, (5, 5), (1, 1), 'same', l2_reg, True)
    x, _ = GhostBlock(x, 240, 40, (5, 5), (1, 1), 'same', l2_reg, True)
    x, _ = GhostBlock(x, 120, 48, (5, 5), (1, 1), 'same', l2_reg, True)
    x, _ = GhostBlock(x, 144, 48, (5, 5), (1, 1), 'same', l2_reg, True)
    x = keras.layers.Concatenate()([x_prev, x])
    stage4 = x
    x_prev = depthwiseconv_bn(x, (3, 3), (2, 2), 'same', l2_reg)
    x, _ = GhostBlock(x, 288, 96, (5, 5), (2, 2), 'same', l2_reg, True)
    x, _ = GhostBlock(x, 576, 96, (5, 5), (1, 1), 'same', l2_reg, True)
    x, _ = GhostBlock(x, 576, 96, (5, 5), (1, 1), 'same', l2_reg, True)
    x = keras.layers.Concatenate()([x_prev, x])
    x = conv2d_bn(x, 576, (1, 1), (1, 1), 'same', l2_reg, False)
    stage5 = x
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Reshape((1, 1, -1))(x)
    x = keras.layers.Conv2D(filters=1280, kernel_size=(1, 1), strides=(1, 1),
                            kernel_regularizer=l2_reg)(x)
    x = keras.layers.Activation('relu')(x)
    stage6 = x
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(filters=num_classes, kernel_size=(1, 1), strides=(1, 1),
                            kernel_regularizer=l2_reg, activation='softmax')(x)
    x = keras.layers.Reshape((-1,))(x)

    return keras.models.Model(input_tensor, x), stage1, stage2, stage3, stage4, stage5, stage6


def GhostNet_CRELU_CSP_Large(input_shape, num_classes=1000, weight_decay=0.0005, tensor=None):
    l2_reg = keras.regularizers.l2(weight_decay)
    if tensor is None:
        input_tensor = keras.layers.Input(input_shape)
    else:
        input_tensor = keras.layers.Input(tensor=tensor, shape=input_shape)

    # 112,112,16
    x = conv2d_bn(input_tensor, 64, (3, 3), (2, 2), 'same', l2_reg, cRelu=True)
    stage1 = x
    # 56,56,16
    x_prev = depthwiseconv_bn(x, (3, 3), (2, 2), 'same', l2_reg)
    x, _ = GhostBlock(x, 16, 16, (3, 3), (1, 1), 'same', l2_reg, True)
    x, _ = GhostBlock(x, 48, 24, (3, 3), (2, 2), 'same', l2_reg, True)
    x = keras.layers.Concatenate()([x_prev, x])
    stage2 = x
    # 28,28,24
    x_prev = depthwiseconv_bn(x, (3, 3), (2, 2), 'same', l2_reg)
    x, _ = GhostBlock(x, 72, 24, (3, 3), (1, 1), 'same', l2_reg, True)
    x, _ = GhostBlock(x, 88, 40, (3, 3), (2, 2), 'same', l2_reg, True)
    x = keras.layers.Concatenate()([x_prev, x])
    stage3 = x
    # 14,14,40
    x_prev = depthwiseconv_bn(x, (3, 3), (2, 2), 'same', l2_reg)
    x, _ = GhostBlock(x, 200, 80, (5, 5), (1, 1), 'same', l2_reg, True)
    x, _ = GhostBlock(x, 184, 80, (5, 5), (1, 1), 'same', l2_reg, True)
    x, _ = GhostBlock(x, 184, 80, (5, 5), (1, 1), 'same', l2_reg, True)
    x, _ = GhostBlock(x, 480, 112, (5, 5), (1, 1), 'same', l2_reg, True)
    x, _ = GhostBlock(x, 672, 112, (5, 5), (1, 1), 'same', l2_reg, True)
    x, _ = GhostBlock(x, 672, 160, (5, 5), (2, 2), 'same', l2_reg, True)
    x = keras.layers.Concatenate()([x_prev, x])
    stage4 = x
    x_prev = depthwiseconv_bn(x, (3, 3), (2, 2), 'same', l2_reg)
    x, _ = GhostBlock(x, 960, 160, (5, 5), (1, 1), 'same', l2_reg, True)
    x, _ = GhostBlock(x, 960, 160, (5, 5), (1, 1), 'same', l2_reg, True)
    x, _ = GhostBlock(x, 960, 160, (5, 5), (1, 1), 'same', l2_reg, True)
    x, _ = GhostBlock(x, 960, 160, (5, 5), (2, 2), 'same', l2_reg, True)
    x = keras.layers.Concatenate()([x_prev, x])
    x = conv2d_bn(x, 960, (1, 1), (1, 1), 'same', l2_reg, False)
    stage5 = x
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Reshape((1, 1, -1))(x)
    x = keras.layers.Conv2D(filters=1280, kernel_size=(1, 1), strides=(1, 1),
                            kernel_regularizer=l2_reg)(x)
    x = keras.layers.Activation('relu')(x)
    stage6 = x
    x = keras.layers.Dropout(0.2)(x)
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


    keras.backend.set_learning_phase(1)
    model, stage1, stage2, stage3, stage4, stage5, stage6 = GhostNet_CRELU_CSP_Large((256, 256, 3))
    model.summary()
    print(get_flops(model))
    print(stage1, stage2, stage3, stage4, stage5, stage6)
