import tensorflow as tf
import keras
import math
import keras.backend as K


def conv2d_bn(inputs, filters, kernel_size, strides, padding, regularizer, use_se=False, activation='relu',
              cRelu=False, name=None, zero_gamma=False):
    if cRelu:
        filters = int(filters / 2)
    tensor = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                 strides=strides, padding=padding, use_bias=False,
                                 kernel_regularizer=regularizer,
                                 kernel_initializer=keras.initializers.he_normal())(inputs)
    if zero_gamma:
        tensor = keras.layers.BatchNormalization(gamma_initializer=keras.initializers.Zeros())(tensor)
    else:
        tensor = keras.layers.BatchNormalization()(tensor)

    if activation == 'h-swish':
        tensor = keras.layers.Activation(h_swish)(tensor)
    elif activation == 'relu6':
        tensor = keras.layers.Activation(tf.nn.relu6)(tensor)
    elif activation == 'relu':
        if cRelu:
            tensor = ModifiedCReLU()(tensor)
        else:
            tensor = keras.layers.Activation('relu')(tensor)
    else:
        tensor = keras.layers.Activation(activation)(tensor)

    if use_se:
        n, h, w, c = tensor.shape.as_list()
        avg_pool = keras.layers.GlobalAveragePooling2D()(tensor)
        avg_pool = keras.layers.Reshape((1, 1, c))(avg_pool)
        squeeze = keras.layers.Conv2D(filters=int(c / 4), kernel_size=(1, 1),
                                      use_bias=False, kernel_regularizer=regularizer,
                                      kernel_initializer=keras.initializers.he_normal())(avg_pool)
        squeeze = keras.layers.BatchNormalization()(squeeze)
        squeeze = keras.layers.Activation('relu')(squeeze)

        excitation = keras.layers.Conv2D(filters=c, kernel_size=(1, 1),
                                         use_bias=False, kernel_regularizer=regularizer,
                                         kernel_initializer=keras.initializers.he_normal())(squeeze)
        excitation = keras.layers.BatchNormalization()(excitation)
        excitation = keras.layers.Activation('sigmoid')(excitation)
        tensor = keras.layers.Multiply()([excitation, tensor])
    if name is not None:
        tensor = keras.layers.Lambda(lambda x: x, name=name)(tensor)
    return tensor


def depthwiseconv_bn(inputs, kernel_size, strides, padding, regularizer, activation='relu', depth_multiplier=1,
                     dilation_rate=1, zero_gamma=False):
    tensor = keras.layers.DepthwiseConv2D(kernel_size=kernel_size,
                                          strides=strides, padding=padding,
                                          kernel_regularizer=regularizer,
                                          kernel_initializer=keras.initializers.he_normal(),
                                          use_bias=False,
                                          depth_multiplier=depth_multiplier,
                                          dilation_rate=dilation_rate)(inputs)
    if zero_gamma:
        tensor = keras.layers.BatchNormalization(gamma_initializer=keras.initializers.Zeros())(tensor)
    else:
        tensor = keras.layers.BatchNormalization()(tensor)
    if activation == 'h-swish':
        tensor = keras.layers.Activation(h_swish)(tensor)
    elif activation == 'relu6':
        tensor = keras.layers.Activation(tf.nn.relu6)(tensor)
    else:
        tensor = keras.layers.Activation(activation)(tensor)
    return tensor


def MBBlock_linear(inputs, exp, out, kernel_size, strides, padding, regularizer, use_se=False, drop_out=False,
                   cRelu=False, activation='relu'):
    n, h, w, c = inputs.shape.as_list()
    tensor = conv2d_bn(inputs, exp, (1, 1), (1, 1), 'same', regularizer, cRelu=cRelu, activation=activation)
    tensor = depthwiseconv_bn(tensor, kernel_size, strides, padding, regularizer, activation=activation)
    if use_se:
        pool = keras.layers.GlobalAveragePooling2D()(tensor)
        pool = keras.layers.Reshape((1, 1, exp))(pool)
        squeeze = keras.layers.Conv2D(filters=int(exp / 4), kernel_size=(1, 1),
                                      use_bias=False, kernel_regularizer=regularizer,
                                      kernel_initializer=keras.initializers.he_normal())(pool)
        squeeze = keras.layers.BatchNormalization()(squeeze)
        squeeze = keras.layers.Activation('relu')(squeeze)

        excitation = keras.layers.Conv2D(filters=exp, kernel_size=(1, 1),
                                         use_bias=False, kernel_regularizer=regularizer,
                                         kernel_initializer=keras.initializers.he_normal())(squeeze)
        excitation = keras.layers.BatchNormalization()(excitation)
        excitation = keras.layers.Activation('sigmoid')(excitation)
        tensor = keras.layers.Multiply()([tensor, excitation])

    if strides == (1, 1) and c == out:  # Residual connection
        tensor = conv2d_bn(tensor, out, (1, 1), (1, 1), 'same', regularizer, activation='linear', zero_gamma=True)
        if drop_out:
            tensor = keras.layers.Dropout(0.5)(tensor)
        tensor = keras.layers.Add()([inputs, tensor])
    else:
        tensor = conv2d_bn(tensor, out, (1, 1), (1, 1), 'same', regularizer, activation='linear', zero_gamma=False)
    return tensor


def G_module(inputs, filters, kernel_size, strides, padding, regularizer, trans_size=(3, 3), ratio=2,
             activation='relu', cRelu=False, zero_gamma=False):
    if cRelu:
        activation = 'linear'
        filters = filters // 2
    init_channels = math.ceil(filters / ratio)
    x = conv2d_bn(inputs, filters=init_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                  regularizer=regularizer, activation=activation, zero_gamma=zero_gamma)
    if ratio == 1:
        return x
    dw = depthwiseconv_bn(x, kernel_size=trans_size, strides=(1, 1), padding='same', regularizer=regularizer,
                          activation=activation, depth_multiplier=ratio - 1, zero_gamma=zero_gamma)
    x = keras.layers.Concatenate()([x, dw])
    if cRelu:
        x = ModifiedCReLU()(x)
    return x


def GhostBlock(inputs, exp, out, kernel_size, strides, padding, regularizer, use_se=False,
               use_eSE=False,
               activation='relu', ratio=2):
    n, h, w, c = inputs.shape.as_list()
    tensor = G_module(inputs, filters=exp, kernel_size=(1, 1), strides=(1, 1), padding='same', regularizer=regularizer,
                      ratio=ratio, activation=activation)
    exp_layer = tensor
    if strides == (2, 2):
        tensor = depthwiseconv_bn(tensor, kernel_size, strides, padding, regularizer, activation='linear')

    if use_se:
        pool = keras.layers.GlobalAveragePooling2D()(tensor)
        pool = keras.layers.Reshape((1, 1, exp))(pool)
        squeeze = keras.layers.Conv2D(filters=int(exp / 4), kernel_size=(1, 1),
                                      use_bias=False, kernel_regularizer=regularizer,
                                      kernel_initializer=keras.initializers.he_normal())(pool)
        squeeze = keras.layers.BatchNormalization()(squeeze)
        squeeze = keras.layers.Activation('relu')(squeeze)

        excitation = keras.layers.Conv2D(filters=exp, kernel_size=(1, 1),
                                         use_bias=False, kernel_regularizer=regularizer,
                                         kernel_initializer=keras.initializers.he_normal())(squeeze)
        excitation = keras.layers.BatchNormalization()(excitation)
        excitation = keras.layers.Activation('sigmoid')(excitation)
        tensor = keras.layers.Multiply()([tensor, excitation])

    tensor = G_module(tensor, filters=out, kernel_size=(1, 1), strides=(1, 1), padding='same', regularizer=regularizer,
                      activation='linear', ratio=ratio)
    if strides == (1, 1):
        if c != out:
            inputs = conv2d_bn(inputs, out, (1, 1), strides, padding, regularizer, activation='linear')

    if strides == (2, 2):
        inputs = depthwiseconv_bn(inputs, kernel_size, (2, 2), padding, regularizer, activation='linear')
        inputs = conv2d_bn(inputs, out, (1, 1), (1, 1), padding, regularizer, activation='linear')
    tensor = keras.layers.Add()([inputs, tensor])
    return tensor, exp_layer


def GhostBlockRES(inputs, exp, out, kernel_size, strides, padding, regularizer, use_se=False,
                  activation='relu', ratio=2):
    n, h, w, c = inputs.shape.as_list()
    tensor = G_module(inputs, filters=exp, kernel_size=(1, 1), strides=(1, 1), padding='same', regularizer=regularizer,
                      ratio=ratio, activation=activation)
    exp_layer = tensor
    if strides == (2, 2):
        tensor = depthwiseconv_bn(tensor, kernel_size, strides, padding, regularizer, activation='linear')

    if use_se:
        pool = keras.layers.GlobalAveragePooling2D()(tensor)
        pool = keras.layers.Reshape((1, 1, exp))(pool)
        squeeze = keras.layers.Conv2D(filters=int(exp / 4), kernel_size=(1, 1),
                                      use_bias=False, kernel_regularizer=regularizer,
                                      kernel_initializer=keras.initializers.he_normal())(pool)
        squeeze = keras.layers.BatchNormalization()(squeeze)
        squeeze = keras.layers.Activation('relu')(squeeze)

        excitation = keras.layers.Conv2D(filters=exp, kernel_size=(1, 1),
                                         use_bias=False, kernel_regularizer=regularizer,
                                         kernel_initializer=keras.initializers.he_normal())(squeeze)
        excitation = keras.layers.BatchNormalization()(excitation)
        excitation = keras.layers.Activation('sigmoid')(excitation)
        tensor = keras.layers.Multiply()([tensor, excitation])

    tensor = G_module(tensor, filters=out, kernel_size=(1, 1), strides=(1, 1), padding='same', regularizer=regularizer,
                      activation='linear', ratio=ratio)
    if strides == (1, 1):
        if c != out:
            inputs = G_module(inputs, out, (1, 1), strides, padding, regularizer, activation='linear', ratio=ratio)

    if strides == (2, 2):
        inputs = depthwiseconv_bn(inputs, kernel_size, (2, 2), padding, regularizer, activation='linear')
        inputs = conv2d_bn(inputs, out, (1, 1), (1, 1), padding, regularizer, activation='linear')
    tensor = keras.layers.Add()([inputs, tensor])
    return tensor, exp_layer


def GhostBlock_HSE(inputs, exp, out, kernel_size, strides, padding, regularizer, use_se=False,
                   activation='relu', ratio=2):
    n, h, w, c = inputs.shape.as_list()
    tensor = G_module(inputs, filters=exp, kernel_size=(1, 1), strides=(1, 1), padding='same', regularizer=regularizer,
                      ratio=ratio, activation=activation)
    exp_layer = tensor
    if strides == (2, 2):
        tensor = depthwiseconv_bn(tensor, kernel_size, strides, padding, regularizer, activation='linear')

    if use_se:
        pool = keras.layers.GlobalAveragePooling2D()(tensor)
        pool = keras.layers.Reshape((1, 1, exp))(pool)
        squeeze = keras.layers.Conv2D(filters=int(exp / 4), kernel_size=(1, 1),
                                      use_bias=False, kernel_regularizer=regularizer,
                                      kernel_initializer=keras.initializers.he_normal())(pool)
        squeeze = keras.layers.BatchNormalization()(squeeze)
        squeeze = keras.layers.Activation('relu')(squeeze)

        excitation = keras.layers.Conv2D(filters=exp, kernel_size=(1, 1),
                                         use_bias=False, kernel_regularizer=regularizer,
                                         kernel_initializer=keras.initializers.he_normal())(squeeze)
        excitation = keras.layers.BatchNormalization()(excitation)
        excitation = keras.layers.Activation('hard_sigmoid')(excitation)
        tensor = keras.layers.Multiply()([tensor, excitation])

    tensor = G_module(tensor, filters=out, kernel_size=(1, 1), strides=(1, 1), padding='same', regularizer=regularizer,
                      activation='linear', ratio=ratio)
    if strides == (1, 1):
        if c != out:
            inputs = conv2d_bn(inputs, out, (1, 1), strides, padding, regularizer, activation='linear')

    if strides == (2, 2):
        inputs = depthwiseconv_bn(inputs, kernel_size, (2, 2), padding, regularizer, activation='linear')
        inputs = conv2d_bn(inputs, out, (1, 1), (1, 1), padding, regularizer, activation='linear')
    tensor = keras.layers.Add()([inputs, tensor])
    return tensor, exp_layer


def CBAM(x, regularizer):
    def shared_mlp(inputs, features, r=8):
        reduction = int(features / r)
        mlp_1 = keras.layers.Dense(reduction, activation='relu', kernel_regularizer=regularizer)(inputs)
        mlp_2 = keras.layers.Dense(features, activation='relu', kernel_regularizer=regularizer)(mlp_1)
        return keras.models.Model(inputs, mlp_2)

    # channel attention
    n, h, w, c = x.shape.as_list()
    mlp = shared_mlp(keras.layers.Input(shape=(None, c)), c)
    max_pool = keras.layers.GlobalMaxPooling2D()(x)
    avg_pool = keras.layers.GlobalAveragePooling2D()(x)
    max_pool = mlp(max_pool)
    avg_pool = mlp(avg_pool)
    channel_attention = keras.layers.Add()([max_pool, avg_pool])
    channel_attention = keras.layers.Activation('sigmoid')(channel_attention)
    channel_attention = keras.layers.Reshape((1, 1, c))(channel_attention)
    x = keras.layers.Multiply()([x, channel_attention])

    # spatial attention
    max_pool = keras.layers.Lambda(lambda inputs: tf.reduce_max(inputs, axis=-1, keep_dims=True))(x)
    avg_pool = keras.layers.Lambda(lambda inputs: tf.reduce_mean(inputs, axis=-1, keep_dims=True))(x)
    spatial_attention = keras.layers.Concatenate(axis=-1)([max_pool, avg_pool])
    spatial_attention = keras.layers.Conv2D(1, (7, 7), padding='same', kernel_regularizer=regularizer,
                                            activation='sigmoid')(spatial_attention)
    x = keras.layers.Multiply()([x, spatial_attention])
    return x


def h_swish(x):
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0


class ModifiedCReLU(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ModifiedCReLU, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        neg_input = tf.negative(inputs)
        neg_input = tf.add(tf.multiply(neg_input, self.scale), self.shift)
        tensor = tf.concat([inputs, neg_input], axis=-1)
        tensor = tf.nn.relu(tensor)
        return tensor

    def build(self, input_shape):
        self.scale = self.add_weight('scale', shape=(1, 1, 1, input_shape[3]),
                                     dtype='float32', initializer=keras.initializers.ones())
        self.shift = self.add_weight('shift', shape=(1, 1, 1, input_shape[3]),
                                     dtype='float32', initializer=keras.initializers.zeros())

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2], input_shape[3] * 2


class PartialResidual(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PartialResidual, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        tensor_1 = inputs[0]
        tensor_2 = inputs[1]
        _, _, _, c1 = tensor_1.shape.as_list()
        _, _, _, c2 = tensor_2.shape.as_list()
        c = c1 // 2

        if c > c2:
            c = c2
        t1 = tensor_1[:, :, :, :c]
        t2 = tensor_1[:, :, :, c:]
        t1 = tf.add(t1, tensor_2[:, :, :, :c])
        tensor = tf.concat([t1, t2], axis=-1)

        return tensor

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def GhostNextBlock(inputs, filters, cardinality, kernel_size, strides, regularizer):
    n, h, w, c = inputs.shape.as_list()
    card_filters = filters // cardinality
    tensors = []
    for i in range(cardinality):
        x = G_module(inputs, card_filters, (1, 1), (1, 1), 'same', regularizer, trans_size=kernel_size)
        if strides == (2, 2):
            x = depthwiseconv_bn(x, kernel_size, strides, 'same', regularizer, activation='linear')
        tensors.append(x)
    x = keras.layers.Concatenate()(tensors)
    x = G_module(x, filters, (1, 1), (1, 1), 'same', regularizer, trans_size=kernel_size)

    if strides == (1, 1):
        if c != filters:
            inputs = conv2d_bn(inputs, filters, (1, 1), strides, 'same', regularizer, activation='linear')

    if strides == (2, 2):
        inputs = depthwiseconv_bn(inputs, kernel_size, (2, 2), 'same', regularizer, activation='linear')
        inputs = conv2d_bn(inputs, filters, (1, 1), (1, 1), 'same', regularizer, activation='linear')
    x = keras.layers.Add()([inputs, x])
    return x
