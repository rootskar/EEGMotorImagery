#!/usr/bin/env python
# -*- coding: utf-8 -*-

import model
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.models import Model


def EEGNet(nb_classes, Chans=64, Samples=128,
           dropoutRate=0.5, kernLength=64, F1=8,
           D=2, F2=16, norm_rate=0.25, dropoutType='Dropout', cpu=False):
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    if cpu:
        input_shape = (Samples, Chans, 1)
        conv_filters = (kernLength, 1)
        depth_filters = (1, Chans)
        pool_size = (6, 1)
        pool_size2 = (12, 1)
        separable_filters = (20, 1)
        axis = -1
    else:
        input_shape = (1, Chans, Samples)
        conv_filters = (1, kernLength)
        depth_filters = (Chans, 1)
        pool_size = (1, 6)
        pool_size2 = (1, 12)
        separable_filters = (1, 20)
        axis = 1

    input1 = Input(shape=input_shape)
    block1 = Conv2D(F1, conv_filters, padding='same',
                    input_shape=input_shape,
                    use_bias=False)(input1)
    block1 = BatchNormalization(axis=axis)(block1)
    block1 = DepthwiseConv2D(depth_filters, use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization(axis=axis)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D(pool_size)(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, separable_filters,
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization(axis=axis)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D(pool_size2)(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten')(block2)

    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)


def DeepConvNet(nb_classes, Chans=64, Samples=256,
                dropoutRate=0.5, cpu=False):
    if cpu:
        input_shape = (Samples, Chans, 1)
        input_main = Input(input_shape)
        conv_filters = (2, 1)
        conv_filters2 = (1, Chans)
        pool = (2, 1)
        strides = (2, 1)
        axis = -1
    else:
        input_shape = (1, Chans, Samples)
        input_main = Input(input_shape)
        conv_filters = (1, 2)
        conv_filters2 = (Chans, 1)
        pool = (1, 2)
        strides = (1, 2)
        axis = 1

    # start the model
    block1 = Conv2D(25, conv_filters,
                    input_shape=input_shape,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(25, conv_filters2,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=axis, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=pool, strides=strides)(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = Conv2D(50, conv_filters,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block2 = BatchNormalization(axis=axis, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D(pool_size=pool, strides=strides)(block2)
    block2 = Dropout(dropoutRate)(block2)

    block3 = Conv2D(100, conv_filters,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block3 = BatchNormalization(axis=axis, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling2D(pool_size=pool, strides=strides)(block3)
    block3 = Dropout(dropoutRate)(block3)

    block4 = Conv2D(200, conv_filters,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block4 = BatchNormalization(axis=axis, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)
    block4 = MaxPooling2D(pool_size=pool, strides=strides)(block4)
    block4 = Dropout(dropoutRate)(block4)

    flatten = Flatten()(block4)

    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


# need these for ShallowConvNet
def square(x):
    return K.square(x)


def log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))


def ShallowConvNet(nb_classes, Chans=64, Samples=128, dropoutRate=0.5, cpu=False):
    if cpu:
        input_shape = (Samples, Chans, 1)
        conv_filters = (25, 1)
        conv_filters2 = (1, Chans)
        pool_size = (45, 1)
        strides = (15, 1)
        axis = -1
    else:
        input_shape = (1, Chans, Samples)
        conv_filters = (1, 20)
        conv_filters2 = (Chans, 1)
        pool_size = (1, 45)
        strides = (1, 15)
        axis = 1

    input_main = Input(input_shape)
    block1 = Conv2D(20, conv_filters,
                    input_shape=input_shape,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(20, conv_filters2, use_bias=False,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=axis, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=pool_size, strides=strides)(block1)
    block1 = Activation(log)(block1)
    block1 = Dropout(dropoutRate)(block1)
    flatten = Flatten()(block1)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


from tensorflow.keras.layers import concatenate


def EEGNet_fusion(nb_classes, Chans=64, Samples=128,
                  dropoutRate=0.5, norm_rate=0.25, dropoutType='Dropout', cpu=False):
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    if cpu:
        input_shape = (Samples, Chans, 1)
        conv_filters = (64, 1)
        conv_filters2 = (96, 1)
        conv_filters3 = (128, 1)

        depth_filters = (1, Chans)
        pool_size = (4, 1)
        pool_size2 = (8, 1)
        separable_filters = (8, 1)
        separable_filters2 = (16, 1)
        separable_filters3 = (32, 1)

        axis = -1
    else:
        input_shape = (1, Chans, Samples)
        conv_filters = (1, 64)
        conv_filters2 = (1, 96)
        conv_filters3 = (1, 128)

        depth_filters = (Chans, 1)
        pool_size = (1, 4)
        pool_size2 = (1, 8)
        separable_filters = (1, 8)
        separable_filters2 = (1, 16)
        separable_filters3 = (1, 32)

        axis = 1

    F1 = 8
    F1_2 = 16
    F1_3 = 32
    F2 = 16
    F2_2 = 32
    F2_3 = 64
    D = 2
    D2 = 2
    D3 = 2

    input1 = Input(shape=input_shape)
    block1 = Conv2D(F1, conv_filters, padding='same',
                    input_shape=input_shape,
                    use_bias=False)(input1)
    block1 = BatchNormalization(axis=axis)(block1)
    block1 = DepthwiseConv2D(depth_filters, use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization(axis=axis)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D(pool_size)(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, separable_filters,
                             use_bias=False, padding='same')(block1)  # 8
    block2 = BatchNormalization(axis=axis)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D(pool_size2)(block2)
    block2 = dropoutType(dropoutRate)(block2)
    block2 = Flatten()(block2)  # 13

    # 8 - 13

    input2 = Input(shape=input_shape)
    block3 = Conv2D(F1_2, conv_filters2, padding='same',
                    input_shape=input_shape,
                    use_bias=False)(input2)
    block3 = BatchNormalization(axis=axis)(block3)
    block3 = DepthwiseConv2D(depth_filters, use_bias=False,
                             depth_multiplier=D2,
                             depthwise_constraint=max_norm(1.))(block3)
    block3 = BatchNormalization(axis=axis)(block3)
    block3 = Activation('elu')(block3)
    block3 = AveragePooling2D(pool_size)(block3)
    block3 = dropoutType(dropoutRate)(block3)

    block4 = SeparableConv2D(F2_2, separable_filters2,
                             use_bias=False, padding='same')(block3)  # 22
    block4 = BatchNormalization(axis=axis)(block4)
    block4 = Activation('elu')(block4)
    block4 = AveragePooling2D(pool_size2)(block4)
    block4 = dropoutType(dropoutRate)(block4)
    block4 = Flatten()(block4)  # 27
    # 22 - 27

    input3 = Input(shape=input_shape)
    block5 = Conv2D(F1_3, conv_filters3, padding='same',
                    input_shape=input_shape,
                    use_bias=False)(input3)
    block5 = BatchNormalization(axis=axis)(block5)
    block5 = DepthwiseConv2D(depth_filters, use_bias=False,
                             depth_multiplier=D3,
                             depthwise_constraint=max_norm(1.))(block5)
    block5 = BatchNormalization(axis=axis)(block5)
    block5 = Activation('elu')(block5)
    block5 = AveragePooling2D(pool_size)(block5)
    block5 = dropoutType(dropoutRate)(block5)

    block6 = SeparableConv2D(F2_3, separable_filters3,
                             use_bias=False, padding='same')(block5)  # 36
    block6 = BatchNormalization(axis=axis)(block6)
    block6 = Activation('elu')(block6)
    block6 = AveragePooling2D(pool_size2)(block6)
    block6 = dropoutType(dropoutRate)(block6)
    block6 = Flatten()(block6)  # 41

    # 36 - 41

    merge_one = concatenate([block2, block4])
    merge_two = concatenate([merge_one, block6])

    flatten = Flatten()(merge_two)

    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)

    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=[input1, input2, input3], outputs=softmax)


def get_models(trial_type, nb_classes, samples, use_cpu):
    if use_cpu:
        K.set_image_data_format('channels_last')
    else:
        K.set_image_data_format('channels_first')
    return {
        'EEGNet_fusion': model.Model('EEGNet_fusion', trial_type, [(0, 8), (14, 22), (28, 36)],
                                     EEGNet_fusion(nb_classes, Samples=samples, cpu=use_cpu), multi_branch=True),
        'EEGNet': model.Model('EEGNet', trial_type, [(0, 8)], EEGNet(nb_classes, Samples=samples, cpu=use_cpu)),
        'ShallowConvNet': model.Model('ShallowConvNet', trial_type, [(0, 2)],
                                      ShallowConvNet(nb_classes, Samples=samples, cpu=use_cpu)),
        'DeepConvNet': model.Model('DeepConvNet', trial_type, [(0, 8), (14, 22), (28, 36)],
                                   DeepConvNet(nb_classes, Samples=samples, cpu=use_cpu)),
    }
