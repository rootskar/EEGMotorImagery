#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

def EEGNet(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout', cpu=False):
    
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
        pool_size = (4, 1)
        pool_size2 = (8, 1)
        separable_filters = (16, 1)
    else:
        input_shape = (1, Chans, Samples)
        conv_filters = (1, kernLength)
        depth_filters = (Chans, 1)
        pool_size = (1, 6)
        pool_size2 = (1, 12)
        separable_filters = (1, 20)

    ##################################################################
    input1   = Input(shape = input_shape)
    block1       = Conv2D(F1, conv_filters, padding = 'same',
                                   input_shape = input_shape,
                                   use_bias = False)(input1)
    block1       = BatchNormalization(axis = 1)(block1)
    block1       = DepthwiseConv2D(depth_filters, use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization(axis = 1)(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D(pool_size)(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, separable_filters,
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization(axis = 1)(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D(pool_size2)(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)


def DeepConvNet(nb_classes, Chans = 64, Samples = 256,
                dropoutRate = 0.5, cpu=False):

    if cpu:
        input_shape = (Samples, Chans, 1)
        input_main   = Input(input_shape)
        conv_filters = (5, 1)
        conv_filters2 = (1, Chans)
        pool = (2, 1)
        strides = (2, 1)
    else:
        input_shape = (1, Chans, Samples)
        input_main   = Input(input_shape)
        conv_filters = (1, 5)
        conv_filters2 = (Chans, 1)
        pool = (1, 2)
        strides = (1, 2)

    # start the model
    block1       = Conv2D(25, conv_filters, 
                                 input_shape=input_shape,
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = Conv2D(25, conv_filters2,
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1       = Activation('elu')(block1)
    block1       = MaxPooling2D(pool_size=pool, strides=strides)(block1)
    block1       = Dropout(dropoutRate)(block1)
  
    block2       = Conv2D(50, conv_filters,
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block2       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2       = Activation('elu')(block2)
    block2       = MaxPooling2D(pool_size=pool, strides=strides)(block2)
    block2       = Dropout(dropoutRate)(block2)
    
    block3       = Conv2D(100, conv_filters,
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block2)
    block3       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3       = Activation('elu')(block3)
    block3       = MaxPooling2D(pool_size=pool, strides=strides)(block3)
    block3       = Dropout(dropoutRate)(block3)
    
    block4       = Conv2D(200, conv_filters,
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block3)
    block4       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4       = Activation('elu')(block4)
    block4       = MaxPooling2D(pool_size=pool, strides=strides)(block4)
    block4       = Dropout(dropoutRate)(block4)
    
    flatten      = Flatten()(block4)
    
    dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)


# need these for ShallowConvNet
def square(x):
    return K.square(x)

def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000))   


def ShallowConvNet(nb_classes, Chans = 64, Samples = 128, dropoutRate = 0.5, cpu = False):

    if cpu:
        input_shape = (Samples, Chans, 1)
        conv_filters = (13, 1)
        conv_filters2 = (1, Chans)
        pool_size = (35, 1)
        strides = (7, 1)
    else:
        input_shape = (1, Chans, Samples)
        conv_filters = (1, 25)
        conv_filters2 = (Chans, 1)
        pool_size = (1, 45)
        strides = (1, 15)

    # start the model
    input_main   = Input(input_shape)
    block1       = Conv2D(40, conv_filters, 
                                 input_shape=input_shape,
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = Conv2D(40, conv_filters2, use_bias=False, 
                          kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1       = Activation(square)(block1)
    block1       = AveragePooling2D(pool_size=pool_size, strides=strides)(block1)
    block1       = Activation(log)(block1)
    block1       = Dropout(dropoutRate)(block1)
    flatten      = Flatten()(block1)
    dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)

from tensorflow.keras.layers import Conv3D

def ConvNet3D(nb_classes, Chans = 64, Chunks = 8, Samples = 80, dropoutRate = 0.45, cpu = False):

    if cpu:
        input_shape = (Samples, Chunks, Chans, 1)
        conv_filters = (5, 3, 3)
        conv_strides = (4, 2, 2)
        conv_filters2 = (3, 2, 2)
        conv_strides2 = (2, 2, 2)
        conv_filters3 = (3, 1, 2)
        conv_strides3 = (2, 2, 2)
    else:
        input_shape = (1, Chans, Chunks, Samples)
        conv_filters = (3, 3, 5)
        conv_strides = (2, 2, 4)
        conv_filters2 = (2, 2, 3)
        conv_strides2 = (2, 2, 2)
        conv_filters3 = (2, 1, 3)
        conv_strides3 = (2, 2, 2)

    input_main = Input(shape=input_shape)

    block1 = Conv3D(16, conv_filters, strides=conv_strides)(input_main)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = Conv3D(32, conv_filters2, strides=conv_strides2)(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = Dropout(dropoutRate)(block2)

    block3 = Conv3D(64, conv_filters3, strides=conv_strides3)(block2)
    block3 = BatchNormalization()(block3)
    block3 = Activation('elu')(block3)
    block3 = Dropout(dropoutRate)(block3)

    flatten = Flatten()(block3)
    
    out = Dense(nb_classes)(flatten)
    out = Activation('softmax')(out)

    return Model(inputs=input_main, outputs=out)

def DeeperConvNet(nb_classes, Chans = 64, Samples = 640, dropoutRate = 0.45, cpu = False):

    if cpu:
        input_shape = (Samples, Chans, 1)
    else:
        input_shape = (1, Chans, Samples)

    x = Input(shape=input_shape)
    c1 = Conv2D(64, (3, 3), padding='same')(x)
    a1 = Activation('relu')(c1)
    c2 = Conv2D(64, (3, 3), padding='same')(a1)
    a2 = Activation('relu')(c2)
    b1 = BatchNormalization()(a2)
    m1 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(b1)
    d1 = Dropout(0.2)(m1)

    c3 = Conv2D(128, (3, 3), padding='same')(d1)
    a3 = Activation('relu')(c3)
    c4 = Conv2D(128, (3, 3), padding='same')(a3)
    a4 = Activation('relu')(c4)
    b2 = BatchNormalization()(a4)
    m2 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(b2)
    d2 = Dropout(0.2)(m2)

    c5 = Conv2D(256, (3, 3), padding='same')(d2)
    a5 = Activation('relu')(c5)
    c6 = Conv2D(256, (3, 3), padding='same')(a5)
    a6 = Activation('relu')(c6)
    c7 = Conv2D(256, (3, 3), padding='same')(a6)
    a7 = Activation('relu')(c7)
    b3 = BatchNormalization()(a7)
    m3 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(b3)
    d3 = Dropout(0.2)(m3)

    c8 = Conv2D(512, (3, 3), padding='same')(d3)
    a8 = Activation('relu')(c8)
    c9 = Conv2D(512, (3, 3), padding='same')(a8)
    a9 = Activation('relu')(c9)
    c10 = Conv2D(512, (3, 3), padding='same')(a9)
    a10 = Activation('relu')(c10)
    b4 = BatchNormalization()(a10)
    m4 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(b4)
    d4 = Dropout(0.2)(m4)

    c11 = Conv2D(512, (3, 3), padding='same')(d4)
    a11 = Activation('relu')(c11)
    c12 = Conv2D(512, (3, 3), padding='same')(a11)
    a12 = Activation('relu')(c12)
    c13 = Conv2D(512, (3, 3), padding='same')(a12)
    a13 = Activation('relu')(c13)
    b5 = BatchNormalization()(a13)
    m5 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(b5)
    d5 = Dropout(0.2)(m5)

    f = Flatten()(d5)
    de1 = Dense(units=4096, activation='relu')(f)
    de2 = Dense(units=4096, activation='relu')(de1)
    d6 = Dropout(0.5)(de2)
    de3 = Dense(units=nb_classes, activation='softmax')(d6)

    return Model(inputs=x, outputs=de3)

def ConvNet2D(nb_classes, Chans = 64, Samples = 640, dropoutRate = 0.45, cpu = False):

    if cpu:
        input_shape = (Samples, Chans, 1)
        conv_filters = (4, 2)
        conv_filters2 = (4, 2)
        pool_size = (4, 2)
        pool_size2 = (4, 4)
    else:
        input_shape = (1, Chans, Samples)
        conv_filters = (2, 4)
        conv_filters2 = (2, 4)
        pool_size = (2, 4)
        pool_size2 = (4, 4)
    
    input_layer = Input(shape=input_shape)
    x = Conv2D(64, conv_filters)(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=pool_size)(x)

    y = Conv2D(128, conv_filters2)(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = MaxPooling2D(pool_size=pool_size2)(y)

    f = Flatten()(y)

    k = Dense(1024)(f)
    k = BatchNormalization()(k)
    k = Activation('relu')(k)
    k = Dropout(0.45)(k)

    out = Dense(nb_classes)(k)
    out = Activation('softmax')(out)

    return Model(inputs=input_layer, outputs=out)

def ConvNet3D_2(nb_classes, Chans = 64, Chunks = 8, Samples = 80, dropoutRate = 0.45, cpu = False):
    if cpu:
        input_shape = (Samples, Chunks, Chans, 1)
        conv_filters = (5, 3, 3)
        conv_strides = (4, 2, 2)
        conv_filters2 = (3, 2, 2)
        conv_strides2 = (2, 2, 2)
        conv_filters3 = (3, 1, 2)
        conv_strides3 = (2, 2, 2)
    else:
        input_shape = (1, Chans, Chunks, Samples)
        conv_filters = (3, 3, 5)
        conv_strides = (2, 2, 4)
        conv_filters2 = (2, 2, 3)
        conv_strides2 = (2, 2, 2)
        conv_filters3 = (2, 1, 3)
        conv_strides3 = (2, 2, 2)
    
    input_layer = Input(shape=input_shape)
    x = Conv3D(16, conv_filters, strides=conv_strides)(input_layer)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(dropoutRate)(x)

    y = Conv3D(32, conv_filters2, strides=conv_strides2)(x)
    y = BatchNormalization()(y)
    y = Activation('elu')(y)
    y = Dropout(dropoutRate)(y)

    z = Conv3D(64, conv_filters3, strides=conv_strides3)(y)
    z = BatchNormalization()(z)
    z = Activation('elu')(z)
    z = Dropout(dropoutRate)(z)

    f = Flatten()(z)

    h = Dense(64)(f)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    k = Dense(128)(h)
    k = BatchNormalization()(k)
    k = Activation('relu')(k)

    out = Dense(nb_classes)(k)
    out = Activation('softmax')(out)

    return Model(inputs=input_layer, outputs=out)

from tensorflow.keras.layers import concatenate

def EEGNet_fusion(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, norm_rate = 0.25, dropoutType = 'Dropout'):
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
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
    F1 = 8
    F1_2 = 16
    F1_3 = 32
    F2 = 16
    F2_2 = 32
    F2_3 = 64
    D = 2
    D2 = 2
    D3 = 2
    

    ##################################################################
    input1       = Input(shape = input_shape)
    block1       = Conv2D(F1, conv_filters, padding = 'same',
                                   input_shape = input_shape,
                                   use_bias = False)(input1)
    block1       = BatchNormalization(axis = 1)(block1)
    block1       = DepthwiseConv2D(depth_filters, use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization(axis = 1)(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D(pool_size)(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, separable_filters,
                                   use_bias = False, padding = 'same')(block1) # 8
    block2       = BatchNormalization(axis = 1)(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D(pool_size2)(block2)
    block2       = dropoutType(dropoutRate)(block2)
    block2       = Flatten()(block2) # 13
    
    # 8 - 13
    
    input2       = Input(shape = input_shape) 
    block3       = Conv2D(F1_2, conv_filters2, padding = 'same',
                                   input_shape = input_shape,
                                   use_bias = False)(input2)
    block3       = BatchNormalization(axis = 1)(block3)
    block3       = DepthwiseConv2D(depth_filters, use_bias = False, 
                                   depth_multiplier = D2,
                                   depthwise_constraint = max_norm(1.))(block3)
    block3       = BatchNormalization(axis = 1)(block3)
    block3       = Activation('elu')(block3)
    block3       = AveragePooling2D(pool_size)(block3)
    block3       = dropoutType(dropoutRate)(block3)
    
    block4       = SeparableConv2D(F2_2, separable_filters2,
                                   use_bias = False, padding = 'same')(block3) #22
    block4       = BatchNormalization(axis = 1)(block4)
    block4       = Activation('elu')(block4)
    block4       = AveragePooling2D(pool_size2)(block4)
    block4       = dropoutType(dropoutRate)(block4)
    block4       = Flatten()(block4) # 27
    # 22 - 27
    
    input3       = Input(shape = input_shape)
    block5       = Conv2D(F1_3, conv_filters3, padding = 'same',
                                   input_shape = input_shape,
                                   use_bias = False)(input3)
    block5       = BatchNormalization(axis = 1)(block5)
    block5       = DepthwiseConv2D(depth_filters, use_bias = False, 
                                   depth_multiplier = D3,
                                   depthwise_constraint = max_norm(1.))(block5)
    block5       = BatchNormalization(axis = 1)(block5)
    block5       = Activation('elu')(block5)
    block5       = AveragePooling2D(pool_size)(block5)
    block5       = dropoutType(dropoutRate)(block5)
    
    block6       = SeparableConv2D(F2_3, separable_filters3,
                                   use_bias = False, padding = 'same')(block5) # 36
    block6       = BatchNormalization(axis = 1)(block6)
    block6       = Activation('elu')(block6)
    block6       = AveragePooling2D(pool_size2)(block6)
    block6       = dropoutType(dropoutRate)(block6)
    block6       = Flatten()(block6) # 41
    
    # 36 - 41
    
    merge_one    = concatenate([block2, block4])
    merge_two    = concatenate([merge_one, block6])
    
    flatten      = Flatten()(merge_two)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)

    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=[input1, input2, input3], outputs=softmax)
