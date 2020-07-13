# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 20:43:48 2020

@author: ValerioC
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import warnings


from keras import layers
from keras.models import Model


from keras import backend as K



def identity_block(input_tensor, kernel_size, filters, stage, block):

    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):

    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def ResNet50(include_top=True,
             starting_filters=64,
             first_conv_kernel_size=5,
             first_conv_stride=1,
             max_Pooling=False,
             input_tensor=None,
             input_shape=96,
             pooling=None,
             num_classes=1000,
             **kwargs):

    global layers, models, keras_utils

    
    
    if input_tensor is None:
        img_input = layers.Input((input_shape, input_shape, 1))    
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(starting_filters, (first_conv_kernel_size, first_conv_kernel_size),
                      strides=(first_conv_stride, first_conv_stride),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    if max_Pooling==True:
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [starting_filters, starting_filters, starting_filters*4], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [starting_filters, starting_filters, starting_filters*4], stage=2, block='b')
    x = identity_block(x, 3, [starting_filters, starting_filters, starting_filters*4], stage=2, block='c')

    x = conv_block(x, 3, [starting_filters*2, starting_filters*2, starting_filters*8], stage=3, block='a')
    x = identity_block(x, 3, [starting_filters*2, starting_filters*2, starting_filters*8], stage=3, block='b')
    x = identity_block(x, 3, [starting_filters*2, starting_filters*2, starting_filters*8], stage=3, block='c')
    x = identity_block(x, 3, [starting_filters*2, starting_filters*2, starting_filters*8], stage=3, block='d')

    x = conv_block(x, 3, [starting_filters*4, starting_filters*4, starting_filters*16], stage=4, block='a')
    x = identity_block(x, 3, [starting_filters*4, starting_filters*4, starting_filters*16], stage=4, block='b')
    x = identity_block(x, 3, [starting_filters*4, starting_filters*4, starting_filters*16], stage=4, block='c')
    x = identity_block(x, 3, [starting_filters*4, starting_filters*4, starting_filters*16], stage=4, block='d')
    x = identity_block(x, 3, [starting_filters*4, starting_filters*4, starting_filters*16], stage=4, block='e')
    x = identity_block(x, 3, [starting_filters*4, starting_filters*4, starting_filters*16], stage=4, block='f')


    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(num_classes, activation='softmax', name='output_layer')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
        else:
            warnings.warn('The output shape of `ResNet50(include_top=False)` '
                          'has been changed since Keras 2.2.0.')

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')

    

    return model

