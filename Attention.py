"""
This file is the implementation of all sub-units and blocks of the concurrent dual attention module in the iDAAM architecture.
This file contains:
    1. Committee of multi-feature attention module.
    2. Simultaneous excitation module.
    3. Concurrent dual attention module.
"""

# Import libraries

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization, Dense, Activation, Add, Multiply, GlobalAveragePooling2D, Conv2D

# Defining Self-attention block

def self_attention(input_feature, num_channel, base):

    """
    args:
    1. input_feature: Input to the self-attention block.
    2. num_channel: Number of channels after the dense operation.
    3. base: layer name identifier.
    """

    bn_1 = BatchNormalization(axis = -1, name = base + '/bn_1')(input_feature)
    dense_1 = Dense(num_channel, name = base + '/dense_1')(bn_1)
    act_1 = Activation('relu', name = base + '/act_1')(dense_1)

    bn_2 = BatchNormalization(axis = -1, name = base + '/bn_2')(input_feature)
    dense_2 = Dense(num_channel, name = base + '/dense_2')(bn_2)
    act_2 = Activation('relu', name = base + '/act_2')(dense_2)

    bn_3 = BatchNormalization(axis = -1, name = base + '/bn_3')(input_feature)
    dense_3 = Dense(num_channel, name = base + '/dense_3')(bn_3)
    act_3 = Activation('relu', name = base + '/act_3')(dense_3)

    mul_1 = Multiply(name = base + '/mul_1')([act_2, act_3])
    mask_part = Activation('softmax', name = base + '/act_4')(mul_1)
    mul_2 = Multiply(name = base + '/mul_2')([act_1, mask_part])

    output_feature = Add(name = base + '/add_1')([mul_2, input_feature])

    return output_feature


# Committee of multi-feature attention module

def CMFA(input_feature, num_channel, base):

    """
    args:
    1. input_feature: Input to the CMFA module.
    2. num_channel: Number of channels after the dense operation.
    3. base: layer name identifier.
    """

    out_1 = self_attention(input_feature, num_channel, base + '/self_att_1')
    out_2 = self_attention(input_feature, num_channel, base + '/self_att_2')
    out_3 = self_attention(input_feature, num_channel, base + '/self_att_3')
    out_4 = self_attention(input_feature, num_channel, base + '/self_att_4')

    output_feature = Add(name = base + '/add')([out_1, out_2, out_3, out_4])

    return output_feature


# Simultaneous excitation module

def SEM(input_feature, num_channel, base):

    """
    args:
    1. input_feature: Input to the SEM module.
    2. num_channel: Number of channels after the convolution operation.
    3. base: layer name identifier.
    """

    # Channel attention

    GAP_output = GlobalAveragePooling2D(name = base + '/gap_layer')(input_feature)

    bn_1 = BatchNormalization(axis = -1, name = base + '/bn_1')(GAP_output)
    dense_1 = Dense(int(num_channel/4), name = base + '/dense_1')(bn_1)
    act_1 = Activation('relu', name = base + '/act_1')(dense_1)

    bn_2 = BatchNormalization(axis = -1, name = base + '/bn_2')(act_1)
    dense_2 = Dense(num_channel, name = base + '/dense_2')(bn_2)
    act_2 = Activation('sigmoid', name = base + '/act_2')(dense_2)

    out_channel = Multiply(name = base + '/mul_1')([act_2, input_feature])

    # Spatial attention

    strides = (1,1)

    bn_3 = BatchNormalization(axis = -1, name = base + '/bn_3')(input_feature)
    cn_3 = Conv2D(num_channel, (3,3), strides = strides, padding = 'same', name = base + '/conv_1')(bn_3)
    an_3 = Activation('relu', name = base + '/act_3')(cn_3)

    bn_4 = BatchNormalization(axis = -1, name = base + '/bn_4')(an_3)
    cn_4 = Conv2D(1, (1,1), strides = strides, padding = 'same', name = base + '/conv_2')(bn_4)
    an_4 = Activation('sigmoid', name = base + '/act_4')(cn_4)

    out_spatial = Multiply(name = base + '/mul_2')([an_4, input_feature])

    output_response = Add(name = base + '/add')([input_feature, out_channel, out_spatial])

    return output_response
