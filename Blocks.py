"""
This file is the implementation of all sub-units and blocks of the fine-grained dense module in the iDAAM architecture.
This file contains:
    1. Residual block.
    2. Fine-grained dense module.
"""


# Import libraries

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add


# Defining Residual Block

def residual_block(input_feature, num_channel, base):

    """
    args:
    1. input_feature: Input to the residual block.
    2. num_channel: Number of channels after the convolution operation.
    3. base: layer name identifier.
    """

    strides = (1,1)

    bn_1 = BatchNormalization(axis = -1, name = base + '/bn_1')(input_feature)
    cn_1 = Conv2D(num_channel, (1,1), strides = strides, padding = 'same', name = base + '/conv_1')(bn_1)
    an_1 = Activation('relu', name = base + '/act_1')(cn_1)

    bn_2 = BatchNormalization(axis = -1, name = base + '/bn_2')(an_1)
    cn_2 = Conv2D(num_channel, (3,3), strides = strides, padding = 'same', name = base + '/conv_2')(bn_2)
    an_2 = Activation('relu', name = base + '/act_2')(cn_2)

    bn_3 = BatchNormalization(axis = -1, name = base + '/bn_3')(an_2)
    cn_3 = Conv2D(num_channel, (1,1), strides = strides, padding = 'same', name = base + '/conv_3')(bn_3)
    an_3 = Activation('relu', name = base + '/act_3')(cn_3)

    output_feature = Add(name = base + '/add')([input_feature, an_3])
    
    return output_feature


# Defining Fine-grained dense module

def FGDM(input_feature, num_channel, base):

    """
    args:
    1. input_feature: Input to the FGDM.
    2. num_channel: Number of channels after the convolution operation.
    3. base: layer name identifier.
    """

    # Initial conv layer for channel number variation
    bn_1 = BatchNormalization(axis = -1, name = base + '/bn_1')(input_feature)
    cn_1 = Conv2D(num_channel, (1,1), strides = (1,1), padding = 'same', name = base + '/conv_1')(bn_1)
    an_1 = Activation('relu', name = base + '/act_1')(cn_1)

    # First residual block
    res_1 = residual_block(an_1, num_channel, base + '/res_1')

    # Second residual block
    res_2 = residual_block(res_1, num_channel, base + '/res_2')
    res_2 = Add(name = base + '/add_1')([res_1, res_2])

    # Third residual block
    res_3 = residual_block(res_2, num_channel, base + '/res_3')
    res_3 = Add(name = base + '/add_2')([res_1, res_2, res_3])

    # Fourth residual block
    res_4 = residual_block(res_3, num_channel, base + '/res_4')
    res_4 = Add(name = base + '/add_3')([res_1, res_2, res_3, res_4])

    # Final conv layer for spatial dimension variation
    bn_2 = BatchNormalization(axis = -1, name = base + '/bn_2')(res_4)
    cn_2 = Conv2D(num_channel, (3,3), strides = (2,2), padding = 'same', name = base + '/conv_2')(bn_2)
    output_feature = Activation('relu', name = base + '/act_2')(cn_2)

    return output_feature
