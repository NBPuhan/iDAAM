import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, Activation, Add, Multiply, GlobalAveragePooling2D, Conv2D, Dropout

from Blocks import FGDM
from Attention import CDAM

def iDAAM_part(X_input):

    """
    X_input: input tensor
    """
    
    bn_1 = BatchNormalization(axis = -1, name = 'bn_1')(X_input)
    cn_1 = Conv2D(32, (3,3), strides = (1,1), padding = 'same', name = 'conv_1')(bn_1)
    X = Activation('relu', name = 'act_1')(cn_1)
    
    # Dense block 1 
    X = FGDM(X, 32, 'FGDM_1')
    # Attention block 1
    X = CDAM(X, 32, 'CDAM_1')
    # Dense block 2
    X = FGDM(X, 64, 'FGDM_2')
    # Attention block 2
    X = CDAM(X, 64, 'CDAM_2')
    # Dense block 3
    X = FGDM(X, 128, 'FGDM_3')
    # Attention block 3
    X = CDAM(X, 128, 'CDAM_3')
    # Dense block 4
    X = FGDM(X, 256, 'FGDM_4')
    # Global average pooling operation
    X = GlobalAveragePooling2D(name = 'gap_1')(X)
    X = Dense(100, activation = 'relu', name = 'Dense_layer_1')(X)
    X = Dropout(0.25)(X)
    X = Dense(100, activation = 'relu', name = 'Dense_layer_2')(X)
    X = Dropout(0.25)(X)
    
    return X


def iDAAM_full(X_input, dataset_name):

    """
    X_input: Input tensor
    dataset_name: Name of the dataset for investigation
    """

    if (dataset_name == 'codebrim'):

        X = iDAAM_part(X_input)
        X = Dense(5, name = 'final_dense')(X)
        output_feature = Activation('sigmoid', name = 'act_final')(X)

    if (dataset_name == 'sdnet'):

        X = iDAAM_part(X_input)
        X = Dense(2, name = 'final_dense')(X)
        output_feature = Activation('sigmoid', name = 'act_final')(X)

    if (dataset_name == 'concrete_crack_image'):

        X = iDAAM_part(X_input)
        X = Dense(2, name = 'final_dense')(X)
        output_feature = Activation('sigmoid', name = 'act_final')(X)

    return output_feature
