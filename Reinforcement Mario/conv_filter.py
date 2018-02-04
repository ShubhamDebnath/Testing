import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc

def conv_model(X):
    X = Conv2D(filters=16, kernel_size=(2, 2), strides=(2, 2), padding='same',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides = (2, 2))

    X = Conv2D(filters=16, kernel_size=(2, 2), strides=(2, 2), padding='same',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))

    X = Conv2D(filters=16, kernel_size=(2, 2), strides=(2, 2), padding='same',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))

    X = Conv2D(filters=16, kernel_size=(2, 2), strides=(2, 2), padding='same',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))

    X = Conv2D(filters=16, kernel_size=(2, 2), strides=(2, 2), padding='same',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))

    return(X)