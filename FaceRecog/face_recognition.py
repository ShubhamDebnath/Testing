from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
import pickle
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from utility import *
from inception_blocks_v2 import *

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

np.set_printoptions(threshold=np.nan)


#Using an ConvNet to compute encodings
FRmodel = faceRecoModel(input_shape=(3, 96, 96))

# print("Total Params:", FRmodel.count_params())

def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    ### START CODE HERE ### (≈ 4 lines)
    # Step 1: Compute the (encoding) distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(y_pred[1] - y_pred[0]), axis = -1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(y_pred[2] - y_pred[0]), axis = -1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = pos_dist - neg_dist + alpha
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    ### END CODE HERE ###
    
    return loss

## Loading the trained model
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)


def open_database():
    try:
        with open('database/user_database.pickle', 'rb') as data:
        	user_db = pickle.load(data)
    except:
        print("error opening previous database\n creating new dictionary")
        user_db = {}
    finally:
        return user_db


def add_to_database(name, img_path, user_db):
    user_db[name] = img_to_encoding(img_path, FRmodel)
    print("added {} to the database".format(name))
    save_database(user_db)	
    return user_db


def delete_from_database(name, user_db):
	del user_db[name]
	save_database(user_db)
	return user_db

def save_database(user_db):
	with open('database/user_database.pickle', 'wb') as handle:
                pickle.dump(user_db, handle, protocol=pickle.HIGHEST_PROTOCOL)

def identify(image_path, database, model, min_threshold):
    """
    Implements face recognition by finding who is the person on the image_path image.
    
    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras
    
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    
    ### START CODE HERE ### 
    
    ## Step 1: Compute the target "encoding" for the image
    encoding = img_to_encoding(image_path, model)
    
    ## Step 2: Find the closest encoding ##
    
    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100
    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current "emb" from the database
        dist = np.linalg.norm(np.subtract(encoding ,database[name]))

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    if min_dist > min_threshold:
        print("Not in the database.")
        identity = "Unknown"
    print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        
    return min_dist, identity
    ### END CODE HERE ###

take_picture_no_webcam()
user_db = open_database()
user_db = add_to_database("Shubham", 'to_detect_faces/1.jpg', user_db)
realtime_face_recognition_no_webcam(user_db, FRmodel, 0.7)