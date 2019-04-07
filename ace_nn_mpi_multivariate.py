#!/usr/bin/python3
# author: xiangxiang-xu, zhaofeng-shu33
# description: implementation of Alternating Conditional Expectation Algorithm using Neural Network on mpi_faust dataset
# joint training, multivariate version
import keras
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input, Lambda, Dense
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten
from sklearn.preprocessing import OneHotEncoder
from keras import backend as K
import numpy as np

import pdb
def neg_hscore_multivariate(x_list):
    # call neg_hscore
    total = tf.constant(0.0)
    for i in range(len(x_list)):
        for j in range(i+1, len(x_list)):
            one_tf_value = neg_hscore([x_list[i], x_list[j]])
            total = tf.add(total, one_tf_value)
    return total
def neg_hscore(x):
    """
    negative hscore calculation
    """
    f = x[0]
    g = x[1]
    f0 = f - K.mean(f, axis = 0)
    g0 = g - K.mean(g, axis = 0)
    corr = tf.reduce_mean(tf.reduce_sum(tf.multiply(f0, g0), 1))
    cov_f = K.dot(K.transpose(f0), f0) / K.cast(K.shape(f0)[0] - 1, dtype = 'float32')
    cov_g = K.dot(K.transpose(g0), g0) / K.cast(K.shape(g0)[0] - 1, dtype = 'float32')
    return - corr + tf.trace(K.dot(cov_f, cov_g)) / 2
def feature_extraction_layer(ns, input_tensor):
    hidden_layer_num = 64
    fdim = ns 
    activation_function = 'relu'
    shape_list = input_tensor.shape.as_list()
    input_shape = (shape_list[1], shape_list[2], 1)
    conv1 = Conv2D(32, kernel_size=(3, 3),
                     activation=activation_function,
                     input_shape=input_shape)(input_tensor)
    conv2 = Conv2D(64, (3, 3), activation=activation_function)(conv1)
    pool = MaxPooling2D(pool_size=(2, 2))(conv2)    
    f_internal = Dropout(0.25)(pool)
    f_internal_2 = Flatten()(f_internal)
    f_internal_3 = Dense(hidden_layer_num, activation=activation_function)(f_internal_2)
    f = Dense(fdim, activation=activation_function)(f_internal_3)
    return f
    
def ace_nn_mpi_multivariate(x_list, ns = 26, epochs = 12, verbose = 1):
    ''' 
    Uses the alternating conditional expectations algorithm
    to find the transformations of each class variable that maximise the summation of pairwise hscore

    Parameters
    ----------
    x_list : list of array_like
        x_list[class_id] = [i, x_h, x_w] i is the index of the image, where the last two dims form one image.
        x_list[j,:,:,:] has the same dimension
    epochs : float, optional
        termination threshold (the default is 300). iteration epochs for
        neural network fitting.
    ns : int, optional
        number of eigensolutions (sets of transformations, the default is 1).
    verbose: Integer. 0, 1, or 2. Verbosity mode.
        0 = silent, 1 = progress bar(default), 2 = one line per epoch.   

    Returns
    -------
    pairewise h_score matrix : array_like
    h_m[i,j] = h_score of class_i and class_j
    '''
    
    batch_size = 20
    class_num = len(x_list)

    x_internal_list_item_shape = (x_list[0].shape[0], x_list[0].shape[1], x_list[0].shape[2], 1)
    x_internal_list = [np.zeros(x_internal_list_item_shape) for i in range(class_num)]
    for i in range(class_num):        
        x_internal_list[i][:,:,:,0] = x_list[i]
    
    input_shape_x = (x_list[0].shape[1], x_list[0].shape[2], 1)
    input_tensor_list = []
    feature_tensor_list = []
    for i in range(class_num):
        input_tensor = Input(shape = input_shape_x)
        input_tensor_list.append(input_tensor)
        f = feature_extraction_layer(ns, input_tensor)
        feature_tensor_list.append(f)
        
    loss = Lambda(neg_hscore_multivariate)(feature_tensor_list)
    model = Model(inputs = input_tensor_list, outputs = loss)
    # y_pred is loss
    model.compile(optimizer='sgd', loss = lambda y_true,y_pred: y_pred)
    model.fit(x_internal_list, np.zeros([x_list[0].shape[0]]), verbose=verbose,
        batch_size = batch_size, epochs = epochs)
    # after fitting, calculate pairewise hscore numerically.
    h_score_matrix = np.zeros([class_num, class_num])
    for i in range(class_num):
        for j in range(i+1, class_num):
            model_f = Model(inputs = input_tensor_list[i], outputs = feature_tensor_list[i])
            model_g = Model(inputs = input_tensor_list[j], outputs = feature_tensor_list[j])
            feature_f = model_f.predict(x_internal_list[i])
            feature_g = model_g.predict(x_internal_list[j])
            tensor_result = neg_hscore([tf.convert_to_tensor(feature_f), tf.convert_to_tensor(feature_g)])
            sess = tf.Session()
            h_score_matrix[i, j] = sess.run(tensor_result)
    return h_score_matrix + h_score_matrix.T

if __name__ == '__main__':
    POSE_NUM = 2
    x_list = []
    for i in range(POSE_NUM):
        x = np.load('output/%d.npx.npy'%i).astype('float')
        x_list.append(x)
    sim_matrix = ace_nn_mpi_multivariate(x_list, epochs = 1)
    np.save('sim_matrix_multi', sim_matrix)
