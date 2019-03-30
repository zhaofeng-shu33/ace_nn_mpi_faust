#!/usr/bin/python3
# author: xiangxiang-xu, zhaofeng-shu33
# description: implementation of Alternating Conditional Expectation Algorithm using Neural Network on mpi_faust dataset

import keras
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input, Lambda, Dense
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten
from sklearn.preprocessing import OneHotEncoder
from keras import backend as K
import numpy as np

import pdb
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
def ace_nn_mpi(x, y, ns = 26, epochs = 12, verbose = 1, return_hscore = False):
    ''' 
    Uses the alternating conditional expectations algorithm
    to find the transformations of y and x that maximise the 
    correlation between image class x and image class y.

    Parameters
    ----------
    x : array_like
        [i, x_h, x_w] i is the index of the image, where the last two dims form one image.
    y : array_like
        [i, y_h, y_w] constraint is len(x[:,0,0]) == len(y[:,0,0])
    epochs : float, optional
        termination threshold (the default is 300). iteration epochs for
        neural network fitting.
    ns : int, optional
        number of eigensolutions (sets of transformations, the default is 1).
    verbose: Integer. 0, 1, or 2. Verbosity mode.
        0 = silent, 1 = progress bar(default), 2 = one line per epoch.   

    Returns
    -------
    tx : array_like
        the transformed x values.
    ty : array_like
        the transformed y values.
    '''
    
    batch_size = 20
    hidden_layer_num = 64
    fdim = ns 
    gdim = fdim
    activation_function = 'relu'

    x_internal = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
    y_internal = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)
    # channel last image format
    input_shape_x = (x.shape[1], x.shape[2], 1)
    input_shape_y = (y.shape[1], y.shape[2], 1)

    dense_layer = Dense(fdim, activation=activation_function)
    input_x = Input(shape = input_shape_x)
    conv1 = Conv2D(32, kernel_size=(3, 3),
                     activation=activation_function,
                     input_shape=input_shape_x)(input_x)
    conv2 = Conv2D(64, (3, 3), activation=activation_function)(conv1)
    pool = MaxPooling2D(pool_size=(2, 2))(conv2)    
    f_internal = Dropout(0.25)(pool)
    f_internal_2 = Flatten()(f_internal)
    f_internal_3 = Dense(hidden_layer_num, activation=activation_function)(f_internal_2)
    f = Dense(fdim, activation=activation_function)(f_internal_3)


    input_y = Input(shape = input_shape_y)
    conv1_y = Conv2D(32, kernel_size=(3, 3),
                     activation=activation_function,
                     input_shape=input_shape_x)(input_y)
    conv2_y = Conv2D(64, (3, 3), activation=activation_function)(conv1_y)
    pool_y = MaxPooling2D(pool_size=(2, 2))(conv2_y)    
    g_internal = Dropout(0.25)(pool_y)
    g_internal_2 = Flatten()(g_internal)
    g_internal_3 = Dense(hidden_layer_num, activation=activation_function)(g_internal_2)
    g = Dense(gdim, activation=activation_function)(g_internal_3)
    
    loss = Lambda(neg_hscore)([f, g])
    model = Model(inputs = [input_x, input_y], outputs = loss)
    # y_pred is loss
    model.compile(optimizer='sgd', loss = lambda y_true,y_pred: y_pred)
    model_f = Model(inputs = input_x, outputs = f)
    model_g = Model(inputs = input_y, outputs = g)
    shape_fake = [x_internal.shape[0], x_internal.shape[1], x_internal.shape[2], 1]
    model.fit([x_internal, y_internal], np.zeros(shape_fake), verbose=verbose,
        batch_size = batch_size, epochs = epochs)
    h_score = -model.history.history['loss'][-1] # consider taking the average of last five
    t_x = model_f.predict(x_internal)
    t_y = model_g.predict(y_internal)
    if(return_hscore):
        return h_score
    return (t_x, t_y)
 
if __name__ == '__main__':
    sim_matrix = np.zeros([10,10])
    for i in range(10):
        for j in range(i+1, 10):
            x = np.load('output/%d.npx.npy'%i).astype('float')
            y = np.load('output/%d.npx.npy'%j).astype('float')
            # pdb.set_trace()
            h_score = ace_nn_mpi(x, y, return_hscore=True, epochs=24)
            sim_matrix[i,j] = h_score
    np.save('sim_matrix', sim_matrix)
