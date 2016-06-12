'''Example of using residual_blocks.py by Keunwoo Choi (keunwoo.choi@qmul.ac.uk), on Keras 0.3.2
It's copy-and-pasted from the code I am using, so it wouldn't run.
Just take a look to understand how to use residual blocks. 

The whole structure is...

   -------Residual Model (keras.models.Sequential())-------
   |                                                      |
   |     ---- Residual blocks ------------------------|   |
   |     |    (keras.layers.containers.Sequential())  |   |
   |     |                                            |   |
   |     |     -- Many Residual blocks ------------   |   |
   |     |     | (keras.layers.containers.Graph())|   |   |
   |     |     |                                  |   |   |
   |     |     |__________________________________|   |   |
   |     |____________________________________________|   |
   |                                                      |
   |______________________________________________________|

'''
from __future__ import print_function
import sys
sys.setrecursionlimit(99999)
import pdb

import numpy as np
np.random.seed(1337)  # for reproducibility

import keras

from keras.datasets import mnist, cifar10
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D, Convolution2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization

import residual_blocks


batch_size = 64
nb_classes = 10
nb_epoch = 20

def compute_padding_length(length_before, stride, length_conv):
    ''' Assumption: you want the subsampled result has a length of floor(original_length/stride).
    '''
    N = length_before
    F = length_conv
    S = stride
    if S == F:
        return 0
    if S == 1:
        return (F-1)/2
    for P in range(S):
        if (N-F+2*P)/S + 1 == N/S:
            return P
    return None

def design_for_residual_blocks(num_channel_input=1):
    ''''''
    model = Sequential() # it's a CONTAINER, not MODEL
    # set numbers
    num_big_blocks = 3
    image_patch_sizes = [[3,3]]*num_big_blocks
    pool_sizes = [(2,2)]*num_big_blocks
    n_features = [128, 256, 512, 512, 1024]
    n_features_next = [256, 512, 512, 512, 1024]
    height_input = 32
    width_input = 32
    for conv_idx in range(num_big_blocks):    
        n_feat_here = n_features[conv_idx]
        # residual block 0
        model.add(residual_blocks.building_residual_block(  (num_channel_input, height_input, width_input),
                                                            n_feat_here,
                                                            kernel_sizes=image_patch_sizes[conv_idx]
                                                            ))

        # residual block 1 (you can add it as you want (and your resources allow..))
        if False:
            model.add(residual_blocks.building_residual_block(  (n_feat_here, height_input, width_input),
                                                                n_feat_here,
                                                                kernel_sizes=image_patch_sizes[conv_idx]
                                                                ))
        
        # the last residual block N-1
        # the last one : pad zeros, subsamples, and increase #channels
        pad_height = compute_padding_length(height_input, pool_sizes[conv_idx][0], image_patch_sizes[conv_idx][0])
        pad_width = compute_padding_length(width_input, pool_sizes[conv_idx][1], image_patch_sizes[conv_idx][1])
        model.add(ZeroPadding2D(padding=(pad_height,pad_width))) 
        height_input += 2*pad_height
        width_input += 2*pad_width
        n_feat_next = n_features_next[conv_idx]
        model.add(residual_blocks.building_residual_block(  (n_feat_here, height_input, width_input),
                                                            n_feat_next,
                                                            kernel_sizes=image_patch_sizes[conv_idx],
                                                            is_subsample=True,
                                                            subsample=pool_sizes[conv_idx]
                                                            ))

        height_input, width_input = model.output_shape[2:]
        # width_input  = int(width_input/pool_sizes[conv_idx][1])
        num_channel_input = n_feat_next

    # Add average pooling at the end:
    print('Average pooling, from (%d,%d) to (1,1)' % (height_input, width_input))
    model.add(AveragePooling2D(pool_size=(height_input, width_input)))

    return model

def get_residual_model(is_mnist=True, img_channels=1, img_rows=28, img_cols=28):
    model = keras.models.Sequential()
    first_layer_channel = 128
    if is_mnist: # size to be changed to 32,32
        model.add(ZeroPadding2D((2,2), input_shape=(img_channels, img_rows, img_cols))) # resize (28,28)-->(32,32)
        # the first conv 
        model.add(Convolution2D(first_layer_channel, 3, 3, border_mode='same'))
    else:
        model.add(Convolution2D(first_layer_channel, 3, 3, border_mode='same', input_shape=(img_channels, img_rows, img_cols)))

    model.add(Activation('relu'))
    # [residual-based Conv layers]
    residual_blocks = design_for_residual_blocks(num_channel_input=first_layer_channel)
    model.add(residual_blocks)
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    # [Classifier]    
    model.add(Flatten())
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    # [END]
    return model

if __name__ =='__main__':
    
    is_mnist = False
    is_cifar10 = not is_mnist
    if is_mnist:
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        img_rows, img_cols = 28, 28
        img_channels = 1
        print(' == MNIST ==')
    else:
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        img_rows, img_cols = 32, 32
        img_channels = 3
        print(' == CIFAR10 ==')

    X_train = X_train.reshape(X_train.shape[0], img_channels, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_channels, img_rows, img_cols)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # X_train /= 255
    # X_test /= 255
    X_train = (X_train - np.mean(X_train))/np.std(X_train)
    X_test = (X_test - np.mean(X_test))/np.std(X_test)
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    model = get_residual_model(is_mnist=is_mnist, img_channels=img_channels, img_rows=img_rows, img_cols=img_cols)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # autosave best Model
    best_model_file = "./my_model_weights.h5"
    best_model = ModelCheckpoint(best_model_file, verbose=1, save_best_only=True)
    
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test))#, callbacks=[best_model])
    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

