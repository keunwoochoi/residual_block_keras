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

import numpy as np
np.random.seed(1337)  # for reproducibility

import keras
from keras.datasets import mnist
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

import residual_blocks

batch_size = 128
nb_classes = 10
nb_epoch = 12
img_rows, img_cols = 28, 28

def design_for_residual_blocks():
    ''''''
    model = keras.layers.containers.Sequential() # it's a CONTAINER, not MODEL. It should be replaced with Sequential
    # set numbers
    num_big_blocks = 3
    image_patch_sizes = [[3,3]]*num_big_blocks
    # vgg_modi_weight, pool_sizes = get_residual_weights(num_big_blocks=num_big_blocks)
    pool_sizes = [(2,2)]*num_big_blocks
    n_features = [128, 256, 512]
    n_features_next = [256, 512, 512]
    height_input = 32
    width_input = 32
    num_channel_input = 1

    for conv_idx in range(num_big_blocks):
    
        n_feat_here = n_features[conv_idx]
        
        # residual block 0
        this_node_name = 'residual_block_%d_0' % conv_idx
        name_prefix = 'Conv_%d_0' % conv_idx
        model.add(residual_blocks.building_residual_block(name_prefix,
                                                            input_shape=(num_channel_input, height_input, width_input),
                                                            n_feature_maps=n_feat_here,
                                                            kernel_sizes=image_patch_sizes[conv_idx]
                                                            ))
        last_node_name = this_node_name

        # residual block 1 (you can add it as you want (and your resources allow..))
        if False:
            this_node_name = 'residual_block_%d_1' % conv_idx
            name_prefix = 'Conv_%d_1' % conv_idx
            model.add(residual_blocks.building_residual_block(name_prefix,
                                                                input_shape=(n_feat_here, height_input, width_input),
                                                                n_feature_maps=n_feat_here,
                                                                kernel_sizes=image_patch_sizes[conv_idx]
                                                                ))
            last_node_name = this_node_name
        
        # the last residual block N-1
        # the last one : subsamples and increase #channels
        this_node_name = 'residual_block_%d_last' % conv_idx
        n_feat_next = n_features_next[conv_idx]
        name_prefix = 'Conv_%d_last' % conv_idx
        model.add(residual_blocks.building_residual_block(name_prefix,
                                                            input_shape=(n_feat_here, height_input, width_input),
                                                            n_feature_maps=n_feat_next,
                                                            kernel_sizes=image_patch_sizes[conv_idx],
                                                            is_subsample=True,
                                                            subsample=pool_sizes[conv_idx]
                                                            ))
        last_node_name = this_node_name

        height_input = int(height_input/pool_sizes[conv_idx][0])
        width_input  = int(width_input/pool_sizes[conv_idx][1])
        num_channel_input = n_feat_next

    # Add average pooling at the end: (4,4) --> (1,1)
    model.add(AveragePooling2D(pool_size=(height_input, width_input)))

    this_node_name = 'residual_block_output'

    return model

def design_residual_model():
    ''''''
    #-------------- design_residual_model -------------------#
    #--------------------------------------------------------#
    n_skips = 2
    
    #--------------------------------------------------------#
    
    # start the model!
    model = keras.models.Sequential() # 
    model.add(ZeroPadding2D(padding=(2,2), input_shape=(1, img_rows, img_cols))) # resize (28,28)-->(32,32)
    # [residual-based Conv layers]
    residual_blocks = design_for_residual_blocks()
    model.add(residual_blocks)
    # [Prepare to add classifier]
    residual_output_shape = residual_blocks.output_shape
    classifier_input_shape = residual_output_shape[1:]
    # [Classifier]
    model.add(Flatten())
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    # [END]
    return model

if __name__ =='__main__':
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = design_residual_model()

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # autosave best Model
    fBestModel = "./my_model_weights.h5"
    best_model = ModelCheckpoint(fBestModel, verbose=1, save_best_only=True)

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              show_accuracy=True, verbose=1, validation_data=(X_test, Y_test), callbacks=[best_model])
    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

