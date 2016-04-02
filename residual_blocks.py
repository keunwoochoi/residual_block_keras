'''Residual block by Keunwoo Choi (keunwoo.choi@qmul.ac.uk)

It is based on "Deep Residual Learning for Image Recognition" (http://arxiv.org/abs/1512.03385)
and "Identity Mappings in Deep Residual Networks" (http://arxiv.org/abs/1603.05027).
'''
import keras
from keras.layers.containers import Sequential, Graph
from keras.layers.core import Layer, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization


class Identity(Layer):
    def get_output(self, train):
        return self.get_input(train)

def building_residual_block(name_prefix, input_shape, n_feature_maps, kernel_sizes=None, n_skip=2, is_subsample=False, subsample=None):
    '''
    [1] Building block of layers for residual learning.
        Code based on https://github.com/ndronen/modeling/blob/master/modeling/residual.py
        , but modification of (perhaps) incorrect relu(f)+x thing and it's for conv layer

    [2] MaxPooling is used instead of strided convolution to make it easier 
        to set size(output of short-cut) == size(output of conv-layers).
        If you want to remove MaxPooling,
           i) change (border_mode in Convolution2D in shortcut), 'same'-->'valid'
           ii) uncomment ZeroPadding2D in conv layers.
               (Then the following Conv2D is not the first layer of this container anymore,
                so you can remove the input_shape in the line 101, the line with comment #'OPTION' )

    [3] It can be used for both cases whether it subsamples or not.

    [4] In the short-cut connection, I used 1x1 convolution to increase #channel.
        It occurs when is_expand_channels == True 

    name_prefix : prefix for layer names.  
    input_shape = (num_channel, height, width) 
    n_feature_maps: number of feature maps. In ResidualNet it increases whenever image is downsampled.
    kernel_sizes : list or tuple, (3,3) or [3,3] for example
    n_skip       : number of layers to skip
    is_subsample : If it is True, the layers subsamples by *subsample* to reduce the size.
    subsample    : tuple, (2,2) or (1,2) for example. Used only if is_subsample==True
    '''
    # ***** VERBOSE_PART ***** 
    print ('    - Create residual building block named %s' % name_prefix)
    print ('      input shape:', input_shape)
    print ('      kernel size:', kernel_sizes)
    # is_expand_channels == True when num_channels increases.
    #    E.g. the very first residual block (e.g. 3->128)
    #    E.g. usually when there's subsampling. 
    is_expand_channels = not (input_shape[0] == n_feature_maps) 
    
    if is_expand_channels:
        print ('      - Input channels: %d ---> num feature maps on out: %d' % (input_shape[0], n_feature_maps))
    if is_subsample:
        print ('      - with subsample:', subsample)

    kernel_row, kernel_col = kernel_sizes

    # ***** INITIATION ***** 
    block = keras.layers.containers.Graph()
    # set input shape
    input_name = '%s_x' % name_prefix
    block.add_input(input_name, input_shape=input_shape)
    prev_output = input_name
    
    # ***** SHORT_CUT_PART ***** 
    shortcut_output = '%s_identity' % name_prefix
    block.add_node(Identity(input_shape=input_shape), name=shortcut_output, 
                                input=prev_output)    
    
    if is_subsample: # subsample (+ channel expansion if needed)
        this_node_name = '%s_shortcut_conv' % name_prefix 
        layer = Convolution2D(n_feature_maps, kernel_sizes[0], kernel_sizes[1], 
                                    subsample=subsample,
                                    border_mode='valid')
        block.add_node(layer, name=this_node_name, input=shortcut_output)
        shortcut_output = this_node_name
    else: # channel expansion only (e.g. the very first layer of the whole networks)
        if is_expand_channels:
            this_node_name = '%s_shortcut_conv' % name_prefix 
            layer = Convolution2D(n_feature_maps, 1, 1, border_mode='same')
            block.add_node(layer, name=this_node_name, input=shortcut_output)
            shortcut_output = this_node_name
        else:
            # if no subsample and no channel expension, there's nothing to add on the shortcut.
            pass
    
    # ***** CONVOLUTION_PART ***** 
    for i in range(n_skip):
        # [BN]
        layer_name = '%s_BN_%d' % (name_prefix, i)
        block.add_node(BatchNormalization(axis=1), name=layer_name, input=prev_output)
        prev_output = layer_name    
        # [ReLU]
        layer_name = '%s_relu_%d' % (name_prefix, i)
        block.add_node(Activation('relu'), name=layer_name, input=prev_output)
        prev_output = layer_name
        # [Conv]
        layer_name = '%s_conv_%d' % (name_prefix, i)
        if i==0 and is_subsample: # [Subsample at layer 0 if needed]
            layer = Convolution2D(n_feature_maps, kernel_row, kernel_col,
                                    subsample=subsample,
                                    border_mode='valid')  
        else:        
            layer = Convolution2D(n_feature_maps, kernel_row, kernel_col, border_mode='same')   
        block.add_node(layer, name=layer_name, input=prev_output)
        prev_output = layer_name

    # output
    layer_name = '%s_output' % name_prefix
    block.add_output(name=layer_name, 
                    inputs=[prev_output, shortcut_output],
                    merge_mode='sum')

    return block
