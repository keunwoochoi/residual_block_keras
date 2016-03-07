'''Example of using my_residual_blocks.py by Keunwoo Choi (keunwoo.choi@qmul.ac.uk)
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
   |                                                      |
   |            |--- classifier block ------|             |
   |            | (not relavent to residual)|             |
   |            |___________________________|             |
   |                                                      |
   |______________________________________________________|


'''

import keras
import residual_blocks
import classifier_blocks
from keras.models import Sequential, Graph
import sys


def design_for_residual_blocks():
    ''''''
    model = keras.layers.containers.Sequential() # it's a CONTAINER, not MODEL. It should be replaced with Sequential
    # set numbers
    image_patch_sizes = [[3,3]]*num_layers
    vgg_modi_weight, pool_sizes = get_residual_weights(num_layers=num_layers)
    setting_dict['vgg_modi_weight'] = vgg_modi_weight
    height_input = height
    width_input = width
    num_channel_input = 1

    for conv_idx in range(num_layers):
        print '= design for mel - conv layer blocks %d will be added = ' % conv_idx
    
        n_feat_here = int(num_stacks[conv_idx]*vgg_modi_weight[conv_idx][0])
        
        # residual block 0
        this_node_name = 'residual_block_%d_0' % conv_idx
        name_prefix = 'Conv_%d_0' % conv_idx
        model.add(my_residual_blocks.building_residual_block(name_prefix,
                                                            input_shape=(num_channel_input, height_input, width_input),
                                                            n_feature_maps=n_feat_here,
                                                            kernel_sizes=image_patch_sizes[conv_idx]
                                                            ))
        last_node_name = this_node_name

        # residual block 1
        this_node_name = 'residual_block_%d_1' % conv_idx
        name_prefix = 'Conv_%d_1' % conv_idx
        model.add(my_residual_blocks.building_residual_block(name_prefix,
                                                            input_shape=(n_feat_here, height_input, width_input),
                                                            n_feature_maps=n_feat_here,
                                                            kernel_sizes=image_patch_sizes[conv_idx]
                                                            ))
        last_node_name = this_node_name
        # residual block 2
        this_node_name = 'residual_block_%d_2' % conv_idx
        name_prefix = 'Conv_%d_2' % conv_idx
        model.add(my_residual_blocks.building_residual_block(name_prefix,
                                                            input_shape=(n_feat_here, height_input, width_input),
                                                            n_feature_maps=n_feat_here,
                                                            kernel_sizes=image_patch_sizes[conv_idx]
                                                            ))
        last_node_name = this_node_name
        
        # residual block 3
        # the last one : subsamples and increase #channels
        this_node_name = 'residual_block_%d_3' % conv_idx
        try:
            n_feat_next = int(num_stacks[conv_idx+1]*vgg_modi_weight[conv_idx+1][0])
        except:
            pass
        name_prefix = 'Conv_%d_3' % conv_idx
        model.add(my_residual_blocks.building_residual_block(name_prefix,
                                                            input_shape=(n_feat_here, height_input, width_input),
                                                            n_feature_maps=n_feat_next,
                                                            kernel_sizes=image_patch_sizes[conv_idx],
                                                            is_subsample=True,
                                                            subsample=pool_sizes[conv_idx]
                                                            ))
        last_node_name = this_node_name

        height_input /= pool_sizes[conv_idx][0]
        width_input  /= pool_sizes[conv_idx][1]
        num_channel_input = n_feat_next

    this_node_name = 'residual_block_output'
    # model.add_output(name=this_node_name, input=last_node_name)

    return model

def design_residual_model(setting_dict):
    ''''''
    #-------------- design_residual_model -------------------#
    #--------------------------------------------------------#
    n_skips = setting_dict['n_skips'] # These are the setting I'm using.
    tf_type = setting_dict['tf_type'] 
    height = setting_dict["height_image"]
    width = setting_dict["width_image"]
    num_labels = setting_dict["dim_labels"]
    num_layers = setting_dict["num_layers"]
    num_fc_layers = setting_dict["num_fc_layers"]
    dropouts_fc_layers = setting_dict["dropouts_fc_layers"]
    nums_units_fc_layers = setting_dict["nums_units_fc_layers"]
    num_stacks = setting_dict["num_feat_maps"]
    dropouts_fc_layers = setting_dict["dropouts_fc_layers"]
    num_channels = 1
    nb_maxout_feature = setting_dict['nb_maxout_feature']
    #--------------------------------------------------------#
    sys.setrecursionlimit(99999)
    # start the model!
    model = keras.models.Sequential() # 
    # [residual-based Conv layers]
    residual_blocks = design_for_residual_blocks()
    model.add(residual_blocks)
    # [Prepare to add classifier]
    residual_output_shape = residual_blocks.output_shape
    classifier_input_shape = residual_output_shape[1:]
    # [Classifier]
    model.add(my_classifier_blocks.building_classifier_block(setting_dict, classifier_input_shape))
    # [END]
    return model
