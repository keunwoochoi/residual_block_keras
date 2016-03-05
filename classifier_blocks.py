import keras
from keras.layers.containers import Graph
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, MaxoutDense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.activations import relu


def building_classifier_block(setting_dict, input_shape):
	''''''
	num_labels = setting_dict["dim_labels"]
	num_fc_layers = setting_dict["num_fc_layers"]
	dropouts_fc_layers = setting_dict["dropouts_fc_layers"]
	nums_units_fc_layers = setting_dict["nums_units_fc_layers"]
	nb_feature = 4 # for maxout
	#--------------------------------------#
	block = keras.layers.containers.Graph()

	# set input shape
	input_name = 'classifier_input' 
	block.add_input(input_name, input_shape=input_shape)
	prev_output = input_name
	
	# Flatten
	layer_name = 'flatten'
	block.add_node(Flatten(input_shape=input_shape), 
					name=layer_name,
					input=prev_output)
	last_node_name = layer_name
	
	# Maxout layers and conv
	for fc_idx in xrange(num_fc_layers):
		print 'Add fc layer %d' % fc_idx
		this_node_name = 'maxout_%d' % fc_idx
		
		block.add_node(MaxoutDense(nums_units_fc_layers[fc_idx], 
									nb_feature=nb_feature),
						input=last_node_name,
						name=this_node_name)
		last_node_name = this_node_name
		# Dropout
		if not dropouts_fc_layers[fc_idx] == 0.0:
			this_node_name = 'dropout_fc_%d' % fc_idx
			block.add_node(Dropout(dropouts_fc_layers[fc_idx]),
							input=last_node_name,
							name=this_node_name)
			last_node_name = this_node_name

		this_node_name = 'bn_fc_%d' % fc_idx
		block.add_node(BatchNormalization(axis=1),
										input=last_node_name,
										name=this_node_name)
		last_node_name = this_node_name

	# final layer
	this_node_name = 'last_dense'
	block.add_node(Dense(num_labels, activation='sigmoid'), 
					input=last_node_name, 
					name=this_node_name)
	last_node_name = this_node_name
	
	#[Output layer]
	this_node_name = 'output'
	block.add_output(input=last_node_name,
					name=this_node_name)

	# return it
	return block

