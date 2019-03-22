#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

# External
import tensorflow as tf
import numpy as np
from collections import OrderedDict

# Internal
import Segm_net_3D as segm3D

#-----------------------------------------------------------------------------#
#--------------- 3D Autoencoder Network Levels -------------------------------#
#-----------------------------------------------------------------------------#
    
def Encode_level(N_fc, input_tensor, internal_size, code_size, phase, layer_name, non_lin_func = tf.nn.relu):
    h = OrderedDict()
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        
        # First the operation layer creation 
        # Flatten input
        #input_flat = tf.layers.Flatten()(input_tensor)
        input_flat = tf.contrib.layers.flatten(input_tensor)
        
        
        if N_fc > 1:
            h_aux = tf.layers.dense(input_flat,
                                   internal_size,
                                   activation=non_lin_func,
                                   use_bias=True,
                                   kernel_initializer=None,
                                   bias_initializer=tf.zeros_initializer(),
                                   kernel_regularizer=None,
                                   bias_regularizer=None,
                                   activity_regularizer=None,
                                   #kernel_constraint=None,
                                   #bias_constraint=None,
                                   trainable=True,
                                   name=layer_name+'FC_0',
                                   reuse=None)

            h[0] = tf.layers.batch_normalization(h_aux, training=phase)


            # If there are mores layers in this level, lets create them
            if N_fc > 2:
                for i in range(1,N_fc-1):
                    # Convolutional layer creation
                    h_aux = tf.layers.dense(h[i-1],
                                   internal_size,
                                   activation=non_lin_func,
                                   use_bias=True,
                                   kernel_initializer=None,
                                   bias_initializer=tf.zeros_initializer(),
                                   kernel_regularizer=None,
                                   bias_regularizer=None,
                                   activity_regularizer=None,
                                   #kernel_constraint=None,
                                   #bias_constraint=None,
                                   trainable=True,
                                   name=layer_name+'FC_%d'%i,
                                   reuse=None)
                    h[i] = tf.layers.batch_normalization(h_aux, training=phase)
            
            # Last code level
            h_aux = tf.layers.dense(h[N_fc-2],
                                    code_size,
                                    activation=non_lin_func,
                                    use_bias=True,
                                    kernel_initializer=None,
                                    bias_initializer=tf.zeros_initializer(),
                                    kernel_regularizer=None,
                                    bias_regularizer=None,
                                    activity_regularizer=None,
                                    #kernel_constraint=None,
                                    #bias_constraint=None,
                                    trainable=True,
                                    name=layer_name+'FC_CODE',
                                    reuse=None)
            h_relu = tf.layers.batch_normalization(h_aux, training=phase)
            
        
        # Single layer
        else:
            h_aux = tf.layers.dense(input_flat,
                                   code_size,
                                   activation=non_lin_func,
                                   use_bias=True,
                                   kernel_initializer=None,
                                   bias_initializer=tf.zeros_initializer(),
                                   kernel_regularizer=None,
                                   bias_regularizer=None,
                                   activity_regularizer=None,
                                   #kernel_constraint=None,
                                   #bias_constraint=None,
                                   trainable=True,
                                   name=layer_name+'FC_0_CODE',
                                   reuse=None)

            h_relu = tf.layers.batch_normalization(h_aux, training=phase)

        # Finaly we return the code
        return (h_relu)
    
def Decode_level(N_fc, code_tensor, internal_size, out_shape, phase, layer_name, non_lin_func = tf.nn.relu):
    
    h = OrderedDict()
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        
        if N_fc > 1:
            # First the operation layer creation 
            h_aux = tf.layers.dense(code_tensor,
                                   internal_size,
                                   activation=non_lin_func,
                                   use_bias=True,
                                   kernel_initializer=None,
                                   bias_initializer=tf.zeros_initializer(),
                                   kernel_regularizer=None,
                                   bias_regularizer=None,
                                   activity_regularizer=None,
                                   #kernel_constraint=None,
                                   #bias_constraint=None,
                                   trainable=True,
                                   name=layer_name+'FC_0',
                                   reuse=None)

            h[0] = tf.layers.batch_normalization(h_aux, training=phase)

            # If there are mores layers in this level, lets create them
            if N_fc > 2:
                for i in range(1,N_fc-1):
                    # Convolutional layer creation
                    h_aux = tf.layers.dense(h[i-1],
                                   internal_size,
                                   activation=non_lin_func,
                                   use_bias=True,
                                   kernel_initializer=None,
                                   bias_initializer=tf.zeros_initializer(),
                                   kernel_regularizer=None,
                                   bias_regularizer=None,
                                   activity_regularizer=None,
                                   #kernel_constraint=None,
                                   #bias_constraint=None,
                                   trainable=True,
                                   name=layer_name+'FC_%d'%i,
                                   reuse=None)
                    h[i] = tf.layers.batch_normalization(h_aux, training=phase)


            # Finally the expansion layer
            expand_size = out_shape[1]*out_shape[2]*out_shape[3]*out_shape[4] # Size of the low-res image
            h_aux = tf.layers.dense(h[N_fc-2],
                                   expand_size,
                                   activation=non_lin_func,
                                   use_bias=True,
                                   kernel_initializer=None,
                                   bias_initializer=tf.zeros_initializer(),
                                   kernel_regularizer=None,
                                   bias_regularizer=None,
                                   activity_regularizer=None,
                                   #kernel_constraint=None,
                                   #bias_constraint=None,
                                   trainable=True,
                                   name=layer_name+'FC_EXPAND',
                                   reuse=None)
            h_relu = tf.layers.batch_normalization(h_aux, training=phase)        

            # Reshape into low-resolution multichannel volume
            h_decode = tf.reshape(h_relu, out_shape)
            
        # Single layer
        else:
            expand_size = out_shape[1]*out_shape[2]*out_shape[3]*out_shape[4] # Size of the low-res image
            h_aux = tf.layers.dense(code_tensor,
                                   expand_size,
                                   activation=non_lin_func,
                                   use_bias=True,
                                   kernel_initializer=None,
                                   bias_initializer=tf.zeros_initializer(),
                                   kernel_regularizer=None,
                                   bias_regularizer=None,
                                   activity_regularizer=None,
                                   #kernel_constraint=None,
                                   #bias_constraint=None,
                                   trainable=True,
                                   name=layer_name+'FC_0_EXPAND',
                                   reuse=None)
            h_relu = tf.layers.batch_normalization(h_aux, training=phase)        

            # Reshape into low-resolution multichannel volume
            h_decode = tf.reshape(h_relu, out_shape)

        # Finaly we return the decoded low-res volume
        return (h_decode)
    
    
    

#-----------------------------------------------------------------------------#
#--------------- Network Levels Building -------------------------------------#
#-----------------------------------------------------------------------------#

def Assemble_Autoencoder(ph_entry, 
                         phase, 
                         input_size, 
                         input_channels, 
                         size_filt_fine, 
                         size_filt_out, 
                         network_depth, 
                         net_channels_down, 
                         net_layers_down, 
                         net_channels_up, 
                         net_layers_up, 
                         net_layers_output,
                         
                         net_layers_encode,
                         net_neurons_encode, 
                         code_size,
                         net_layers_decode,
                         net_neurons_decode,
                         
                         fullycon_code = False,
                         all_outs = False):

    
    # The input tensor must be reshaped as a 5d tensor, with last dimension the 
    # color channels
    x_vol = tf.reshape(ph_entry, [-1, input_size[0], input_size[1], input_size[2], input_channels])
    
    # -- We first construct the encoder
    
    # First level:
    level_channels = net_channels_down[0]
    level_layers = net_layers_down[0]
    h_down = OrderedDict()
    h_relu_down = OrderedDict()
    (h_down[0], h_relu_down[0]) = segm3D.Descendent_level(level_layers, 
                                                   size_filt_fine, 
                                                   x_vol, 
                                                   input_channels, 
                                                   level_channels, 
                                                   phase, 
                                                   "Encoder_Level_0") 
    # Rest of the levels
    for down_path_index in range(1,network_depth):
        previous_level_channels = net_channels_down[down_path_index-1]
        level_channels = net_channels_down[down_path_index]
        level_layers = net_layers_down[down_path_index]
        (h_down[down_path_index], h_relu_down[down_path_index]) = segm3D.Descendent_level(level_layers, 
                                                                                   size_filt_fine, 
                                                                                   h_down[down_path_index-1], 
                                                                                   previous_level_channels, 
                                                                                   level_channels, 
                                                                                   phase, 
                                                                                   "Encoder_Level_%d"%down_path_index)
               
        
    # # Encode Level
    if fullycon_code:
        h_code = Encode_level(net_layers_encode[0],  
                               h_down[network_depth-1], 
                               net_neurons_encode[0], 
                               code_size[0],
                               phase, 
                               "Encode_Level")
        
    else:
        h_code = h_down[network_depth-1]

        
       
    # -- Now we construct the encoder
    
    # # Decode or Base level
    if fullycon_code:
        # Calculate decoded low-res volume size
        out_size = [-1,
                    int(input_size[0]/(2**(network_depth))),
                    int(input_size[1]/(2**(network_depth))),
                    int(input_size[2]/(2**(network_depth))),
                    int(net_channels_down[-1])]

        h_decode = Decode_level(net_layers_decode[0],  
                                           h_code, 
                                    net_neurons_encode[0],
                                    out_size,
                                          phase, 
                                           "Decode_Level")
    else:
        h_decode = h_code


    # -- We start the upward path

    network_depth = len(net_channels_up)


    # -- We complete the upward path

    # First level:
    previous_level_channels = net_channels_down[-1] # Incoming channels
    level_channels = net_channels_up[network_depth-1] # this chanel levels
    level_layers = net_layers_up[network_depth-1] # this chanelslayers
    h_up = OrderedDict()
    h_relu_up = OrderedDict()
    (h_up[network_depth-1], h_relu_up[network_depth-1]) = segm3D.Base_level(level_layers, 
                                                                            size_filt_fine, 
                                                                            h_decode, 
                                                                            previous_level_channels, 
                                                                            level_channels, 
                                                                            phase, 
                                                                            "Decoder_Level_%d"%(network_depth-1),
                                                                            non_lin_func = tf.nn.relu)

    # Rest of the levels
    for up_path_index in range(network_depth-2,-1,-1):
        # All previous level channels are divided by 2 by the up convolution
        previous_level_channels = net_channels_up[up_path_index+1]//2 
        level_channels = net_channels_up[up_path_index]
        level_layers = net_layers_up[up_path_index]
        (h_up[up_path_index], h_relu_up[up_path_index]) = segm3D.Base_level(level_layers, 
                                                                                   size_filt_fine, 
                                                                                   h_up[up_path_index+1], 
                                                                                   previous_level_channels, 
                                                                                   level_channels, 
                                                                                   phase, 
                                                                                   "Decoder_Level_%d"%up_path_index,
                                                                            non_lin_func = tf.nn.relu)


    # -- Finally we construct the output labels
    (h_recon_raw) = segm3D.Output_layer(net_layers_output[0], 
                                      size_filt_out, 
                                      h_up[0], 
                                      net_channels_up[0]//2, 
                                      input_channels, 
                                      phase, 
                                      "Image_construction_level",
                                     non_lin_func = None)

    # Pass logits through sigmoid to get reconstructed image
    with tf.name_scope('softmax_node'):
        sigmoid_out = tf.nn.sigmoid(h_recon_raw)
  
        
    # And we return the network topology
    if all_outs:
        return sigmoid_out, h_down, h_relu_down, h_base, h_relu_base, h_up, h_relu_up, h_relu_out, h_recon_raw, h_decode, h_code
    else:
        return sigmoid_out, h_recon_raw