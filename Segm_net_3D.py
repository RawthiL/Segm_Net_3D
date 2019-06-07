#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

# External
import tensorflow as tf
import numpy as np
from collections import OrderedDict

# Internal
import Geometrias_3D as geo3D


#-----------------------------------------------------------------------------#
#--------------- Variable generation -----------------------------------------#
#-----------------------------------------------------------------------------#

def weight_variable(shape, nombre):
    initial = tf.truncated_normal(shape, stddev=0.1)
    #return tf.Variable(initial)
    return tf.get_variable(nombre, dtype=tf.float32, initializer=initial)


def bias_variable(shape, nombre):
    initial = tf.constant(0.1, shape=shape)
    #return tf.Variable(initial)
    return tf.get_variable(nombre, dtype=tf.float32, initializer=initial)


#-----------------------------------------------------------------------------#
#--------------- Convolution and pooling Layers ------------------------------#
#-----------------------------------------------------------------------------#



def conv3d_fine(x, W):
    paso = 1
    return tf.nn.conv3d(x, W, strides=[1, paso, paso, paso, 1], padding='SAME')

def conv3d_down(x, W):
    with tf.name_scope('Down_conv_2'):
        paso = 2
        return tf.nn.conv3d(x, W, strides=[1, paso, paso, paso, 1], padding='SAME')

def max_pool_down(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1],
                        strides=[1, 2, 2, 2, 1], padding='SAME')

# Convolution with fractional strides
def conv3d_up(x, W):
    with tf.name_scope('Up_conv_2'):
        paso = 2
        x_shape = tf.shape(x)
        output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]*2, x_shape[4]//2])
        return tf.nn.conv3d_transpose(x, W, output_shape, strides=[1, paso, paso, paso, 1], padding='SAME')

# 3D Convolutional layer
def nn_3D_conv_layer(input_tensor, input_size, output_size, filt_size, phase, layer_name, non_linear_function = tf.nn.relu):
    # Weight creation
    with tf.name_scope('weights'):
        W = weight_variable([filt_size[0], filt_size[1], filt_size[2], input_size, output_size], layer_name+'_weights')
        variable_summaries(W)
    # Bias creation
    with tf.name_scope('biases'):
        b = bias_variable([output_size], layer_name+'_biases')
        variable_summaries(b)
    # 3D Convolution operation
    h_conv= conv3d_fine(input_tensor, W)
    # Batch normalization
    h_BN = tf.layers.batch_normalization(tf.nn.bias_add(h_conv, b), training=phase);
    # Alineality
    if non_linear_function == None:
        h_alin = h_BN
    else:
        h_alin = non_linear_function(h_BN)
    return h_alin
    

#-----------------------------------------------------------------------------#
#--------------- 3D Segmentation Network Levels ------------------------------#
#-----------------------------------------------------------------------------#
    
def Descendent_level(N_conv, filt_size, input_tensor, input_size, internal_size, phase, layer_name, non_lin_func = tf.nn.relu):
    h = OrderedDict()
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # First convolution layer creation 
        h[0] = nn_3D_conv_layer(input_tensor, input_size, internal_size, filt_size, phase, layer_name+"_0", non_linear_function = non_lin_func)
        # If there are mores layers in this level, lets create them
        if N_conv > 1:
            for i in range(1,N_conv):
                # Convolutional layer creation
                h[i] = nn_3D_conv_layer(h[i-1], internal_size, internal_size, filt_size, phase, layer_name+"_%d"%N_conv, non_linear_function = non_lin_func)
                
        h_relu = h[N_conv-1]
            
        # After creating all internal layers, we perform the down-convolution
        # Weight creation
        with tf.name_scope('weights_down'):
            W = weight_variable([2, 2, 2, internal_size, internal_size], layer_name+'_weights_down_weights')
            variable_summaries(W)
        # Down-convolution
        h_out = conv3d_down(h_relu, W)

        # Finaly we return the down-convolutioned (downsampled) tensor and
        # the last output tensor of "internal_size", used in the ascending levels
        return (h_out, h_relu)
    
def Base_level(N_conv, filt_size, input_tensor, input_size, internal_size, phase, layer_name, non_lin_func = tf.nn.relu):
    h = OrderedDict()
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # First the convolution layer creation 
        h[0] = nn_3D_conv_layer(input_tensor, input_size, internal_size, filt_size, phase, layer_name+"_0", non_linear_function = non_lin_func)
        # If there are mores layers in this level, lets create them
        if N_conv > 1:
            for i in range(1,N_conv):
                # Convolutional layer creation
                h[i] = nn_3D_conv_layer(h[i-1], internal_size, internal_size, filt_size, phase, layer_name+"_%d"%N_conv, non_linear_function = non_lin_func)
                
        h_relu = h[N_conv-1]

        # After creating all internal layers, we perform the up-convolution
        # Weight creation
        with tf.name_scope('weights_up'):
            W = weight_variable([2, 2, 2, internal_size//2, internal_size], layer_name+'_weights_up_weights')
            variable_summaries(W)
        # Up-convolution
        h_out = conv3d_up(h_relu, W)

        # Finaly we return the up-convolutioned tensor and
        # the last output tensor of "internal_size"
        return (h_out, h_relu)
    
    
def Ascendent_level(N_conv, filt_size, input_tensor, input_size, internal_size, detail_tensor, phase, layer_name, non_lin_func = tf.nn.relu):
    h = OrderedDict()
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # First the input must be concatenated with the detal vector
        h_cat = tf.concat([detail_tensor, input_tensor], 4) 
        # Now the convolution layer creation 
        h[0] = nn_3D_conv_layer(h_cat, input_size, internal_size, filt_size, phase, layer_name+"_0", non_linear_function = non_lin_func)
        # If there are mores layers in this level, lets create them
        if N_conv > 1:
            for i in range(1,N_conv):
                # Convolutional layer creation
                h[i] = nn_3D_conv_layer(h[i-1], internal_size, internal_size, filt_size, phase, layer_name+"_%d"%N_conv, non_linear_function = non_lin_func)
            
        h_relu = h[N_conv-1]
        
        # After creating all internal layers, we perform the up-convolution
        # Weight creation
        with tf.name_scope(layer_name+'weights_up'):
            W = weight_variable([2, 2, 2, internal_size//2, internal_size], layer_name+'_weights_up_weights')
            variable_summaries(W)
        # Up-convolution
        h_out = conv3d_up(h_relu, W)

        # Finaly we return the up-convolutioned (upsampled) tensor and
        # the last output tensor of "internal_size"
        return (h_out, h_relu)
    
    
def Output_layer(N_conv, filt_size, input_tensor, input_size, out_chanels, phase, layer_name, non_lin_func = tf.nn.relu):
    h = OrderedDict()
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # First the convolution layer creation 
        h[0] = nn_3D_conv_layer(input_tensor, input_size, out_chanels, filt_size, phase, layer_name+"_0", non_linear_function = non_lin_func)
        # If there are mores layers in this level, lets create them
        if N_conv > 1:
            for i in range(1,N_conv):
                # Convolutional layer creation
                h[i] = nn_3D_conv_layer(h[i-1], out_chanels, out_chanels, filt_size, phase, layer_name+"_%d"%N_conv, non_linear_function = non_lin_func)
        # Finally we return the output maps
        return (h[N_conv-1])
    

#-----------------------------------------------------------------------------#
#--------------- Network Levels Building -------------------------------------#
#-----------------------------------------------------------------------------#

def Assemble_Tenxture_Network(ph_entry, 
                                 phase, 
                                 input_size, 
                                 input_channels, 
                                 size_filt_fine, 
                                 size_filt_out,
                                 network_depth, 
                                 net_channels_down, 
                                 net_layers_down, 
                                 net_channels_base, 
                                 net_layers_base, 
                                 net_channels_up, 
                                 net_layers_up, 
                                 net_layers_output,
                                 num_textures = 1 ):
    
    # Get basic V-Net Structure
    v_out = Assemble_Network(ph_entry, 
                         phase, 
                         input_size, 
                         input_channels, 
                         num_textures, 
                         size_filt_fine, 
                         size_filt_out,
                         network_depth, 
                         net_channels_down, 
                         net_layers_down, 
                         net_channels_base, 
                         net_layers_base, 
                         net_channels_up, 
                         net_layers_up, 
                         net_layers_output, 
                         all_outs = False) 
    
    with tf.name_scope('sigmoid_node'):
        sigmoid_out = tf.nn.sigmoid(v_out)
        
    return sigmoid_out, v_out




def Assemble_Segmentation_Network(ph_entry, 
                                 phase, 
                                 input_size, 
                                 input_channels, 
                                 num_clases, 
                                 size_filt_fine, 
                                 size_filt_out,
                                 network_depth, 
                                 net_channels_down, 
                                 net_layers_down, 
                                 net_channels_base, 
                                 net_layers_base, 
                                 net_channels_up, 
                                 net_layers_up, 
                                 net_layers_segm):
    
    # Get basic V-Net Structure
    v_out = Assemble_Network(ph_entry, 
                         phase, 
                         input_size, 
                         input_channels, 
                         num_clases, 
                         size_filt_fine, 
                         size_filt_out,
                         network_depth, 
                         net_channels_down, 
                         net_layers_down, 
                         net_channels_base, 
                         net_layers_base, 
                         net_channels_up, 
                         net_layers_up, 
                         net_layers_segm, 
                         all_outs = False) 
    
    with tf.name_scope('softmax_node'):
        soft_out = tf.nn.softmax(v_out,-1)
        
    return soft_out, v_out


def Assemble_Mixed_Segment_Texture_Network(ph_entry, 
                                           phase, 
                                           input_size, 
                                           input_channels, 
                                           num_clases, 
                                           size_filt_fine, 
                                           size_filt_out,
                                           network_depth, 
                                           net_channels_down, 
                                           net_layers_down, 
                                           net_channels_base, 
                                           net_layers_base, 
                                           net_channels_up, 
                                           net_layers_up, 
                                           net_layers_segm,
                                           net_texture_neurons,
                                           net_layers_texture):



    with tf.variable_scope('Base_network'):

        (h_relu_out,
         h_down, 
         h_relu_down, 
         h_base,
         h_relu_base, 
         h_up, 
         h_relu_up)  = Assemble_Network(ph_entry, 
                                        phase, 
                                        input_size, 
                                        input_channels, 
                                        num_clases, 
                                        size_filt_fine, 
                                        size_filt_out,
                                        network_depth, 
                                        net_channels_down, 
                                        net_layers_down, 
                                        net_channels_base, 
                                        net_layers_base, 
                                        net_channels_up, 
                                        net_layers_up, 
                                        net_layers_segm, 
                                        all_outs = True)
    
    with tf.variable_scope('Segmentation_softmax_node'):
        soft_out = tf.nn.softmax(h_relu_out,-1)
    
    # Add aditional layers for texturing 
    with tf.variable_scope("Terxturing_layers"):
        acivation_fcn = tf.nn.sigmoid
        h_nn = OrderedDict()
        h_nn[0] = tf.contrib.layers.fully_connected(h_relu_up[0],
                                          net_texture_neurons,
                                          activation_fn=acivation_fcn,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          biases_initializer=tf.zeros_initializer(),
                                          reuse=tf.AUTO_REUSE,
                                          scope='Input_layer',
                                          trainable=True)
        for idx_layer in range(1,net_layers_texture):

            h_nn[idx_layer] = tf.contrib.layers.fully_connected(h_nn[idx_layer-1],
                                                  net_texture_neurons,
                                                  activation_fn=acivation_fcn,
                                                  weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                  biases_initializer=tf.zeros_initializer(),
                                                  reuse=tf.AUTO_REUSE,
                                                  scope='Hidden_layers',
                                                  trainable=True)
            
        h_nn[net_layers_texture] = tf.contrib.layers.fully_connected(h_nn[net_layers_texture-1],
                                          1,
                                          activation_fn=acivation_fcn,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          biases_initializer=tf.zeros_initializer(),
                                          reuse=tf.AUTO_REUSE,
                                          scope='Output_layer',
                                          trainable=True)
        
        texture_tensor = h_nn[net_layers_texture]
        
        
    
    return soft_out, h_relu_out, texture_tensor

def Assemble_Network(ph_entry, 
                     phase, 
                     input_size, 
                     input_channels, 
                     num_clases, 
                     size_filt_fine, 
                     size_filt_out,
                     network_depth, 
                     net_channels_down, 
                     net_layers_down, 
                     net_channels_base, 
                     net_layers_base, 
                     net_channels_up, 
                     net_layers_up, 
                     net_layers_output, 
                     all_outs = False):

    
    # The input tensor must be reshaped as a 5d tensor, with last dimension the 
    # color channels
    x_vol = tf.reshape(ph_entry, [-1, input_size[0], input_size[1], input_size[2], input_channels])
    
    # -- We first construct the downward path
    
    # First level:
    level_channels = net_channels_down[0]
    level_layers = net_layers_down[0]
    h_down = OrderedDict()
    h_relu_down = OrderedDict()
    (h_down[0], h_relu_down[0]) = Descendent_level(level_layers, 
                                                   size_filt_fine, 
                                                   x_vol, 
                                                   input_channels, 
                                                   level_channels, 
                                                   phase, 
                                                   "Level_0_down") 
    # Rest of the levels
    for down_path_index in range(1,network_depth):
        previous_level_channels = net_channels_down[down_path_index-1]
        level_channels = net_channels_down[down_path_index]
        level_layers = net_layers_down[down_path_index]
        (h_down[down_path_index], h_relu_down[down_path_index]) = Descendent_level(level_layers, 
                                                                                   size_filt_fine, 
                                                                                   h_down[down_path_index-1], 
                                                                                   previous_level_channels, 
                                                                                   level_channels, 
                                                                                   phase, 
                                                                                   "Level_%d_down"%down_path_index)
               
        
    # -- Now we place the base level
    (h_base, h_relu_base) = Base_level(net_layers_base[0], 
                                       size_filt_fine, 
                                       h_down[network_depth-1], 
                                       net_channels_down[network_depth-1],
                                       net_channels_base[0], 
                                       phase, 
                                       "Base_Level")

    
    # -- We start the upward path
    
    # First we connect the base layer
    level_channels = net_channels_up[network_depth-1]
    level_layers = net_layers_up[network_depth-1]
    h_up = OrderedDict()
    h_relu_up = OrderedDict()
    (h_up[network_depth-1], h_relu_up[network_depth-1]) = Ascendent_level(level_layers, 
                                                                          size_filt_fine, 
                                                                          h_base, 
                                                                          net_channels_base[0], 
                                                                          level_channels, 
                                                                          h_relu_down[network_depth-1], 
                                                                          phase, 
                                                                          "Level_%d_up"%(network_depth-1))
    # Rest of the levels
    for up_path_index in range(network_depth-2,-1,-1):
        previous_level_channels = net_channels_up[up_path_index+1]
        level_channels = net_channels_up[up_path_index]
        level_layers = net_layers_up[up_path_index]
        (h_up[up_path_index], h_relu_up[up_path_index]) = Ascendent_level(level_layers, 
                                                                          size_filt_fine, 
                                                                          h_up[up_path_index+1], 
                                                                          previous_level_channels, 
                                                                          level_channels, 
                                                                          h_relu_down[up_path_index], 
                                                                          phase, 
                                                                          "Level_%d_up"%up_path_index)

    
    # -- Finally we construct the output layer
    (h_relu_out) = Output_layer(net_layers_output[0], 
                                      size_filt_out, 
                                      h_relu_up[0], 
                                      net_channels_up[0], 
                                      num_clases, 
                                      phase, 
                                      "Output_level")
  



        
    # And we return the network topology
    if all_outs:
        return h_relu_out, h_down, h_relu_down, h_base, h_relu_base, h_up, h_relu_up
    else:
        return h_relu_out

    
#-----------------------------------------------------------------------------#
#--------------- TensorBoard summaries ---------------------------------------#
#-----------------------------------------------------------------------------#

def variable_summaries(var):
    # """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

        

#-----------------------------------------------------------------------------#
#--------------- Objective Functions -----------------------------------------#
#-----------------------------------------------------------------------------#
        
def dice_loss(output_map, ojective_map):
    with tf.name_scope('Dice_loss'):
        # Element-wise multiplication of output and truth
        inter_multi = tf.multiply(output_map, ojective_map)

        # Sumation
        mult_sum = tf.reduce_sum(inter_multi,[1,2,3,4])

        # Inter-class
        sum_class_1 = tf.reduce_sum(tf.cast(tf.pow(output_map,2), tf.float32),[1,2,3,4])
        sum_class_2 = tf.reduce_sum(tf.cast(tf.pow(ojective_map,2), tf.float32),[1,2,3,4])
        
        # Dice-loss
        return tf.div(tf.multiply(tf.cast(2, tf.float32),mult_sum) , tf.add(sum_class_1,sum_class_2))

    