#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

# External
import tensorflow as tf
import numpy as np
from collections import OrderedDict

# Internal
import Geometrias_3D as geo3D


TF_DTYPE_USE = tf.float32
DO_NOT_CREATE_SUMMARIES = False

#-----------------------------------------------------------------------------#
#--------------- Variable generation -----------------------------------------#
#-----------------------------------------------------------------------------#

def weight_variable(shape_in, nombre, Initializer = 'xavier'):
    
    if Initializer == 'normal':
        initial = tf.truncated_normal(shape_in, stddev=0.1, dtype=TF_DTYPE_USE)
        #return tf.Variable(initial)
        w_var =  tf.get_variable(nombre, dtype=TF_DTYPE_USE, initializer=initial)
        
    elif Initializer == 'xavier':        
        w_var =  tf.get_variable(nombre, dtype=TF_DTYPE_USE, shape=shape_in,  initializer=tf.contrib.layers.xavier_initializer())
        
    else:
        print('Initializer not found')
        error()
    
    return w_var


def bias_variable(shape, nombre):
    initial = tf.constant(0.1, shape=shape, dtype=TF_DTYPE_USE)
    #return tf.Variable(initial)
    return tf.get_variable(nombre, dtype=TF_DTYPE_USE, initializer=initial)


def trilinear_upsample_kernel_weight_variable(shape, nombre):
    
    # Get the size of the filter
    kernel_size = shape[0]
    
    # Calculate the center anf factor
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    # Create a list of vectors in each axis
    og = np.ogrid[:kernel_size, :kernel_size, :kernel_size]
    # Create the filter by multiplication of the three vecors, 
    # each vector contains a linear interpolator from the center
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor) * \
           (1 - abs(og[2] - center) / factor)
    
    # Expand the filter to the desired shape by repeating its structure
    filt_expanded = filt[:,:,:,np.newaxis,np.newaxis]
    filt_expanded = np.repeat(filt_expanded, shape[3], axis=3)
    filt_expanded = np.repeat(filt_expanded, shape[4], axis=4)
    
    # Cast to selected type
    if TF_DTYPE_USE == tf.float32:
        filt_expanded = filt_expanded.astype(np.float32)
    elif TF_DTYPE_USE == tf.float16:
        filt_expanded = filt_expanded.astype(np.float16)
    
    # Initialize the variables using the filter values
    return tf.get_variable(nombre, dtype=TF_DTYPE_USE, initializer=filt_expanded)
    
    
    
    

#-----------------------------------------------------------------------------#
#--------------- Variable normalization --------------------------------------#
#-----------------------------------------------------------------------------#

def spectral_norm(w, iteration=1, nombre='u'):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    with tf.variable_scope(nombre, reuse=tf.AUTO_REUSE):
        u = tf.get_variable(nombre, [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)


    return w_norm

#-----------------------------------------------------------------------------#
#--------------- Variable loading --------------------------------------------#
#-----------------------------------------------------------------------------#

# from tensorflow.python.tools import inspect_checkpoint as inch
# inch.print_tensors_in_checkpoint_file(CHECKPOINT_PATH_AE_CLASS_0, '', all_tensors=True)

def create_restore_dict(graph_keys, old_scope, new_scope):
    
    # Create dictionary
    restore_dictinary = dict()
    
    
    if old_scope == new_scope:
        # If the model is the same, just create a trivial dictionary
        for a in tf.get_collection(graph_keys, scope=new_scope):
            restore_dictinary[a.name[:-2]] = a
    else:
        # Iterat throug all tensors in the scope
        for a in tf.get_collection(graph_keys, scope=new_scope):
            # Split the tensor name by the first '/' to get the raw tensor name
            partition_nom = a.name[:-2].partition('/')
            # Construct the old tensor name using the old scope name
            if old_scope=='':
                old_name = partition_nom[-1]
            else:
                old_name = old_scope+'/'+partition_nom[-1]
            # Assing the net tensor to the dictionary entry of the old name
            restore_dictinary[old_name] = a
        

    return restore_dictinary

def assign_and_convert_halfPrecision(restore_dictinary, CHECKPOINT_PATH):
    
    # Iterate over the dictionary containing the variables to load
    for variable_name_old, varible_new in restore_dictinary.items():
        
        # Load the variable from the checkpoint
        var = tf.contrib.framework.load_variable(CHECKPOINT_PATH, variable_name_old)
        
        # Assign to new graph
        if(var.dtype == np.float32) and (varible_new.dtype == np.float16):
            # If the variable is float16 in the new graph, we cast it
            tf.add_to_collection('assignOps', varible_new.assign(tf.cast(var, tf.float16)))
        else:
            # If the variable in the old graph is float16 or the new variable is float32, 
            # we load it directly
            tf.add_to_collection('assignOps', varible_new.assign(var))
        
   
    # Return the operation
    return tf.get_collection('assignOps')


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
def nn_3D_conv_layer(input_tensor, input_size, output_size, filt_size, phase, layer_name, non_linear_function = tf.nn.relu, use_Spect_Norm = False):
    # Weight creation
    with tf.name_scope('weights'):
        W = weight_variable([filt_size[0], filt_size[1], filt_size[2], input_size, output_size], layer_name+'_weights')
        if use_Spect_Norm:
            W = spectral_norm(W, nombre=layer_name+'_u')
        variable_summaries(W)
    # Bias creation
    with tf.name_scope('biases'):
        b = bias_variable([output_size], layer_name+'_biases')
        variable_summaries(b)
    # 3D Convolution operation
    h_conv= conv3d_fine(input_tensor, W)
    # Batch normalization
    if TF_DTYPE_USE == tf.float32:
        h_BN = tf.layers.batch_normalization(tf.nn.bias_add(h_conv, b), training=phase);
    else:
        h_BN = tf.dtypes.cast(tf.layers.batch_normalization(tf.dtypes.cast(tf.nn.bias_add(h_conv, b),dtype=tf.float32), training=phase),dtype=tf.float16);
        #h_BN = tf.layers.batch_normalization(tf.nn.bias_add(h_conv, b), training=phase, fused=False);
    # Alineality
    if non_linear_function == None:
        h_alin = h_BN
    else:
        h_alin = non_linear_function(h_BN)
    return h_alin
    

#-----------------------------------------------------------------------------#
#--------------- 3D Segmentation Network Levels ------------------------------#
#-----------------------------------------------------------------------------#
    
def Descendent_level(N_conv, filt_size, input_tensor, input_size, internal_size, phase, layer_name, non_lin_func = tf.nn.relu, use_Spect_Norm = False):
    h = OrderedDict()
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # First convolution layer creation 
        h[0] = nn_3D_conv_layer(input_tensor, input_size, internal_size, filt_size, phase, layer_name+"_0", non_linear_function = non_lin_func, use_Spect_Norm = use_Spect_Norm)
        # If there are mores layers in this level, lets create them
        if N_conv > 1:
            for i in range(1,N_conv):
                # Convolutional layer creation
                h[i] = nn_3D_conv_layer(h[i-1], internal_size, internal_size, filt_size, phase, layer_name+"_%d"%i, non_linear_function = non_lin_func, use_Spect_Norm = use_Spect_Norm)
                
        h_relu = h[N_conv-1]
            
        # After creating all internal layers, we perform the down-convolution
        # Weight creation
        with tf.name_scope('weights_down'):
            #W = weight_variable([2, 2, 2, internal_size, internal_size], layer_name+'_weights_down_weights')
            W = trilinear_upsample_kernel_weight_variable([2, 2, 2, internal_size, internal_size], 
                                                          layer_name+'_weights_down_weights')
            
            if use_Spect_Norm:
                W = spectral_norm(W, nombre=layer_name+'_u')
                
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
                h[i] = nn_3D_conv_layer(h[i-1], internal_size, internal_size, filt_size, phase, layer_name+"_%d"%i, non_linear_function = non_lin_func)
                
        h_relu = h[N_conv-1]

        # After creating all internal layers, we perform the up-convolution
        # Weight creation
        with tf.name_scope('weights_up'):
            #W = weight_variable([2, 2, 2, internal_size//2, internal_size], layer_name+'_weights_up_weights')
            W = trilinear_upsample_kernel_weight_variable([2, 2, 2, internal_size//2, internal_size], 
                                                          layer_name+'_weights_up_weights')
            variable_summaries(W)
        # Up-convolution
        h_out = conv3d_up(h_relu, W)

        # Finaly we return the up-convolutioned tensor and
        # the last output tensor of "internal_size"
        return (h_out, h_relu)
    
    
def Ascendent_level(N_conv, filt_size, input_tensor, input_size, internal_size, detail_tensor, phase, layer_name, non_lin_func = tf.nn.relu, last_level=False):
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
                h[i] = nn_3D_conv_layer(h[i-1], internal_size, internal_size, filt_size, phase, layer_name+"_%d"%i, non_linear_function = non_lin_func)
            
        h_relu = h[N_conv-1]
        
        # After creating all internal layers, we perform the up-convolution if this is not the last level
        if not last_level:
            # Weight creation
            with tf.name_scope(layer_name+'weights_up'):
                #W = weight_variable([2, 2, 2, internal_size//2, internal_size], layer_name+'_weights_up_weights')
                W = trilinear_upsample_kernel_weight_variable([2, 2, 2, internal_size//2, internal_size], 
                                                              layer_name+'_weights_up_weights')
                variable_summaries(W)
            # Up-convolution
            h_out = conv3d_up(h_relu, W)
        else:
            h_out = h_relu

        # Finaly we return the up-convolutioned (upsampled) tensor and
        # the last output tensor of "internal_size"
        return (h_out, h_relu)
    
    
def Output_layer(N_conv, filt_size, input_tensor, input_size, out_chanels, phase, layer_name, non_lin_func = tf.nn.relu):
    h = OrderedDict()
    h_raw = OrderedDict()
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # First the convolution layer creation 
        h_raw[0] = nn_3D_conv_layer(input_tensor, input_size, out_chanels, filt_size, phase, layer_name+"_0", non_linear_function = None)
        h[0] = non_lin_func(h_raw[0])
        # If there are mores layers in this level, lets create them
        if N_conv > 1:
            for i in range(1,N_conv):
                # Convolutional layer creation
                h_raw[i] = nn_3D_conv_layer(h[i-1], out_chanels, out_chanels, filt_size, phase, layer_name+"_%d"%i, non_linear_function = None)
                h[i] = non_lin_func(h_raw[i])
        # Finally we return the output maps
        return (h[N_conv-1], h_raw[N_conv-1])
    

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
                                 num_textures = 1,
                             all_outs = False):
    
    # Get basic V-Net Structure
    (v_out,
     h_out,
     h_down, 
     h_relu_down, 
     h_base,
     h_relu_base, 
     h_up, 
     h_relu_up) = Assemble_Network(ph_entry, 
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
                         all_outs = True) 
    
    #with tf.name_scope('sigmoid_node'):
    #    h_alin_out = tf.nn.sigmoid(h_out)
    
    with tf.name_scope('tanh_relu_node'):
        h_alin_out = tf.nn.tanh(v_out)
        
    if all_outs:
        return h_alin_out, h_out, h_down, h_relu_down, h_base, h_relu_base, h_up, h_relu_up
    else:
        return h_alin_out, h_out




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
                                 net_layers_segm,
                                 all_outs = False):
    
    # Get basic V-Net Structure
    (v_out,
     h_out,
     h_down, 
     h_relu_down, 
     h_base,
     h_relu_base, 
     h_up, 
     h_relu_up) = Assemble_Network(ph_entry, 
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
    
    with tf.name_scope('softmax_node'):
        soft_out = tf.nn.softmax(h_out,-1)
        
    if all_outs:
        return soft_out, h_out, h_down, h_relu_down, h_base, h_relu_base, h_up, h_relu_up
    else:
        return soft_out, h_out


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
                                           net_layers_texture,
                                           acivation_fcn = tf.nn.sigmoid,
                                          all_outs = False):



    with tf.name_scope('Base_network'):

        (h_relu_out,
         h_out,
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
    
    with tf.name_scope('Segmentation_softmax_node'):
        soft_out = tf.nn.softmax(h_out,-1)
    
    # Add aditional layers for texturing 
    with tf.name_scope("Terxturing_layers"):
        h_nn = OrderedDict()

        h_nn[0] = tf.layers.dense(h_relu_up[0],
                                   net_texture_neurons,
                                   activation=None,
                                   use_bias=True,
                                   kernel_initializer=None,
                                   bias_initializer=tf.zeros_initializer(),
                                   kernel_regularizer=None,
                                   bias_regularizer=None,
                                   activity_regularizer=None,
                                   #kernel_constraint=None,
                                   #bias_constraint=None,
                                   trainable=True,
                                   name='Input_layer',
                                   reuse=None)
        
        # Batch normalization
        if TF_DTYPE_USE == tf.float32:
            h_nn[0] = tf.layers.batch_normalization(h_nn[0], training=phase);
        else:
            h_nn[0] = tf.dtypes.cast(tf.layers.batch_normalization(tf.dtypes.cast(h_nn[0],dtype=tf.float32), training=phase),dtype=tf.float16);
#             h_nn[0] = tf.layers.batch_normalization(h_nn[0], training=phase, fused=False);
        # Activation
        h_nn[0] = acivation_fcn(h_nn[0])
        
        # ---- Hidden Layers
        for idx_layer in range(1,net_layers_texture):



            h_nn[idx_layer] = tf.layers.dense(h_nn[idx_layer-1],
                                   net_texture_neurons,
                                   activation=None,
                                   use_bias=True,
                                   kernel_initializer=None,
                                   bias_initializer=tf.zeros_initializer(),
                                   kernel_regularizer=None,
                                   bias_regularizer=None,
                                   activity_regularizer=None,
                                   #kernel_constraint=None,
                                   #bias_constraint=None,
                                   trainable=True,
                                   name='Hidden_layer_%d'%idx_layer,
                                   reuse=None)
            # Batch normalization
            if TF_DTYPE_USE == tf.float32:
                h_nn[idx_layer] = tf.layers.batch_normalization(h_nn[idx_layer], training=phase);
            else:
                h_nn[idx_layer] = tf.dtypes.cast(tf.layers.batch_normalization(tf.dtypes.cast(h_nn[idx_layer],dtype=tf.float32), training=phase),dtype=tf.float16);
#                 h_nn[idx_layer] = tf.layers.batch_normalization(h_nn[idx_layer], training=phase, fused=False);
            # Activation
            h_nn[idx_layer] = acivation_fcn(h_nn[idx_layer])
         
        
        # ----- Output layer
        h_nn[net_layers_texture] = tf.layers.dense(h_nn[net_layers_texture-1],
                                   1,
                                   activation=None,
                                   use_bias=True,
                                   kernel_initializer=None,
                                   bias_initializer=tf.zeros_initializer(),
                                   kernel_regularizer=None,
                                   bias_regularizer=None,
                                   activity_regularizer=None,
                                   #kernel_constraint=None,
                                   #bias_constraint=None,
                                   trainable=True,
                                   name='Output_layer',
                                   reuse=None)
        
        # Batch normalization
        if TF_DTYPE_USE == tf.float32:
            h_nn[net_layers_texture] = tf.layers.batch_normalization(h_nn[net_layers_texture], training=phase);
        else:
            h_nn[net_layers_texture] = tf.dtypes.cast(tf.layers.batch_normalization(tf.dtypes.cast(h_nn[net_layers_texture],dtype=tf.float32), training=phase),dtype=tf.float16);
#             h_nn[net_layers_texture] = tf.layers.batch_normalization(h_nn[net_layers_texture], training=phase, fused=False);
        # Activation
        #texture_tensor = acivation_fcn(h_nn[net_layers_texture])
        #texture_tensor = tf.nn.sigmoid(h_nn[net_layers_texture])
        texture_tensor = tf.nn.tanh(tf.nn.relu(h_nn[net_layers_texture]))
        
        
    if all_outs:
        return soft_out, h_out, texture_tensor, h_down, h_relu_down, h_base, h_relu_base, h_up, h_relu_up
    else:
        return soft_out, h_out, texture_tensor





def Assemble_Labeled_Texture_Network(ph_entry, 
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
                                           net_layers_out_text,
                                           acivation_fcn = tf.nn.sigmoid, 
                                     all_outs = False):



    with tf.name_scope('Texture_network'):
        
        (texture_tensor, 
        h_out,
        h_down, 
        h_relu_down, 
        h_base, 
        h_relu_base, 
        h_up, 
        h_relu_up ) = Assemble_Tenxture_Network(ph_entry, 
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
                                                          net_layers_out_text,
                                                          num_textures = 1,
                                                         all_outs = True)

        
    with tf.name_scope('Segmentation_network'):
        # This s the convolutional segmentation layer, takes the 
        # texture image and computes the labels from it
        (h_relu_out, h_out) = Output_layer(net_layers_segm[0], 
                                           size_filt_out, 
                                           texture_tensor, 
                                           1, 
                                           num_clases, 
                                           phase, 
                                           "Output_level_segmentation")
        
        
        with tf.name_scope('softmax_node'):
            soft_out = tf.nn.softmax(h_out,-1)
    
    if all_outs:
        return soft_out, h_out, texture_tensor, h_down, h_relu_down, h_base, h_relu_base, h_up, h_relu_up
    else:
        return soft_out, h_out, texture_tensor


def Assemble_Segmentation_and_Texture_FullyConvolutional(ph_entry, 
                                                         phase, 
                                                         input_size, 
                                                         input_channels, 
                                                         num_clases, 
                                                         num_tex,
                                                         size_filt_fine, 
                                                         size_filt_out,
                                                         size_filt_tex,
                                                         network_depth, 
                                                         net_channels_down, 
                                                         net_layers_down, 
                                                         net_channels_base, 
                                                         net_layers_base, 
                                                         net_channels_up, 
                                                         net_layers_up, 
                                                         net_layers_out,
                                                         net_layers_text,
                                                         acivation_fcn = tf.nn.sigmoid, 
                                                         sigmoid_overshot = 0.3,
                                                         all_outs = False):
    
    with tf.name_scope('Base_network'):

        (h_relu_out,
         h_out_full,
         h_down, 
         h_relu_down, 
         h_base,
         h_relu_base, 
         h_up, 
         h_relu_up)  = Assemble_Network(ph_entry, 
                                        phase, 
                                        input_size, 
                                        input_channels, 
                                        num_clases+net_layers_text, 
                                        size_filt_fine, 
                                        size_filt_out,
                                        network_depth, 
                                        net_channels_down, 
                                        net_layers_down, 
                                        net_channels_base, 
                                        net_layers_base, 
                                        net_channels_up, 
                                        net_layers_up, 
                                        net_layers_out, 
                                        all_outs = True)
        # Split
        h_out_segm, h_out_text = tf.split(h_out_full, [num_clases, net_layers_text], axis = 4)
            
    with tf.name_scope('Segmentation_layers'):

        # First 3 outs are segmentation
        h_soft_segm_out = tf.nn.softmax(h_out_segm,-1)
            
    with tf.name_scope('Texturing_Layers'):
        # Last outs are texture
        # we add couple more convolutional layers
        
        if net_layers_text > 1:
            (h_out_text_mid, h_out_text_mid_raw) = Output_layer(net_layers_text, 
                                           size_filt_tex, 
                                           h_out_text, 
                                           net_layers_text, 
                                           net_layers_text, 
                                           phase, 
                                           "Texturing_Layers_mid")
        else:
            h_out_text_mid = h_out_text
            h_out_text_mid_raw = h_out_text

        (_, h_out_text_fine_raw) = Output_layer(1, 
                                       size_filt_tex, 
                                       h_out_text_mid, 
                                       net_layers_text, 
                                       num_tex, 
                                       phase, 
                                       "Texturing_Layers_out")

        h_out_text_fine = acivation_fcn(h_out_text_fine_raw)
        
        if acivation_fcn == tf.nn.sigmoid:
            h_out_text_fine = h_out_text_fine*(1+sigmoid_overshot)

                        
    if all_outs:
        return h_soft_segm_out, h_out_text_fine, h_out_segm, h_out_text_fine_raw, h_out_text, h_down, h_relu_down, h_base, h_relu_base, h_up, h_relu_up
    else:
        return h_soft_segm_out, h_out_text_fine, h_out_segm, h_out_text_fine_raw


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
    last_level_flag=False
    for up_path_index in range(network_depth-2,-1,-1):
        if up_path_index== 0:
            last_level_flag=True
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
                                                                          "Level_%d_up"%up_path_index,
                                                                         last_level = last_level_flag)

    
    # -- Finally we construct the output layer
    (h_relu_out, h_out) = Output_layer(net_layers_output[0], 
                                       size_filt_out, 
                                       h_relu_up[0], 
                                       net_channels_up[0], 
                                       num_clases, 
                                       phase, 
                                       "Output_level")
  



        
    # And we return the network topology
    if all_outs:
        return h_relu_out, h_out, h_down, h_relu_down, h_base, h_relu_base, h_up, h_relu_up
    else:
        return h_relu_out, h_out
    
    
def Assemble_Classification_Featuring_Network(ph_entry, 
                                         phase, 
                                         input_size, 
                                         input_channels, 
                                         size_filt_fine, 
                                         network_depth, 
                                         net_channels_down, 
                                         net_layers_down, 
                                         spectral_normalization = False):
    
    # The input tensor must be reshaped as a 5d tensor, with last dimension the 
    # color channels
    x_vol = tf.reshape(ph_entry, [-1, input_size[0], input_size[1], input_size[2], input_channels])
    
    # Feature extraction levels


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
                                                   "Level_0_down",
                                                   use_Spect_Norm = spectral_normalization) 
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
                                                                                   "Level_%d_down"%down_path_index,
                                                                                   use_Spect_Norm = spectral_normalization)

    # The base level is a fully connected layer for clasification

    h_flat = tf.contrib.layers.flatten(h_down[network_depth-1])


    return x_vol, h_down, h_relu_down, h_flat



def Assemble_Muliscale_Classification_Featuring_Network(list_entry, 
                                                        phase, 
                                                        input_size, 
                                                        input_channels, 
                                                        size_filt_fine, 
                                                        network_depth, 
                                                        net_channels_down, 
                                                        net_layers_down, 
                                                        spectral_normalization = False):
    
    # "list_entry" contains a list of all the expansions of the input volume (as tensors)
    full_scale_entry = list_entry[0]
    
    # The input tensor must be reshaped as a 5d tensor, with last dimension the 
    # color channels
    x_vol = tf.reshape(full_scale_entry, [-1, input_size[0], input_size[1], input_size[2], input_channels])
    
    # Feature extraction levels
    with tf.variable_scope('Convolutional_layers'):

        # First level (no concatenation, just the input):
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
                                                       "Level_0_down",
                                                       use_Spect_Norm = spectral_normalization) 
        # Rest of the levels
        h_scalled_inputs = list()
        for down_path_index in range(1,network_depth):

            previous_level_channels = net_channels_down[down_path_index-1]
            level_channels = net_channels_down[down_path_index]
            level_layers = net_layers_down[down_path_index]
            
            # Convert low resolution sample to input channels
            level_scaled_entry = nn_3D_conv_layer(list_entry[down_path_index], 
                                                  1, 
                                                  previous_level_channels, 
                                                  [1,1,1], 
                                                  phase, 
                                                  "image2feature_%d"%down_path_index, 
                                                  non_linear_function = None)
            # Queep list
            h_scalled_inputs.append(level_scaled_entry)
            
            # Concatenate
            h_cat = tf.concat([level_scaled_entry, h_down[down_path_index-1]], 4) 
            
            # Apply
            (h_down[down_path_index], h_relu_down[down_path_index]) = Descendent_level(level_layers, 
                                                                                       size_filt_fine, 
                                                                                       h_cat, 
                                                                                       previous_level_channels*2, 
                                                                                       level_channels, 
                                                                                       phase, 
                                                                                       "Level_%d_down"%down_path_index,
                                                                                       use_Spect_Norm = spectral_normalization)

        # The base level is a fully connected layer for clasification

        h_flat = tf.contrib.layers.flatten(h_down[network_depth-1])


    return x_vol, h_down, h_relu_down, h_flat



def Assemble_Classification_FC(h_flat,
                               phase,
                               num_clases,
                               net_neurons_base,
                               net_layers_base,
                               net_base_activation_fcn = tf.nn.sigmoid,
                              spectral_normalization = False):
    
    # -- Initial layer
    h_nn = OrderedDict()
    with tf.variable_scope('FC_Input_layer'):

        # Set kernel contraint if needed
        kern_const = None
        if spectral_normalization:
            kern_const = lambda x: spectral_norm(x, iteration=1, nombre = 'FC_Input_layer_u')

        h_nn[0] = tf.layers.dense(h_flat,
                                  net_neurons_base,
                                  activation=None,
                                  use_bias=True,
                                  kernel_initializer=None,
                                  bias_initializer=tf.zeros_initializer(),
                                  kernel_regularizer=None,
                                  bias_regularizer=None,
                                  activity_regularizer=None,
                                  kernel_constraint=kern_const,
                                  #bias_constraint=None,
                                  trainable=True,
                                  name='FC_Input_layer',
                                  reuse=tf.AUTO_REUSE)

        # Batch normalization
        if TF_DTYPE_USE == tf.float32:
            h_nn[0] = tf.layers.batch_normalization(h_nn[0], training=phase);
        else:
            h_nn[0] = tf.dtypes.cast(tf.layers.batch_normalization(tf.dtypes.cast(h_nn[0],dtype=tf.float32), training=phase),dtype=tf.float16);
    #         h_nn[0] = tf.layers.batch_normalization(h_nn[0], training=phase, fused=False);
        # Activation
        h_nn[0] = net_base_activation_fcn(h_nn[0])

    # -- Rest of the layers
    for idx_layer in range(1,net_layers_base):

        with tf.variable_scope('FC_Hidden_layer_%d'%idx_layer):
            # Set kernel contraint if needed
            kern_const = None
            if spectral_normalization:
                kern_const = lambda x: spectral_norm(x, iteration=1, nombre = 'FC_Hidden_layer_%d'%idx_layer+'_u')

            h_nn[idx_layer] = tf.layers.dense(h_nn[idx_layer-1],
                                              net_neurons_base,
                                              activation=None,
                                              use_bias=True,
                                              kernel_initializer=None,
                                              bias_initializer=tf.zeros_initializer(),
                                              kernel_regularizer=None,
                                              bias_regularizer=None,
                                              activity_regularizer=None,
                                              kernel_constraint=kern_const,
                                              #bias_constraint=None,
                                              trainable=True,
                                              name='FC_Hidden_layer_%d'%idx_layer,
                                              reuse=tf.AUTO_REUSE)


            # Batch normalization
            if TF_DTYPE_USE == tf.float32:
                h_nn[idx_layer] = tf.layers.batch_normalization(h_nn[idx_layer], training=phase);
            else:
                h_nn[idx_layer] = tf.dtypes.cast(tf.layers.batch_normalization(tf.dtypes.cast(h_nn[idx_layer],dtype=tf.float32), training=phase),dtype=tf.float16);
    #             h_nn[idx_layer] = tf.layers.batch_normalization(h_nn[idx_layer], training=phase, fused=False);
            # Activation
            h_nn[idx_layer] = net_base_activation_fcn(h_nn[idx_layer])

    # -- Final classification
    with tf.variable_scope('FC_Output_layer'):

        # Set kernel contraint if needed
        kern_const = None
        if spectral_normalization:
            kern_const = lambda x: spectral_norm(x, iteration=1, nombre = 'FC_Output_layer_u')


        h_nn[net_layers_base] = tf.layers.dense(h_nn[net_layers_base-1],
                                                num_clases,
                                                activation=None,
                                                use_bias=True,
                                                kernel_initializer=None,
                                                bias_initializer=tf.zeros_initializer(),
                                                kernel_regularizer=None,
                                                bias_regularizer=None,
                                                activity_regularizer=None,
                                                kernel_constraint=kern_const,
                                                #bias_constraint=None,
                                                trainable=True,
                                                name='FC_Output_layer',
                                                reuse=tf.AUTO_REUSE)


        # Batch normalization
        if TF_DTYPE_USE == tf.float32:
            h_nn[net_layers_base] = tf.layers.batch_normalization(h_nn[net_layers_base], training=phase);
        else:
            h_nn[net_layers_base] = tf.dtypes.cast(tf.layers.batch_normalization(tf.dtypes.cast(h_nn[net_layers_base],dtype=tf.float32), training=phase),dtype=tf.float16);
    #         h_nn[net_layers_base] = tf.layers.batch_normalization(h_nn[net_layers_base], training=phase, fused=False);


    return h_nn
    
    
    
    

def Assemble_Classification_Network(ph_entry, 
                                    phase, 
                                    input_size, 
                                    input_channels, 
                                    num_clases, 
                                    size_filt_fine, 
                                    network_depth, 
                                    net_channels_down, 
                                    net_layers_down, 
                                    net_neurons_base, 
                                    net_layers_base,
                                    net_base_activation_fcn = tf.nn.sigmoid,
                                    spectral_normalization = False,
                                    all_outs = False):
    
    with tf.variable_scope('Convolutional_layers'):
        (x_vol,
         h_down, 
         h_relu_down, 
         h_flat) = Assemble_Classification_Featuring_Network(ph_entry, 
                                                             phase, 
                                                             input_size, 
                                                             input_channels, 
                                                             size_filt_fine, 
                                                             network_depth, 
                                                             net_channels_down, 
                                                             net_layers_down, 
                                                             spectral_normalization = spectral_normalization)

    with tf.variable_scope('Dense_layers'):

            
        h_nn = Assemble_Classification_FC(h_flat,
                                          phase,
                                          num_clases,
                                          net_neurons_base,
                                          net_layers_base,
                                          net_base_activation_fcn = tf.nn.sigmoid,
                                         spectral_normalization = spectral_normalization)
    

    with tf.variable_scope('output_prob'):
        # Probability output output
        if num_clases > 1:
            softmax_out = tf.nn.softmax(net_base_activation_fcn(h_nn[net_layers_base]),-1)
        else:
            # If there is only one class, the output must be sigmoid
            softmax_out = tf.nn.sigmoid(h_nn[net_layers_base])

    
    # And we return the network topology
    if all_outs:
        return softmax_out, h_nn[net_layers_base], h_nn, h_flat, h_down, h_relu_down
    else:
        return softmax_out, h_nn[net_layers_base]
    
    
    


def Assemble_MultiScale_Classification_Network(list_entry, 
                                               phase, 
                                               input_size, 
                                               input_channels, 
                                               num_clases, 
                                               size_filt_fine, 
                                               network_depth, 
                                               net_channels_down, 
                                               net_layers_down, 
                                               net_neurons_base, 
                                               net_layers_base,
                                               net_base_activation_fcn = tf.nn.sigmoid,
                                               spectral_normalization = False,
                                               all_outs = False):
    
    
    with tf.variable_scope('Convolutional_layers'):
        (x_vol,
         h_down, 
         h_relu_down, 
         h_flat) = Assemble_Muliscale_Classification_Featuring_Network(list_entry, 
                                                                       phase, 
                                                                       input_size, 
                                                                       input_channels, 
                                                                       size_filt_fine, 
                                                                       network_depth, 
                                                                       net_channels_down, 
                                                                       net_layers_down, 
                                                                       spectral_normalization = spectral_normalization)
      
        
    with tf.variable_scope('Dense_layers'):
            
        h_nn = Assemble_Classification_FC(h_flat,
                                          phase,
                                          num_clases,
                                          net_neurons_base,
                                          net_layers_base,
                                          net_base_activation_fcn = tf.nn.sigmoid,
                                          spectral_normalization = spectral_normalization)
    

    with tf.variable_scope('output_prob'):
        # Probability output output
        if num_clases > 1:
            softmax_out = tf.nn.softmax(net_base_activation_fcn(h_nn[net_layers_base]),-1)
        else:
            # If there is only one class, the output must be sigmoid
            softmax_out = tf.nn.sigmoid(h_nn[net_layers_base])
    
    
    # And we return the network topology
    if all_outs:
        return softmax_out, h_nn[net_layers_base], h_nn, h_flat, h_down, h_relu_down
    else:
        return softmax_out, h_nn[net_layers_base]
    
    
def Pixel_ClassNet(pixel_in, num_hidden_units, num_layers, num_clases_out, acivation_fcn = tf.nn.sigmoid):
    
    h_nn = OrderedDict()
    
    # Input layer
    with tf.variable_scope("Input_layer"):
        h_nn[0] = tf.contrib.layers.fully_connected(pixel_in,
                                          num_hidden_units,
                                          activation_fn=acivation_fcn,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          biases_initializer=tf.zeros_initializer(),
                                          reuse=tf.AUTO_REUSE,
                                          scope='Input_layer',
                                          trainable=True)

    with tf.variable_scope("Hidden_layers"):
        for idx_layer in range(1,num_layers):

            h_nn[idx_layer] = tf.contrib.layers.fully_connected(h_nn[idx_layer-1],
                                                  num_hidden_units,
                                                  activation_fn=acivation_fcn,
                                                  weights_initializer= tf.contrib.layers.xavier_initializer(),
                                                  biases_initializer=tf.zeros_initializer(),
                                                  reuse=tf.AUTO_REUSE,
                                                  scope='Hidden_layer_%d'%idx_layer,
                                                  trainable=True)

    with tf.variable_scope("Output_layer"):
        
        h_nn[num_layers] = tf.contrib.layers.fully_connected(h_nn[num_layers-1],
                                                             num_clases_out,
                                                             activation_fn=None,
                                                             weights_initializer= tf.contrib.layers.xavier_initializer(),
                                                             biases_initializer= tf.zeros_initializer(),
                                                             reuse=tf.AUTO_REUSE,
                                                             scope='Output_layer',
                                                             trainable=True)
        
    with tf.variable_scope("class_layer"):
        
        if num_clases_out == 1:
            out_soft = tf.nn.sigmoid(h_nn[num_layers])
        else:
            out_soft = tf.nn.softmax(tf.nn.relu(h_nn[num_layers]), axis = -1)
            
    return out_soft
        

def assemble_CT_SegmNet(CT_input_tensor, num_hidden_units, num_layers, num_clases_out, acivation_fcn = tf.nn.relu):
    
    map_function = lambda x: Pixel_ClassNet(x, 
                                            num_hidden_units, 
                                            num_layers, 
                                            num_clases_out, 
                                            acivation_fcn = acivation_fcn)
    
    return tf.map_fn(map_function, CT_input_tensor)
    
    
    
#-----------------------------------------------------------------------------#
#--------------- TensorBoard summaries ---------------------------------------#
#-----------------------------------------------------------------------------#

def variable_summaries(var):
    if not DO_NOT_CREATE_SUMMARIES:
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
        sum_class_1 = tf.reduce_sum(tf.cast(tf.pow(output_map,2), TF_DTYPE_USE),[1,2,3,4])
        sum_class_2 = tf.reduce_sum(tf.cast(tf.pow(ojective_map,2), TF_DTYPE_USE),[1,2,3,4])
        
        # Dice-loss
        return tf.div(tf.multiply(tf.cast(2, TF_DTYPE_USE),mult_sum) , tf.add(sum_class_1,sum_class_2))

    