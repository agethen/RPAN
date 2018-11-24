import tensorflow as tf
import numpy

# Please adjust the path to the downloaded weights here.
FILENAME_WEIGHTS = '/home/phd/01/agethen/DCN-Models/resnet_v2.npy'

try:
  print "Util.py: Loading from", FILENAME_WEIGHTS
  param_dict = numpy.load( FILENAME_WEIGHTS ).item()
except:
  print "NOT loading weights, as file", FILENAME_WEIGHTS, "was not found. Please edit util.py."
  print "Continuing with random weights."
  param_dict = {}

# Notes: We do not use any regularizer on bias (Bishop PRML)

def fc( bottom, num_out, name, use_bias = True ):
  shape = bottom.get_shape().as_list()
  if len(shape) > 2:
    bottom = tf.contrib.layers.flatten( bottom )

  num_in = bottom.get_shape().as_list()[-1]

  init_W = tf.contrib.layers.xavier_initializer()
  init_B = tf.zeros_initializer()

  if name + "/weights" in param_dict.keys():
    init_W = tf.constant_initializer(value=param_dict[name + "/weights"], dtype=tf.float32)
    print "Loading", name
  if name + "/biases"  in param_dict.keys() and use_bias:
    init_B = tf.constant_initializer(value=param_dict[name + "/biases"], dtype=tf.float32)
    print "loading", name

  W = tf.get_variable( name + "/weights", shape = [num_in, num_out], initializer = init_W, regularizer = tf.contrib.layers.l2_regularizer( 1.0 ) )

  if use_bias:
    b = tf.get_variable( name + "/biases", shape = [num_out], initializer = init_B )
    return tf.matmul( bottom, W ) + b
  else:
    return tf.matmul( bottom, W )

def conv2d( bottom, ksize, stride = [1,1,1,1], padding = 'SAME', name = "", use_bias = True ):
  num_c = bottom.get_shape().as_list()[-1]
  kernel = [ksize[0], ksize[1], num_c, ksize[2]]

  init_W = tf.contrib.layers.xavier_initializer_conv2d()
  init_B = tf.zeros_initializer()

  if name + "/weights" in param_dict.keys():
    init_W = tf.constant_initializer( value=param_dict[name + "/weights"], dtype=tf.float32)

  if name + "/biases" in param_dict.keys() and use_bias:
    init_B = tf.constant_initializer( value=param_dict[name + "/biases"],  dtype=tf.float32)

  W = tf.get_variable( name + "/weights", shape = kernel, initializer = init_W, regularizer = tf.contrib.layers.l2_regularizer( 1.0 ) )

  if use_bias:
    b = tf.get_variable( name + "/biases",  shape = [ksize[2]], initializer = init_B )
    return tf.nn.conv2d( bottom, W, strides = stride, padding = padding ) + b
  else:
    return tf.nn.conv2d( bottom, W, strides = stride, padding = padding )

def conv2d_transpose( bottom, ksize, stride, name ):
  shape = bottom.get_shape().as_list()
  num_c = shape[-1]
  kernel = [ksize[0], ksize[1], ksize[2], num_c]
  output_shape = [shape[0], shape[1]*stride[1], shape[2]*stride[2], ksize[2]]

  W = tf.get_variable( name + "/weights", shape = kernel,     initializer = tf.contrib.layers.xavier_initializer_conv2d(), regularizer = tf.contrib.layers.l2_regularizer( 1.0 ) )
  b = tf.get_variable( name + "/biases",  shape = [ksize[2]], initializer = tf.zeros_initializer() )
  return tf.nn.conv2d_transpose( bottom, W, output_shape, strides = stride ) + b

def batch_norm( input, name, phase, center=True, scale=True, axis=3 ):

  init_mean   = tf.zeros_initializer()
  init_std    = tf.ones_initializer()
  init_beta   = tf.zeros_initializer()
  init_gamma  = tf.ones_initializer()

  if name + "/moving_mean" in param_dict.keys():
    init_mean = tf.constant_initializer( value = param_dict[name + "/moving_mean"], dtype = tf.float32 )
    # print "Load BN MM", name
  if name + "/moving_variance" in param_dict.keys():
    init_std  = tf.constant_initializer( value = param_dict[name + "/moving_variance"], dtype = tf.float32 )
    # print "Load BN MV", name
  if name + "/beta" in param_dict.keys():
    init_beta = tf.constant_initializer( value = param_dict[name + "/beta"], dtype = tf.float32 )
    # print "Load BN BETA", name
  if name + "/gamma" in param_dict.keys():
    init_gamma= tf.constant_initializer( value = param_dict[name + "/gamma"], dtype = tf.float32 )
    # print "Load BN GAMMA", name

  return tf.layers.batch_normalization( input,  axis=axis, center=center, scale=scale, epsilon=1e-5,
                                        moving_mean_initializer = init_mean, moving_variance_initializer = init_std,
				                                beta_initializer = init_beta, gamma_initializer = init_gamma, training = phase, name = name, fused=True )

def conv3d( bottom, ksize, stride, padding, name, use_bias = True ):
  num_c  = bottom.get_shape().as_list()[-1]
  kernel = [ksize[0], ksize[1], ksize[2], num_c, ksize[3]]

  init_W = tf.contrib.layers.xavier_initializer()
  init_B = tf.zeros_initializer()

  if name + "/weights" in param_dict.keys():
    init_W = tf.constant_initializer( value = param_dict[name + "/weights"], dtype = tf.float32 )
  if name + "/biases" in param_dict.keys() and use_bias:
    init_B = tf.constant_initializer( value = param_dict[name + "/biases"], dtype = tf.float32 )
  W = tf.get_variable( name + "/weights", shape = kernel, initializer = init_W, regularizer = tf.contrib.layers.l2_regularizer( 1.0 ) )

  if use_bias:
    b = tf.get_variable( name + "/biases",  shape = [ksize[3]], initializer = init_B )
    return tf.nn.conv3d( bottom, W, strides = stride, padding = padding ) + b
  else:
    return tf.nn.conv3d( bottom, W, strides = stride, padding = padding )
