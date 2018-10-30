import tensorflow as tf
import util

class ResNet():

  def __init__( self ):
    self.phase = tf.placeholder( tf.bool )
    self.with_classifier = False  # Do we need IMAGENET classifier

  def pad( self, data, ksize ):
    total = ksize-1
    pb = total//2
    pe = total-pb
    return tf.pad( data, [[0,0], [pb,pe], [pb,pe], [0,0]])

  def projection_shortcut( self, input, num_conv, strides, name ):
    h0 = util.conv2d( input, [1,1,num_conv], strides, name=name + "/shortcut", use_bias = False, padding = ('SAME' if strides[1] == 1 else 'VALID') )
    return h0

  def resnet_v2_bottleneck_block( self, input, num_conv, strides, name, projection_shortcut = False ):

    shortcut = input

    bn1 = util.batch_norm( input, name + "/bn1", self.phase )
    bn1 = tf.nn.relu( bn1 )

    # 1x1 Conv layer
    if projection_shortcut:
      shortcut = self.projection_shortcut( bn1, num_conv*4, strides, name )

    conv1 = util.conv2d( bn1, [1,1,num_conv], [1,1,1,1], name = name + "/conv1", use_bias = False )
    bn2   = util.batch_norm( conv1, name + "/bn2", self.phase )
    bn2   = tf.nn.relu( bn2 )

    if strides[1] > 1:
      bn2 = self.pad( bn2, 3 )

    conv2 = util.conv2d( bn2, [3,3,num_conv], strides, name=name + "/conv2", use_bias = False, padding = ('SAME' if strides[1] == 1 else 'VALID') )
    bn3   = util.batch_norm( conv2, name + "/bn3", self.phase )
    bn3   = tf.nn.relu( bn3 )

    conv3 = util.conv2d( bn3, [1,1,num_conv*4], [1,1,1,1], name=name + "/conv3", use_bias = False )

    return conv3 + shortcut

  # ResNet 50 (v2)
  # input is of shape BATCH x 224 x 224 x 3, ordered R, G, B (NOT B, G, R !)
  def resnet_v2( self, input ):

    strides   = [1,2,2,2]
    blocks    = [3,4,6,3]
    num_conv  = [64, 128, 256, 512]

    input = self.pad( input, 7 )
    res = util.conv2d( input, [7,7,64], stride=[1,2,2,1], padding = 'VALID', name = "conv_pre", use_bias = False )
    res = tf.nn.max_pool( res, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME' )

    for j,b in enumerate( blocks ):
      block_stride = [1, strides[j], strides[j], 1]

      res = self.resnet_v2_bottleneck_block( res, num_conv = num_conv[j], strides = block_stride, name = "block" + str(j+1) + "-1", projection_shortcut = True )

      for i in range( 1, b ):
        res = self.resnet_v2_bottleneck_block( res, num_conv = num_conv[j], strides = [1,1,1,1], name = "block" + str(j+1) + "-" + str(i+1) )

    res = util.batch_norm( res, "post_bn", self.phase )
    res = tf.nn.relu( res )

    self.spatial = res

    # Average Pooling over both spatial dimensions
    res = tf.reduce_mean( res, axis=[1,2] )

    # With ImageNet classifier
    if self.with_classifier:
      res = util.fc( res, 1001, "imagenet_dense" )


    return res
