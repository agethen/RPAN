import tensorflow as tf 
import numpy

import data_handler as data 
import model_simple as model

NUM_EPOCHS  = 10    # Number of epochs to train
BATCH       = 1     # How many items per iteration on GPU?
ACCUM       = 16    # Update gradients after how many iterations?
T           = 1     # Load how many frames from video?
LR          = 1e-3  # Initial learning rate
GAMMA       = 1e-5  # Regularization weights (L2)

# Load data handler
# PREFIX_RGB: Path to RGB frames (structured as PREFIX_RGB/video_id/frame_000001.jpg etc)
# PREFIX_POSE: Path to Pose files (PREFIX_POSE/video_id.npy)
# annotation: Annotation file, e.g., example.csv
hnd_train = data.DataHandler( PREFIX_RGB = "/path/to/folder/", PREFIX_POSE = "/path/to/folder/", \
                              annotation = "example.csv", T=T, is_test=False, do_resize=None )

hnd_test  = data.DataHandler( PREFIX_RGB = "/path/to/folder/", PREFIX_POSE = "/path/to/folder/", \
                              annotation = "example.csv", T=T, is_test=True, do_resize=None )

num_train = hnd_train.num()
num_test  = hnd_test.num()

C         = hnd_train.num_classes()

# Build graph
graph = model.Graph( T, C )
graph.build_graph()


conf = tf.ConfigProto(  gpu_options = tf.GPUOptions( allow_growth = True ),
                        device_count = { 'GPU': 1 } )

with tf.Session( config = conf ) as sess:
  tf.global_variables_initializer().run()

  train_range = zip( range(0, num_train, BATCH), range(BATCH, num_train+1, BATCH) )
  test_range  = zip( range(0, num_test,  BATCH), range(BATCH, num_test+1,  BATCH) )

  # Size of datasets may not always be multiples of BATCH. Ensure we do not "forget" data
  if num_train%BATCH > 0:
    train_range.append( (num_train-(num_train%BATCH), num_train+1) )
  if num_test%BATCH > 0:
    test_range.append( (num_test-(num_test%BATCH), num_test+1) )


  # Main loop
  for epoch in range( NUM_EPOCHS ):
    
    hnd_train.shuffle()
    sess.run( graph.zero_ops ) # Clear gradient accumulators

    # Train phase
    cnt_iter_train = 0
    for start, end in train_range:

      data, label = hnd_train.load_rgb( start, end )
      posemaps    = hnd_train.load_pose_map( start, end )

      _, loss     = sess.run( [graph.accum_ops, graph.total_loss] , 
                              feed_dict = { graph.X : data, graph.Y : label, graph.P : posemaps, graph.gamma : GAMMA, graph.phase : False } )

      cnt_iter_train += 1

      # Update gradients after ACCUM minibatches
      if cnt_iter_train%ACCUM == 0:
        sess.run( graph.op, feed_dict = { graph.LR : LR } )         # Apply gradients
        sess.run( graph.zero_ops )   # Zero gradient buffer


    # Test phase
    cnt_iter_test = 0
    for start, end in test_range:

      data, label = hnd_test.load_rgb( start, end )
      posemaps    = hnd_test.load_pose_map( start, end )

      loss, res   = sess.run( [graph.total_loss, graph.result], 
                              feed_dict = { graph.X : data, graph.Y : label, graph.P : posemaps, graph.gamma : GAMMA, graph.phase : False } )

      cnt_iter_test += 1
