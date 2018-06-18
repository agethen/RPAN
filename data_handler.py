import random
import numpy
import csv
import cv2

# This class allows the loading of RGB frames and generation of the pose joint maps.
# We assume the RGB frames are available as jpeg files, and named starting from "frame_000001.jpg".
# The frames of video xxx are located in a subfolder of same name, which located in PREFIX_RGB.

# The annotation file requires fields `id` (folder name of video),
# `actions` (';' seperated list of <action time_start time_end> tuples), and `length` (the video length in seconds).

class DataHandler():
  # Read annotations from file `annotation`.
  # T: Number of frames to load from each video.
  # C: Number of action classes.
  # is_test: Does this class handle a test or a training dataset?
  def __init__( self, PREFIX_RGB, PREFIX_POSE, annotation, T, C, is_test = False ):

    self.annotations = csv.DictReader( open( annotation ) )
    self.is_test     = is_test
    self.mean        = numpy.array( [104., 117., 123.] )  # B, G, R mean

    self.PREFIX_RGB   = PREFIX_RGB    # The directory containing RGB frames (as .jpg). For each video, expect a subfolder.
    self.PREFIX_POSE  = PREFIX_POSE   # We keep CMU poses in numpy files. Each file has shape NUM_FRAMES x 18 x 3, and contains pose_coordinates in [0,1].

    self.T            = T             # How many frames per video
    self.C            = C             # Number of classes
    self.J            = 18            # Number of joints in CMU
    self.stepsize     = 5             # Sample every `stepsize`-th frame

    self.actions = []

    for row in self.annotations:
      fps   = 25.0                              # TODO: Read actual FPS instead of assuming 25 fps.
      v_len = int( float(row["length"]) * fps)  # Length of video in frames

      if row["actions"] != "":
        ra = row["actions"].split(';')
        for a in ra:
          cid = int( a.split(' ')[0][1:] )                 # Classes are annotated as "cxxx", where xxx is the class-id.
          ts  = int( fps * float( a.split(' ')[1] ) ) + 1  # Frames are numbered starting at 1.
          te  = int( fps * float( a.split(' ')[2] ) ) + 1

          if te >= v_len:
            continue

          if ts >= te:
            continue

          self.actions.append( (cid, row["id"], ts, te ) ) # Class, Video-ID, Start, End

    # Permutation (for shuffling)
    self.perm = range( len(self.actions) )

  # Return number of items in dataset
  def num( self ):
    return len(self.actions)

  # Shuffle dataset. Here, we shuffle a permutation index instead of the actual data.
  def shuffle( self ):
    random.shuffle( self.perm )

  # Draws a gaussian kernel with stddev=`sig` at position (`off_x`, `off_y`).
  # The resulting bitmap as dimensions (`l` x `l`).
  def gkern( self, l=224, sig=5., off_x = 0, off_y = 0):

    ax = numpy.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = numpy.meshgrid(ax+off_x, ax+off_y)

    kernel = numpy.exp(-(xx**2 + yy**2) / (2. * sig**2))

    # Avoid division by 0
    # Occasionally, the locations are so far outside the crop, that this value takes 0.
    # We just return an empty map then.
    if numpy.sum( kernel ) == 0:
      return numpy.zeros( (l, l) )

    kmap   = kernel / numpy.sum(kernel)

    return kmap

  # Given human pose annotations (in coordinate form), render the groundtruth pose maps.
  def load_pose_map( self, start, end ):
    
    POSE_SIZE = 7 # Size of final pose map

    posemaps  = numpy.zeros( [end-start, self.T, POSE_SIZE, POSE_SIZE, self.J], dtype=numpy.float32 )

    for b in range( start, end ):
      # Find video.
      pb     = self.perm[b]
      action = self.actions[pb]
      video  = action[1]

      # Load corresponding pose file.
      posefile = numpy.load( self.PREFIX_POSE + video + ".npy" )

      # The currently used crop and shape of RGB image.
      c        = self.last_crops[b-start]
      s        = self.last_shapes[b-start]

      # The currently used timestamp in the video
      ts   = self.last_ts[b-start]

      for t in range( self.T ):

        # Note that we assume that poses were sampled at the same fps.
        tt = ts[t]-1
        tt = min( posefile.shape[0], tt )

        pose = posefile[ tt ]

        for j in range( self.J ):

          coord = pose[j,0:2]                 # Note that coordinates are saved as (x,y)
          if coord[0] == 0 and coord[1] == 0: # Joints that are (0,0) were not detected.
            continue

          # Transform coordinates from [0,1] range to image range
          coord[0] *= s[1]
          coord[1] *= s[0]
          
          # Apply crop.
          coord -= numpy.array( [c[1], c[0]], dtype=numpy.float32 )

          # gkern(..) assumes origin in center of image.
          coord = 112-coord 

          m     = self.gkern( l=224, off_x = coord[0], off_y = coord[1] )

          # Resize m from (224x224) --> (7x7)
          m     = cv2.resize( m, (POSE_SIZE, POSE_SIZE) )

          # Normalize, if not fully 0.
          if m.max() > 0:
            m = m * (1./m.max())

          posemaps[b-start,t,:,:,j] = m

    return posemaps

  # Load a set of RGB frames. Uses OpenCV.
  def load_rgb( self, start, end ):

    data    = numpy.zeros( [end-start, self.T, 224, 224, 3], dtype=numpy.float32 )
    label   = numpy.zeros( [end-start, self.T], dtype=numpy.int32 )

    self.last_crops = []
    self.last_shapes= []
    self.last_ts    = []

    for b in range( start, end ):

      pb            = self.perm[b]
      video, ts, te = self.actions[pb][1:]

      label[b-start, :] = class_dict[action[0]]

      # Sample strategy depends on whether we are in training or test phase.
      if self.is_test == False:
        t_off     = random.randint( ts, max( te-self.stepsize*self.T, ts ) )
        frame_pos = range( t_off, t_off + self.stepsize*self.T, self.stepsize )
      else:
        frame_pos = range( ts, ts + self.stepsize*self.T, self.stepsize )

      # This will be the first frame
      self.last_ts.append( frame_pos )

      # Read one frame to determine size
      frame     = cv2.imread( self.PREFIX_RGB + video + "/" + "frame" + "_" + str( 1 ).zfill(6) + ".jpg" )

      if frame is None:
        print "Could not read video", video
        self.last_crops.append( (0,0) )
        continue

      # Generate a random crop
      sh    = [256., 256.] # frame.shape
      crop  = (random.randint( 0, sh[0]-224 ), random.randint( 0, sh[1]-224 ))

      self.last_shapes.append( sh )
      self.last_crops.append( crop )

      # Load RGB data
      for t in range( self.T ):
        frame = cv2.imread( self.PREFIX_RGB + video + "/" + "frame" + "_" + str( frame_pos[t] ).zfill(6) + ".jpg" )

        if frame is None:
          data[b-start, t]  = self.mean 	# I.e., the data will be all zeros after mean substraction.
        else:
          frame             = cv2.resize( frame, (sh[0],sh[1]) )
          data[b-start, t]  = frame[ crop[0] : crop[0] + 224, crop[1] : crop[1] + 224 ]

    data -= self.mean

    # Note that cv2 loads images as BGR. Resnet was however trained on RGB.
    # Transpose data: BGR --> RGB
    data  = data[:,:,:,:, ::-1 ]

    return data, label