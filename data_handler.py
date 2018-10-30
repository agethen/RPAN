import random
import numpy
import csv
import cv2

# This class allows the loading of RGB frames and generation of the pose joint maps.
# We assume the RGB frames (sampled at 25 fps) are available as jpeg files, and named starting from "frame_000001.jpg".
# The frames of video xxx are located in a subfolder of same name, which located in PREFIX_RGB.

# Note that we drop missing classes
# Example: Classes 3,4,1 exist.
# Resulting mapping: class 1 --> 0, class 3 --> 1, class 4 --> 2

class DataHandler():
  # Read annotations from file `annotation`.
  def __init__( self, PREFIX_RGB, PREFIX_POSE, annotation, T, is_test = False, do_resize = None ):

    self.annotations = csv.DictReader( open( annotation ) )
    self.is_test     = is_test
    self.mean        = numpy.array( [104., 117., 123.] )  # B, G, R mean

    self.PREFIX_RGB   = PREFIX_RGB    # The directory containing RGB frames (as .jpg). For each video, expect a subfolder.
    self.PREFIX_POSE  = PREFIX_POSE   # We keep CMU poses in numpy files. Each file has shape NUM_FRAMES x 18 x 3, and contains pose_coordinates in [0,1].
    self.do_resize    = do_resize

    self.T            = T             # How many frames per video
    self.J            = 18            # Number of joints in CMU
    self.stepsize     = 5             # Sample every `stepsize`-th frame

    self.actions = []
    self.known_classes = []

    # Annotation reader for format in example.csv
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

          if cid not in self.known_classes:
            self.known_classes.append( cid )

          self.actions.append( (cid, row["id"], ts, te ) ) # Class, Video-ID, Start, End

    # Permutation (for shuffling)
    self.perm = range( len(self.actions) )
    self.C    = self.num_classes()

    self.video_shapes = {}
    self.video_crops  = {}
    self.video_ts     = {}

    # Cleanup class annotations
    self.known_classes= sorted( self.known_classes )
    self.class_map    = { c : i for i,c in enumerate(self.known_classes) }

  # Note: We do not protect against invalid labels (values >= C) at this moment
  def num_classes( self ):
    return len(self.known_classes)

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
    
    POSE_SIZE = 7   # Spatial dimensions of pose map will be POSE_SIZE x POSE_SIZE

    posemaps  = numpy.zeros( [end-start, self.T, POSE_SIZE, POSE_SIZE, self.J], dtype=numpy.float32 )

    for b in range( start, end ):

      # Find video.
      pb            = self.perm[b]
      _,video,_, _  = self.actions[pb]

      # Load corresponding pose file.
      try:
        posefile = numpy.load( self.PREFIX_POSE + video + ".npy" )
      except:
        print "Could not open poses for", video
        continue

      # The currently used crop and shape of RGB image.
      c        = self.video_crops[pb]
      s        = self.video_shapes[pb]

      # The currently used timestamp in the video
      ts   = self.video_ts[pb]

      for t in range( self.T ):

        # Note that we assume that poses were sampled at the same fps.
        tt = ts[t]-1
        try:
          pose = posefile[ tt ]
        except:
          print "Could not read pose in", video, "at t=", tt
          continue

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

          # Normalize, unless all 0.
          if m.max() > 0:
            m = m * (1./m.max())

          posemaps[b-start,t,:,:,j] = m

    return posemaps

  # Load a set of RGB frames. Uses OpenCV.
  def load_rgb( self, start, end ):

    data    = numpy.zeros( [end-start, self.T, 224, 224, 3], dtype=numpy.float32 )
    label   = numpy.zeros( [end-start, self.T], dtype=numpy.int32 )

    for b in range( start, end ):

      pb               = self.perm[b]
      cid, vid, ts, te = self.actions[pb]
      
      label[b-start, :] = self.class_map[cid]

      # Sample strategies:
      # Train phase: Pick random offset in annotated action, such that we can load T frames
      # Test phase:  Load T frames beginning at first frame of annotated action
      if self.is_test == False:
        t_off     = random.randint( ts, max( te-self.stepsize*self.T, ts ) )
        frame_pos = range( t_off, t_off + self.stepsize*self.T, self.stepsize )
      else:
        frame_pos = range( ts, ts + self.stepsize*self.T, self.stepsize )

      self.video_ts[pb] = frame_pos

      # Read one frame to determine size
      frame     = cv2.imread( self.PREFIX_RGB + vid + "/" + "frame" + "_" + str( 1 ).zfill(6) + ".jpg" )

      if frame is None:
        print "Could not read video", vid
        self.video_shapes[pb] = (0,0)
        continue
      else:
        self.video_shapes[pb] = frame.shape if self.do_resize is None else self.do_resize

      # Generate a random 224x224 crop
      sh    = self.video_shapes[pb]
      crop  = (random.randint( 0, sh[0]-224 ), random.randint( 0, sh[1]-224 ))

      self.video_crops[pb] = crop

      # Load RGB data
      for t in range( self.T ):
        frame = cv2.imread( self.PREFIX_RGB +  + "/" + "frame" + "_" + str( frame_pos[t] ).zfill(6) + ".jpg" )

        if frame is None:
          print "I/O error reading from", vid, ", t=", frame_pos[t]
          data[b-start, t]  = self.mean 	# I.e., the data will be all zeros.
        else:
          if self.do_resize is not None:
            frame             = cv2.resize( frame, (sh[0],sh[1]) )
          data[b-start, t]  = frame[ crop[0] : crop[0] + 224, crop[1] : crop[1] + 224 ]

    data -= self.mean
    data  = data[:,:,:,:, ::-1 ] # Note that cv2 loads images as BGR. Transpose: BGR --> RGB

    return data, label
