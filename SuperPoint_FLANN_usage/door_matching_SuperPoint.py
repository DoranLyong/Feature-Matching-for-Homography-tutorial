# -*- coding: utf-8 -*-
"""
% CopyRight %
This code is based on the SuperPoint in following links: 
    * https://github.com/magicleap/SuperPointPretrainedNetwork
    * https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/demo_superpoint.py
"""



#%% 
import sys 
import os 
import os.path as osp 
import glob 
import time
from pathlib import Path 

import cv2 
import numpy as np 
import torch 

from SuperPoint import SuperPointFrontend, PointTracker
from cfg import opt 


""" Check OpenCV version 
""" 
if int(cv2.__version__[0]) <3 : 
    sys.exit("Warning: you need to install OpenCV version over 3")  # (ref) https://dololak.tistory.com/690


# Jet colormap for visualization.
myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])



# %%
class VideoStreamer(object):
  """ Class to help process image streams. Three types of possible inputs:"
    1.) USB Webcam.
    2.) A directory of images (files in directory matching 'img_glob').
    3.) A video file, such as an .mp4 or .avi file.
  """
  def __init__(self, basedir, camid, height, width, skip, img_glob):
    self.cap = []
    self.camera = False
    self.video_file = False
    self.listing = []
    self.sizer = [height, width]
    self.i = 0
    self.skip = skip
    self.maxlen = 1000000
    # If the "basedir" string is the word camera, then use a webcam.
    if basedir == "camera/" or basedir == "camera":
      print('==> Processing Webcam Input.')
      self.cap = cv2.VideoCapture(camid)
      self.listing = range(0, self.maxlen)
      self.camera = True
    else:
      # Try to open as a video.
      self.cap = cv2.VideoCapture(basedir)
      lastbit = basedir[-4:len(basedir)]
      if (type(self.cap) == list or not self.cap.isOpened()) and (lastbit == '.mp4'):
        raise IOError('Cannot open movie file')
      elif type(self.cap) != list and self.cap.isOpened() and (lastbit != '.txt'):
        print('==> Processing Video Input.')
        num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.listing = range(0, num_frames)
        self.listing = self.listing[::self.skip]
        self.camera = True
        self.video_file = True
        self.maxlen = len(self.listing)
      else:
        print('==> Processing Image Directory Input.')
        search = os.path.join(basedir, img_glob)
        self.listing = glob.glob(search)
        self.listing.sort()
        self.listing = self.listing[::self.skip]
        self.maxlen = len(self.listing)
        if self.maxlen == 0:
          raise IOError('No images were found (maybe bad \'--img_glob\' parameter?)')

  def read_image(self, impath, img_size):
    """ Read image as grayscale and resize to img_size.
    Inputs
      impath: Path to input image.
      img_size: (W, H) tuple specifying resize size.
    Returns
      grayim: float32 numpy array sized H x W with values in range [0, 1].
    """
    grayim = cv2.imread(impath, 0)
    if grayim is None:
      raise Exception('Error reading image %s' % impath)
    # Image is resized via opencv.
    interp = cv2.INTER_AREA
    grayim = cv2.resize(grayim, (img_size[1], img_size[0]), interpolation=interp)
    grayim = (grayim.astype('float32') / 255.)
    return grayim

  def next_frame(self):
    """ Return the next frame, and increment internal counter.
    Returns
       image: Next H x W image.
       status: True or False depending whether image was loaded.
    """
    if self.i == self.maxlen:
      return (None, False)
    if self.camera:
      ret, input_image = self.cap.read()
      if ret is False:
        print('VideoStreamer: Cannot get image from camera (maybe bad --camid?)')
        return (None, False)
      if self.video_file:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])
      input_image = cv2.resize(input_image, (self.sizer[1], self.sizer[0]),
                               interpolation=cv2.INTER_AREA)
      input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
      input_image = input_image.astype('float')/255.0
    else:
      image_file = self.listing[self.i]
      input_image = self.read_image(image_file, self.sizer)
    # Increment internal counter.
    self.i = self.i + 1
    input_image = input_image.astype('float32')
    return (input_image, True)


#%% 
if __name__ == '__main__':
    
    """ This class helps load input images from different sources.
    """
    vs = VideoStreamer(opt['input'], opt['camid'], opt['H'], opt['W'], opt['skip'], opt['img_glob'])


    """ This class runs the SuperPoint network and processes its outputs.
    """
    print('==> Loading pre-trained network.')
    fe = SuperPointFrontend(weights_path=opt['weights_path'],
                            nms_dist=opt['nms_dist'],
                            conf_thresh=opt['conf_thresh'],
                            nn_thresh=opt['nn_thresh'],
                            cuda=opt['cuda'])

    print('==> Successfully loaded pre-trained network.')


    """ This class helps merge consecutive point matches into tracks.
    """
    tracker = PointTracker(opt['max_length'], nn_thresh=fe.nn_thresh)


  # Create a window to display the demo.
    if not opt['no_display']:
        win = 'SuperPoint Tracker'
        cv2.namedWindow(win)
    else:
        print('Skipping visualization, will not show a GUI.')

    # Font parameters for visualizaton.
    font = cv2.FONT_HERSHEY_DUPLEX
    font_clr = (255, 255, 255)
    font_pt = (4, 12)
    font_sc = 0.4
  
    # Create output directory if desired.
    if opt['write']:
      print('==> Will write outputs to %s' % opt['write_dir'])
      if not os.path.exists(opt['write_dir']):
        os.makedirs(opt['write_dir'])
  
    print('==> Running Demo.')
    while True:
  
      start = time.time()
  
      # Get a new image.
      img, status = vs.next_frame()
      if status is False:
        break
  
      # Get points and descriptors.
      start1 = time.time()
      pts, desc, heatmap = fe.run(img)
      end1 = time.time()
  
      # Add points and descriptors to the tracker.
      tracker.update(pts, desc)
  
      # Get tracks for points which were match successfully across all frames.
      tracks = tracker.get_tracks(opt['min_length'])
  
      # Primary output - Show point tracks overlayed on top of input image.
      out1 = (np.dstack((img, img, img)) * 255.).astype('uint8')
      tracks[:, 1] /= float(fe.nn_thresh) # Normalize track scores to [0,1].
      tracker.draw_tracks(out1, tracks)
      if opt['show_extra']:
        cv2.putText(out1, 'Point Tracks', font_pt, font, font_sc, font_clr, lineType=16)
  
      # Extra output -- Show current point detections.
      out2 = (np.dstack((img, img, img)) * 255.).astype('uint8')
      for pt in pts.T:
        pt1 = (int(round(pt[0])), int(round(pt[1])))
        cv2.circle(out2, pt1, 1, (0, 255, 0), -1, lineType=16)
      cv2.putText(out2, 'Raw Point Detections', font_pt, font, font_sc, font_clr, lineType=16)
  
      # Extra output -- Show the point confidence heatmap.
      if heatmap is not None:
        min_conf = 0.001
        heatmap[heatmap < min_conf] = min_conf
        heatmap = -np.log(heatmap)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + .00001)
        out3 = myjet[np.round(np.clip(heatmap*10, 0, 9)).astype('int'), :]
        out3 = (out3*255).astype('uint8')
      else:
        out3 = np.zeros_like(out2)
      cv2.putText(out3, 'Raw Point Confidences', font_pt, font, font_sc, font_clr, lineType=16)
  
      # Resize final output.
      if opt['show_extra']:
        out = np.hstack((out1, out2, out3))
        out = cv2.resize(out, (3*opt['display_scale']*opt['W'], opt['display_scale']*opt['H']))
      else:
        out = cv2.resize(out1, (opt['display_scale']*opt['W'], opt['display_scale']*opt['H']))
  
      # Display visualization image to screen.
      if not opt['no_display']:
        cv2.imshow(win, out)
        key = cv2.waitKey(opt['waitkey']) & 0xFF
        if key == ord('q'):
          print('Quitting, \'q\' pressed.')
          break
  
      # Optionally write images to disk.
      if opt['write']:
        out_file = os.path.join(opt['write_dir'], 'frame_%05d.png' % vs.i)
        print('Writing image to %s' % out_file)
        cv2.imwrite(out_file, out)
  
      end = time.time()
      net_t = (1./ float(end1 - start))
      total_t = (1./ float(end - start))
      if opt['show_extra']:
        print('Processed image {} (net+post_process: %.2f FPS, total: %.2f FPS).'\
              % (vs.i, net_t, total_t))
  
    # Close any remaining windows.
    cv2.destroyAllWindows()
  
    print('==> Finshed Demo.')
  
  
#   %%
  