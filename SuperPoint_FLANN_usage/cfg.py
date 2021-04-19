opt = dict()


opt['input']: str = 'scene_video.mp4'   #'Image directory or movie file or "camera" (for webcam).'
opt['weights_path'] : str = 'superpoint_v1.pth'  # 'Path to pretrained weights file (default: superpoint_v1.pth).'
opt['img_glob'] : str = '*.png' # 'Glob match if directory of images is specified (default: \'*.png\').')
opt['skip'] : int = 1 # 'Images to skip if input is movie or directory (default: 1).'
opt['show_extra'] = False # 'Show extra debug outputs (default: False).'
opt['H'] : int = 120  # 'Input image height (default: 120).'
opt['W'] : int = 160  # 'Input image width (default:160).'

opt['display_scale'] : int = 2   # 'Factor to scale output visualization (default: 2).'
opt['min_length'] : int = 2   # 'Minimum length of point tracks (default: 2).'
opt['max_length'] : int = 5   # 'Maximum length of point tracks (default: 5).'
opt['nms_dist'] : int = 4     # 'Non Maximum Suppression (NMS) distance (default: 4).'
opt['conf_thresh'] : float = 0.015  # 'Detector confidence threshold (default: 0.015).'
opt['nn_thresh'] : float = 0.7  # 'Descriptor matching threshold (default: 0.7).'
opt['camid'] : int = 0   # 'OpenCV webcam video capture ID, usually 0 or 1 (default: 0).'

opt['waitkey'] : int = 1 # 'OpenCV waitkey time in ms (default: 1).'
opt['cuda'] = False   # 'Use cuda GPU to speed up network processing speed (default: False)'
opt['no_display'] = False # 'Do not display images to screen. Useful if running remotely (default: False).'
opt['write'] = False # 'Save output frames to a directory (default: False)'
opt['write_dir'] : str = 'tracker_outputs/'   # 'Directory where to write output frames (default: tracker_outputs/).'

