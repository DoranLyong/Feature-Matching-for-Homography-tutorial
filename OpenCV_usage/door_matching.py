# -*- coding: utf-8 -*-
"""
(ref) Feature Matching + Homography ; https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html

"""


#%%
import sys 
import os 
import os.path as osp 
from pathlib import Path 

import cv2 
import numpy as np 


parent_path = Path(__file__).parent.parent #(ref)https://stackoverflow.com/questions/2817264/how-to-get-the-parent-dir-location
DataPath = osp.join(parent_path, 'data')

DoorPath = osp.join(DataPath, 'query', 'door_RGB.png' )
ScenePath = osp.join(DataPath, 'scene', 'scene_RGB.png' )



# ================================================================= #
#                      1. Feature Extractor - SIFT                  #
# ================================================================= #
# %%
sift = cv2.SIFT_create()  # init. SIFT dectector 


def feature_matching(query, scene):

    """ find the keypoints and descriptors with SIFT 
    """
    kp1, des1 = sift.detectAndCompute(query, None)
    kp2, des2 = sift.detectAndCompute(scene, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    """store all the good matches as per Lowe's ratio test.
    """
    good = [] 
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)


    """ Homography 
    """
    MIN_MATCH_COUNT = 3
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        
        h, w = query.shape[:2]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        scene = cv2.polylines(scene,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0),  # draw matches in green color
                       singlePointColor = None, 
                       matchesMask = matchesMask, # draw only inliers 
                       flags = 2  )
    img3 = cv2.drawMatches(query, kp1, scene, kp2, good, None, **draw_params)

    return img3

# ================================================================= #
#                               2. Main                             #
# ================================================================= #
# %%

if __name__ == '__main__':
    
    door_query = cv2.imread(DoorPath, cv2.IMREAD_GRAYSCALE)
    scene_img = cv2.imread(ScenePath, cv2.IMREAD_GRAYSCALE)


    if door_query  is None: 
        print("Loading image failed...")
        sys.exit()

    
    H, W = door_query.shape[:2]
    print(H, W)

    """ Door area 
    """
    up_end = 200, 30   # (x, y)-order
    down_end = W-150, H-100

    door_area = door_query[ up_end[1]:down_end[1], up_end[0]:down_end[0]]


    img3 = feature_matching(door_area, scene_img)

    
    cv2.imshow("Query", door_area)
    cv2.imshow("Scene", scene_img )
    cv2.imshow("homography", img3)
    cv2.waitKey()
    cv2.destroyAllWindows()



# %%
