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


parent_path = Path(__file__).parent #(ref)https://stackoverflow.com/questions/2817264/how-to-get-the-parent-dir-location
DataPath = osp.join(parent_path, '..','data')

DoorPath = osp.join(DataPath, 'query', 'door_RGB.png' )
ScenePath = osp.join(DataPath, 'scene', 'scene_RGB.png' )
ScenePath_video = osp.join(DataPath, 'scene', 'scene_video.mp4' )


""" Video capture params 
"""
fourcc = cv2.VideoWriter_fourcc(*'XVID')
record = False



""" Show the door area 
"""
def imshow(img_src: np.array, title:str, door_bbox: list = None):
    

    if door_bbox is not None:
        xmin, ymin, xmax, ymax = door_bbox
        cv2.rectangle(img_src,(xmin, ymin), (xmax, ymax), (255, 255, 0 ), thickness=2, lineType=cv2.LINE_8 )    
    cv2.imshow(title, img_src)


# ================================================================= #
#                      1. Feature Extractor - SIFT                  #
# ================================================================= #
# %%
sift = cv2.SIFT_create()  # init. SIFT dectector 

def FLANN_Homography(door_area:np.array, sample_img:np.array, factor:float):
    
    kp1, des1 = sift.detectAndCompute(door_area, None)
    kp2, des2 = sift.detectAndCompute(sample_img, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = [] 
    for m, n in matches:
        if m.distance < factor*n.distance:
            good.append(m)    

    """ Start Homography 
    """
    MIN_MATCH_COUNT = 3
    
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
#
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
#
        h, w = door_area.shape
        pts = np.float32([ [0,0], [0, h-1], [w-1, h-1], [w-1,0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
#
        sample_img = cv2.polylines(sample_img,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    else: 
        print("Not enough matches are found - %d/%d" %(len(good), MIN_MATCH_COUNT))
        matchesMask = None 
    draw_params = dict(matchColor = (0,255,0),  # draw matches in green color
                       singlePointColor = None, 
                       matchesMask = matchesMask, # draw only inliers 
                       flags = 2  )
    img3 = cv2.drawMatches(door_area, kp1, sample_img, kp2, good, None, **draw_params)
    cv2.imshow("FLANN with Homography", img3)
    return img3

# ================================================================= #
#                               2. Main                             #
# ================================================================= #
# %%

if __name__ == '__main__':
    
    door_query = cv2.imread(DoorPath, cv2.IMREAD_GRAYSCALE)
    scene_img = cv2.imread(ScenePath, cv2.IMREAD_GRAYSCALE)
    vs = cv2.VideoCapture(ScenePath_video)

    

    if door_query  is None: 
        print("Loading image failed...")
        sys.exit()

    
    H, W = door_query.shape[:2]
    print(H, W)

    """ Door area 
    """
    up_end = 200, 30   # (x, y)-order
    down_end = W-150, H-100

    door_bbox = [ up_end[0], up_end[1], down_end[0], down_end[1]]
    door_area = door_query[ up_end[1]:down_end[1], up_end[0]:down_end[0]]


    imshow(door_query, "door", door_bbox)
    imshow(door_area, "Query_img")
    cv2.imshow("Scene_img", scene_img )


    
    img3 = FLANN_Homography(door_area, scene_img, 0.95)

    
    cv2.imshow("homography_img", img3)



    """ Apply to video 
    """
    if (vs.isOpened() == False):
        print("opening video stream failed.")    
        sys.exit()
    
    while True: 
        vs_frame = vs.read()
        vs_frame = vs_frame[1]  # grab the next frame if we are reading from Videocapture 


        if ScenePath_video is not None and vs_frame is None:  # if we are viewing a video and we didn't grab a frame then we have reached the end of the video
            break      


        frame = cv2.cvtColor(vs_frame, cv2.COLOR_BGR2GRAY) 
        key = cv2.waitKey(1)

        homo_vs = FLANN_Homography(door_area=door_area, sample_img=frame, factor=0.95)

        

        if key == 27: # _Press ESC on keyboard to exit 
            break

        elif key == ord('v'): # press 'v'
            print("Recording start...")
            record = True 
            video = cv2.VideoWriter(f"./output/IR_right_full.avi",fourcc, 20.0,( homo_vs.shape[1], homo_vs.shape[0]))

        elif key == 32 : # press 'SPACE' 
            print("Recording stop...")
            record = False 
            video.release() 
        
        if record == True: 
            print("Video recording...")
            video.write(homo_vs)
        



    cv2.waitKey()
    cv2.destroyAllWindows()



# %%
