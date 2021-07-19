from pathlib import Path
import logging 
import os.path as osp
import os 
import sys 

import cv2
import numpy as np
#import matplotlib.cm as cm # (ref) https://matplotlib.org/stable/api/cm_api.html
import torch

from tools import (image2tensor, 
                    plot_keypoints, 
                    plot_matches, 
                    plot_homography)
from superpoint.superpoint import SuperPoint


""" Path checking 
"""
python_ver = sys.version
script_path = os.path.abspath(__file__)
cwd = Path(script_path).parent
os.chdir(cwd) #changing working directory 

print(f"Python version: {python_ver}")
print(f"The path of the running script: {script_path}")
print(f"CWD is changed to: {cwd}")



"""
(ref) 올바른 매칭점 찾기: https://bkshin.tistory.com/entry/OpenCV-29-%EC%98%AC%EB%B0%94%EB%A5%B8-%EB%A7%A4%EC%B9%AD%EC%A0%90-%EC%B0%BE%EA%B8%B0
(ref) Python-VO: https://github.com/Shiaoming/Python-VO
"""

#%% 
class SuperPointDetector(object):
    default_config = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": -1,
        "remove_borders": 4,
        "path": Path(__file__).parent / "superpoint/superpoint_v1.pth",
        "cuda": True
    }

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        logging.info("SuperPoint detector config: ")
        logging.info(self.config)

#        self.device = 'cuda' if torch.cuda.is_available() and self.config["cuda"] else 'cpu'
        self.device = 'cpu'

        logging.info("creating SuperPoint detector...")
        self.superpoint = SuperPoint(self.config).to(self.device)

    def __call__(self, image):
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        logging.debug("detecting keypoints with superpoint...")
        image_tensor = image2tensor(image, self.device)
        pred = self.superpoint({'image': image_tensor})

        ret_dict = {
            "image_size": np.array([image.shape[0], image.shape[1]]),
            "torch": pred,
            "keypoints": pred["keypoints"][0].cpu().detach().numpy(),
            "scores": pred["scores"][0].cpu().detach().numpy(),
            "descriptors": pred["descriptors"][0].cpu().detach().numpy().transpose()
        }

        return ret_dict





#%%
if __name__ == '__main__':

    """ SuperPoint for an feature extractor 
    """
    detector = SuperPointDetector()



    """ Image loader 
    """
    Q_img = cv2.imread(osp.join("data", "door_query.png"))
    G_img = cv2.imread(osp.join("data", "door_gallery.png"))    


    kptdescs = {}
    imgs = {}

    imgs['ref'] = Q_img
    kptdescs['ref'] = detector(Q_img)
    imgs['cur']  = G_img
    kptdescs['cur'] = detector(G_img)


    kp1, des1 = kptdescs['ref']['keypoints'], kptdescs['ref']['descriptors']
    kp2, des2 = kptdescs['cur']['keypoints'], kptdescs['cur']['descriptors']
    

  
    """ FLANN matching 
    """
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)  # feature matcher 


    """ store all good matches 
    """
    good_points = [] 
    for m,n in matches:
        if m.distance < 0.75 * n.distance :
            """ append the points according to distnace of descriptors 
            """
            good_points.append(m)


    """ Find Homography 
    """
    MIN_MATCH_COUNT = 10
    print(f"Good match points: {len(good_points)}")

    if len(good_points) > MIN_MATCH_COUNT:
        query_pts = np.float32([ kp1[m.queryIdx] for m in good_points]).reshape(-1, 1, 2)
        gallery_pts = np.float32([ kp2[m.trainIdx] for m in good_points]).reshape(-1, 1, 2)   

        Matrix , mask = cv2.findHomography(query_pts, gallery_pts, cv2.RANSAC, 5.0  )  # finding perspective transformation 
                                                                                  # between two planes 
        matches_mask = mask.ravel().tolist()         



        h, w = Q_img.shape[:2]    
    
        pts = np.float32(  [ [0, 0] , 
                             [0, h] , 
                             [w, h] , 
                             [w, 0] ,
                            ]).reshape(-1, 1, 2) # saving all points in pts 
        
        dst = cv2.perspectiveTransform(pts, Matrix) # applying perspective algorithm 
    
    
    
        """ See the output 
        """                
        homography = cv2.polylines(G_img, [np.int32(dst)], True, (255, 0, 0), 3) 
    
        cv2.imshow("Homography", homography)



        matching_img = plot_matches(Q_img, G_img,
                                    kp1, kp2, 
                                    )

        cv2.imshow("matching", matching_img )


        cv2.waitKey(0)
    

    else: 
        print("Not enough matches are found - %d/%d" %(len(good_points), MIN_MATCH_COUNT))
        matches_mask= None 

    


