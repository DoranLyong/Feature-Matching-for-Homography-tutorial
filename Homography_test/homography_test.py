#!/opt/conda/bin/python 

#%% 
import sys 
import os 
import os.path as osp 

import numpy as np 
import cv2 
from colorama import Back, Style # assign color options on your text(ref) https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal


#%%
""" Path checking 
"""
python_ver = sys.version
script_path = osp.abspath(__file__)
cwd = os.getcwd()
os.chdir(cwd) #changing working directory 

print(f"Script Path: {Back.GREEN}{script_path}{Style.RESET_ALL}")
print(f"CWD : {Back.MAGENTA}{cwd}{Style.RESET_ALL}")

#%%
queryPath = osp.join(cwd, 'data', 'query.png' )
galleryPath = osp.join(cwd, 'data', 'gallery.png' )



def find_kps(extractor, *imgs):
    
    Q_img, G_img = imgs 

    # find the keypoints and descriptors
    kp1, des1 = extractor.detectAndCompute(Q_img,None)
    kp2, des2 = extractor.detectAndCompute(G_img,None)


    return kp1, kp2, des1, des2





if __name__ == "__main__":

    Q_img = cv2.imread(queryPath, cv2.IMREAD_GRAYSCALE)
    G_img = cv2.imread(galleryPath, cv2.IMREAD_GRAYSCALE)


    """ Feature extractor using SIFT 
    """
    sift = cv2.xfeatures2d.SIFT_create()  
    kp1, kp2, des1, des2 = find_kps(sift, *[Q_img, G_img])


    """ FLANN 
    """
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)



    

