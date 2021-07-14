#!/opt/conda/bin/python 


""" (ref) http://www.gisdeveloper.co.kr/?p=6832
    (ref) https://www.geeksforgeeks.org/python-opencv-object-tracking-using-homography/
"""



#%% 
import sys 
import os 
import os.path as osp 

import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
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
queryPath = osp.join(cwd, 'data', 'query.jpg' )
galleryPath = osp.join(cwd, 'data', 'gallery.jpg' )



def find_kps(extractor, *imgs):
    
    Q_img, G_img = imgs 

    # find the keypoints and descriptors
    kp1, des1 = extractor.detectAndCompute(Q_img,None)
    kp2, des2 = extractor.detectAndCompute(G_img,None)


    return kp1, des1, kp2, des2





if __name__ == "__main__":

    Q_img = cv2.imread(queryPath, cv2.IMREAD_GRAYSCALE)
    G_img = cv2.imread(galleryPath, cv2.IMREAD_GRAYSCALE)


    """ Feature extractor using SIFT 
    """
    sift = cv2.SIFT_create()  # init. SIFT dectector   
    kp1, des1, kp2, des2 = find_kps(sift, *[Q_img, G_img])  # find the keypoints and descriptors 


    """ FLANN 
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
        if m.distance < 0.3 * n.distance :
            """ append the points according to distnace of descriptors 
            """
            good_points.append(m)


    """ Find Homography 
    """
    print(f"Good match points: {len(good_points)}")
    query_pts = np.float32([ kp1[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
    gallery_pts = np.float32([ kp2[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)


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

#    cv2.imshow("Homography", homography)
#    cv2.waitKey(0)

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matches_mask, # draw only inliers
                   flags = 2)

    img3 = cv2.drawMatches(Q_img,kp1, G_img,kp2, good_points, None, **draw_params)             

    cv2.imshow("matching", img3)
    cv2.waitKey(0)
    
    #plt.imshow(img3, 'gray'),plt.show()



    






    

