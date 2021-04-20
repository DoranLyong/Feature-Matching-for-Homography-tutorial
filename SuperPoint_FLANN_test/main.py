from pathlib import Path
import logging 
import os.path as osp

import cv2
import numpy as np
#import matplotlib.cm as cm # (ref) https://matplotlib.org/stable/api/cm_api.html
import torch

from tools import image2tensor, plot_keypoints, plot_matches, plot_homography
from superpoint.superpoint import SuperPoint


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
class FrameByFrameMatcher(object):
    default_config = {
        "type": "FLANN",
        "KNN": {
            "HAMMING": True,  # For ORB Binary descriptor, only can use hamming matching
            "first_N": 300,  # For hamming matching, use first N min matches
        },
        "FLANN": {
            "kdTrees": 5,
            "searchChecks": 50
        },
        "distance_ratio": 0.75
    }

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        logging.info("Frame by frame matcher config: ")
        logging.info(self.config)

        if self.config["type"] == "KNN":
            logging.info("creating brutal force matcher...")
            if self.config["KNN"]["HAMMING"]:
                logging.info("brutal force with hamming norm.")
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            else:
                self.matcher = cv2.BFMatcher()
        elif self.config["type"] == "FLANN":
            logging.info("creating FLANN matcher...")
            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=self.config["FLANN"]["kdTrees"])
            search_params = dict(checks=self.config["FLANN"]["searchChecks"])  # or pass empty dictionary
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError(f"Unknown matcher type: {self.matcher_type}")

    def match(self, kptdescs):
        self.good = []
        # get shape of the descriptor
        self.descriptor_shape = kptdescs["ref"]["descriptors"].shape[1]

        if self.config["type"] == "KNN" and self.config["KNN"]["HAMMING"]:
            logging.debug("KNN keypoints matching...")
            matches = self.matcher.match(kptdescs["ref"]["descriptors"], kptdescs["cur"]["descriptors"])
            # Sort them in the order of their distance.
            matches = sorted(matches, key=lambda x: x.distance)
            # self.good = matches[:self.config["KNN"]["first_N"]]
            for i in range(self.config["KNN"]["first_N"]):
                self.good.append([matches[i]])
        else:
            logging.debug("FLANN keypoints matching...")
            matches = self.matcher.knnMatch(kptdescs["ref"]["descriptors"], kptdescs["cur"]["descriptors"], k=2)
            # Apply ratio test
            for m, n in matches:
                if m.distance < self.config["distance_ratio"] * n.distance:
                    self.good.append([m])
            # Sort them in the order of their distance.
            self.good = sorted(self.good, key=lambda x: x[0].distance)
        return self.good

    def get_good_keypoints(self, kptdescs):
        logging.debug("getting matched keypoints...")
        kp_ref = np.zeros([len(self.good), 2])
        kp_cur = np.zeros([len(self.good), 2])
        match_dist = np.zeros([len(self.good)])
        for i, m in enumerate(self.good):
            kp_ref[i, :] = kptdescs["ref"]["keypoints"][m[0].queryIdx]
            kp_cur[i, :] = kptdescs["cur"]["keypoints"][m[0].trainIdx]
            match_dist[i] = m[0].distance

        ret_dict = {
            "ref_keypoints": kp_ref,
            "cur_keypoints": kp_cur,
            "match_score": self.normalised_matching_scores(match_dist)
        }
        return ret_dict

    def __call__(self, kptdescs):
        self.match(kptdescs)
        return self.get_good_keypoints(kptdescs)

    def normalised_matching_scores(self, match_dist):

        if self.config["type"] == "KNN" and self.config["KNN"]["HAMMING"]:
            # ORB Hamming distance
            best, worst = 0, self.descriptor_shape * 8  # min and max hamming distance
            worst = worst / 4  # scale
        else:
            # for non-normalized descriptor
            if match_dist.max() > 1:
                best, worst = 0, self.descriptor_shape * 2  # estimated range
            else:
                best, worst = 0, 1

        # normalise the score!
        match_scores = match_dist / worst
        # range constraint
        match_scores[match_scores > 1] = 1
        match_scores[match_scores < 0] = 0
        # 1: for best match, 0: for worst match
        match_scores = 1 - match_scores

        return match_scores

    def draw_matched(self, img0, img1):
        pass





#%%
if __name__ == "__main__":


    """ Matcher 
    """
    matcher = FrameByFrameMatcher({"type": "FLANN"})

    """ SuperPoint 
    """
    detector = SuperPointDetector()

    """ Image loader 
    """
    img0 = cv2.imread(osp.join("data", "door_query.png"))
    img1 = cv2.imread(osp.join("data", "scene_RGB.png"))


    kptdescs = {}
    imgs = {}

    imgs['cur'] = img1
    kptdescs['cur'] = detector(img1)

    """ Get door area
    """
    H, W = img0 .shape[:2]
    up_end = 280, 0   # (x, y)-order
    down_end = W-150, H-100

    imgs['ref'] = img0[ up_end[1]:down_end[1], up_end[0]:down_end[0]]
#    imgs['ref'] = img0
    kptdescs['ref'] = detector(img0[ up_end[1]:down_end[1], up_end[0]:down_end[0]])

    matches = matcher(kptdescs)
    img = plot_matches(imgs['ref'], imgs['cur'],
                        matches['ref_keypoints'][0:200], matches['cur_keypoints'][0:200],
                        matches['match_score'][0:200], layout='lr')

    cv2.imshow("track", img)



    img_homography= plot_homography(imgs['ref'], imgs['cur'],
                                    kptdescs['ref']['keypoints'], kptdescs['cur']['keypoints'],
                                    kptdescs['ref']['descriptors'], kptdescs['cur']['descriptors']
                                    )

    cv2.imshow("FLANN with Homography", img_homography)

    cv2.waitKey(0)

    

#    kptdescs = detector(img0)
#    img0 = plot_keypoints(img0, kptdescs["keypoints"], kptdescs["scores"])
#    cv2.imshow("Query_SuperPoint", img0)
#    cv2.waitKey()
