import cv2
import numpy as np
import torch
import collections
import matplotlib.cm as cm

"""
(ref) 올바른 매칭점 찾기: https://bkshin.tistory.com/entry/OpenCV-29-%EC%98%AC%EB%B0%94%EB%A5%B8-%EB%A7%A4%EC%B9%AD%EC%A0%90-%EC%B0%BE%EA%B8%B0
(ref) Python-VO: https://github.com/Shiaoming/Python-VO

"""




def image2tensor(frame, device):
    return torch.from_numpy(frame / 255.).float()[None, None].to(device)


# --- VISUALIZATION ---
# based on: https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/utils.py
def plot_keypoints(image, kpts, scores=None):
    kpts = np.round(kpts).astype(int)

    if scores is not None:
        # get color
        smin, smax = scores.min(), scores.max()
        assert (0 <= smin <= 1 and 0 <= smax <= 1)

        color = cm.gist_rainbow(scores * 0.4)
        color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
        # text = f"min score: {smin}, max score: {smax}"

        for (x, y), c in zip(kpts, color):
            c = (int(c[0]), int(c[1]), int(c[2]))
            cv2.drawMarker(image, (x, y), tuple(c), cv2.MARKER_CROSS, 6)

    else:
        for x, y in kpts:
            cv2.drawMarker(image, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 6)

    return image


# based on: https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/utils.py
def plot_matches(image0, image1, kpts0, kpts1, scores=None, layout="lr"):
    """
    plot matches between two images. If score is nor None, then red: bad match, green: good match
    :param image0: reference image
    :param image1: current image
    :param kpts0: keypoints in reference image
    :param kpts1: keypoints in current image
    :param scores: matching score for each keypoint pair, range [0~1], 0: worst match, 1: best match
    :param layout: 'lr': left right; 'ud': up down
    :return:
    """
    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]

    if layout == "lr":
        H, W = max(H0, H1), W0 + W1
        out = 255 * np.ones((H, W, 3), np.uint8)
        out[:H0, :W0, :] = image0
        out[:H1, W0:, :] = image1
    elif layout == "ud":
        H, W = H0 + H1, max(W0, W1)
        out = 255 * np.ones((H, W, 3), np.uint8)
        out[:H0, :W0, :] = image0
        out[H0:, :W1, :] = image1
    else:
        raise ValueError("The layout must be 'lr' or 'ud'!")

    kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)

    # get color
    if scores is not None:
        smin, smax = scores.min(), scores.max()
        assert (0 <= smin <= 1 and 0 <= smax <= 1)

        color = cm.gist_rainbow(scores * 0.4)
        color = (np.array(color[:, :3]) * 255).astype(int)[:, ::-1]
    else:
        color = np.zeros((kpts0.shape[0], 3), dtype=int)
        color[:, 1] = 255

    for (x0, y0), (x1, y1), c in zip(kpts0, kpts1, color):
        c = c.tolist()
        if layout == "lr":
            cv2.line(out, (x0, y0), (x1 + W0, y1), color=c, thickness=1, lineType=cv2.LINE_AA)
            # display line end-points as circles
            cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1 + W0, y1), 2, c, -1, lineType=cv2.LINE_AA)
        elif layout == "ud":
            cv2.line(out, (x0, y0), (x1, y1 + H0), color=c, thickness=1, lineType=cv2.LINE_AA)
            # display line end-points as circles
            cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x1, y1 + H0), 2, c, -1, lineType=cv2.LINE_AA)

    return out



def plot_homography(query_area:np.array, current_scene:np.array, kpts0, kpts1, des0, des1):

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    

    """ Matching
    """
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des0, des1, k=2)

    good = [] 
    for m, n in matches:
        if m.distance < 0.95*n.distance:
            good.append(m)   

    print(f"good matches: {len(good)}/{len(matches)}")

    """ Start Homography
    """
    MIN_MATCH_COUNT = 10

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kpts0[m.queryIdx] for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([ kpts1[m.queryIdx] for m in good]).reshape(-1, 1, 2)


        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 6.0)
        
        matchesMask = mask.ravel().tolist()


        h, w = query_area.shape[0:2]
        pts = np.float32([ [0,0], [0, h-1], [w-1, h-1], [w-1,0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        current_scene = cv2.polylines(current_scene,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else: 
        print("Not enough matches are found - %d/%d" %(len(good), MIN_MATCH_COUNT))
        matchesMask = None 

    draw_params = dict(matchColor = (0,255,0),  # draw matches in green color
                       singlePointColor = None, 
                       matchesMask = matchesMask, # draw only inliers 
                       flags = 2  )

#    img3 = cv2.drawMatches(query_area, kp1, current_scene, kp2, good, None, **draw_params)
    
    return current_scene