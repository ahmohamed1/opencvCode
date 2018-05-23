import numpy as np
import cv2

# def kaze_match(im1_path, im2_path):
def kaze_match(img1, img2):
    # load the image and convert it to grayscale
    # im1 = cv2.imread(im1_path)
    # im2 = cv2.imread(im2_path)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


    # initialize the AKAZE descriptor, then detect keypoints and extract
    # local invariant descriptors from the image
    detector = cv2.AKAZE_create()
    (kp1, descs1) = detector.detectAndCompute(gray1, None)
    (kp2, descs2) = detector.detectAndCompute(gray2, None)

    print("keypoints: {}, descriptors: {}".format(len(kp1), descs1.shape))
    print("keypoints: {}, descriptors: {}".format(len(kp2), descs2.shape))

    # Match the features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descs1,descs2, k=2)    # typo fixed

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.9*n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 5

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
        print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None)
    cv2.imshow("AKAZE matching", img3)
    cv2.waitKey(10)
    return img3

def FLANNBasedMatcher(img1,img2):
    # Initiate SIFT detector
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray1,None)
    kp2, des2 = sift.detectAndCompute(gray2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in xrange(len(matches))]

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]

    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0)

    MIN_MATCH_COUNT = 5

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
        print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None)
    cv2.imshow("AKAZE matching", img3)
    cv2.waitKey(10)
    return img3
    return img3

def BruteForceMatchingwithSIFTDescriptorsandRatioTest(img1,img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray1,None)
    kp2, des2 = sift.detectAndCompute(gray2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2)
    cv2.imshow("AKAZE matching", img3)
    cv2.waitKey(3)
    return img3

right = "E:/PHD/005. Work prograse/011. vergence matching data/NewDataMultiTarget/" + str(3) + "right.jpg"
left = "E:/PHD/005. Work prograse/011. vergence matching data/NewDataMultiTarget/" + str(3) + "left.jpg"


cap = cv2.VideoCapture(1)

state = False
template = None
while(True):
    ret, frame = cap.read()

    if template is None and state:
        r = cv2.selectROI('Frame', frame)
        template = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        state = False

    if template is not None:
        frame = kaze_match(frame, template)
        # frame = FLANNBasedMatcher(frame, template)
        # frame = BruteForceMatchingwithSIFTDescriptorsandRatioTest(frame, template)

    cv2.imshow('Frame', frame)
    ikey = cv2.waitKey(10)
    if ikey == ord('q'):
        break
    elif ikey == ord('n'):
        template = None
        state = True
