import numpy as np
import cv2
import time
import os
from matplotlib import pyplot as plt

def FlannBasedMatcher():

    imgname1 = 'img_1.jpg'
    imgname2 = 'img_2.jpg'

    sift = cv2.SIFT_create()

    # FLANN params initialization
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)

    # read and preprocess
    img1 = cv2.imread(imgname1) # read image
    img2 = cv2.imread(imgname2)

    if img1.shape[0] > img1.shape[1]:
        img1 = cv2.resize(img1, (720,960), cv2.INTER_AREA) # resize, warn that opposite
    else:
        img1 = cv2.resize(img1, (960,720), cv2.INTER_AREA)
    if img2.shape[0] > img2.shape[1]:
        img2 = cv2.resize(img2, (720,960), cv2.INTER_AREA)
        img3 = cv2.resize(img2, (720,960), cv2.INTER_AREA)
    else:
        img2 = cv2.resize(img2, (960,720), cv2.INTER_AREA)
        img3 = cv2.resize(img2, (960,720), cv2.INTER_AREA)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # turn to gray
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    print(len(kp1))
    print(len(kp2))

    img1 = cv2.drawKeypoints(img1,kp1,img1,color=(255,0,255)) # drawing kp on source image
    img2 = cv2.drawKeypoints(img2,kp2,img2,color=(255,0,255))

    start = time.time() # start timing
    print(start)
    print(time.localtime( time.time() ))
    print(time.asctime( time.localtime(time.time()) ))


    # Flann knnMatcher
    matches = flann.knnMatch(des1, des2, k=2) # match within des1 & des2
    matchesMask = [[0,0] for i in range(len(matches))]

    good = [] # if matching ratio larger than threshhold, append to good[]
    goodx = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance: # usually 0.5 or 0.75
            good.append([m]) # use in drawing
            goodx.append(m) # use in findHomography

    # img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=2) # before 
    # img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2) # after 
    # cv2.imshow("FLANN", img5)
    
    end = time.time() # end timing
    print(end)
    print(time.localtime( time.time() ))
    print(time.asctime( time.localtime(time.time()) ))

    # cv2.waitKey(0)

    # print(len(goodx))

    if len(goodx) > 4:
        ptsA = np.float32([kp1[m.queryIdx].pt for m in goodx]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in goodx]).reshape(-1, 1, 2)
        ransacReprojThreshold = 4
        H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold) # H: transform matrix

        # draw rectangle
        pts = np.float32([ [0,0],[0,img1.shape[0]-1],[img1.shape[1]-1,img1.shape[0]-1],[img1.shape[1]-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, H)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,(0,255,0),2,cv2.LINE_AA)
        # cv2.imshow("img2", img2)

        img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
        cv2.imwrite('FLANN_sift_withrec_r=0.75.jpg', img5)
        # cv2.imwrite('img2_rec.jpg', img2)
        # cv2.waitKey(0)
 
        # img6 = cv2.warpPerspective(img3, H, (img1.shape[1],img1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP) # change perspective
        # cv2.imshow("find", img6)
        # cv2.imwrite('trans_persp.jpg', img6)
        # cv2.waitKey(0)
    
    

def BFmatcher():

    imgname1 = 'img_1.jpg'
    imgname2 = 'img_2.jpg'

    sift = cv2.SIFT_create()

    img1 = cv2.imread(imgname1)
    img2 = cv2.imread(imgname2)

    if img1.shape[0] > img1.shape[1]:
        img1 = cv2.resize(img1, (720,960), cv2.INTER_AREA) # resize, warn that opposite
    else:
        img1 = cv2.resize(img1, (960,720), cv2.INTER_AREA)
    if img2.shape[0] > img2.shape[1]:
        img2 = cv2.resize(img2, (720,960), cv2.INTER_AREA)
        img3 = cv2.resize(img2, (720,960), cv2.INTER_AREA)
    else:
        img2 = cv2.resize(img2, (960,720), cv2.INTER_AREA)
        img3 = cv2.resize(img2, (960,720), cv2.INTER_AREA) 

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(img1, None)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(img2, None)

    print(len(kp1))
    print(len(kp2))

    img3 = cv2.drawKeypoints(img1,kp1,img1,color=(255,0,255))
    img4 = cv2.drawKeypoints(img2,kp2,img2,color=(255,0,255))

    start = time.time() # start timing
    print(start)
    print(time.localtime( time.time() ))
    print(time.asctime( time.localtime(time.time()) ))

    # BFMatcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    good = []
    goodx = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
            goodx.append(m)

    end = time.time() # end timing
    print(end)
    print(time.localtime( time.time() ))
    print(time.asctime( time.localtime(time.time()) ))

    # img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=2)
    # img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
    
    # cv2.imshow("BFmatch", img5)
    # cv2.imwrite('BFmatcher_sift_r=0.3.jpg',img5)

    if len(goodx) > 4:
        ptsA = np.float32([kp1[m.queryIdx].pt for m in goodx]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in goodx]).reshape(-1, 1, 2)
        ransacReprojThreshold = 4
        H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold) # H: transform matrix

        # draw rectangle
        pts = np.float32([ [0,0],[0,img1.shape[0]-1],[img1.shape[1]-1,img1.shape[0]-1],[img1.shape[1]-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, H)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,(0,255,0),2,cv2.LINE_AA)
        # cv2.imshow("img2", img2)

        img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
        cv2.imwrite('BF_sift_withrec_r=0.75.jpg', img5)
        # cv2.imwrite('img2_rec.jpg', img2)
        # cv2.waitKey(0)
 
        # img6 = cv2.warpPerspective(img3, H, (img1.shape[1],img1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP) # change perspective
        # cv2.imshow("find", img6)
        # cv2.imwrite('trans_persp.jpg', img6)
        # cv2.waitKey(0)


def BFmatcher_SURF():

    imgname1 = 'img_1.jpg'
    imgname2 = 'img_2.jpg'

    surf = cv2.SURF_create(400) # do not exist
    # surf = cv.xfeatures2d.SURF_create(400)

    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks=50)
    # flann = cv2.FlannBasedMatcher(index_params,search_params)

    img1 = cv2.imread(imgname1)
    img2 = cv2.imread(imgname2)

    if img1.shape[0] > img1.shape[1]:
        img1 = cv2.resize(img1, (720,960), cv2.INTER_AREA) # resize, warn that opposite
    else:
        img1 = cv2.resize(img1, (960,720), cv2.INTER_AREA)
    if img2.shape[0] > img2.shape[1]:
        img2 = cv2.resize(img2, (720,960), cv2.INTER_AREA)
        img3 = cv2.resize(img2, (720,960), cv2.INTER_AREA)
    else:
        img2 = cv2.resize(img2, (960,720), cv2.INTER_AREA)
        img3 = cv2.resize(img2, (960,720), cv2.INTER_AREA) 

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    kp1, des1 = surf.detectAndCompute(img1, None)

    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp2, des2 = surf.detectAndCompute(img2, None)

    print(len(kp1))
    print(len(kp2))

    img1 = cv2.drawKeypoints(img1,kp1,img1,color=(255,0,255))
    img2 = cv2.drawKeypoints(img2,kp2,img2,color=(255,0,255))

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    good = []
    goodx = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
            goodx.append(m)
    
    # img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=2)
    # img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

    # cv2.imshow("SURF", img5)
    # cv2.imwrite('out.jpg',img5)

    if len(goodx) > 4:
        ptsA = np.float32([kp1[m.queryIdx].pt for m in goodx]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in goodx]).reshape(-1, 1, 2)
        ransacReprojThreshold = 4
        H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold) # H: transform matrix

        # draw rectangle
        pts = np.float32([ [0,0],[0,img1.shape[0]-1],[img1.shape[1]-1,img1.shape[0]-1],[img1.shape[1]-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, H)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,(0,255,0),2,cv2.LINE_AA)
        # cv2.imshow("img2", img2)

        img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
        cv2.imwrite('BF_surf_withrec_r=0.75.jpg', img5)
        # cv2.imwrite('img2_rec.jpg', img2)
        # cv2.waitKey(0)
 
        # img6 = cv2.warpPerspective(img3, H, (img1.shape[1],img1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP) # change perspective
        # cv2.imshow("find", img6)
        # cv2.imwrite('trans_persp.jpg', img6)
        # cv2.waitKey(0)


def BFmatcher_ORB():

    imgname1 = 'img_1.jpg'
    imgname2 = 'img_2.jpg'

    orb = cv2.ORB_create()

    img1 = cv2.imread(imgname1)
    img2 = cv2.imread(imgname2)

    if img1.shape[0] > img1.shape[1]:
        img1 = cv2.resize(img1, (720,960), cv2.INTER_AREA) # resize, warn that opposite
    else:
        img1 = cv2.resize(img1, (960,720), cv2.INTER_AREA)
    if img2.shape[0] > img2.shape[1]:
        img2 = cv2.resize(img2, (720,960), cv2.INTER_AREA)
        img3 = cv2.resize(img2, (720,960), cv2.INTER_AREA)
    else:
        img2 = cv2.resize(img2, (960,720), cv2.INTER_AREA)
        img3 = cv2.resize(img2, (960,720), cv2.INTER_AREA) 

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(img1,None)

    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp2, des2 = orb.detectAndCompute(img2,None)

    img1 = cv2.drawKeypoints(img1,kp1,img1,color=(255,0,255))
    img2 = cv2.drawKeypoints(img2,kp2,img2,color=(255,0,255))

    # BFMatcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    
    good = []
    goodx = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
            goodx.append(m)

    # img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=2)
    # img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

    # cv2.imshow("ORB", img5)
    # cv2.imwrite('BFmatcher_orb.jpg',img5)

    if len(goodx) > 4:
        ptsA = np.float32([kp1[m.queryIdx].pt for m in goodx]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in goodx]).reshape(-1, 1, 2)
        ransacReprojThreshold = 4
        H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold) # H: transform matrix

        # draw rectangle
        pts = np.float32([ [0,0],[0,img1.shape[0]-1],[img1.shape[1]-1,img1.shape[0]-1],[img1.shape[1]-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, H)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,(0,255,0),2,cv2.LINE_AA)
        # cv2.imshow("img2", img2)

        img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
        cv2.imwrite('BF_orb_withrec_r=0.75.jpg', img5)
        # cv2.imwrite('img2_rec.jpg', img2)
        # cv2.waitKey(0)
 
        # img6 = cv2.warpPerspective(img3, H, (img1.shape[1],img1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP) # change perspective
        # cv2.imshow("find", img6)
        # cv2.imwrite('trans_persp.jpg', img6)
        # cv2.waitKey(0)

def findscene():
    img1 = cv2.imread('input.jpg') # input image
    
    best_match_name = ''
    best_match_size = 0
    
    for file in os.listdir(): 
        print(file)
        if file.endswith('.jpg') and file != 'input.jpg':
            sift = cv2.SIFT_create()

            # FLANN params initialization
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params,search_params)

            img2 = cv2.imread(file)

            if img1.shape[0] > img1.shape[1]:
                img1 = cv2.resize(img1, (720,960), cv2.INTER_AREA) # resize, warn that opposite
            else:
                img1 = cv2.resize(img1, (960,720), cv2.INTER_AREA)
            if img2.shape[0] > img2.shape[1]:
                img2 = cv2.resize(img2, (720,960), cv2.INTER_AREA)
                # img3 = cv2.resize(img2, (720,960), cv2.INTER_AREA)
            else:
                img2 = cv2.resize(img2, (960,720), cv2.INTER_AREA)
                # img3 = cv2.resize(img2, (960,720), cv2.INTER_AREA)

            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # turn to gray
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            kp1, des1 = sift.detectAndCompute(img1, None) # kp & des
            kp2, des2 = sift.detectAndCompute(img2, None)

            img1 = cv2.drawKeypoints(img1,kp1,img1,color=(255,0,255)) # drawing kp on source image
            img2 = cv2.drawKeypoints(img2,kp2,img2,color=(255,0,255))

            # Flann knnMatcher
            matches = flann.knnMatch(des1, des2, k=2) # match within des1 & des2
            matchesMask = [[0,0] for i in range(len(matches))]
        
            good = [] # if matching ratio larger than threshhold, append to good[]
            # goodx = []
            for m,n in matches:
                if m.distance < 0.5 * n.distance: # usually 0.5 or 0.75
                    good.append([m]) # use in drawing
        
            print(len(good))
            print(best_match_size)
            if len(good) > best_match_size:
                best_match_size = len(good)
                best_match_name = file
    
    print(best_match_name)
    img_out = cv2.imread(best_match_name)
    cv2.imshow('output',img_out)
    cv2.waitKey(0)



def main():
    # FlannBasedMatcher()
    # BFmatcher()
    # BFmatcher_SURF()
    # BFmatcher_ORB()
    findscene()


if __name__ == '__main__':
    main()