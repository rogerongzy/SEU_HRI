import cv2
import numpy as np

img1 = cv2.imread('1.jpg')
img2 = cv2.imread('2.jpg')
# img3 = cv2.imread('lx3.jpg')
# img4 = cv2.imread('lx4.jpg')
# img5 = cv2.imread('5.jpg')

img1 = cv2.resize(img1, (1680,630), cv2.INTER_AREA)
# up = np.hstack((img1, img2)) #水平拼接
# down = np.hstack((img3, img4)) #水平拼接
hes = np.vstack((img1,img2))

cv2.imshow('show', hes)
cv2.imwrite('repo5.jpg', hes)
cv2.waitKey(0)

