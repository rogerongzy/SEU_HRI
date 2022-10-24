import cv2
import numpy as np
import random
import os
import time
 
#加载样本
def loadImageList(dirName, fileListPath):
    imageList = []
    file = open(dirName + r'/' + fileListPath)
    imageName = file.readline()
    while imageName != '':
        imageName = dirName + r'/' + imageName.split('/', 1)[1].strip('\n')
		#print imageName
        imageList.append(cv2.imread(imageName)) #将读取的照片塞到imageList里
        imageName = file.readline()
    return imageList
 
#获取正样本，从(16,16)截取大小为(128,64)的区域
def getPosSample(imageList):
    posList = []
    for i in range(len(imageList)):
        roi = imageList[i][16:16+128, 16:16+64] ######为什么从(16,16)开始？######
        posList.append(roi)
    return posList
 
#获取负样本，从没有行人的图片中，随机裁剪出10张大小为(128,64)的区域
def getNegSample(imageList):
	negList = []
	random.seed(1)
	for i in range(len(imageList)):
		for j in range(10): # 10 times
			y = int(random.random() * (len(imageList[i]) - 128))
			x = int(random.random() * (len(imageList[i][0]) - 64))
			negList.append(imageList[i][y:y + 128, x:x + 64])
	return negList


##### selfmade #####
#自己加文件夹路径
def mySample(imgdir):
    imgList = []
    for file in os.listdir(imgdir):
        path = imgdir + '/' + file
        imgList.append(cv2.imread(path))
        # print(file)
    return imgList

##### selfmade #####


#计算HOG特征
def getHOGList(imageList):
	HOGList = []
	hog = cv2.HOGDescriptor()
	for i in range(len(imageList)):
		gray = cv2.cvtColor(imageList[i], cv2.COLOR_BGR2GRAY)
		HOGList.append(hog.compute(gray))
	return HOGList
 
#获取检测子
def getHOGDetector(svm):
	sv = svm.getSupportVectors()
	rho, _, _ = svm.getDecisionFunction(0)
	sv = np.transpose(sv)
	return np.append(sv, [[-rho]], 0)
 
#获取Hard example， 使用训练好的模型对负样本进行测试，如果有识别出来的结果，将这些结果resize为(64.128)后加入到负样本中重新训练
def getHardExamples(negImageList, svm):
    hardNegList = []
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(getHOGDetector(svm))
    for i in range(len(negImageList)):
        rects, wei = hog.detectMultiScale(negImageList[i], winStride=(4, 4),padding=(8, 8), scale=1.05)
        for (x,y,w,h) in rects:
            hardExample = negImageList[i][y:y+h, x:x+w]
            hardNegList.append(cv2.resize(hardExample,(64,128)))
    return hardNegList
 
#非极大值抑制
def fastNonMaxSuppression(boxes, sc, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    # initialize the list of picked indexes
    pick = []
 
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = sc
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the score of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)
 
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
 
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick]


def trainsvm():

    labels = []
    posImageList = []
    posList = []
    posImageList = []
    posList = []
    hosList = []
    tem = []
    hardNegList = []
    #加载含行人的图片
    # posImageList = loadImageList(r"./INRIAPerson/train_64x128_H96", "pos.lst")
    # print ("posImageList:", len(posImageList))
    #剪裁图片，获取正样本
    # posList = getPosSample(posImageList)
    posList = mySample('./ppl_pos')
    print ("posList", len(posList))
    #获取正样本的HOG
    hosList = getHOGList(posList)
    print ("hosList", len(hosList))
    #添加所有正样本对应的label
    [labels.append(+1) for _ in range(len(posList))]
    
    #加载不含行人的图片
    negImageList = loadImageList(r"./INRIAPerson/train_64x128_H96", "neg.lst")
    print ("negImageList:", len(negImageList))
    #随机裁剪获取负样本
    negList = getNegSample(negImageList)
    # negList = mySample('./ppl_neg')
    print ("negList", len(negList))
    #获取负样本HOG，并添加到整体HOG特征list中
    hosList.extend(getHOGList(negList))
    print ("hosList", len(hosList))
    #添加所有负样本对应的label
    [labels.append(-1) for _ in range(len(negList))]
    print ("labels", len(labels))
 
    # parameter
    #-d degree：核函数中的degree设置(针对多项式核函数)(默认3)
    #-g r(gama)：核函数中的gamma函数设置(针对多项式/rbf/sigmoid核函数)(默认1/ k)
    #-r coef0：核函数中的coef0设置(针对多项式/sigmoid核函数)((默认0)
    #-c cost：设置C-SVC，e -SVR和v-SVR的参数(损失函数)(默认1)
    #-n nu：设置v-SVC，一类SVM和v- SVR的参数(默认0.5)
    #-p p：设置e -SVR 中损失函数p的值(默认0.1)
    #-m cachesize：设置cache内存大小，以MB为单位(默认40)
    #-e eps：设置允许的终止判据(默认0.001)
    #-h shrinking：是否使用启发式，0或1(默认1)
    #-wi weight：设置第几类的参数C为weight*C(C-SVC中的C)(默认1)
    #-v n: n-fold交互检验模式，n为fold的个数，必须大于等于2
 
    svm = cv2.ml.SVM_create()
    svm.setCoef0(0.0)
    svm.setDegree(3)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)#终止条件
    svm.setTermCriteria(criteria)
    svm.setGamma(0)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setNu(0.5)
    svm.setP(0.1)  # for EPSILON_SVR, epsilon in loss function?
    svm.setC(0.01)  # From paper, soft classifier 软间隔
    svm.setType(cv2.ml.SVM_EPS_SVR)  # C_SVC # EPSILON_SVR # may be also NU_SVR # do regression task
    print('====Start training for the first time.====')
    
    print(time.time())
    svm.train(np.array(hosList), cv2.ml.ROW_SAMPLE, np.array(labels))
    print(time.time())

    #根据初始训练结果获取hard example
    hardNegList = getHardExamples(negImageList, svm) #裁剪前的负样本
    # hardNegList = getHardExamples(negList, svm) ######直接使用裁剪后的负样本是否会有问题？######
    hosList.extend(getHOGList(hardNegList))
    print ("hosList: ", len(hosList))
    [labels.append(-1) for _ in range(len(hardNegList))]
    
    #添加hard example后，重新训练
    print('====Start retraining.====')
    print(time.time())
    svm.train(np.array(hosList), cv2.ml.ROW_SAMPLE, np.array(labels))
    print(time.time())
    
    #保存模型
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(getHOGDetector(svm))
    hog.save('myHogDector.bin')
    print('==========Model saved!!!==========')
 
# 1 使用切割好的正负样本(ppl_pos/neg)，再训练直接使用的切割好的负样本；有较多的误识别
# 2 使用切割好的正样本，负样本为随机切割，再训练使用负样本源图；误识别效果有改善，仍有部分无法识别

def humandetect(imgpath):
    #行人检测
    hog = cv2.HOGDescriptor()
    # hog.load('myHogDector1.bin') # my
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) # default
    image = cv2.imread(imgpath)
    rects, scores = hog.detectMultiScale(image, winStride=(4, 4),padding=(8, 8), scale=1.05)

    for i in range(len(rects)):
        r = rects[i]
        rects[i][2] = r[0] + r[2]
        rects[i][3] = r[1] + r[3]

    sc = [score[0] for score in scores]
    sc = np.array(sc)
    
    pick = []
    # print('rects_len',len(rects))
    pick = fastNonMaxSuppression(rects, sc, overlapThresh = 0.3)
    # print('pick_len = ',len(pick))
    
    for (x, y, xx, yy) in pick:
    	print (x, y, xx, yy)
    	cv2.rectangle(image, (int(x), int(y)), (int(xx), int(yy)), (0, 255, 0), 2)
    	# cv2.imshow('eachrec', image)
    	# cv2.waitKey(0)
    
    cv2.imshow('detect_result', image)
    cv2.waitKey(0)
    # return image

def humandetect_multi(imgdir):
    for file in os.listdir(imgdir):
        # if file.endswith('.jpg'):
        humandetect(imgdir + '/' + file)
        # img_dst = humandetect(imgdir + '/' + file)
        # cv2.imwrite('./dst/default/' + file, img_dst)

def humandetect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detected = humandetect(frame)
        cv2.imshow("capture", detected)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



def detect_alert(imgpath, imgdpath):
    hog = cv2.HOGDescriptor()
    hog.load('myHogDector2.bin')

    img = cv2.imread(imgpath)
    imgd = cv2.imread(imgdpath, cv2.IMREAD_UNCHANGED)

    rects, scores = hog.detectMultiScale(img, winStride=(4, 4),padding=(8, 8), scale=1.05)

    for i in range(len(rects)):
        r = rects[i]
        rects[i][2] = r[0] + r[2]
        rects[i][3] = r[1] + r[3]

    sc = [score[0] for score in scores]
    sc = np.array(sc)
    
    pick = []
    pick = fastNonMaxSuppression(rects, sc, overlapThresh = 0.3)
    for (x, y, xx, yy) in pick:
        xc = int((x + xx) / 2)
        yc = int((y + yy) / 2)
        # print(x, y, xx, yy)
        print(imgd[xc][yc])
        cv2.rectangle(img, (int(x), int(y)), (int(xx), int(yy)), (0, 255, 0), 2)
        if(imgd[xc][yc]<3000):
            print('ALERT! At the distance of ' + str(imgd[xc][yc]) + ' there are people!')

    cv2.imshow('detect_result', img)
    cv2.waitKey(0)



def main():
    # trainsvm()
    # humandetect_multi('./ppl_test1')
    detect_alert('demo1.jpg','demo2.pfm')

if __name__ == '__main__':
    main()