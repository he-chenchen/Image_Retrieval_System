# Image retrieval system
import numpy as np
import cv2 as cv


# get bag_of_words (bow)
def getBOW(img_paths):
    bowList = []
    imageList = []
    _flag = 0
    for path in img_paths:
        img = cv.imread(path)
        imageList.append(img)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sift = cv.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        # img = cv.drawKeypoints(gray, kp, img)
        descriptors = np.array(des)
        bowList.append(descriptors)
        if _flag == 0:
            bow = descriptors
            _flag = _flag+1
        else:
            bow = np.vstack((bow, descriptors))
    return bow, bowList, imageList


def cluster(k, bagOfWords):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv.kmeans(bagOfWords, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    return center


def represent(clust_rep, k):
    vector = np.zeros(k)
    for idx in range(k):
        for num in clust_rep:
            if (num-1) == idx:
                vector[idx] = vector[idx]+1
    return vector


def transform(centers, k, bowList):
    numImage = len(bowList)
    img_rep = np.zeros([numImage, k])
    # transform image into vector representaion one by one
    for i in range(numImage):
        img = bowList[i]
        clustRep = np.zeros([img.shape[0]])
        for idx in range(img.shape[0]):
            distance = np.zeros(k)
            for c in range(k):
                distance[c] = np.linalg.norm(img[idx]-centers[c])
            clustRep[idx] = np.argmin(distance)
        img_rep[i] = represent(clustRep, k)
    return img_rep


def compare(target, imageSet):
    similarity = np.zeros(imageSet.shape[0])
    for idx in range(imageSet.shape[0]):
        similarity[idx] = np.linalg.norm(target-imageSet[idx])
    index = np.argmin(similarity)
    return index


def invert(img_rep, k):
    invmatrix = []
    for i in range(k):
        intvector = []
        for index in range(img_rep.shape[0]):
            if img_rep[index][i] != 0:
                intvector.append(index)
        vector = np.array(intvector)
        invmatrix.append(vector)
    return invmatrix


def candidIndex(target, invertMatrix):
    candidates = []
    for index in target:
        if invertMatrix[index] is not None:
            candidates.append(invertMatrix[index])
    return candidIndex


def findCandid(candidIndex, numImg):
    counts = np.zeros(numImg)
    for i in range(numImg):
        for lst in candidIndex:
            counts[i] = counts[i]+lst.count(i)
    index = [i for i in range(len(counts)) if counts[i] != 0]
    maxindex = np.argmax(counts)
    return index, maxindex


def nonZeroidx(array):
    return [index for index in range(len(array)) if array[index] != 0]
