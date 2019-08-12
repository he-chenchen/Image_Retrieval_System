import imageRetrieval as IR
import numpy as np
import matplotlib.pyplot as plt
import glob
print('hello python!')

# ----------------------Representing trainig images-----------------
# set the path of images
image_path = '*.jpg'
img_paths = glob.glob(image_path)

# print(img_paths)
# get_bag_of_words
bow, bowList, imageList = IR.getBOW(img_paths)

# clustering the bag_of_words with k centroids
k = 5
centers = IR.cluster(k, bow)


# transform the representation of images into vectors
img_rep = IR.transform(centers, k, bowList)
print('img_rep.shape: ', img_rep.shape)


# ----------------------Representing test image-----------------
testImage_path = '115602.jpg'
testImg_paths = glob.glob(testImage_path)
tbow, tbowList, timageList = IR.getBOW(testImg_paths)
testimg_rep = IR.transform(centers, k, tbowList)

# ----------------------Retrieving target image-----------------
index = IR.compare(testimg_rep, img_rep)
print('retrieval result: ')
print(img_paths[index])
