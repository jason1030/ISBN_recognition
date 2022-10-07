import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.special
import math

dir_name = "test_img"
pic_num = 0
for filename in os.listdir("./" + dir_name):
    pic_num = pic_num + 1
    print(pic_num)
    img = cv.imread(dir_name + "/" + filename, 0)
    black = []
    gauss = cv.GaussianBlur(img, (3, 3), 0)
    ret3, B_image = cv.threshold(gauss, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    rows, cols = B_image.shape
    for i in range(cols):
        black.append(np.array([(255-B_image[:, i])/255]).sum())

    # plt.bar(range(len(black)), black)
    # plt.show()
    black = np.array(black)
    if black.sum()/(rows*cols)>0.5:
        print(filename)


