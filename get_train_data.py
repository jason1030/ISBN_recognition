import os
import cv2 as cv
import csv
import numpy as np


def get_train_data(path):
    img_data = []
    img_label = []
    for file_dir_name in os.listdir(path):
        for img_name in os.listdir(path + '/' + file_dir_name):
            true_path = path + '/' + file_dir_name + '/' + img_name
            img = cv.imread(true_path, 0)
            img = cv.resize(img, (28, 28))
            img_data.append(img.flatten())

            img_name = img_name.replace(".jpg", '')
            img_name = img_name.replace(".JPG", '')

            if img_name[-1] == 'X':
                img_label.append(10)
            elif img_name[-1] == 'I':
                img_label.append(11)
            elif img_name[-1] == 'S':
                img_label.append(12)
            elif img_name[-1] == 'B':
                img_label.append(13)
            elif img_name[-1] == 'N':
                img_label.append(14)
            elif img_name[-1] == 'E':
                img_label.append(15)
            elif img_name[-1] == '-':
                img_label.append(16)
            else:
                img_label.append(img_name[-1])
    with open('features.csv', 'w', encoding='utf-8', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerows(np.array(img_data))
    print('x写入完成...')
    with open('labels.csv', 'w', encoding='utf-8', newline='') as fp:
        writer = csv.writer(fp)
        for i in img_label:
            writer.writerow([i])
        # writer.writerows(img_label_list)
    print('y写入完成...')


get_train_data(r'C:\PycharmProjects\opencv_test1\train')
