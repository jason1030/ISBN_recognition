import time
from tensorflow import keras
import cv2 as cv
import numpy as np
import os


# 函数定义：对输入图片进行霍夫直线检测，计算出倾斜角，然后进行仿射变换，将图片旋转正
def rotate_img(edge_img, src_img):
    rows, cols = src_img.shape
    sobel_y = cv.Sobel(edge_img, -1, 0, 1, ksize=5)
    lines = cv.HoughLines(sobel_y, 1, np.pi / 180, 250)
    theta_avg = 0
    line_cnt = 0
    if lines is None:
        pass
    else:
        for line in lines:
            if (abs(line[0][1]) >= np.pi * 0.25) and (abs(line[0][1]) <= np.pi * 0.75):
                line_cnt = line_cnt + 1
                theta_avg = theta_avg + line[0][1]
    if lines is None or len(lines) == 0:
        theta_avg = np.pi / 2
    else:
        theta_avg = theta_avg / line_cnt
    turn_angle = (theta_avg * 180 / np.pi - 90)
    M = cv.getRotationMatrix2D((int((cols - 1) / 2), int((rows - 1) / 2)), turn_angle, 1)
    out_img = cv.warpAffine(src_img, M, (cols, rows), borderValue=255)
    return out_img


def horizontal_cut(input_img):
    input_rows, input_cols = input_img.shape
    min_link_black = 5  # 连通行的黑色像素阈值，大于阈值才算有效
    flag = 0
    link_rows = 0  # 记录当前连通区
    max_link_rows = 0  # 记录最大连通区
    upper_bd = 0  # 当前连通区上界
    max_upper_bd = 0  # 最大连通区的上界
    for i in range(input_rows):
        black = 0
        for j in range(int(0.2 * input_cols), int(0.8 * input_cols)):
            if input_img[i][j] == 0:
                black = black + 1
            if black > min_link_black:
                link_rows = link_rows + 1
                if flag == 0:
                    flag = 1
                    upper_bd = i
                break

        if flag == 1 and (black < min_link_black or i == int(input_rows - 1)):
            if link_rows > max_link_rows:
                max_upper_bd = upper_bd
                max_link_rows = link_rows
            link_rows = 0
            flag = 0

    flag = 0
    end_bd = max_upper_bd - 1  # 以条形码上界作为起点，进行反向遍历寻找ISBN码的上下界
    lower_bd = max_upper_bd - 1
    upper_bd = 0
    for i in range(end_bd):
        black = 0
        for j in range(int(0.2 * input_cols), int(0.8 * input_cols)):
            if input_img[end_bd - i][j] == 0:
                black += 1
            if black > min_link_black and flag == 0:
                flag = 1
                lower_bd = end_bd - i
                break
        if flag == 1 and black < min_link_black:
            upper_bd = end_bd - i
            break
    if upper_bd > 2:
        upper_bd -= 2

    return upper_bd, lower_bd + 2


def vertical_cut(input_img):
    input_rows, input_cols = input_img.shape
    black_num_list = []  # 一维数组，用以记录每列黑像素个数
    for j in range(input_cols):
        black = 0
        for i in range(input_rows):
            if input_img[i][j] == 0:
                black += 1
        black_num_list.append(black)
    l_index, r_index = 0, 0  # 左右边界坐标
    flag = False
    y_cuts_index = []  # 存储坐标信息的数组
    for j, count in enumerate(black_num_list):
        if flag == 0 and count > 0:
            l_index = j
            flag = True
        if flag == 1 and count == 0:
            r_index = j - 1
            flag = False
            if 5 < r_index - l_index < 40:
                y_cuts_index.append([l_index, r_index])
    return y_cuts_index


if __name__ == '__main__':
    start_time = time.time()
    # 载入已经训练好模型
    model = keras.models.load_model('my_ISBN_model')
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    dir_name = "test_img"
    Bimage_dir_name = "B_image"
    cut_dir_name = "cut_Bimage"
    ISBN_cut_dir_name = "ISBN_cut"
    if not os.path.exists(cut_dir_name):
        os.mkdir(cut_dir_name)
    if not os.path.exists(Bimage_dir_name):
        os.mkdir(Bimage_dir_name)
    if not os.path.exists(ISBN_cut_dir_name):
        os.mkdir(ISBN_cut_dir_name)
    # 分别定义总图片数，正确图片数，总ISBN数字数量，识别正确的ISBN数字数量，用以统计正确率与准确率
    pic_num = 0
    true_pic_num = 0
    ISBN_num = 0
    true_ISBN_num = 0
    for filename in os.listdir("./" + dir_name):
        # if pic_num >= 10:
        #     break
        pic_num = pic_num + 1
        print(pic_num)
        ISBN_true = []  # 存储当前ISBN号的真值
        for index in range(len(filename)):
            if '0' <= filename[index] <= '9' or filename[index] == 'X':
                ISBN_true.append(filename[index])
        img = cv.imread(dir_name + "/" + filename, 0)
        temp_rows, temp_cols = img.shape
        width = int(750 * temp_cols / temp_rows)
        height = 750
        img = cv.resize(img, (width, height))
        src_rows, src_cols = img.shape
        gauss = cv.GaussianBlur(img, (5, 5), 0)
        ret3, B_image = cv.threshold(gauss, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        cv.imwrite(Bimage_dir_name + "/" + filename, B_image)
        # 考虑到有黑底白字的图片，对二值图的黑色像素进行统计，超过总像素的1/2判定为黑底，进行一下反色变换
        black_num = []
        for i in range(src_cols):
            black_num.append(np.array([(255 - B_image[:, i]) / 255]).sum())
        black_num = np.array(black_num)
        if black_num.sum() / (src_rows * src_cols) > 0.5:
            B_image = cv.bitwise_not(B_image)

        edges = cv.Canny(B_image, 50, 150, apertureSize=3)
        turn_Bimage = rotate_img(edges, B_image)

        cv.imwrite("turn_img" + "/" + filename, turn_Bimage)
        cut1, cut2 = horizontal_cut(turn_Bimage)
        cut_image = turn_Bimage[cut1:cut2, 0:src_cols]
        cv.imwrite(cut_dir_name + "/" + filename, cut_image)
        ver_cut_index = vertical_cut(cut_image)
        ver_cut_index = np.array(ver_cut_index)
        file_name = os.path.splitext(filename)[0]
        file_name = file_name.replace(' ', '')
        print(file_name)
        if not os.path.exists(ISBN_cut_dir_name + "/" + file_name):
            os.mkdir(ISBN_cut_dir_name + "/" + file_name)

        # 预测部分，在原二值图上切割出字符，对每个字符进行预测，预测值存入ISBN_pre list中
        ISBN_pre = []
        for index in range(len(ver_cut_index)):
            # print(ver_cut_index[index][0], ver_cut_index[index][1])
            temp_img = cut_image[0:cut2 - cut1, ver_cut_index[index][0]:ver_cut_index[index][1]]
            # w_path = ISBN_cut_dir_name + "/" + file_name + "/" + str(index) + "-" + file_name[index] + ".jpg"
            # cv.imwrite(w_path, temp_img)
            temp_img = [cv.resize(temp_img, (28, 28)).astype('float32')]
            temp_img = np.expand_dims(np.array(temp_img) / 255, -1)
            pre_class = np.argmax(model.predict(temp_img), axis=-1)
            if 0 <= pre_class[0] <= 9:
                ISBN_pre.append(str(pre_class[0]))
            elif pre_class[0] == 10:
                ISBN_pre.append('X')
            elif pre_class[0] == 11:
                ISBN_pre.append('I')
            elif pre_class[0] == 12:
                ISBN_pre.append('S')
            elif pre_class[0] == 13:
                ISBN_pre.append('B')
            elif pre_class[0] == 14:
                ISBN_pre.append('N')
            elif pre_class[0] == 15:
                ISBN_pre.append('E')

        temp_true_ISBN_num = 0
        # 如果没有预测到字符，ISBN_pre 存入一个'E'字符，表示错误
        if len(ISBN_pre) == 0:
            ISBN_pre.append('E')
        # 因为只需要后面的数字预测值与真值进行比较，所以以'N'字符为界，舍去前面的字符
        for i in range(len(ISBN_pre)):
            if ISBN_pre[i] == 'N':
                del ISBN_pre[0:i + 1]
                if ISBN_pre[-1] == 'E':
                    del ISBN_pre[-1]
                break
        # 将预测值与真值进行逐一比较，计算图片准确率与字符准确率
        index_pre = 0
        for index in range(len(ISBN_true)):
            if index >= len(ISBN_pre) or index_pre >= len(ISBN_pre):
                break
            elif ISBN_pre[index_pre] == ISBN_true[index]:
                temp_true_ISBN_num += 1
                index_pre += 1
            # 考虑到切割或者识别时的字符缺失，导致后续的字符顺序出现前移，有必要检查下个预测值是否与当前真值相同（只检查一位，没有检查两位）
            elif index + 1 < len(ISBN_true):
                if ISBN_pre[index_pre] == ISBN_true[index + 1] and len(ISBN_pre) < len(ISBN_true):
                    ISBN_pre.insert(index_pre, 'E')  # 将空缺的字符补上error符，对齐数据方便检查
                    index_pre += 1
                else:
                    index_pre += 1
            else:
                index_pre += 1
        if temp_true_ISBN_num == len(ISBN_true):
            true_pic_num += 1
        true_ISBN_num += temp_true_ISBN_num
        ISBN_num += len(ISBN_true)
        # if ISBN_pre[0] == 'N':
        #     del ISBN_pre[0]
        print("ISBN_true:\n", ISBN_true)
        print("ISBN_pre:\n", ISBN_pre)
    print("total_pic_num:", pic_num, "total_ISBN_num:", ISBN_num)
    print("ISBN_pic_acc:", true_pic_num / pic_num)
    print("ISBN_num_acc:", true_ISBN_num / ISBN_num)

    end_time = time.time()
    print("Running time:", end_time - start_time, "sec")
    # if index > len(file_name):
    #     w_path = ISBN_cut_dir_name + "/" + file_name + "/" + str(index) + "-" + "?" + ".jpg"
    #     cv.imwrite(w_path, temp_img)
    #     continue
    # w_path = ISBN_cut_dir_name + "/" + file_name + "/" + str(index) + "-" + file_name[index] + ".jpg"
    # # print(w_path)
    # cv.imwrite(w_path, temp_img)

    # cv.imshow('turn_edges', turn_edges)
    # cv.imshow('turn_Bimage', turn_Bimage)
    # cv.imshow('1', B_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
