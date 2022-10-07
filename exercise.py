import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from operator import itemgetter

np.set_printoptions(threshold=np.inf)


# 实验二-------------------------------------
# 自写一个灰度直方图函数
def histogram(input_image, draw_hist_flag):
    rows, cols = input_image.shape
    ret = np.zeros(256)
    for i in range(rows):
        for j in range(cols):
            ret[input_image[i][j]] += 1
    if draw_hist_flag == 1:
        plt.figure(1)
        plt.plot(range(256), ret)
    return ret


# 函数功能：得到图像分割的阈值，先用均值平滑直方图，当出现双峰图时，取双峰坐标的中点作为分割阈值
def get_segment_threshold(hist, max_iter=100):
    # 判别是否为双峰图
    def is_two_peaks(temp):
        count = 0
        for y in range(1, 255):
            if temp[y - 1] < temp[y] and temp[y + 1] < temp[y]:
                count += 1
                if count > 2:
                    return False

        return True if count == 2 else False

    hist_t1 = hist.astype(np.float32)
    hist_t2 = hist_t1
    iterate = 0
    while not is_two_peaks(hist_t2):
        hist_t2[0] = (hist_t1[0] * 2 + hist_t1[1]) / 3
        for i in range(1, 255):
            hist_t2[i] = (hist_t1[i - 1] + hist_t1[i] + hist_t1[i + 1]) / 3
        hist_t2[255] = (hist_t1[255] * 2 + hist_t1[254]) / 3
        iterate += 1
        # 平滑处理肯定需要多次迭代，这里设定一个迭代上限，如果达到迭代上限依然出现不了双峰，那就直接返回128作为分割阈值
        if iterate > max_iter:
            return 128
        hist_t2 = hist_t1
    plt.figure(2)
    plt.plot(range(256), hist_t2)
    peaks = [0, 0]
    index = 0
    for i in range(1, 255):
        if hist_t2[i - 1] < hist_t2[i] and hist_t2[i + 1] < hist_t2[i]:
            peaks[index] = i
            index += 1
    return (peaks[0] + peaks[1]) / 2


# 基于双峰平均值作为阈值的图像二值化算法
def two_peaks_segmentation(input_img, show_flag):
    img_hist = histogram(input_img, show_flag)
    threshold = get_segment_threshold(img_hist)
    rows, cols = input_img.shape
    out_img = input_img
    for i in range(rows):
        for j in range(cols):
            if input_img[i][j] <= threshold:
                out_img[i][j] = 0
            else:
                out_img[i][j] = 255
    if show_flag:
        cv.imshow('b_image', out_img)
        plt.show()
    return out_img


# 实验三---------------------------------------
# 高斯平滑函数，给图像去除噪声。参数length设定了滤波器算子的宽度
def gauss_smooth(input_image, length, sigma=1.4):
    k = length // 2
    gauss_filter = np.zeros([length, length])
    for i in range(length):
        for j in range(length):
            gauss_filter[i, j] = np.exp(-((i - k) ** 2 + (j - k) ** 2) / (2 * sigma ** 2))
            gauss_filter /= 2 * np.pi * sigma ** 2
            gauss_filter = gauss_filter / np.sum(gauss_filter)

    rows, cols = input_image.shape
    new_rows = rows - k * 2
    new_cols = cols - k * 2
    new_image = np.zeros([new_rows, new_cols])

    for i in range(new_rows):
        for j in range(new_cols):
            new_image[i][j] = np.sum(input_image[i:i + length, j:j + length] * gauss_filter)

    new_image = np.uint8(new_image)
    cv.imshow('gauss_smooth', new_image)
    return new_image


# 求取图像每个点的梯度及方向，这里使用了sobel算子
def get_gradient_and_direction(input_image):
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    rows, cols = input_image.shape
    new_rows = rows - 2
    new_cols = cols - 2
    gradient = np.zeros([new_rows, new_cols])
    direction = np.zeros([new_rows, new_cols])
    for i in range(new_rows):
        for j in range(new_cols):
            dx = np.sum(input_image[i:i + 3, j:j + 3] * Gx)
            dy = np.sum(input_image[i:i + 3, j:j + 3] * Gy)
            gradient_length = dx ** 2 + dy ** 2
            gradient[i, j] = np.math.sqrt(gradient_length)
            if dx == 0:
                direction[i, j] = np.pi / 2
            else:
                direction[i, j] = np.arctan(dy / dx)

    gradient = np.uint8(gradient)
    cv.imshow('gradient', gradient)
    return gradient, direction


# 非最大抑制方法NMS，来保留边界上的梯度极大值，使得边界变得清晰
# 由于实际图像中像素点是离散的数据，任一像素点在其梯度方向两侧的点不一定存在，该不存在点的梯度就需要其两侧的点插值得到
def NMS(gradient, direction):
    rows, cols = gradient.shape
    nms = np.copy(gradient)
    # 由于numpy中tan函数的定义域为( -π/2，π/2），那么可以对中心点的梯度方向角度归为四类：
    # （-π/2, -π/4）， （-π/4, 0）, （0, π/4）,（π/4, π/2），根据每种情况确定插值，算出中心点两侧点的梯度
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            angle = direction[i, j]
            weight = np.tan(angle)
            if angle > np.pi / 4:
                d1 = [0, 1]
                d2 = [1, 1]
                weight = 1 / weight
            elif 0 <= angle < np.pi / 4:
                d1 = [1, 0]
                d2 = [1, 1]
            elif -np.pi / 4 <= 0:
                d1 = [1, 0]
                d2 = [1, -1]
                weight *= -1
            else:
                d1 = [0, -1]
                d2 = [1, -1]
                weight = -1 / weight

            g1 = gradient[i + d1[0], j + d1[1]]
            g2 = gradient[i + d2[0], j + d2[1]]
            g3 = gradient[i - d1[0], j - d1[1]]
            g4 = gradient[i - d2[0], j - d2[1]]

            grade_count1 = g1 * weight + g2 * (1 - weight)
            grade_count2 = g3 * weight + g4 * (1 - weight)
            # 中心点梯度值大于两侧点保留，否则将其置零
            if grade_count1 > gradient[i, j] or grade_count2 > gradient[i, j]:
                nms[i, j] = 0
    return nms


# canny算法中涉及到双阈值设定，梯度值大于强阈值为确定边界点，介于弱阈值和强阈值之间为候选点，通过深度遍历的方式判断候选点是否与确定边界点
# 相连，如果相连则也算入边界点，这么做目的是去除掉一些噪点，保留真正需要的边界
def double_threshold(input_image, threshold1, threshold2):
    visited = np.zeros_like(input_image)
    output_image = input_image.copy()
    rows, cols = output_image.shape

    # 深度遍历dfs，遍历中心点周围八个像素点
    def dfs(i, j):
        if i >= rows or i < 0 or j >= cols or j < 0 or visited[i, j] == 1:
            return
        visited[i, j] = 1
        if output_image[i, j] > threshold1:
            output_image[i, j] = 255
            dfs(i - 1, j - 1)
            dfs(i - 1, j)
            dfs(i - 1, j + 1)
            dfs(i, j - 1)
            dfs(i, j + 1)
            dfs(i + 1, j - 1)
            dfs(i + 1, j)
            dfs(i + 1, j + 1)
        else:
            output_image[i, j] = 0

    for i in range(rows):
        for j in range(cols):
            if visited[i, j] == 1:
                continue
            # 从确定边界点出发，深度遍历寻找与其连通的候选边界点
            if output_image[i, j] >= threshold2:
                dfs(i, j)
            elif output_image[i, j] <= threshold1:
                output_image[i, j] = 0
                visited[i, j] = 1

    for i in range(rows):
        for j in range(cols):
            if visited[i, j] == 0:
                output_image[i, j] = 0
    return output_image


# canny边界检测算法
def canny_edge_extraction(input_image, threshold1, threshold2):
    gauss = gauss_smooth(input_image, 5)
    gradient, direction = get_gradient_and_direction(gauss)
    nms = NMS(gradient, direction)
    output_image = double_threshold(nms, threshold1, threshold2)
    cv.imshow('edge', output_image)
    return output_image


# 实验四--------------------------------------------
# 特征提取：通过八领域方式找到字符的连通域，然后计算出每个字符区域的外接矩形，面积，重心坐标以及边缘周长
OFFSETS_8 = [[-1, -1], [0, -1], [1, -1],
             [-1, 0], [0, 0], [1, 0],
             [-1, 1], [0, 1], [1, 1]]


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

    return max_upper_bd - 5, max_upper_bd + max_link_rows + 2


def find_area(input_image, offsets, reverse=False):
    area_img = np.array(input_image)
    label_index = 0
    max_label = 0
    rows, cols = area_img.shape
    rows_range = [0, rows, 1] if reverse is False else [rows - 1, -1, -1]
    cols_range = [0, cols, 1] if reverse is False else [cols - 1, -1, -1]

    for row in range(rows_range[0], rows_range[1], rows_range[2]):
        for col in range(cols_range[0], cols_range[1], cols_range[2]):
            label = 256
            if area_img[row, col] == 0:
                continue
            for offset in offsets:
                neighbor_row = min(max(0, row + offset[0]), rows - 1)
                neighbor_col = min(max(0, col + offset[1]), cols - 1)
                neighbor_val = area_img[neighbor_row, neighbor_col]
                if neighbor_val == 0:
                    continue
                label = neighbor_val if neighbor_val < label else label
            if label == 255:
                label_index += 1
                label = label_index
            area_img[row, col] = label
            if label > max_label:
                max_label = label
    return area_img, max_label


def feature_extraction(input_b_image, offsets):
    cut1, cut2 = horizontal_cut(input_b_image)
    cut_b_img = b_img[cut1:cut2, 0:b_img.shape[1]]
    cut_rows, cut_cols = cut_b_img.shape
    for i in range(cut_rows):
        for j in range(cut_cols):
            cut_b_img[i, j] = 255 - cut_b_img[i, j]
    area, area_idx = find_area(cut_b_img, offsets, False)
    area, area_idx = find_area(area, offsets, True)
    cv.imshow('label_img', area)
    bound_rectangle = []
    for k in range(1, area_idx + 1):
        area_flag = 0
        up_bd, down_bd, l_bd, r_bd = 0, 0, 0, 0

        for i in range(cut_rows):
            for j in range(cut_cols):
                if area[i][j] == k:
                    if area_flag == 0:
                        area_flag = 1
                        up_bd = i
                        l_bd = j
                        continue
                    up_bd = min(up_bd, i)
                    down_bd = max(down_bd, i)
                    l_bd = min(l_bd, j)
                    r_bd = max(r_bd, j)
        if area_flag == 1:
            bound_rectangle.append([k, up_bd, down_bd, l_bd, r_bd])
    # print(bound_rectangle)
    # print(len(bound_rectangle))
    rectangle_img = cut_b_img
    features = []
    for k in range(len(bound_rectangle)):
        area_idx, up_bd, down_bd, l_bd, r_bd = bound_rectangle[k]
        area_area, core_x, core_y, edge_len = 0, 0, 0, 0
        # print(area_idx, up_bd, down_bd, l_bd, r_bd)
        cv.rectangle(rectangle_img, (l_bd, up_bd), (r_bd, down_bd), 255, 1, 4)
        for i in range(up_bd, down_bd + 1):
            for j in range(l_bd, r_bd + 1):
                if area[i][j] == area_idx:
                    for offset in offsets:
                        neighbor_row = i + offset[0]
                        neighbor_col = j + offset[1]
                        neighbor_val = area[neighbor_row, neighbor_col]
                        if neighbor_val == 0:
                            edge_len += 1
                            break
                    area_area += 1
                    core_x += j
                    core_y += i

        core_x = round(core_x / area_area)
        core_y = round(core_y / area_area)
        features.append((area_area, core_x, core_y, edge_len, up_bd, down_bd, l_bd, r_bd))
    features.sort(key=itemgetter(1))
    # sorted(features, key=itemgetter(1))
    # print(features)
    for k in range(len(features)):
        area_area, core_x, core_y, edge_len, up_bd, down_bd, l_bd, r_bd = features[k]
        print("第", k + 1, "个连通域特征：面积:", area_area, "重心坐标:", (core_x, core_y), "边缘长度:", edge_len, "外接矩形坐标:",
              (l_bd, up_bd), (r_bd, down_bd))

    cv.namedWindow('rectangle_image', cv.WINDOW_NORMAL)
    cv.imshow('rectangle_image', rectangle_img)


if __name__ == '__main__':
    # 实验1 读取图片并灰度化， 并将像素数据存放在一个文件中
    img = cv.imread('test.jpg', 0)
    cv.imshow('src', img)
    np.savetxt('image_data.txt', np.c_[img], fmt='%d', delimiter='\t')
    # 实验2 通过灰度直方图确定阈值进行图像分割
    b_img = two_peaks_segmentation(img, 1)
    # print(np.array(255-b_img))
    # 实验3 边缘检测
    # out_image = canny_edge_extraction(img, 50, 150)
    # 实验4 特征提取
    feature_extraction(b_img, OFFSETS_8)
    # print(area)
    cv.waitKey(0)
    cv.destroyAllWindows()
