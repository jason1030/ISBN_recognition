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