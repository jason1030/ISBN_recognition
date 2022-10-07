# coding=utf-8
import cv2 as cv
import numpy as np
np.set_printoptions(threshold=np.inf)

def horizontal_cut(src_img):
    rows, cols = src_img.shape
    flag = 0
    link_rows = 0   #记录当前连通区
    max_link_rows = 0      #记录最大连通区
    upper_bd = 0
    lower_bd = 0
    for i in range(rows):
        black = 0
        for j in range(int(0.25*cols), int(0.75*cols)):
            if turn_edges[i][j] == 0:
                black = black +1
            if black > 5:
                link_rows = link_rows + 1
                if flag == 0:
                    flag = 1
                    upper_bd = i
                break

        if flag == 1 and (black < 5 or i ==int(0.5*rows)):
            if link_rows > max_link_rows:
                max_link_rows = link_rows
                link_rows = 0
                flag = 0

    flag = 0
    end_bd = upper_bd - 10 #以条形码上界作为起点，进行反向遍历寻找ISBN码的上下界
    lower_bd = upper_bd - 10
    upper_bd = 0
    for  i in range(lower_bd):
        black = 0
        for j in range(int(0.25*cols), int(0.75*cols)):
            if turn_Bimage[end_bd - i][j] == 0:
                black += 1
            if black > 5 and flag == 0:
                flag = 1
                lower_bd =end_bd - i
                break
        if flag and black < 5:
            upper_bd = end_bd -i
            break
    return upper_bd,lower_bd
img = cv.imread('5.jpg')
height, width = img.shape[:2]
img = cv.resize(img, (0, 0), None, 0.5, 0.5)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
src_rows, src_cols = gray.shape
gauss = cv.GaussianBlur(gray, (3, 3), 0)
cv.imshow("1", gauss)
ret3, B_image = cv.threshold(gauss, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imshow("11", B_image)
edges = cv.Canny(B_image, 50, 150, apertureSize=3)
cv.imshow("2", edges)

sobel_y = cv.Sobel(edges, -1, 0, 1, ksize=5)
cv.imshow("3", sobel_y)
lines = cv.HoughLines(sobel_y, 1, np.pi / 180, 250)
theta_avg = 0
line_cnt = 0
#print(len(lines))
for line in lines:
    if (abs(line[0][1]) >= np.pi*0.25) and (abs(line[0][1]) <= np.pi*0.75):
        line_cnt = line_cnt + 1
        theta_avg = theta_avg + line[0][1]
if len(lines) == 0:
    theta_avg = np.pi / 2
else:
    theta_avg = theta_avg / line_cnt
    print(theta_avg)
theta_angle = (theta_avg * 180 / np.pi - 90)
#print(theta_angle)
M = cv.getRotationMatrix2D(((src_cols-1)/2, (src_rows-1)/2), theta_angle, 1)
dst = cv.warpAffine(img, M, (src_cols, src_rows))

turn_edges = cv.warpAffine(edges, M, (src_cols, src_rows))
turn_Bimage = cv.warpAffine(B_image, M, (src_cols, src_rows))
cv.imshow('turn_edges', turn_edges)
cv.imshow('turn_Bimage', turn_Bimage)



for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)
print(edges)

cv.imshow("hough_line", img)
cv.imshow('dst', dst)
cv.waitKey()
cv.destroyAllWindows()

