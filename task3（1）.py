import numpy as np
import cv2 as cv

#创建窗口
cv.namedWindow('Rotated Image', cv.WINDOW_NORMAL)
cv.resizeWindow('Rotated Image', 800, 600)
cv.namedWindow('dst', cv.WINDOW_NORMAL)
cv.resizeWindow('dst', 800, 600)
cv.namedWindow('dst1', cv.WINDOW_NORMAL)
cv.resizeWindow('dst1', 800, 600)

#蓝色范围
lower_blue = np.array([100, 140, 140])
upper_blue = np.array([140, 255, 255])

#初步处理图片
img = cv.imread('frame_65.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

mask_blue = cv.inRange(hsv,lower_blue,upper_blue)
kernel = np.ones((5, 5), np.uint8)
mask_blue = cv.morphologyEx(mask_blue, cv.MORPH_CLOSE, kernel)#去噪

#画出轮廓
contours_blue, _ = cv.findContours(mask_blue, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
for contour in contours_blue:
    #过滤
    if cv.contourArea(contour) > 200:  
        # 计算重心
        M = cv.moments(contour)
        if M["m00"] != 0:
            cx_blue = int(M["m10"] / M["m00"])
            cy_blue = int(M["m01"] / M["m00"])
            # 绘制轮廓和重心
            cv.drawContours(img, [contour], -1, (0, 0, 255), 2)  # 蓝色轮廓
            cv.circle(img, (cx_blue, cy_blue), 5, (255, 0, 0), -1)  # 蓝色重心

# 取旋转矩形
cnt = contours_blue[0]
rect = cv.minAreaRect(cnt)
theta = rect[2]
box = cv.boxPoints(rect)
box = np.int32(box)
cv.drawContours(img,[box],0,(0,0,255),2)

# 获取图像的尺寸（行数和列数）
rows, cols = img.shape[:2]
# 旋转角度
angle = (theta)
# 计算旋转矩阵，中心为图像中心，角度为旋转角度，比例为1（即大小不变）
M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
# 旋转
dst = cv.warpAffine(img, M, (cols, rows))
#利用极点判断正逆向
top = contours_blue[0][contours_blue[0][:, :, 1].argmin()][0]  # 最上点
bottom = contours_blue[0][contours_blue[0][:, :, 1].argmax()][0]  # 最下点
#取出y坐标
top_y = int(top[1])
bottom_y = int(bottom[1])
M = cv.moments(contours_blue[0], True)
cy = int(M['m01'] / M['m00'])
d_top = cy- top_y
d_down = bottom_y - cy
if d_top >= d_down:
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
    dst1 = cv.warpAffine(dst, M, (cols, rows))
else:
# 计算旋转矩阵，中心为图像中心，角度为旋转角度，比例为1（即大小不变）
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), 0, 1)
    dst1 = cv.warpAffine(dst, M, (cols, rows))

cv.imshow('res', img)
cv.imshow('dst', dst)
cv.imshow('dst1', dst1)
cv.waitKey(0)
cv.destroyAllWindows()



