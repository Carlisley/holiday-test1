import cv2 as cv
import numpy as np

#创建窗口
cv.namedWindow('dst3', cv.WINDOW_NORMAL)
cv.resizeWindow('dst3', 800, 600)
cv.namedWindow('dst', cv.WINDOW_NORMAL)
cv.resizeWindow('dst', 800, 600)
cv.namedWindow('res', cv.WINDOW_NORMAL)
cv.resizeWindow('res', 800, 600)
cv.namedWindow('num', cv.WINDOW_NORMAL)
cv.resizeWindow('num', 800, 600)

#蓝色范围
lower_blue = np.array([100, 100, 100])
upper_blue = np.array([140, 255, 255])
#图像初步处理
img = cv.imread('1.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
mask_blue = cv.inRange(hsv,lower_blue,upper_blue)
kernel = np.ones((5, 5), np.uint8)
mask_blue = cv.morphologyEx(mask_blue, cv.MORPH_CLOSE, kernel)

contours_blue, _ = cv.findContours(mask_blue, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contour_max = max(contours_blue,key=cv.contourArea)
M = cv.moments(contour_max)
print(M['m00'])
cx_blue = int(M["m10"] / M["m00"])
cy_blue = int(M["m01"] / M["m00"])
cv.drawContours(img, [contour_max], -1, (0, 0, 255), 2) 
# cv.circle(img, (cx_blue, cy_blue), 5, (255, 0, 0), -1)
#取旋转矩形
cnt = contour_max
rect = cv.minAreaRect(cnt)
theta = rect[2]
box = cv.boxPoints(rect)
box = np.int32(box)
cv.drawContours(img,[box],0,(0,0,255),2)
# 绘制旋转矩形
cv.drawContours(img, [box], -1, (0, 255, 0), 2)

# 获取图像的尺寸（行数和列数）
rows, cols = img.shape[:2]
# 旋转角度
angle = (theta)
print(angle)
# 计算旋转矩阵，中心为图像中心，角度为旋转角度，比例为1（即大小不变）
M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
# 旋转
dst1= cv.warpAffine(img, M, (cols, rows))
contour_max1 = cv.transform(contour_max, M)

#利用长宽比判断横竖
(w,h) = rect[1]
aspect_ratio = float(w)/h
if aspect_ratio < 1:
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), 0, 1)
    dst2 = cv.warpAffine(dst1, M, (cols, rows))
    contour_max2 = cv.transform(contour_max1, M)
else:
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    dst2 = cv.warpAffine(dst1, M, (cols, rows))
    contour_max2 = cv.transform(contour_max1, M)

#利用极点判断正逆向
top = contour_max2[contour_max2[:, :, 1].argmin()][0]  # 最上点
bottom = contour_max2[contour_max2[:, :, 1].argmax()][0]  # 最下点
#取出y坐标
top_y = int(top[1])
bottom_y = int(bottom[1])
M = cv.moments(contour_max2, True)
cy = int(M['m01'] / M['m00'])
d_top = cy- top_y
d_down = bottom_y - cy
if d_top <= d_down:
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
    dst3 = cv.warpAffine(dst2, M, (cols, rows))
else:
# 计算旋转矩阵，中心为图像中心，角度为旋转角度，比例为1（即大小不变）
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), 0, 1)
    dst3 = cv.warpAffine(dst2, M, (cols, rows))

img1 = dst3
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
hsv = cv.cvtColor(img1, cv.COLOR_BGR2HSV)
mask_bule = cv.inRange(hsv,lower_blue,upper_blue)
bule_region = cv.bitwise_and(img1,img1,mask=mask_bule)
gray = cv.cvtColor(bule_region, cv.COLOR_BGR2GRAY)
#二值图
ret,dst = cv.threshold(gray,0,255,cv.THRESH_BINARY)
contours,hierarchy = cv.findContours(dst,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
#填充
dst = cv.drawContours(dst, contours, -1, 255, cv.FILLED)
#将其与掩模相减
res = cv.absdiff(dst,mask_bule)
res = cv.GaussianBlur(res,(5,5),0)
kernel = np.ones((3, 3), np.uint8)
res = cv.morphologyEx(res, cv.MORPH_CLOSE, kernel)
num = cv.bitwise_and(gray1, gray1, mask=res)

contours, hierarchy = cv.findContours(num, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
for contour in contours:
    # 获取轮廓的边界框
    x, y, w, h = cv.boundingRect(contour)
    
    # 在原图像中裁剪出轮廓所在区域
    cropped_image = num[y:y+h, x:x+w]
    
    # 显示或保存裁剪的图像
    cv.imshow('Cropped Contour', cropped_image)
    cv.waitKey(0)
cv.imshow('dst3',dst3)
cv.imshow('dst', dst)
cv.imshow('res', dst)
cv.imshow('num', num)
cv.waitKey(0)
cv.destroyAllWindows()