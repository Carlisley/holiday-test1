import cv2 as cv
import numpy as np

#创建窗口
cv.namedWindow('img', cv.WINDOW_NORMAL)
cv.resizeWindow('img', 800, 600)
cv.namedWindow('dst', cv.WINDOW_NORMAL)
cv.resizeWindow('dst', 800, 600)
cv.namedWindow('res', cv.WINDOW_NORMAL)
cv.resizeWindow('res', 800, 600)
cv.namedWindow('num', cv.WINDOW_NORMAL)
cv.resizeWindow('num', 800, 600)

#颜色   
lower_blue = np.array([100, 100, 100])
upper_blue = np.array([140, 255, 255])

#初步处理
img = cv.imread('1.jpg')
gray1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
mask_bule = cv.inRange(hsv,lower_blue,upper_blue)
bule_region = cv.bitwise_and(img,img,mask=mask_bule)
cv.imshow('img', mask_bule)
#灰度图
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

cv.imshow('dst', dst)
cv.imshow('res', dst)
cv.imshow('num', num)
cv.waitKey(0)
cv.destroyAllWindows()




    
