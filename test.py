import numpy as np
import cv2 as cv


cv.namedWindow('frame', cv.WINDOW_NORMAL)
cv.resizeWindow('frame', 800, 600)

cv.namedWindow('mask', cv.WINDOW_NORMAL)
cv.resizeWindow('mask', 800, 600)

cv.namedWindow('res', cv.WINDOW_NORMAL)
cv.resizeWindow('res', 800, 600)

frame_count = 0
cap = cv.VideoCapture('task4_level3.mp4')
cap.set(cv.CAP_PROP_EXPOSURE, -1)

while(cap.isOpened()):
    ret, frame = cap.read()
    frame_count += 1
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    
    lower_blue = np.array([100, 140, 140])
    upper_blue = np.array([140, 255, 255])

    mask_blue = cv.inRange(hsv,lower_blue,upper_blue)
    kernel = np.ones((5, 5), np.uint8)
    mask_blue = cv.morphologyEx(mask_blue, cv.MORPH_CLOSE, kernel)

    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    res = cv.bitwise_and(frame,frame, mask= mask_blue)

    contours_blue, _ = cv.findContours(mask_blue, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours_blue:
        if cv.contourArea(contour) > 100:  # 过滤掉小的区域
            # 计算重心
            M = cv.moments(contour)
            if M["m00"] != 0:
                cx_blue = int(M["m10"] / M["m00"])
                cy_blue = int(M["m01"] / M["m00"])
                # 绘制轮廓和重心
                cv.drawContours(frame, [contour], -1, (255, 0, 0), 2)  # 蓝色轮廓
                cv.circle(frame, (cx_blue, cy_blue), 5, (255, 0, 0), -1)  # 蓝色重心
                if frame_count % 5 == 0 :
                 cv.imwrite(f'frame_{frame_count}.jpg', frame)
                 print(f"保存图像：frame_{frame_count}.jpg")

    # 显示结果
    cv.imshow("红蓝色块检测", frame)

    # 按 'q' 键退出
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv.destroyAllWindows()


    #cv.imshow('frame',frame)
    #cv.imshow('mask',mask_blue)
    #cv.imshow('res',res)
    #k = cv.waitKey(5) & 0xFF
    #if k == ord('q'):
       #break


    #cv.imshow('video',frame)
    #if cv.waitKey(60) & 0xFF == ord('q'):
        #break
