import cv2 as cv
import numpy as np
import os

# 创建保存裁剪图像的文件夹
save_folder = 'cropped_images'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 创建窗口
cv.namedWindow('frame', cv.WINDOW_NORMAL)
cv.resizeWindow('frame', 800, 600)
# cv.namedWindow('dst3', cv.WINDOW_NORMAL)
# cv.resizeWindow('dst3', 800, 600)
# cv.namedWindow('dst', cv.WINDOW_NORMAL)
# cv.resizeWindow('dst', 800, 600)
cv.namedWindow('res', cv.WINDOW_NORMAL)
cv.resizeWindow('res', 800, 600)
# cv.namedWindow('num', cv.WINDOW_NORMAL)
# cv.resizeWindow('num', 800, 600)

# 蓝色范围
lower_blue = np.array([100, 100, 100])
upper_blue = np.array([140, 255, 255])

# 打开视频文件或摄像头
cap = cv.VideoCapture('task4_level1.mov')  # 如果是视频文件
# cap = cv.VideoCapture(0)  # 如果是摄像头

frame_count = 0  # 用来为保存的图像命名

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("无法接收帧，退出...")
        break

    img = frame.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask_blue = cv.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((5, 5), np.uint8)
    mask_blue = cv.morphologyEx(mask_blue, cv.MORPH_CLOSE, kernel)

    contours_blue, _ = cv.findContours(mask_blue, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours_blue) == 0:
        continue  # 如果没有蓝色物体，跳过这一帧

    contour_max = max(contours_blue, key=cv.contourArea)
    M = cv.moments(contour_max)
    if M["m00"] == 0:
        continue  # 如果没有轮廓的质量，跳过这一帧

    cx_blue = int(M["m10"] / M["m00"])
    cy_blue = int(M["m01"] / M["m00"])
    cv.drawContours(img, [contour_max], -1, (0, 0, 255), 2)

    # 获取旋转矩形
    cnt = contour_max
    rect = cv.minAreaRect(cnt)
    theta = rect[2]
    box = cv.boxPoints(rect)
    box = np.int32(box)
    cv.drawContours(img, [box], 0, (0, 0, 255), 2)

    # 获取图像尺寸
    rows, cols = img.shape[:2]
    angle = theta  # 旋转角度

    # 计算旋转矩阵并旋转图像
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    dst1 = cv.warpAffine(img, M, (cols, rows))
    contour_max1 = cv.transform(contour_max, M)

    # 根据长宽比判断是否需要旋转
    (w, h) = rect[1]
    aspect_ratio = float(w) / h
    if aspect_ratio < 1:
        M = cv.getRotationMatrix2D((cols / 2, rows / 2), 0, 1)
        dst2 = cv.warpAffine(dst1, M, (cols, rows))
        contour_max2 = cv.transform(contour_max1, M)
    else:
        M = cv.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
        dst2 = cv.warpAffine(dst1, M, (cols, rows))
        contour_max2 = cv.transform(contour_max1, M)

    # 利用极点判断正逆向
    top = contour_max2[contour_max2[:, :, 1].argmin()][0]
    bottom = contour_max2[contour_max2[:, :, 1].argmax()][0]
    top_y = int(top[1])
    bottom_y = int(bottom[1])
    M = cv.moments(contour_max2, True)
    cy = int(M['m01'] / M['m00'])
    d_top = cy - top_y
    d_down = bottom_y - cy
    if d_top <= d_down:
        M = cv.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
        dst3 = cv.warpAffine(dst2, M, (cols, rows))
    else:
        M = cv.getRotationMatrix2D((cols / 2, rows / 2), 0, 1)
        dst3 = cv.warpAffine(dst2, M, (cols, rows))

    # 显示处理后的帧
    img1 = dst3
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(img1, cv.COLOR_BGR2HSV)
    mask_bule = cv.inRange(hsv, lower_blue, upper_blue)
    bule_region = cv.bitwise_and(img1, img1, mask=mask_bule)
    gray = cv.cvtColor(bule_region, cv.COLOR_BGR2GRAY)

    # 二值化图像
    ret, dst = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(dst, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 填充轮廓
    dst = cv.drawContours(dst, contours, -1, 255, cv.FILLED)

    # 将填充后的轮廓与掩模相减
    res = cv.absdiff(dst, mask_bule)
    res = cv.GaussianBlur(res, (5, 5), 0)
    kernel = np.ones((3, 3), np.uint8)
    res = cv.morphologyEx(res, cv.MORPH_CLOSE, kernel)
    num = cv.bitwise_and(gray1, gray1, mask=res)

# 统计白色像素数量
    white_pixels = cv.countNonZero(res)

# 如果白色像素数量小于100，则跳过当前图像
    if white_pixels < 400:
     continue
    contours, _ = cv.findContours(num, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # 获取轮廓的边界框
        x, y, w, h = cv.boundingRect(contour)

        # 在原图像中裁剪出轮廓所在区域
        cropped_image = num[y:y+h, x:x+w]
        
        if cropped_image.size < 400:  # 如果像素数小于100，则跳过
            continue

        # # 保存裁剪图像
        # frame_count += 10
        # cropped_image_filename = os.path.join(save_folder, f"cropped_{frame_count}.png")
        # cv.imwrite(cropped_image_filename, cropped_image)
        # print(f"裁剪图像已保存为 {cropped_image_filename}")

        # 显示裁剪图像
        cv.imshow('Cropped Num', cropped_image)

    # 显示帧
    cv.imshow('frame', frame)
    # cv.imshow('dst3', dst3)
    # cv.imshow('dst', dst)
    cv.imshow('res', res)
    # cv.imshow('num', num)

    # 按1毫秒延时显示图像，并按'q'退出
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

# 释放视频捕获对象并关闭所有窗口
cap.release()
cv.destroyAllWindows()