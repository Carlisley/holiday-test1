import cv2
import time

# 打开摄像头，0通常是默认的摄像头
cap = cv2.VideoCapture('video.mp4')

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 设置视频帧宽度和高度
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS,29)
# 记录上一帧的时间
prev_time = time.time()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("无法读取视频帧")
        break

    # 获取当前时间
    curr_time = time.time()

    # 计算帧率
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # 在帧上绘制帧率
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示视频帧
    cv2.imshow("摄像头视频", frame)

    # 按'q'键退出
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
