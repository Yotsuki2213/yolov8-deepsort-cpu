import cv2
import os

# 打开视频文件
video_path = "./video/input2.mp4"
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 创建保存帧的目录
output_dir = "./video_cut"
os.makedirs(output_dir, exist_ok=True)

frame_count = 0
saved_frame_count = 0

while True:
    ret, frame = cap.read()

    # 如果读取帧失败，则退出循环
    if not ret:
        break

    # 每隔30帧保存一次
    if frame_count % 30 == 0:
        frame_filename = os.path.join(output_dir, f"frame_{saved_frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_frame_count += 1

    frame_count += 1

# 释放视频对象
cap.release()
print(f"总共保存了 {saved_frame_count} 帧图像。")
