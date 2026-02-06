from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train4/weights/best.pt")  # 加载模型

# 输入输出视频路径
input_video = './video/input2.mp4'
output_video = './video/output2.mp4'

# 打开视频文件
cap = cv2.VideoCapture(input_video)

# 获取视频属性（用于设置输出视频）
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# 逐帧处理
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用YOLO检测当前帧
    results = model(frame, conf=0.5)  # 只显示置信度≥50%的检测

    # 在帧上绘制检测结果
    annotated_frame = results[0].plot()  # 自动绘制边界框和标签

    # 写入输出视频
    out.write(annotated_frame)

# 释放资源
cap.release()
out.release()
print(f"处理完成！结果已保存到 {output_video}")
