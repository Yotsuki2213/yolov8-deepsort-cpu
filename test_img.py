from ultralytics import YOLO

model = YOLO("runs/detect/train4/weights/best.pt")  # 加载模型
print(f"检测类别: {model.names}")
# 执行检测并保存结果（默认存到runs/detect/predict）
model.predict(source='./images/caoyu_fish3.jpg', save=True, conf=0.6)
