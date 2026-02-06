from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort


def main():
    # 加载预训练模型
    model = YOLO('./runs/detect/train4/weights/best.pt')

    # 初始化跟踪器
    tracker = DeepSort(
        max_age=15,  # 减少最大未匹配帧数
        nn_budget=100,  # 特征向量池大小
        n_init=3,  # 确认轨迹前需要的连续检测次数
        max_cosine_distance=0.3  # 余弦距离阈值
    )
    cap = cv2.VideoCapture("./video/input3.mp4")

    # 获取视频属性并创建VideoWriter
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./video/output_tracking.mp4', fourcc, fps, (width, height))

    # 存储轨迹点
    track_history = {}  # {track_id: [(center_x, center_y), ...]}
    max_trajectory_length = 30  # 最大轨迹长度

    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 对当前帧进行检测
            results = model(source=frame, conf=0.65, verbose=False)

            detections = []
            detected_objects = []  # 存储检测到的对象信息

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()  # 坐标 [x1, y1, x2, y2]
                scores = results[0].boxes.conf.cpu().numpy()  # 置信度
                class_ids = results[0].boxes.cls.cpu().numpy() if results[0].boxes.cls is not None else None

                # 获取类别名称
                names = model.names

                # 构造检测输入：[[left, top, width, height], confidence]
                for i, (box, score, cls_id) in enumerate(zip(boxes, scores, class_ids)):
                    x1, y1, x2, y2 = map(int, box[:4])
                    w, h = x2 - x1, y2 - y1
                    detections.append([[x1, y1, w, h], score])

                    # 记录检测对象信息
                    obj_info = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': score,
                        'class_id': int(cls_id),
                        'class_name': names[int(cls_id)] if names else f"类别{int(cls_id)}"
                    }
                    detected_objects.append(obj_info)

            # 执行跟踪更新
            tracks = tracker.update_tracks(detections, frame=frame)

            # 按照新格式打印检测结果
            if detected_objects:
                # 统计各类别的数量和置信度
                class_counts = {}
                class_confidences = {}

                for obj in detected_objects:
                    class_name = obj['class_name']
                    confidence = obj['confidence']

                    if class_name not in class_counts:
                        class_counts[class_name] = 0
                        class_confidences[class_name] = []

                    class_counts[class_name] += 1
                    class_confidences[class_name].append(confidence)

                # 打印格式："当前帧/总帧数"帧  识别到：数量个类别  置信度：iou
                for class_name, count in class_counts.items():
                    avg_confidence = sum(class_confidences[class_name]) / len(class_confidences[class_name])
                    print(f"{frame_count}/{total_frames}帧  识别到：{count}个{class_name}  置信度：{avg_confidence:.2f}")

            # 可视化确认状态的轨迹和类别名
            for track in tracks:
                if not track.is_confirmed():
                    continue
                bbox = track.to_ltrb()

                # 计算中心点
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                track_id = track.track_id
                class_name = "Unknown"

                # 匹配检测框和跟踪ID以获取类别名
                for det_obj in detected_objects:
                    det_bbox = det_obj['bbox']
                    track_bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

                    # 检查检测框与跟踪框是否匹配（简单重叠判断）
                    if (abs(det_bbox[0] - track_bbox[0]) < 20 and
                            abs(det_bbox[1] - track_bbox[1]) < 20 and
                            abs(det_bbox[2] - track_bbox[2]) < 20 and
                            abs(det_bbox[3] - track_bbox[3]) < 20):
                        class_name = det_obj['class_name']
                        break

                # 更新轨迹点
                if track_id not in track_history:
                    track_history[track_id] = []
                track_history[track_id].append((center_x, center_y))

                # 限制轨迹长度
                if len(track_history[track_id]) > max_trajectory_length:
                    track_history[track_id].pop(0)

                # 绘制轨迹线
                if len(track_history[track_id]) > 1:
                    for j in range(1, len(track_history[track_id])):
                        pt1 = track_history[track_id][j - 1]
                        pt2 = track_history[track_id][j]
                        cv2.line(frame, pt1, pt2, (255, 255, 255), 2)

                # 绘制跟踪框和ID
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                cv2.putText(frame, f'{class_name} ID: {track_id}', (int(bbox[0]), int(bbox[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # 写入输出视频
            out.write(frame)

            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # 释放资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"\n视频处理完成！总帧数: {frame_count}")


if __name__ == "__main__":
    main()
