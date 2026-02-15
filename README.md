<<<<<<< HEAD
# 鱼类识别模型训练教程

## 前言

本项目处于开发中，更多细节可以联系作者邮箱，希望对您有帮助！(如果可以点个star的话，瓦塔西什么都会做的~)

> 提示：YOLO 官方文档推荐时间充裕时阅读 [Ultralytics YOLO 文档](https://docs.ultralytics.com/zh/)。

## 数据集准备（重点）

鱼类识别的核心在于数据集质量。模型只能识别你“喂”给它的内容，例如：
- 如果训练集中没有深海鱼样本，模型就无法识别深海鱼；
- 如果背景复杂且缺乏多样性，模型容易误判背景为鱼类。

因此，优化数据集是提升模型性能的关键。

### 数据采集建议

1. **光照条件多样化**：包括水下昏暗、阳光直射、黄昏微光等场景。
2. **鱼的姿态全面**：包含正面、侧面、倾斜、游动等多种姿态。
3. **拍摄视角多样**：水平视角、俯视、仰视等角度全覆盖。
4. **水域环境丰富**：淡水、海水、养殖池、河流等不同环境。
5. **鱼类大小差异**：近景（大鱼）、中景（普通大小）、远景（小鱼）。
6. **遮挡场景补充**：模拟水草、岩石等遮挡情况。
7. **多鱼共存场景**：采集画面中出现多种鱼类的样本。

## 基础环境

- Python 环境：Python 3.12

## 准备阶段

### 代码获取

项目源码已托管至 GitHub：
- GitHub 地址：[yolov8-fish-detection](https://github.com/example/yolov8-fish-detection)
- 直接下载：[点击下载](https://example.com/download)

### 环境安装

推荐使用 conda 创建独立环境：
```
conda create -n fish-detect python=3.12
conda activate fish-detect
pip install torch torchvision
pip install -r requirements.txt
```


## 训练模型

### 数据准备

将鱼类图像和标签放入 `data` 目录：
- `train`:训练集
- `val`：验证集
- `test`：测试集

标签格式需符合 YOLO 要求，推荐使用 LabelImg 或 CVAT 标注工具。

### 修改配置文件

```yaml
train: ./data/train/images
val: ./data/val/images
test: ./data/test/images
nc: 3  # 类别数量
names:
  - 'salmon'
  - 'tuna'
  - 'mackerel'
```


### 启动训练

打开[train.py]
```
from ultralytics import YOLO

model = YOLO('./yolov8n.pt')
model.train(
    data='./data/data.yaml',
    epochs=50,
    imgsz=640,
    device='CPU',
    batch=8
)
```

## 测试模型

### 图像推理测试

```python
model = YOLO("./runs/detect/train/weights/best.pt")
model.predict(source='./images/fish_sample.jpg', save=True, conf=0.6)
```


### 视频流测试（可选）

支持实时视频流检测，需配合 ffmpeg 和 mediamtx 工具。

## 训练结果分析

### 核心指标

- **精确率 (Precision)**：预测为鱼的样本中，实际是鱼的比例。
- **召回率 (Recall)**：实际是鱼的样本中，被正确预测的比例。
- **mAP50**：IoU 阈值为 0.5 时的平均精度。
- **mAP50-95**：多个 IoU 阈值下的平均精度。

### 结果图表

- 混淆矩阵：显示预测与真实的匹配情况。
- F1 曲线：展示不同置信度下的 F1 分数。
- P/R 曲线：精确率与召回率随置信度变化的趋势。

## 优化建议

### 数据集优化

- 增加样本数量，覆盖更多鱼类种类和环境。
- 清洗低质量样本，修正标注错误。
- 平衡各类别样本数量。

### 超参数调优

使用 `model.tune()` 自动优化超参数：
```python
model.tune(data='./data/data.yaml', epochs=50, iterations=10)
```


### 模型结构优化
- 若部署在边缘设备，选择轻量级模型如 YOLOv8n。
- 使用迁移学习，基于公开数据集预训练模型进行微调。
--- 

=======
测试测试测试
>>>>>>> 92737bd38872043a0f3ec8eb8fbeb2b61199ca6a
