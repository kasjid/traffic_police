# 在test目录下运行
from ultralytics import YOLO

model = YOLO("models/baseline/model/yolo26s-pose_baseline.yaml") # 相对于脚本运行时的目录

results = model.train(
    data="datasets/coco-pose.yaml", # 相对于脚本运行时的目录
    imgsz=640,
    epochs=400,
    batch=128,
    workers=16,
    device=1,
    pretrained=False,
    project="/2024212456/workspace/yolo26/exp/models/baseline/logs/test", # 改路径
    name="coco_baseline", # 改名字
)
