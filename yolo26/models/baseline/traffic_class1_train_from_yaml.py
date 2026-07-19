from ultralytics import YOLO

# 用预训练模型
# model = YOLO("/2024212456/workspace/thesis-code/yolo-v26s/exp/models/baseline/logs/coco_full/coco_baseline/weights/best.pt")

# 从头训练
model = YOLO("models/baseline/model/yolo26s-pose_baseline.yaml") 

results = model.train(
    data="datasets/traffic_class1.yaml", # 相对于脚本运行时的目录
    imgsz=640,
    epochs=300,
    batch=32,
    workers=16,
    device=0,
    pretrained=False,
    project="/2024212456/workspace/thesis-code/yolo-v26s/exp/models/baseline/logs/traffic_class1_from_yaml",                                                                      
    name="traffic_class1_baseline_from_yaml_bs32_epoch200", 
)