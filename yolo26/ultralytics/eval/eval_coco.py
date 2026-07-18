
from ultralytics import YOLO

# Load a model
model = YOLO("./yolo26s-pose.pt")  # load an official model

# Validate the model
metrics = model.val(
    data="coco-pose.yaml",
    imgsz=640,
    device=0,
    project="./runs/eval/coco",  # 强制指定保存在当前终端运行目录下的 runs/pose
    name="v26s",
)  # no arguments needed, dataset and settings remembered
print(f"map50-95: {metrics.box.map}")  # map50-95
print(f"map50: {metrics.box.map50}")  # map50
print(f"map75: {metrics.box.map75}")  # map75
# print(f"{metrics.box.maps}")  # a list containing mAP50-95 for each category
# print(f"{metrics.box.image_metrics}")  # per-image metrics dictionary for box with precision, recall, F1, TP, FP, and FN
print(f"map50-95(P): {metrics.pose.map}")  # map50-95(P)
print(f"map50(P): {metrics.pose.map50 }") # map50(P)
print(f"map75(P): {metrics.pose.map75}")  # map75(P)
# print(f"{metrics.pose.maps}")  # a list containing mAP50-95(P) for each category
# print(f"{metrics.pose.image_metrics}")  # per-image metrics dictionary for pose with precision, recall, F1, TP, FP, and FN