
from ultralytics import YOLO

# Load a model
model = YOLO("/test/yolo26/workspace/thesis-code/yolo-v26s/exp/yolo26s-pose.pt") 

# Validate the model
metrics = model.val(
    data="datasets/coco-pose.yaml",
    imgsz=640,
    batch=256,
    device=0,
)  # no arguments needed, dataset and settings remembered
print(f"map50-95(B): {metrics.box.map}")  # map50-95
print(f"map50(B): {metrics.box.map50}")  # map50
print(f"map75(B): {metrics.box.map75}")  # map75
# print(f"{metrics.box.maps}")  # a list containing mAP50-95 for each category
# print(f"{metrics.box.image_metrics}")  # per-image metrics dictionary for box with precision, recall, F1, TP, FP, and FN
print(f"map50-95(P): {metrics.pose.map}")  # map50-95(P)
print(f"map50(P): {metrics.pose.map50 }") # map50(P)
print(f"map75(P): {metrics.pose.map75}")  # map75(P)
# print(f"{metrics.pose.maps}")  # a list containing mAP50-95(P) for each category
# print(f"{metrics.pose.image_metrics}")  # per-image metrics dictionary for pose with precision, recall, F1, TP, FP, and FN