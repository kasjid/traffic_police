from ultralytics import YOLO

# 1. 加载模型
# 可以加载官方预训练模型，也可以加载你自定义训练出来的最佳权重 (如 runs/detect/train/weights/best.pt)
model = YOLO('/2024212456/workspace/yolo26/exp/models/baseline/logs/coco_baseline/coco_baseline/weights/best.pt')  # 加载自定义训练的权重文件

# 2. 导出为 ONNX 格式
# export() 函数会返回导出文件的路径
success = model.export(
    format='onnx',      # 指定导出格式为 onnx
    imgsz=640,          # 输入图像的尺寸 (高, 宽)，默认通常是 640
    half=False,         # 是否使用 FP16 半精度导出 (有助于加快推理速度并减小显存占用)
    dynamic=False,      # 是否启用动态输入尺寸 (如果你的输入图片大小不固定，建议设为 True)
    simplify=True,      # 强烈建议开启！使用 onnxsim 简化计算图，避免后续部署引擎解析报错
    opset=12            # ONNX 的算子集版本，TensorRT 通常推荐 12 或 13
)

print(f"ONNX 模型导出成功，路径为: {success}")