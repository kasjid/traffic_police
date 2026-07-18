../# 在exp目录下运行
import sys
from pathlib import Path

# 添加exp目录到Python路径
test_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(test_dir))

# 1. 注册自定义模块（必须在导入YOLO之前）
from register_custom_modules import register_modules
register_modules('models/XXX/vx/model')  # 扫描 model 目录，注册 TDF.py

# 2. 导入YOLO
from ultralytics import YOLO

# 3. 加载模型
model = YOLO("models/XXXvx/model/yolo26s-pose_TDFvx.yaml")

# 4. 训练 (参数与 baseline 保持一致, 确保公平对比)
results = model.train(
    data="datasets/coco-pose.yaml.yaml",
    imgsz=640,                            # 输入图像尺寸
    epochs=400,                           # 训练轮数
    batch=128,                             # 批次大小
    workers=16,                           # 数据加载线程数
    device=0,                             # GPU 设备号
    pretrained=False,                     # 不使用预训练权重 (公平对比)
    project="/test/yolo26/exp/models/XXX/logs/coco_baseline",
    name="coco_XXXvx",
)

# 打印训练信息
print(f"\n{'='*60}")
print(f"初始学习率: {model.trainer.args.lr0}")
print(f"最终学习率: {model.trainer.optimizer.param_groups[0]['lr']:.6f}")
print(f"{'='*60}\n")
