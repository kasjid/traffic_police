"""通用的自定义模块动态注册脚本

使用方法：
    在训练脚本开头导入：
    from scripts.register_custom_modules import register_modules
    register_modules('models/CBAM')  # 指定模块目录

功能：
    - 自动扫描指定目录下的Python模块
    - 动态注册到ultralytics，无需修改源码
    - 支持多个模块目录
    - 自动处理导入错误
"""

import sys
import importlib.util
from pathlib import Path


def register_modules(*module_dirs):
    """动态注册自定义模块到ultralytics

    Args:
        *module_dirs: 一个或多个模块目录路径（相对于test目录）
                     例如: 'models/CBAM', 'models/Transformer'

    Returns:
        dict: 注册成功的模块字典 {模块名: 模块类}
    """
    try:
        from ultralytics.nn import tasks
    except ImportError as e:
        print(f"❌ 无法导入ultralytics: {e}")
        return {}

    # 获取test目录的绝对路径
    test_dir = Path(__file__).parent.parent.resolve()

    registered_modules = {}


    for module_dir in module_dirs:
        # 转换为绝对路径
        abs_module_dir = test_dir / module_dir

        if not abs_module_dir.exists():
            print(f"⚠️  模块目录不存在: {abs_module_dir}")
            continue

        print(f"\n📦 扫描模块目录: {module_dir}")

        # 扫描目录下的所有.py文件（排除__init__.py和register相关）
        py_files = [
            f for f in abs_module_dir.glob("*.py")
            if f.stem not in ['__init__', 'register_modules']
            and not f.stem.startswith('_')
        ]

        # 1. 扫描指定目录的 .py 文件
        for py_file in py_files:
            module_name = py_file.stem

            try:
                # 2.动态导入模块
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                # 3.提取模块中的所有类（以大写字母开头的）
                module_classes = [
                    (name, obj) for name, obj in vars(module).items()
                    if isinstance(obj, type) and name[0].isupper()
                    and not name.startswith('_')
                ]

                # 4.注册到ultralytics.nn.tasks
                for class_name, class_obj in module_classes:
                    setattr(tasks, class_name, class_obj) # 把 class_obj 这个类，放进 tasks.py 文件里，并取名为 class_name。
                    registered_modules[class_name] = class_obj
                    print(f"  ✅ {class_name}")

            except Exception as e:
                print(f"  ❌ 导入 {module_name} 失败: {e}")
                continue

    if registered_modules:
        print(f"\n🎉 成功注册 {len(registered_modules)} 个模块")
    else:
        print("\n⚠️  没有注册任何模块")

    return registered_modules


def list_registered_modules():
    """列出所有已注册的自定义模块"""
    try:
        from ultralytics.nn import tasks

        # 获取tasks模块中的所有类
        custom_modules = [
            name for name in dir(tasks)
            if not name.startswith('_') and name[0].isupper()
        ]

        print("\n📋 已注册的模块:")
        for name in sorted(custom_modules):
            print(f"  - {name}")

        return custom_modules
    except Exception as e:
        print(f"❌ 无法列出模块: {e}")
        return []
