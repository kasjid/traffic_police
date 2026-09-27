#!/usr/bin/env python
"""统计 Hyper-GCN 模型结构的参数量和 GFLOPs。

使用方法：
    1. 修改下面“用户配置”区域里的 CONFIG_PATH、INPUT_SHAPE 和 DEVICE。
    2. 在 Hyper-GCN 项目根目录运行：
           python cacu_params_gflops.py

示例：
    CONFIG_PATH = 'config/base/nturgbd-cross-subject/hyper_joint.yaml'
    INPUT_SHAPE = (1, 3, 64, 25, 2)

输入形状约定：
    Hyper-GCN 输入: (N, C, T, V, M)
    NTU60 常用:     (1, 3, 64, 25, 2)

注意：
    参数量和 FLOPs 只由模型结构与输入形状决定，不需要加载 .pt/.pth 权重。
"""
import os
import sys

import torch
import torch.nn as nn
import yaml


ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ===================== 用户配置 =====================
CONFIG_PATH = 'config/base/nturgbd-cross-subject/hyper_joint.yaml'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Hyper-GCN 骨架输入形状: (N, C, T, V, M)
INPUT_SHAPE = (1, 3, 64, 25, 2)
# ====================================================


def import_class(import_str):
    mod_str, _, class_str = import_str.rpartition('.')
    if not mod_str:
        raise ValueError(f'模型类路径无效: {import_str}')
    __import__(mod_str)
    return getattr(sys.modules[mod_str], class_str)


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT, path)


def human_count(value):
    for unit in ('', 'K', 'M', 'G', 'T'):
        if abs(value) < 1000:
            return f'{value:.3f}{unit}'
        value /= 1000.0
    return f'{value:.3f}P'


class HyperGCNForwardWrapper(nn.Module):
    """只返回分类结果，避免 fvcore 处理模型返回的辅助超关节点列表。"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out, _hyper_joints = self.model(x)
        return out


def patch_cpu_cuda_call():
    """兼容 Hyper-GCN 在 forward 里写死 .cuda(x.get_device()) 的情况。"""
    if DEVICE.startswith('cuda'):
        return None

    origin_cuda = torch.Tensor.cuda

    def safe_cuda(tensor, device=None, non_blocking=False, memory_format=torch.preserve_format):
        if device is None or device == -1:
            return tensor
        return origin_cuda(tensor, device=device, non_blocking=non_blocking, memory_format=memory_format)

    torch.Tensor.cuda = safe_cuda
    return origin_cuda


def main():
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count
    except ImportError:
        print('统计失败: 当前 Python 环境未安装 fvcore，请先安装 fvcore。')
        print('安装示例: pip install fvcore')
        return

    config_path = resolve_path(CONFIG_PATH)
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model_name = cfg.get('model')
    model_args = cfg.get('model_args', {})
    if not model_name:
        raise ValueError('配置文件中缺少 model 字段。')

    Model = import_class(model_name)
    model = Model(**model_args)
    model.eval().to(DEVICE)
    wrapper = HyperGCNForwardWrapper(model).eval().to(DEVICE)

    dummy_input = torch.randn(*INPUT_SHAPE, device=DEVICE)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    fvcore_params = parameter_count(model)['']

    print('=' * 70)
    print(f'配置文件: {config_path}')
    print(f'模型类: {model_name}')
    print(f'计算设备: {DEVICE}')
    print(f'输入形状: {INPUT_SHAPE}')
    print('-' * 70)
    print(f'可训练参数量: {trainable_params} ({human_count(trainable_params)})')
    print(f'总参数量: {total_params} ({human_count(total_params)})')
    print(f'fvcore 统计参数量: {fvcore_params} ({human_count(fvcore_params)})')

    origin_cuda = patch_cpu_cuda_call()
    try:
        with torch.no_grad():
            flops = FlopCountAnalysis(wrapper, dummy_input)
            total_flops = flops.total()
    except Exception as exc:
        print('-' * 70)
        print(f'FLOPs 计算失败: {exc}')
        print('请检查 CONFIG_PATH、DEVICE 和 INPUT_SHAPE 是否适配当前 Hyper-GCN 模型。')
        return
    finally:
        if origin_cuda is not None:
            torch.Tensor.cuda = origin_cuda

    print('-' * 70)
    print(f'FLOPs: {total_flops} ({total_flops / 1e9:.4f} GFLOPs)')
    print(f'FLOPs，按 Gi 单位换算: {total_flops / 1024 / 1024 / 1024:.4f} G')
    print('=' * 70)
    print('注意: fvcore 可能会忽略部分不支持的自定义/复杂算子，FLOPs 适合作为近似值。')
    print('未支持的算子:', flops.unsupported_ops())


if __name__ == '__main__':
    main()
