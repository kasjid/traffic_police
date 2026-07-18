import torch
import torch.nn as nn
import torch
from torch import nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect, Pose, Pose26, RealNVP

__all__ = ("Module",)

class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
    
    def forward(self, x):
       return x
    
    @staticmethod
    def parse_channel(ch, f, args):
        """自定义通道解析函数 —
        """
        # f = [17, 21, 25]，取各层输出通道
        c2 = args[0]  # 对 Detect 子类 c2 不影响实际行为
        c1 = c2
        return c2, args
    