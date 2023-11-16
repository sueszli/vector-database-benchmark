import os
import random
import torch
import torch.distributed as dist
import torch.nn as nn
from .base_exp import BaseExp

class Exp(BaseExp):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.num_classes = 80
        self.depth = 1.0
        self.width = 1.0
        self.act = 'silu'
        self.test_size = (640, 640)
        self.test_conf = 0.01
        self.nmsthre = 0.65

    def get_model(self):
        if False:
            return 10
        from ..models import YOLOX, YOLOPAFPN, YOLOXHead

        def init_yolo(M):
            if False:
                while True:
                    i = 10
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 0.001
                    m.momentum = 0.03
        if getattr(self, 'model', None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            self.model = YOLOX(backbone, head)
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(0.01)
        self.model.train()
        return self.model