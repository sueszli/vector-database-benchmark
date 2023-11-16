from abc import ABC, abstractmethod
import torch

class LudwigTorchWrapper(torch.nn.Module):
    """Base class that establishes the contract for exporting to different file formats."""

    def __init__(self, model):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.model = model

    def forward(self, x):
        if False:
            return 10
        return self.model({'image_path': x})

class BaseModelExporter(ABC):

    @abstractmethod
    def export(self, model_path, export_path, export_args_override):
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def check_model_export(self, path):
        if False:
            while True:
                i = 10
        pass