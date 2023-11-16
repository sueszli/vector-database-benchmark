from abc import ABC, abstractmethod

import torch


class LudwigTorchWrapper(torch.nn.Module):
    """Base class that establishes the contract for exporting to different file formats."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model({"image_path": x})


class BaseModelExporter(ABC):
    @abstractmethod
    def export(self, model_path, export_path, export_args_override):
        pass

    @abstractmethod
    def check_model_export(self, path):
        pass
