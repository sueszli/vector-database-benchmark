import inspect
import torch.nn as nn
import nni.nas.nn.pytorch.layers as nas_nn
_original_classes = {}

def remove_inject_pytorch_nn():
    if False:
        return 10
    for name in _original_classes:
        setattr(nn, name, _original_classes[name])

def inject_pytorch_nn():
    if False:
        return 10
    for name in dir(nn):
        if inspect.isclass(getattr(nn, name)) and issubclass(getattr(nn, name), nn.Module):
            _original_classes[name] = getattr(nn, name)
            setattr(nn, name, getattr(nas_nn, name))