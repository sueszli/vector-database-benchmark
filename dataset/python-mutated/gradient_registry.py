"""Global GradientRegistry."""
from tensorflow.python.framework.experimental import _tape
_GRADIENT_REGISTRY_GLOBAL = _tape.GradientRegistry()

def get_global_registry():
    if False:
        while True:
            i = 10
    return _GRADIENT_REGISTRY_GLOBAL