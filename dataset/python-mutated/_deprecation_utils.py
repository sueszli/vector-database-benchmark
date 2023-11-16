from typing import List, Callable
import importlib
import warnings
_MESSAGE_TEMPLATE = "Usage of '{old_location}' is deprecated; please use '{new_location}' instead."

def lazy_deprecated_import(all: List[str], old_module: str, new_module: str) -> Callable:
    if False:
        return 10
    'Import utility to lazily import deprecated packages / modules / functional.\n\n    The old_module and new_module are also used in the deprecation warning defined\n    by the `_MESSAGE_TEMPLATE`.\n\n    Args:\n        all: The list of the functions that are imported. Generally, the module\'s\n            __all__ list of the module.\n        old_module: Old module location\n        new_module: New module location / Migrated location\n\n    Returns:\n        Callable to assign to the `__getattr__`\n\n    Usage:\n\n        # In the `torch/nn/quantized/functional.py`\n        from torch.nn.utils._deprecation_utils import lazy_deprecated_import\n        _MIGRATED_TO = "torch.ao.nn.quantized.functional"\n        __getattr__ = lazy_deprecated_import(\n            all=__all__,\n            old_module=__name__,\n            new_module=_MIGRATED_TO)\n    '
    warning_message = _MESSAGE_TEMPLATE.format(old_location=old_module, new_location=new_module)

    def getattr_dunder(name):
        if False:
            print('Hello World!')
        if name in all:
            warnings.warn(warning_message, RuntimeWarning)
            package = importlib.import_module(new_module)
            return getattr(package, name)
        raise AttributeError(f'Module {new_module!r} has no attribute {name!r}.')
    return getattr_dunder