from bigdl.nano.utils.common import invalidInputError
from bigdl.nano.utils.pytorch import TORCH_VERSION_LESS_1_12
from typing import Optional

def check_ccl(process_group_backend: Optional[str]):
    if False:
        while True:
            i = 10
    "\n    Check whether process group backend is None or 'ccl'.\n\n    :param process_group_backend: process group backend\n    "
    if process_group_backend is not None:
        invalidInputError(process_group_backend == 'ccl', f"Process group backends supported now are None and 'ccl'", f' but got {process_group_backend}.')
        pkg_name = ''
        try:
            if TORCH_VERSION_LESS_1_12:
                pkg_name = 'torch_ccl'
                import torch_ccl
            else:
                pkg_name = 'oneccl_bindings_for_pytorch'
                import oneccl_bindings_for_pytorch
        except Exception as _e:
            invalidInputError(False, f'Failed to import {pkg_name}, maybe you should install it first: pip install oneccl_bind_pt=<your pytroch version> -f https://developer.intel.com/ipex-whl-stable-cpu')