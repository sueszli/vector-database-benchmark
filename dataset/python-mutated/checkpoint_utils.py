import os
import shutil
import tempfile
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple
import torch.distributed as dist

def with_temp_dir(func: Optional[Callable]=None) -> Optional[Callable]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Wrapper to initialize temp directory for distributed checkpoint.\n    '
    assert func is not None

    @wraps(func)
    def wrapper(self, *args: Tuple[object], **kwargs: Dict[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        if dist.get_rank() == 0:
            temp_dir = tempfile.mkdtemp()
            print(f'Using temp directory: {temp_dir}')
        else:
            temp_dir = ''
        object_list = [temp_dir]
        os.sync()
        dist.broadcast_object_list(object_list)
        self.temp_dir = object_list[0]
        os.sync()
        try:
            func(self, *args, **kwargs)
        finally:
            if dist.get_rank() == 0:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
    return wrapper