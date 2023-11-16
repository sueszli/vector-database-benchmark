import os
from ludwig.backend.ray import initialize_ray
try:
    import ray
except ImportError:
    raise ImportError(' ray is not installed. In order to use auto_train please run pip install ludwig[ray]')

def _ray_init():
    if False:
        for i in range(10):
            print('nop')
    if ray.is_initialized():
        return
    os.environ.setdefault('TUNE_FORCE_TRIAL_CLEANUP_S', '120')
    initialize_ray()