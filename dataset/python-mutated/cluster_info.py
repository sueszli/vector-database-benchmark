from functools import lru_cache
from pathlib import Path

@lru_cache()
def _is_ray_cluster():
    if False:
        while True:
            i = 10
    'Checks if the bootstrap config file exists.\n\n    This will always exist if using an autoscaling cluster/started\n    with the ray cluster launcher.\n    '
    return Path('~/ray_bootstrap_config.yaml').expanduser().exists()