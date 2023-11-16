from typing import Any
import ray
CACHED_FUNCTIONS = {}

def cached_remote_fn(fn: Any, **ray_remote_args) -> Any:
    if False:
        print('Hello World!')
    'Lazily defines a ray.remote function.\n\n    This is used in Datasets to avoid circular import issues with ray.remote.\n    (ray imports ray.data in order to allow ``ray.data.read_foo()`` to work,\n    which means ray.remote cannot be used top-level in ray.data).\n\n    Note: Dynamic arguments should not be passed in directly,\n    and should be set with ``options`` instead:\n    ``cached_remote_fn(fn, **static_args).options(**dynamic_args)``.\n    '
    if fn not in CACHED_FUNCTIONS:
        default_ray_remote_args = {'scheduling_strategy': 'DEFAULT', 'max_retries': -1}
        CACHED_FUNCTIONS[fn] = ray.remote(**{**default_ray_remote_args, **ray_remote_args})(fn)
    return CACHED_FUNCTIONS[fn]