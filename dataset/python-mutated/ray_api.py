def create_ray_multiprocessing_backend():
    if False:
        print('Hello World!')
    from bigdl.nano.deps.ray.ray_backend import RayBackend
    return RayBackend()

def create_ray_strategy(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Create ray strategy.'
    from .ray_distributed import RayStrategy
    return RayStrategy(*args, **kwargs)

def distributed_ray(*args, **kwargs):
    if False:
        return 10
    from bigdl.nano.utils.common import invalidInputError
    invalidInputError(False, 'bigdl-nano no longer support ray backend when using pytorch lightning 1.4, please upgrade your pytorch lightning to 1.6.4')