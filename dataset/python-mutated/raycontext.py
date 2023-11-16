import os
from threading import Lock
from bigdl.dllib.utils.log4Error import invalidInputError
from typing import Optional

class OrcaRayContext(object):
    _active_ray_context = None
    _lock = Lock()

    def __init__(self, runtime: str='spark', cores: int=2, num_nodes: int=1, **kwargs) -> None:
        if False:
            return 10
        import sys
        if not hasattr(sys.stdout, 'fileno'):
            sys.stdout.fileno = lambda : 1
        self.runtime = runtime
        self.initialized = False
        if runtime == 'spark':
            from bigdl.orca.ray import RayOnSparkContext
            self._ray_on_spark_context = RayOnSparkContext(**kwargs)
            self.is_local = self._ray_on_spark_context.is_local
        elif runtime == 'ray':
            self.is_local = False
            self.ray_args = kwargs.copy()
            self.num_ray_nodes = num_nodes
            self.ray_node_cpu_cores = cores
        else:
            invalidInputError(False, f'Unsupported runtime: {runtime}. Runtime must be spark or ray')
        OrcaRayContext._active_ray_context = self

    def init(self, driver_cores: int=0):
        if False:
            return 10
        if self.runtime == 'ray':
            import ray
            import ray.ray_constants as ray_constants
            address_env_var = os.environ.get(ray_constants.RAY_ADDRESS_ENVIRONMENT_VARIABLE)
            if 'address' not in self.ray_args and address_env_var is None:
                print('Creating a local Ray instance.')
                results = ray.init(num_cpus=self.ray_node_cpu_cores, **self.ray_args)
            else:
                print('Connecting to an existing ray cluster, num_cpus must not be provided.')
                results = ray.init(**self.ray_args)
        else:
            results = self._ray_on_spark_context.init(driver_cores=driver_cores)
            self.num_ray_nodes = self._ray_on_spark_context.num_ray_nodes
            self.ray_node_cpu_cores = self._ray_on_spark_context.ray_node_cpu_cores
            self.address_info = self._ray_on_spark_context.address_info
            self.redis_address = self._ray_on_spark_context.redis_address
            self.redis_password = self._ray_on_spark_context.redis_password
            self.sc = self._ray_on_spark_context.sc
        self.initialized = True
        return results

    def stop(self) -> None:
        if False:
            i = 10
            return i + 15
        if not self.initialized:
            print('The Ray cluster has not been launched.')
            return
        import ray
        ray.shutdown()
        self.initialized = False
        with OrcaRayContext._lock:
            OrcaRayContext._active_ray_context = None

    @classmethod
    def get(cls, initialize: bool=True) -> Optional['OrcaRayContext']:
        if False:
            print('Hello World!')
        if OrcaRayContext._active_ray_context:
            ray_ctx = OrcaRayContext._active_ray_context
            if initialize and (not ray_ctx.initialized):
                ray_ctx.init()
            return ray_ctx
        else:
            invalidInputError(False, 'No active RayContext. Please call init_orca_context to create a RayContext.')
        return None