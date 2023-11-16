import ray
import dask.array as da
z = da.ones(100)
from ray.util.dask import RayDaskCallback, ray_dask_get
from timeit import default_timer as timer

class MyTimerCallback(RayDaskCallback):

    def _ray_pretask(self, key, object_refs):
        if False:
            while True:
                i = 10
        start_time = timer()
        return start_time

    def _ray_posttask(self, key, result, pre_state):
        if False:
            i = 10
            return i + 15
        execution_time = timer() - pre_state
        print(f'Execution time for task {key}: {execution_time}s')
with MyTimerCallback():
    z.compute(scheduler=ray_dask_get)

def my_presubmit_cb(task, key, deps):
    if False:
        i = 10
        return i + 15
    print(f'About to submit task {key}!')
with RayDaskCallback(ray_presubmit=my_presubmit_cb):
    z.compute(scheduler=ray_dask_get)

class MyPresubmitCallback(RayDaskCallback):

    def _ray_presubmit(self, task, key, deps):
        if False:
            print('Hello World!')
        print(f'About to submit task {key}!')
with MyPresubmitCallback():
    z.compute(scheduler=ray_dask_get)
with MyTimerCallback(), MyPresubmitCallback():
    z.compute(scheduler=ray_dask_get)

@ray.remote
class SimpleCacheActor:

    def __init__(self):
        if False:
            return 10
        self.cache = {}

    def get(self, key):
        if False:
            i = 10
            return i + 15
        return self.cache[key]

    def put(self, key, value):
        if False:
            return 10
        self.cache[key] = value

class SimpleCacheCallback(RayDaskCallback):

    def __init__(self, cache_actor_handle, put_threshold=10):
        if False:
            i = 10
            return i + 15
        self.cache_actor = cache_actor_handle
        self.put_threshold = put_threshold

    def _ray_presubmit(self, task, key, deps):
        if False:
            while True:
                i = 10
        try:
            return ray.get(self.cache_actor.get.remote(str(key)))
        except KeyError:
            return None

    def _ray_pretask(self, key, object_refs):
        if False:
            print('Hello World!')
        start_time = timer()
        return start_time

    def _ray_posttask(self, key, result, pre_state):
        if False:
            for i in range(10):
                print('nop')
        execution_time = timer() - pre_state
        if execution_time > self.put_threshold:
            self.cache_actor.put.remote(str(key), result)
cache_actor = SimpleCacheActor.remote()
cache_callback = SimpleCacheCallback(cache_actor, put_threshold=2)
with cache_callback:
    z.compute(scheduler=ray_dask_get)