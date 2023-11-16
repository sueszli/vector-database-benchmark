import ray
ray.init(job_config=ray.job_config.JobConfig(code_search_path=['/path/to/code']))
import ray
ray.init(job_config=ray.job_config.JobConfig(code_search_path='/path/to/jars:/path/to/pys'))
import ray
with ray.init(job_config=ray.job_config.JobConfig(code_search_path=['/path/to/code'])):
    counter_class = ray.cross_language.java_actor_class('io.ray.demo.Counter')
    counter = counter_class.remote()
    obj_ref1 = counter.increment.remote()
    assert ray.get(obj_ref1) == 1
    obj_ref2 = counter.increment.remote()
    assert ray.get(obj_ref2) == 2
    add_function = ray.cross_language.java_function('io.ray.demo.Math', 'add')
    obj_ref3 = add_function.remote(1, 2)
    assert ray.get(obj_ref3) == 3
import ray

@ray.remote
class Counter(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.value = 0

    def increment(self):
        if False:
            i = 10
            return i + 15
        self.value += 1
        return self.value

@ray.remote
def add(a, b):
    if False:
        return 10
    return a + b
import ray

@ray.remote
def py_return_input(v):
    if False:
        print('Hello World!')
    return v
import ray

@ray.remote
def raise_exception():
    if False:
        return 10
    1 / 0
import ray
with ray.init(job_config=ray.job_config.JobConfig(code_search_path=['/path/to/ray_exception'])):
    obj_ref = ray.cross_language.java_function('io.ray.demo.MyRayClass', 'raiseExceptionFromPython').remote()
    ray.get(obj_ref)