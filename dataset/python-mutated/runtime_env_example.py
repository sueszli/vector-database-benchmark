"""
This file holds code for the runtime envs documentation.

FIXME: We switched our code formatter from YAPF to Black. Check if we can enable code
formatting on this module and update the paragraph below. See issue #21318.

It ignores yapf because yapf doesn't allow comments right after code blocks,
but we put comments right after code blocks to prevent large white spaces
in the documentation.
"""
import ray
runtime_env = {'pip': ['emoji'], 'env_vars': {'TF_WARNINGS': 'none'}}
from ray.runtime_env import RuntimeEnv
runtime_env = RuntimeEnv(pip=['emoji'], env_vars={'TF_WARNINGS': 'none'})
ray.init(runtime_env=runtime_env)

@ray.remote
def f_job():
    if False:
        for i in range(10):
            print('nop')
    pass

@ray.remote
class Actor_job:

    def g(self):
        if False:
            return 10
        pass
ray.get(f_job.remote())
a = Actor_job.remote()
ray.get(a.g.remote())
ray.shutdown()
ray.init()

@ray.remote
def f():
    if False:
        print('Hello World!')
    pass

@ray.remote
class SomeClass:
    pass
f.options(runtime_env=runtime_env).remote()
actor = SomeClass.options(runtime_env=runtime_env).remote()

@ray.remote(runtime_env=runtime_env)
def g():
    if False:
        return 10
    pass

@ray.remote(runtime_env=runtime_env)
class MyClass:
    pass