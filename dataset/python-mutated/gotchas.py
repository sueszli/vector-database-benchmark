import ray
import os
ray.init()

@ray.remote
def myfunc():
    if False:
        i = 10
        return i + 15
    myenv = os.environ.get('FOO')
    print(f'myenv is {myenv}')
    return 1
ray.get(myfunc.remote())
ray.shutdown()
ray.init(runtime_env={'env_vars': {'FOO': 'bar'}})

@ray.remote
def myfunc():
    if False:
        for i in range(10):
            print('nop')
    myenv = os.environ.get('FOO')
    print(f'myenv is {myenv}')
    return 1
ray.get(myfunc.remote())