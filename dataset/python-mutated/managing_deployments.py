from ray import serve
import time
import os

@serve.deployment(name='my_deployment', num_replicas=1)
class SimpleDeployment:
    pass
serve.run(SimpleDeployment.bind())

@serve.deployment(name='my_deployment', num_replicas=2)
class SimpleDeployment:
    pass
serve.run(SimpleDeployment.bind())
serve.run(SimpleDeployment.options(num_replicas=2).bind())

@serve.deployment(num_replicas=1)
def func(*args):
    if False:
        for i in range(10):
            print('nop')
    pass
serve.run(func.bind())
serve.run(func.options(num_replicas=3).bind())
serve.run(func.options(num_replicas=1).bind())

@serve.deployment(autoscaling_config={'min_replicas': 1, 'initial_replicas': 2, 'max_replicas': 5, 'target_num_ongoing_requests_per_replica': 10})
def func(_):
    if False:
        i = 10
        return i + 15
    time.sleep(1)
    return ''
serve.run(func.bind())

@serve.deployment
class MyDeployment:

    def __init__(self, parallelism: str):
        if False:
            while True:
                i = 10
        os.environ['OMP_NUM_THREADS'] = parallelism

    def __call__(self):
        if False:
            return 10
        pass
serve.run(MyDeployment.bind('12'))