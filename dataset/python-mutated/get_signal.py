import ray
from ray import serve

@serve.deployment
class A:

    def __call__(self):
        if False:
            for i in range(10):
                print('nop')
        signal = ray.get_actor('signal123')
        ray.get(signal.wait.remote())
app = A.bind()