import ray
from ray import serve
signal = ray.get_actor('signal123')
ray.get(signal.wait.remote())

@serve.deployment(ray_actor_options={'num_cpus': 0.1})
def f():
    if False:
        for i in range(10):
            print('nop')
    return 'hello world'
app = f.bind()