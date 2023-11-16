import os
import ray
ray.init(num_gpus=2)

@ray.remote(num_gpus=1)
class GPUActor:

    def ping(self):
        if False:
            while True:
                i = 10
        print('ray.get_gpu_ids(): {}'.format(ray.get_gpu_ids()))
        print('CUDA_VISIBLE_DEVICES: {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

@ray.remote(num_gpus=1)
def use_gpu():
    if False:
        i = 10
        return i + 15
    print('ray.get_gpu_ids(): {}'.format(ray.get_gpu_ids()))
    print('CUDA_VISIBLE_DEVICES: {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
gpu_actor = GPUActor.remote()
ray.get(gpu_actor.ping.remote())
ray.get(use_gpu.remote())

@ray.remote(num_gpus=1)
def use_gpu():
    if False:
        while True:
            i = 10
    import tensorflow as tf
    tf.Session()
ray.shutdown()
ray.init(num_cpus=4, num_gpus=1)

@ray.remote(num_gpus=0.25)
def f():
    if False:
        i = 10
        return i + 15
    import time
    time.sleep(1)
ray.get([f.remote() for _ in range(4)])
ray.shutdown()
ray.init(num_gpus=3)

@ray.remote(num_gpus=0.5)
class FractionalGPUActor:

    def ping(self):
        if False:
            for i in range(10):
                print('nop')
        print('ray.get_gpu_ids(): {}'.format(ray.get_gpu_ids()))
fractional_gpu_actors = [FractionalGPUActor.remote() for _ in range(3)]
[ray.get(fractional_gpu_actors[i].ping.remote()) for i in range(3)]

@ray.remote(num_gpus=1)
def leak_gpus():
    if False:
        while True:
            i = 10
    import tensorflow as tf
    tf.Session()
ray.shutdown()
import ray.util.accelerators
import ray._private.ray_constants as ray_constants
v100_resource_name = f'{ray_constants.RESOURCE_CONSTRAINT_PREFIX}{ray.util.accelerators.NVIDIA_TESLA_V100}'
ray.init(num_gpus=4, resources={v100_resource_name: 1})
from ray.util.accelerators import NVIDIA_TESLA_V100

@ray.remote(num_gpus=1, accelerator_type=NVIDIA_TESLA_V100)
def train(data):
    if False:
        print('Hello World!')
    return 'This function was run on a node with a Tesla V100 GPU'
ray.get(train.remote(1))