import ray

@ray.remote
class Counter:

    def __init__(self):
        if False:
            print('Hello World!')
        self.count = 0

    def inc(self, n):
        if False:
            return 10
        self.count += n

    def get(self):
        if False:
            i = 10
            return i + 15
        return self.count
counter = Counter.options(name='global_counter').remote()
print(ray.get(counter.get.remote()))
counter = ray.get_actor('global_counter')
counter.inc.remote(1)
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
config = AlgorithmConfig().exploration(exploration_config={'type': 'StochasticSampling', 'constructor_arg': 'value'})