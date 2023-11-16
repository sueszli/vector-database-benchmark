import ray
from ray import tune

def objective(*args):
    if False:
        while True:
            i = 10
    ray.data.range(10).show()
ray.init(num_cpus=4)
tuner = tune.Tuner(tune.with_resources(objective, {'cpu': 1}), tune_config=tune.TuneConfig(num_samples=1, max_concurrent_trials=3))
tuner.fit()