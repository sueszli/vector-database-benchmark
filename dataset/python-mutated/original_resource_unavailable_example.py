import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
ray.init(num_cpus=2)
pg = placement_group([{'CPU': 2}])
ray.get(pg.ready())

@ray.remote(num_cpus=2)
def f():
    if False:
        for i in range(10):
            print('nop')
    return True
f.remote()
f.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)).remote()