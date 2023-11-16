import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
ray.init(num_cpus=2)
pg = placement_group([{'CPU': 2}])
ray.get(pg.ready())

@ray.remote(num_cpus=1)
def child():
    if False:
        i = 10
        return i + 15
    import time
    time.sleep(5)

@ray.remote(num_cpus=1)
def parent():
    if False:
        return 10
    ray.get(child.remote())
ray.get(parent.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_capture_child_tasks=True)).remote())

@ray.remote
def parent():
    if False:
        print('Hello World!')
    ray.get(child.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=None)).remote())
try:
    ray.get(parent.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_capture_child_tasks=True)).remote(), timeout=5)
except Exception as e:
    print("Couldn't create a child task!")
    print(e)