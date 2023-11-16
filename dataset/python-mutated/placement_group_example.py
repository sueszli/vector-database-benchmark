from pprint import pprint
import time
from ray.util.placement_group import placement_group, placement_group_table, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import ray
ray.init(num_cpus=2, num_gpus=2)
pg = placement_group([{'CPU': 1, 'GPU': 1}])
ray.get(pg.ready(), timeout=10)
(ready, unready) = ray.wait([pg.ready()], timeout=10)
print(placement_group_table(pg))
pending_pg = placement_group([{'CPU': 1}, {'GPU': 2}])
try:
    ray.get(pending_pg.ready(), timeout=5)
except Exception as e:
    print("Cannot create a placement group because {'GPU': 2} bundle cannot be created.")
    print(e)

@ray.remote(num_cpus=1)
class Actor:

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    def ready(self):
        if False:
            i = 10
            return i + 15
        pass
actor = Actor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)).remote()
ray.get(actor.ready.remote(), timeout=10)

@ray.remote(num_cpus=0, num_gpus=1)
class Actor:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        pass

    def ready(self):
        if False:
            print('Hello World!')
        pass
actor2 = Actor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=0)).remote()
ray.get(actor2.ready.remote(), timeout=10)
remove_placement_group(pg)
time.sleep(1)
pprint(placement_group_table(pg))
"\n{'bundles': {0: {'GPU': 1.0}, 1: {'CPU': 1.0}},\n'name': 'unnamed_group',\n'placement_group_id': '40816b6ad474a6942b0edb45809b39c3',\n'state': 'REMOVED',\n'strategy': 'PACK'}\n"
pg = placement_group([{'CPU': 1}, {'GPU': 1}], strategy='PACK')
remove_placement_group(pg)
pg = placement_group([{'CPU': 1}], lifetime='detached', name='global_name')
ray.get(pg.ready())
remove_placement_group(pg)
pg = placement_group([{'CPU': 1}], name='global_name')
ray.get(pg.ready())
pg = ray.util.get_placement_group('global_name')