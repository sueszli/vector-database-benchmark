import ray

@ray.remote
def a():
    if False:
        return 10
    x_ref = ray.put(1)
    return x_ref
x_ref = ray.get(a.remote())
try:
    print(ray.get(x_ref))
except ray.exceptions.OwnerDiedError:
    pass

@ray.remote
def a():
    if False:
        while True:
            i = 10
    return 1
x_ref = a.remote()
print(ray.get(x_ref))

@ray.remote
def b():
    if False:
        return 10
    return 1
b.options(resources={'node:127.0.0.3': 1}).remote()
b.options(scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(), soft=True)).remote()

@ray.remote
class Actor:

    def read_only(self):
        if False:
            print('Hello World!')
        import sys
        import random
        rand = random.random()
        if rand < 0.2:
            return 2 / 0
        elif rand < 0.3:
            sys.exit(1)
        return 2
actor = Actor.remote()
while True:
    try:
        print(ray.get(actor.read_only.remote()))
        break
    except ZeroDivisionError:
        pass
    except ray.exceptions.RayActorError:
        actor = Actor.remote()