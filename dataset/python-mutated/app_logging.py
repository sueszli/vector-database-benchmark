import ray
ray.init()

@ray.remote
def task():
    if False:
        for i in range(10):
            print('nop')
    print('task')
ray.get(task.remote())

@ray.remote
class Actor:

    def ready(self):
        if False:
            print('Hello World!')
        print('actor')
actor = Actor.remote()
ray.get(actor.ready.remote())