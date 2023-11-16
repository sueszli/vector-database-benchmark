import os
import ray
ray.init()

@ray.remote(max_restarts=4, max_task_retries=-1)
class Actor:

    def __init__(self):
        if False:
            print('Hello World!')
        self.counter = 0

    def increment_and_possibly_fail(self):
        if False:
            while True:
                i = 10
        if self.counter == 10:
            os._exit(0)
        self.counter += 1
        return self.counter
actor = Actor.remote()
for _ in range(50):
    counter = ray.get(actor.increment_and_possibly_fail.remote())
    print(counter)
for _ in range(10):
    try:
        counter = ray.get(actor.increment_and_possibly_fail.remote())
        print(counter)
    except ray.exceptions.RayActorError:
        print('FAILURE')