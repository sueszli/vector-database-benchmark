import ray
import os
import signal
ray.init()

@ray.remote(max_restarts=-1)
class Actor:

    def ping(self):
        if False:
            i = 10
            return i + 15
        return 'hello'

@ray.remote
class Parent:

    def generate_actors(self):
        if False:
            while True:
                i = 10
        self.child = Actor.remote()
        self.detached_actor = Actor.options(name='actor', lifetime='detached').remote()
        return (self.child, self.detached_actor, os.getpid())
parent = Parent.remote()
(actor, detached_actor, pid) = ray.get(parent.generate_actors.remote())
os.kill(pid, signal.SIGKILL)
try:
    print('actor.ping:', ray.get(actor.ping.remote()))
except ray.exceptions.RayActorError as e:
    print('Failed to submit actor call', e)
try:
    print('detached_actor.ping:', ray.get(detached_actor.ping.remote()))
except ray.exceptions.RayActorError as e:
    print('Failed to submit detached actor call', e)