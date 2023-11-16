import ray
ray.init()
global_var = 3

@ray.remote
class Actor:

    def f(self):
        if False:
            while True:
                i = 10
        return global_var + 3
actor = Actor.remote()
global_var = 4
assert ray.get(actor.f.remote()) == 6

@ray.remote
class GlobalVarActor:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.global_var = 3

    def set_global_var(self, var):
        if False:
            i = 10
            return i + 15
        self.global_var = var

    def get_global_var(self):
        if False:
            print('Hello World!')
        return self.global_var

@ray.remote
class Actor:

    def __init__(self, global_var_actor):
        if False:
            for i in range(10):
                print('nop')
        self.global_var_actor = global_var_actor

    def f(self):
        if False:
            for i in range(10):
                print('nop')
        return ray.get(self.global_var_actor.get_global_var.remote()) + 3
global_var_actor = GlobalVarActor.remote()
actor = Actor.remote(global_var_actor)
ray.get(global_var_actor.set_global_var.remote(4))
assert ray.get(actor.f.remote()) == 7