import memray
import ray

@ray.remote
class Actor:

    def __init__(self):
        if False:
            return 10
        memray.Tracker(f'/tmp/ray/session_latest/logs/{ray.get_runtime_context().get_actor_id()}_mem_profile.bin').__enter__()
        self.arr = [bytearray(b'1' * 1000000)]

    def append(self):
        if False:
            return 10
        self.arr.append(bytearray(b'1' * 1000000))
a = Actor.remote()
ray.get(a.append.remote())
import memray
import ray

@ray.remote
def task():
    if False:
        for i in range(10):
            print('nop')
    with memray.Tracker(f'/tmp/ray/session_latest/logs/{ray.get_runtime_context().get_task_id()}_mem_profile.bin'):
        arr = bytearray(b'1' * 1000000)
ray.get(task.remote())