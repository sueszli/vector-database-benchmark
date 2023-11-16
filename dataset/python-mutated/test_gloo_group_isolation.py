from python.ray.util.collective.types import Backend
import ray
import ray.util.collective as col
import time

@ray.remote
class Worker:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def init_gloo_group(rank: int, world_size: int, group_name: str):
        if False:
            i = 10
            return i + 15
        col.init_collective_group(world_size, rank, Backend.GLOO, group_name)
        return True

def test_two_groups_in_one_cluster(ray_start_regular_shared):
    if False:
        print('Hello World!')
    w1 = Worker.remote()
    ret1 = w1.init_gloo_group.remote(1, 0, 'name_1')
    w2 = Worker.remote()
    ret2 = w2.init_gloo_group.remote(1, 0, 'name_2')
    assert ray.get(ret1)
    assert ray.get(ret2)

def test_failure_when_initializing(shutdown_only):
    if False:
        while True:
            i = 10
    ray.init()
    w1 = Worker.remote()
    ret1 = w1.init_gloo_group.remote(2, 0, 'name_1')
    ray.wait([ret1], timeout=1)
    time.sleep(5)
    ray.shutdown()
    ray.init()
    w2 = Worker.remote()
    ret2 = w2.init_gloo_group.remote(1, 0, 'name_1')
    assert ray.get(ret2)
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', '-x', __file__]))