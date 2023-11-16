import sys
import time
import numpy as np
import pytest
import ray
import ray._private.ray_constants as ray_constants
from ray._private.test_utils import get_other_nodes

@pytest.mark.parametrize('ray_start_cluster', [{'num_cpus': 4, 'num_nodes': 3, 'do_init': True}], indirect=True)
def test_actor_creation_node_failure(ray_start_cluster):
    if False:
        i = 10
        return i + 15
    cluster = ray_start_cluster

    @ray.remote
    class Child:

        def __init__(self, death_probability):
            if False:
                for i in range(10):
                    print('nop')
            self.death_probability = death_probability

        def get_probability(self):
            if False:
                print('Hello World!')
            return self.death_probability

        def ping(self):
            if False:
                for i in range(10):
                    print('nop')
            exit_chance = np.random.rand()
            if exit_chance < self.death_probability:
                sys.exit(-1)
    num_children = 25
    death_probability = 0.5
    children = [Child.remote(death_probability) for _ in range(num_children)]
    while len(cluster.list_all_nodes()) > 1:
        for j in range(2):
            children_out = [child.ping.remote() for child in children]
            (ready, _) = ray.wait(children_out, num_returns=len(children_out), timeout=5 * 60.0)
            assert len(ready) == len(children_out)
            for (i, out) in enumerate(children_out):
                try:
                    ray.get(out)
                except ray.exceptions.RayActorError:
                    children[i] = Child.remote(death_probability)
            children_out = [child.get_probability.remote() for child in children]
            (ready, _) = ray.wait(children_out, num_returns=len(children_out), timeout=5 * 60.0)
            assert len(ready) == len(children_out)
        cluster.remove_node(get_other_nodes(cluster, True)[-1])

def test_driver_lives_sequential(ray_start_regular):
    if False:
        return 10
    ray._private.worker._global_node.kill_raylet()
    ray._private.worker._global_node.kill_log_monitor()
    ray._private.worker._global_node.kill_monitor()

def test_driver_lives_parallel(ray_start_regular):
    if False:
        i = 10
        return i + 15
    all_processes = ray._private.worker._global_node.all_processes
    process_infos = all_processes[ray_constants.PROCESS_TYPE_RAYLET] + all_processes[ray_constants.PROCESS_TYPE_LOG_MONITOR] + all_processes[ray_constants.PROCESS_TYPE_MONITOR]
    for process_info in process_infos:
        process_info.process.terminate()
    time.sleep(0.1)
    for process_info in process_infos:
        process_info.process.kill()
    for process_info in process_infos:
        process_info.process.wait()

def test_dying_worker(ray_start_2_cpus):
    if False:
        print('Hello World!')

    @ray.remote(num_cpus=0, max_calls=1)
    def foo():
        if False:
            return 10
        pass
    for _ in range(20):
        ray.get([foo.remote() for _ in range(5)])
    assert ray._private.services.remaining_processes_alive()
if __name__ == '__main__':
    import os
    import pytest
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))