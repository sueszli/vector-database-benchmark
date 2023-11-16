import sys
import time
import numpy as np
import pytest
import ray
import ray._private.ray_constants as ray_constants
from ray._private.test_utils import get_other_nodes

@pytest.mark.skip(reason='No reconstruction for objects placed in plasma yet')
@pytest.mark.parametrize('ray_start_cluster', [{'num_cpus': 1, 'num_nodes': 4, 'object_store_memory': 1000 * 1024 * 1024, '_system_config': {'object_manager_pull_timeout_ms': 1000, 'object_manager_push_timeout_ms': 1000}}], indirect=True)
def test_object_reconstruction(ray_start_cluster):
    if False:
        while True:
            i = 10
    cluster = ray_start_cluster

    @ray.remote
    def large_value():
        if False:
            for i in range(10):
                print('nop')
        time.sleep(0.1)
        return np.zeros(10 * 1024 * 1024)

    @ray.remote
    def g(x):
        if False:
            for i in range(10):
                print('nop')
        return
    time.sleep(0.1)
    worker_nodes = get_other_nodes(cluster)
    assert len(worker_nodes) > 0
    component_type = ray_constants.PROCESS_TYPE_RAYLET
    for node in worker_nodes:
        process = node.all_processes[component_type][0].process
        num_tasks = len(worker_nodes)
        xs = [large_value.remote() for _ in range(num_tasks)]
        for x in xs:
            ray.get(x)
            ray._private.internal_api.free([x], local_only=True)
        process.terminate()
        time.sleep(1)
        process.kill()
        process.wait()
        assert not process.poll() is None
        print('F', xs)
        xs = [g.remote(x) for x in xs]
        print('G', xs)
        ray.get(xs)

@pytest.mark.parametrize('ray_start_cluster', [{'num_cpus': 4, 'num_nodes': 3, 'do_init': True}], indirect=True)
def test_actor_creation_node_failure(ray_start_cluster):
    if False:
        print('Hello World!')
    cluster = ray_start_cluster

    @ray.remote
    class Child:

        def __init__(self, death_probability):
            if False:
                print('Hello World!')
            self.death_probability = death_probability

        def ping(self):
            if False:
                i = 10
                return i + 15
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
        cluster.remove_node(get_other_nodes(cluster, True)[-1])
if __name__ == '__main__':
    import os
    import pytest
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))