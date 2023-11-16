"""Job submission remote multinode test

This tests that Ray Job submission works when submitting jobs
to a remote cluster with multiple nodes.

This file is a driver script to be submitted to a Ray cluster via
the Ray Jobs API.  This is done by specifying `type: job` in
`release_tests.yaml` (as opposed to, say, `type: sdk_command`).

Test owner: architkulkarni
"""
import ray
from ray._private.test_utils import wait_for_condition
ray.init()
NUM_NODES = 5

@ray.remote(num_cpus=1)
def get_node_id():
    if False:
        print('Hello World!')
    return ray.get_runtime_context().get_node_id()
num_expected_nodes = NUM_NODES - 1
node_ids = set(ray.get([get_node_id.remote() for _ in range(100)]))

def check_num_nodes_and_spawn_tasks():
    if False:
        return 10
    node_ids.update(ray.get([get_node_id.remote() for _ in range(10)]))
    return len(node_ids) >= num_expected_nodes
wait_for_condition(check_num_nodes_and_spawn_tasks)