import pytest
import sys
import time
from random import random
try:
    import pytest_timeout
except ImportError:
    pytest_timeout = None
import ray
import ray.cluster_utils
from ray._private.test_utils import wait_for_condition
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

def run_mini_integration_test(cluster, pg_removal=True, num_pgs=999):
    if False:
        print('Hello World!')
    resource_quantity = num_pgs
    num_nodes = 5
    custom_resources = {'pg_custom': resource_quantity}
    num_pg = resource_quantity
    nodes = []
    for _ in range(num_nodes):
        nodes.append(cluster.add_node(num_cpus=3, num_gpus=resource_quantity, resources=custom_resources))
    cluster.wait_for_nodes()
    num_nodes = len(nodes)
    ray.init(address=cluster.address)
    while not ray.is_initialized():
        time.sleep(0.1)
    bundles = [{'GPU': 1, 'pg_custom': 1}] * num_nodes

    @ray.remote(num_cpus=0, num_gpus=1, max_calls=0)
    def mock_task():
        if False:
            for i in range(10):
                print('nop')
        time.sleep(0.1)
        return True

    @ray.remote(num_cpus=0)
    def pg_launcher(num_pgs_to_create):
        if False:
            for i in range(10):
                print('nop')
        print('Creating pgs')
        pgs = []
        for i in range(num_pgs_to_create):
            pgs.append(placement_group(bundles, strategy='STRICT_SPREAD'))
        pgs_removed = []
        pgs_unremoved = []
        if pg_removal:
            print('removing pgs')
        for pg in pgs:
            if random() < 0.5 and pg_removal:
                pgs_removed.append(pg)
            else:
                pgs_unremoved.append(pg)
        print(len(pgs_unremoved))
        tasks = []
        for pg in pgs_unremoved:
            for i in range(num_nodes):
                tasks.append(mock_task.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=i)).remote())
        if pg_removal:
            for pg in pgs_removed:
                remove_placement_group(pg)
        ray.get(tasks)
        for pg in pgs_unremoved:
            remove_placement_group(pg)
    pg_launchers = []
    for _ in range(3):
        pg_launchers.append(pg_launcher.remote(num_pg // 3))
    ray.get(pg_launchers, timeout=240)
    ray.shutdown()
    ray.init(address=cluster.address)
    cluster_resources = ray.cluster_resources()
    cluster_resources.pop('memory')
    cluster_resources.pop('object_store_memory')

    def wait_for_resource_recovered():
        if False:
            for i in range(10):
                print('nop')
        for (resource, val) in ray.available_resources().items():
            if resource in cluster_resources and cluster_resources[resource] != val:
                return False
            if '_group_' in resource:
                return False
        return True
    wait_for_condition(wait_for_resource_recovered)

@pytest.mark.parametrize('execution_number', range(1))
def test_placement_group_create_only(ray_start_cluster, execution_number):
    if False:
        while True:
            i = 10
    'PG mini integration test without remove_placement_group\n\n    When there are failures, this will help identifying if issues are\n    from removal or not.\n    '
    run_mini_integration_test(ray_start_cluster, pg_removal=False, num_pgs=333)

@pytest.mark.parametrize('execution_number', range(3))
def test_placement_group_remove_stress(ray_start_cluster, execution_number):
    if False:
        i = 10
        return i + 15
    'Full PG mini integration test that runs many\n    concurrent remove_placement_group\n    '
    run_mini_integration_test(ray_start_cluster, pg_removal=True, num_pgs=999)
if __name__ == '__main__':
    import os
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))