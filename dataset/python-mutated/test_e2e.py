import os
import sys
import time
from typing import Dict
import pytest
import ray
from ray._private.resource_spec import HEAD_NODE_RESOURCE_NAME
from ray._private.test_utils import run_string_as_driver_nonblocking, wait_for_condition
from ray.autoscaler.v2.sdk import get_cluster_status
from ray.cluster_utils import AutoscalingCluster
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.state.api import list_placement_groups, list_tasks

def is_head_node_from_resource_usage(usage: Dict[str, float]) -> bool:
    if False:
        while True:
            i = 10
    if HEAD_NODE_RESOURCE_NAME in usage:
        return True
    return False

def test_autoscaler_no_churn():
    if False:
        return 10
    num_cpus_per_node = 4
    expected_nodes = 6
    cluster = AutoscalingCluster(head_resources={'CPU': num_cpus_per_node}, worker_node_types={'type-1': {'resources': {'CPU': num_cpus_per_node}, 'node_config': {}, 'min_workers': 0, 'max_workers': 2 * expected_nodes}})
    driver_script = f'\nimport time\nimport ray\n@ray.remote(num_cpus=1)\ndef foo():\n  time.sleep(60)\n  return True\n\nray.init("auto")\n\nprint("start")\nassert(ray.get([foo.remote() for _ in range({num_cpus_per_node * expected_nodes})]))\nprint("end")\n'
    try:
        cluster.start()
        ray.init('auto')
        gcs_address = ray.get_runtime_context().gcs_address

        def tasks_run():
            if False:
                for i in range(10):
                    print('nop')
            tasks = list_tasks()
            assert len(tasks) > 0
            return True
        run_string_as_driver_nonblocking(driver_script)
        wait_for_condition(tasks_run)
        reached_threshold = False
        for _ in range(30):
            status = get_cluster_status(gcs_address)
            has_task_demand = len(status.resource_demands.ray_task_actor_demand) > 0
            assert len(status.active_nodes) <= expected_nodes
            if reached_threshold:
                assert not has_task_demand
            if len(status.active_nodes) == expected_nodes:
                reached_threshold = True
            time.sleep(1)
        assert reached_threshold
    finally:
        ray.shutdown()
        cluster.shutdown()

@pytest.mark.parametrize('mode', ['single_node', 'multi_node'])
def test_scheduled_task_no_pending_demand(mode):
    if False:
        i = 10
        return i + 15
    num_head_cpu = 0 if mode == 'multi_node' else 1
    cluster = AutoscalingCluster(head_resources={'CPU': num_head_cpu}, worker_node_types={'type-1': {'resources': {'CPU': 1}, 'node_config': {}, 'min_workers': 0, 'max_workers': 1}})
    driver_script = '\nimport time\nimport ray\n@ray.remote(num_cpus=1)\ndef foo():\n  return True\n\nray.init("auto")\n\nwhile True:\n    assert(ray.get(foo.remote()))\n'
    try:
        cluster.start()
        ray.init('auto')
        gcs_address = ray.get_runtime_context().gcs_address
        run_string_as_driver_nonblocking(driver_script)

        def tasks_run():
            if False:
                i = 10
                return i + 15
            tasks = list_tasks()
            assert len(tasks) > 0
            return True
        wait_for_condition(tasks_run)
        for _ in range(30):
            status = get_cluster_status(gcs_address)
            has_task_demand = len(status.resource_demands.ray_task_actor_demand) > 0
            has_task_usage = False
            for usage in status.cluster_resource_usage:
                if usage.resource_name == 'CPU':
                    has_task_usage = usage.used > 0
            print(status.cluster_resource_usage)
            print(status.resource_demands.ray_task_actor_demand)
            assert not (has_task_demand and has_task_usage), status
            time.sleep(0.1)
    finally:
        ray.shutdown()
        cluster.shutdown()

def test_placement_group_consistent():
    if False:
        for i in range(10):
            print('nop')
    import time
    cluster = AutoscalingCluster(head_resources={'CPU': 0}, worker_node_types={'type-1': {'resources': {'CPU': 1}, 'node_config': {}, 'min_workers': 0, 'max_workers': 2}})
    driver_script = '\n\nimport ray\nimport time\n# Import placement group APIs.\nfrom ray.util.placement_group import (\n    placement_group,\n    placement_group_table,\n    remove_placement_group,\n)\n\nray.init("auto")\n\n# Reserve all the CPUs of nodes, X= num of cpus, N = num of nodes\nwhile True:\n    pg = placement_group([{"CPU": 1}])\n    ray.get(pg.ready())\n    time.sleep(0.5)\n    remove_placement_group(pg)\n    time.sleep(0.5)\n'
    try:
        cluster.start()
        ray.init('auto')
        gcs_address = ray.get_runtime_context().gcs_address
        run_string_as_driver_nonblocking(driver_script)

        def pg_created():
            if False:
                return 10
            pgs = list_placement_groups()
            assert len(pgs) > 0
            return True
        wait_for_condition(pg_created)
        for _ in range(30):
            status = get_cluster_status(gcs_address)
            has_pg_demand = len(status.resource_demands.placement_group_demand) > 0
            has_pg_usage = False
            for usage in status.cluster_resource_usage:
                has_pg_usage = has_pg_usage or 'bundle' in usage.resource_name
            print(has_pg_demand, has_pg_usage)
            assert not (has_pg_demand and has_pg_usage), status
            time.sleep(0.1)
    finally:
        ray.shutdown()
        cluster.shutdown()

def test_placement_group_removal_idle_node():
    if False:
        print('Hello World!')
    cluster = AutoscalingCluster(head_resources={'CPU': 2}, worker_node_types={'type-1': {'resources': {'CPU': 2}, 'node_config': {}, 'min_workers': 0, 'max_workers': 2}})
    try:
        cluster.start()
        ray.init('auto')
        gcs_address = ray.get_runtime_context().gcs_address
        pg = placement_group([{'CPU': 2}] * 3, strategy='STRICT_SPREAD')
        ray.get(pg.ready())
        time.sleep(2)
        remove_placement_group(pg)
        from ray.autoscaler.v2.sdk import get_cluster_status

        def verify():
            if False:
                while True:
                    i = 10
            cluster_state = get_cluster_status(gcs_address)
            assert len(cluster_state.idle_nodes) == 3
            for node in cluster_state.idle_nodes:
                assert node.node_status == 'IDLE'
                assert node.resource_usage.idle_time_ms >= 1000
            return True
        wait_for_condition(verify)
    finally:
        ray.shutdown()
        cluster.shutdown()

def test_object_store_memory_idle_node(shutdown_only):
    if False:
        i = 10
        return i + 15
    ray.init()
    obj = ray.put('hello')
    gcs_address = ray.get_runtime_context().gcs_address

    def verify():
        if False:
            i = 10
            return i + 15
        state = get_cluster_status(gcs_address)
        for node in state.active_nodes:
            assert node.node_status == 'RUNNING'
            assert node.used_resources()['object_store_memory'] > 0
        assert len(state.idle_nodes) == 0
        return True
    wait_for_condition(verify)
    del obj
    import time
    time.sleep(1)

    def verify():
        if False:
            return 10
        state = get_cluster_status(gcs_address)
        for node in state.idle_nodes:
            assert node.node_status == 'IDLE'
            assert node.used_resources()['object_store_memory'] == 0
            assert node.resource_usage.idle_time_ms >= 1000
        assert len(state.active_nodes) == 0
        return True
    wait_for_condition(verify)

def test_serve_num_replica_idle_node():
    if False:
        print('Hello World!')
    cluster = AutoscalingCluster(head_resources={'CPU': 0}, worker_node_types={'type-1': {'resources': {'CPU': 4}, 'node_config': {}, 'min_workers': 0, 'max_workers': 30}}, idle_timeout_minutes=999)
    from ray import serve

    @serve.deployment(ray_actor_options={'num_cpus': 2})
    class Deployment:

        def __call__(self):
            if False:
                print('Hello World!')
            return 'hello'
    try:
        cluster.start(override_env={'RAY_SERVE_PROXY_MIN_DRAINING_PERIOD_S': '2'})
        serve.run(Deployment.options(num_replicas=10).bind())
        gcs_address = ray.get_runtime_context().gcs_address
        expected_num_workers = 5

        def verify():
            if False:
                while True:
                    i = 10
            cluster_state = get_cluster_status(gcs_address)
            assert len(cluster_state.active_nodes) == expected_num_workers + 1
            for node in cluster_state.active_nodes:
                assert node.node_status == 'RUNNING'
                if not is_head_node_from_resource_usage(node.total_resources()):
                    available = node.available_resources()
                    assert available['CPU'] == 0
            assert len(cluster_state.idle_nodes) == 0
            return True
        wait_for_condition(verify)
        serve.run(Deployment.options(num_replicas=1).bind())

        def verify():
            if False:
                i = 10
                return i + 15
            cluster_state = get_cluster_status(gcs_address)
            expected_idle_workers = expected_num_workers - 1
            assert len(cluster_state.idle_nodes) + len(cluster_state.active_nodes) == expected_num_workers + 1
            idle_nodes = []
            for node in cluster_state.idle_nodes:
                if not is_head_node_from_resource_usage(node.total_resources()):
                    available = node.available_resources()
                    if node.node_status == 'IDLE':
                        assert available['CPU'] == 4
                        idle_nodes.append(node)
            assert len(cluster_state.idle_nodes) == expected_idle_workers
            return True
        wait_for_condition(verify, timeout=15, retry_interval_ms=1000)
    finally:
        ray.shutdown()
        cluster.shutdown()

def test_non_corrupted_resources():
    if False:
        print('Hello World!')
    "\n    Test that when node's local gc happens due to object store pressure,\n    the message doesn't corrupt the resource view on the gcs.\n    See issue https://github.com/ray-project/ray/issues/39644\n    "
    num_worker_nodes = 5
    cluster = AutoscalingCluster(head_resources={'CPU': 2, 'object_store_memory': 100 * 1024 * 1024}, worker_node_types={'type-1': {'resources': {'CPU': 2}, 'node_config': {}, 'min_workers': num_worker_nodes, 'max_workers': num_worker_nodes}})
    driver_script = '\n\nimport ray\nimport time\n\nray.init("auto")\n\n@ray.remote(num_cpus=1)\ndef foo():\n    ray.put(bytearray(1024*1024* 50))\n\n\nwhile True:\n    ray.get([foo.remote() for _ in range(50)])\n'
    try:
        cluster.start(_system_config={'debug_dump_period_milliseconds': 10, 'raylet_report_resources_period_milliseconds': 10000, 'global_gc_min_interval_s': 1, 'local_gc_interval_s': 1, 'high_plasma_storage_usage': 0.2, 'raylet_check_gc_period_milliseconds': 10})
        ctx = ray.init('auto')
        gcs_address = ctx.address_info['gcs_address']
        from ray.autoscaler.v2.sdk import get_cluster_status

        def nodes_up():
            if False:
                i = 10
                return i + 15
            cluster_state = get_cluster_status(gcs_address)
            return len(cluster_state.idle_nodes) == num_worker_nodes + 1
        wait_for_condition(nodes_up)
        run_string_as_driver_nonblocking(driver_script)
        start = time.time()
        while time.time() - start < 10:
            cluster_state = get_cluster_status(gcs_address)
            assert len(cluster_state.idle_nodes) + len(cluster_state.active_nodes) == num_worker_nodes + 1
            assert cluster_state.total_resources()['CPU'] == 2 * (num_worker_nodes + 1)
    finally:
        ray.shutdown()
        cluster.shutdown()
if __name__ == '__main__':
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))