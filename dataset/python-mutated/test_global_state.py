import os
import time
import pytest
import ray
import ray._private.gcs_utils as gcs_utils
import ray._private.ray_constants
from ray._raylet import GcsClient
from ray.core.generated import autoscaler_pb2
from ray._private.test_utils import convert_actor_state, make_global_state_accessor, wait_for_condition
try:
    import pytest_timeout
except ImportError:
    pytest_timeout = None

@pytest.mark.skipif(pytest_timeout is None, reason='Timeout package not installed; skipping test that may hang.')
@pytest.mark.timeout(30)
def test_replenish_resources(ray_start_regular):
    if False:
        for i in range(10):
            print('nop')
    cluster_resources = ray.cluster_resources()
    available_resources = ray.available_resources()
    assert cluster_resources == available_resources

    @ray.remote
    def cpu_task():
        if False:
            print('Hello World!')
        pass
    ray.get(cpu_task.remote())
    resources_reset = False
    while not resources_reset:
        available_resources = ray.available_resources()
        resources_reset = cluster_resources == available_resources
    assert resources_reset

@pytest.mark.skipif(pytest_timeout is None, reason='Timeout package not installed; skipping test that may hang.')
@pytest.mark.timeout(30)
def test_uses_resources(ray_start_regular):
    if False:
        while True:
            i = 10
    cluster_resources = ray.cluster_resources()

    @ray.remote
    def cpu_task():
        if False:
            i = 10
            return i + 15
        time.sleep(1)
    cpu_task.remote()
    resource_used = False
    while not resource_used:
        available_resources = ray.available_resources()
        resource_used = available_resources.get('CPU', 0) == cluster_resources.get('CPU', 0) - 1
    assert resource_used

@pytest.mark.skipif(pytest_timeout is None, reason='Timeout package not installed; skipping test that may hang.')
@pytest.mark.timeout(120)
def test_add_remove_cluster_resources(ray_start_cluster_head):
    if False:
        while True:
            i = 10
    'Tests that Global State API is consistent with actual cluster.'
    cluster = ray_start_cluster_head
    assert ray.cluster_resources()['CPU'] == 1
    nodes = []
    nodes += [cluster.add_node(num_cpus=1)]
    cluster.wait_for_nodes()
    assert ray.cluster_resources()['CPU'] == 2
    cluster.remove_node(nodes.pop())
    cluster.wait_for_nodes()
    assert ray.cluster_resources()['CPU'] == 1
    for i in range(5):
        nodes += [cluster.add_node(num_cpus=1)]
    cluster.wait_for_nodes()
    assert ray.cluster_resources()['CPU'] == 6

def test_global_state_actor_table(ray_start_regular):
    if False:
        print('Hello World!')

    @ray.remote
    class Actor:

        def ready(self):
            if False:
                for i in range(10):
                    print('nop')
            return os.getpid()
    assert len(ray._private.state.actors()) == 0

    def get_actor_table_data(field):
        if False:
            print('Hello World!')
        return list(ray._private.state.actors().values())[0][field]
    a = Actor.remote()
    pid = ray.get(a.ready.remote())
    assert len(ray._private.state.actors()) == 1
    assert get_actor_table_data('Pid') == pid
    del a
    dead_state = convert_actor_state(gcs_utils.ActorTableData.DEAD)
    for _ in range(10):
        if get_actor_table_data('State') == dead_state:
            break
        else:
            time.sleep(0.5)
    assert get_actor_table_data('State') == dead_state

def test_global_state_worker_table(ray_start_regular):
    if False:
        for i in range(10):
            print('nop')

    def worker_initialized():
        if False:
            print('Hello World!')
        workers_data = ray._private.state.workers()
        return len(workers_data) == 1
    wait_for_condition(worker_initialized)

def test_global_state_actor_entry(ray_start_regular):
    if False:
        return 10

    @ray.remote
    class Actor:

        def ready(self):
            if False:
                i = 10
                return i + 15
            pass
    assert len(ray._private.state.actors()) == 0
    a = Actor.remote()
    b = Actor.remote()
    ray.get(a.ready.remote())
    ray.get(b.ready.remote())
    assert len(ray._private.state.actors()) == 2
    a_actor_id = a._actor_id.hex()
    b_actor_id = b._actor_id.hex()
    assert ray._private.state.actors(actor_id=a_actor_id)['ActorID'] == a_actor_id
    assert ray._private.state.actors(actor_id=a_actor_id)['State'] == convert_actor_state(gcs_utils.ActorTableData.ALIVE)
    assert ray._private.state.actors(actor_id=b_actor_id)['ActorID'] == b_actor_id
    assert ray._private.state.actors(actor_id=b_actor_id)['State'] == convert_actor_state(gcs_utils.ActorTableData.ALIVE)

def test_node_name_cluster(ray_start_cluster):
    if False:
        while True:
            i = 10
    cluster = ray_start_cluster
    cluster.add_node(node_name='head_node', include_dashboard=False)
    head_context = ray.init(address=cluster.address, include_dashboard=False)
    cluster.add_node(node_name='worker_node', include_dashboard=False)
    cluster.wait_for_nodes()
    global_state_accessor = make_global_state_accessor(head_context)
    node_table = global_state_accessor.get_node_table()
    assert len(node_table) == 2
    for node in node_table:
        if node['NodeID'] == head_context.address_info['node_id']:
            assert node['NodeName'] == 'head_node'
        else:
            assert node['NodeName'] == 'worker_node'
    global_state_accessor.disconnect()
    ray.shutdown()
    cluster.shutdown()

def test_node_name_init():
    if False:
        for i in range(10):
            print('nop')
    new_head_context = ray.init(_node_name='new_head_node', include_dashboard=False)
    global_state_accessor = make_global_state_accessor(new_head_context)
    node = global_state_accessor.get_node_table()[0]
    assert node['NodeName'] == 'new_head_node'
    ray.shutdown()

def test_no_node_name():
    if False:
        while True:
            i = 10
    new_head_context = ray.init(include_dashboard=False)
    global_state_accessor = make_global_state_accessor(new_head_context)
    node = global_state_accessor.get_node_table()[0]
    assert node['NodeName'] == ray.util.get_node_ip_address()
    ray.shutdown()

@pytest.mark.parametrize('max_shapes', [0, 2, -1])
def test_load_report(shutdown_only, max_shapes):
    if False:
        print('Hello World!')
    resource1 = 'A'
    resource2 = 'B'
    cluster = ray.init(num_cpus=1, resources={resource1: 1}, _system_config={'max_resource_shapes_per_load_report': max_shapes})
    global_state_accessor = make_global_state_accessor(cluster)

    @ray.remote
    def sleep():
        if False:
            i = 10
            return i + 15
        time.sleep(1000)
    sleep.remote()
    for _ in range(3):
        sleep.remote()
        sleep.options(resources={resource1: 1}).remote()
        sleep.options(resources={resource2: 1}).remote()

    class Checker:

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self.report = None

        def check_load_report(self):
            if False:
                i = 10
                return i + 15
            message = global_state_accessor.get_all_resource_usage()
            if message is None:
                return False
            resource_usage = gcs_utils.ResourceUsageBatchData.FromString(message)
            self.report = resource_usage.resource_load_by_shape.resource_demands
            if max_shapes == 0:
                return True
            elif max_shapes == 2:
                return len(self.report) >= 2
            else:
                return len(self.report) >= 3
    checker = Checker()
    wait_for_condition(checker.check_load_report)
    if max_shapes != -1:
        assert len(checker.report) <= max_shapes
    print(checker.report)
    if max_shapes > 0:
        for demand in checker.report:
            if resource2 in demand.shape:
                assert demand.num_infeasible_requests_queued > 0
                assert demand.num_ready_requests_queued == 0
            else:
                assert demand.num_ready_requests_queued > 0
                assert demand.num_infeasible_requests_queued == 0
    global_state_accessor.disconnect()

def test_placement_group_load_report(ray_start_cluster):
    if False:
        while True:
            i = 10
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=4)
    global_state_accessor = make_global_state_accessor(ray.init(address=cluster.address))

    class PgLoadChecker:

        def nothing_is_ready(self):
            if False:
                return 10
            resource_usage = self._read_resource_usage()
            if not resource_usage:
                return False
            if resource_usage.HasField('placement_group_load'):
                pg_load = resource_usage.placement_group_load
                return len(pg_load.placement_group_data) == 2
            return False

        def only_first_one_ready(self):
            if False:
                print('Hello World!')
            resource_usage = self._read_resource_usage()
            if not resource_usage:
                return False
            if resource_usage.HasField('placement_group_load'):
                pg_load = resource_usage.placement_group_load
                return len(pg_load.placement_group_data) == 1
            return False

        def two_infeasible_pg(self):
            if False:
                i = 10
                return i + 15
            resource_usage = self._read_resource_usage()
            if not resource_usage:
                return False
            if resource_usage.HasField('placement_group_load'):
                pg_load = resource_usage.placement_group_load
                return len(pg_load.placement_group_data) == 2
            return False

        def _read_resource_usage(self):
            if False:
                i = 10
                return i + 15
            message = global_state_accessor.get_all_resource_usage()
            if message is None:
                return False
            resource_usage = gcs_utils.ResourceUsageBatchData.FromString(message)
            return resource_usage
    checker = PgLoadChecker()
    pg_feasible = ray.util.placement_group([{'A': 1}])
    pg_infeasible = ray.util.placement_group([{'B': 1}])
    (_, unready) = ray.wait([pg_feasible.ready(), pg_infeasible.ready()], timeout=0)
    assert len(unready) == 2
    wait_for_condition(checker.nothing_is_ready)
    cluster.add_node(resources={'A': 1})
    ray.get(pg_feasible.ready())
    wait_for_condition(checker.only_first_one_ready)
    pg_infeasible_second = ray.util.placement_group([{'C': 1}])
    (_, unready) = ray.wait([pg_infeasible_second.ready()], timeout=0)
    assert len(unready) == 1
    wait_for_condition(checker.two_infeasible_pg)
    global_state_accessor.disconnect()

def test_backlog_report(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    cluster = ray.init(num_cpus=1, _system_config={'max_pending_lease_requests_per_scheduling_category': 1})
    global_state_accessor = make_global_state_accessor(cluster)

    @ray.remote(num_cpus=1)
    def foo(x):
        if False:
            print('Hello World!')
        print('.')
        time.sleep(x)
        return None

    def backlog_size_set():
        if False:
            return 10
        message = global_state_accessor.get_all_resource_usage()
        if message is None:
            return False
        resource_usage = gcs_utils.ResourceUsageBatchData.FromString(message)
        aggregate_resource_load = resource_usage.resource_load_by_shape.resource_demands
        if len(aggregate_resource_load) == 1:
            backlog_size = aggregate_resource_load[0].backlog_size
            print(backlog_size)
            return backlog_size > 0
        return False
    refs = [foo.remote(0.5)]
    refs.extend([foo.remote(1000) for _ in range(9)])
    ray.get(refs[0])
    wait_for_condition(backlog_size_set, timeout=2)
    global_state_accessor.disconnect()

def test_default_load_reports(shutdown_only):
    if False:
        print('Hello World!')
    'Despite the fact that default actors release their cpu after being\n    placed, they should still require 1 CPU for laod reporting purposes.\n    https://github.com/ray-project/ray/issues/26806\n    '
    cluster = ray.init(num_cpus=0)
    global_state_accessor = make_global_state_accessor(cluster)

    @ray.remote
    def foo():
        if False:
            while True:
                i = 10
        return None

    @ray.remote
    class Foo:
        pass

    def actor_and_task_queued_together():
        if False:
            return 10
        message = global_state_accessor.get_all_resource_usage()
        if message is None:
            return False
        resource_usage = gcs_utils.ResourceUsageBatchData.FromString(message)
        aggregate_resource_load = resource_usage.resource_load_by_shape.resource_demands
        print(f'Num shapes {len(aggregate_resource_load)}')
        if len(aggregate_resource_load) == 1:
            num_infeasible = aggregate_resource_load[0].num_infeasible_requests_queued
            print(f'num in shape {num_infeasible}')
            return num_infeasible == 2
        return False
    handle = Foo.remote()
    ref = foo.remote()
    wait_for_condition(actor_and_task_queued_together, timeout=2)
    global_state_accessor.disconnect()
    del handle
    del ref

def test_heartbeat_ip(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    cluster = ray.init(num_cpus=1)
    global_state_accessor = make_global_state_accessor(cluster)
    self_ip = ray.util.get_node_ip_address()

    def self_ip_is_set():
        if False:
            print('Hello World!')
        message = global_state_accessor.get_all_resource_usage()
        if message is None:
            return False
        resource_usage = gcs_utils.ResourceUsageBatchData.FromString(message)
        resources_data = resource_usage.batch[0]
        return resources_data.node_manager_address == self_ip
    wait_for_condition(self_ip_is_set, timeout=2)
    global_state_accessor.disconnect()

def test_next_job_id(ray_start_regular):
    if False:
        return 10
    job_id_1 = ray._private.state.next_job_id()
    job_id_2 = ray._private.state.next_job_id()
    assert job_id_1.int() + 1 == job_id_2.int()

def test_get_draining_nodes(ray_start_cluster):
    if False:
        i = 10
        return i + 15
    cluster = ray_start_cluster
    cluster.add_node()
    ray.init(address=cluster.address)
    cluster.add_node(resources={'worker': 1})
    cluster.wait_for_nodes()

    @ray.remote
    def get_node_id():
        if False:
            return 10
        return ray.get_runtime_context().get_node_id()
    worker_node_id = ray.get(get_node_id.options(resources={'worker': 1}).remote())
    assert ray._private.state.state.get_draining_nodes() == set()

    @ray.remote(num_cpus=1, resources={'worker': 1})
    class Actor:

        def ping(self):
            if False:
                print('Hello World!')
            pass
    actor = Actor.remote()
    ray.get(actor.ping.remote())
    gcs_client = GcsClient(address=ray.get_runtime_context().gcs_address)
    is_accepted = gcs_client.drain_node(worker_node_id, autoscaler_pb2.DrainNodeReason.Value('DRAIN_NODE_REASON_PREEMPTION'), 'preemption')
    assert is_accepted
    wait_for_condition(lambda : ray._private.state.state.get_draining_nodes() == {worker_node_id})
    ray.kill(actor)
    wait_for_condition(lambda : ray._private.state.state.get_draining_nodes() == set())
if __name__ == '__main__':
    import sys
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))