import logging
import time
import pytest
import ray
from ray._private.test_utils import SignalActor, generate_system_config_map, wait_for_condition
from ray.autoscaler._private.monitor import Monitor
from ray.autoscaler.sdk import request_resources
from ray.cluster_utils import Cluster
from ray.util.placement_group import placement_group, remove_placement_group
logger = logging.getLogger(__name__)

def test_cluster():
    if False:
        while True:
            i = 10
    'Basic test for adding and removing nodes in cluster.'
    g = Cluster(initialize_head=False)
    node = g.add_node()
    node2 = g.add_node()
    assert node.remaining_processes_alive()
    assert node2.remaining_processes_alive()
    g.remove_node(node2)
    g.remove_node(node)
    assert not any((n.any_processes_alive() for n in [node, node2]))
    g.shutdown()

def test_shutdown():
    if False:
        print('Hello World!')
    g = Cluster(initialize_head=False)
    node = g.add_node()
    node2 = g.add_node()
    g.shutdown()
    assert not any((n.any_processes_alive() for n in [node, node2]))

@pytest.mark.parametrize('ray_start_cluster_head', [generate_system_config_map(health_check_initial_delay_ms=0, health_check_period_ms=1000, health_check_failure_threshold=3, object_timeout_milliseconds=12345)], indirect=True)
def test_system_config(ray_start_cluster_head):
    if False:
        i = 10
        return i + 15
    'Checks that the internal configuration setting works.\n\n    We set the cluster to timeout nodes after 2 seconds of no timeouts. We\n    then remove a node, wait for 1 second to check that the cluster is out\n    of sync, then wait another 2 seconds (giving 1 second of leeway) to check\n    that the client has timed out. We also check to see if the config is set.\n    '
    cluster = ray_start_cluster_head
    worker = cluster.add_node()
    cluster.wait_for_nodes()

    @ray.remote
    def f():
        if False:
            print('Hello World!')
        assert ray._config.object_timeout_milliseconds() == 12345
        assert ray._config.health_check_initial_delay_ms() == 0
        assert ray._config.health_check_failure_threshold() == 3
        assert ray._config.health_check_period_ms() == 1000
    ray.get([f.remote() for _ in range(5)])
    cluster.remove_node(worker, allow_graceful=False)
    time.sleep(1)
    assert ray.cluster_resources()['CPU'] == 2

    def _node_removed():
        if False:
            print('Hello World!')
        return ray.cluster_resources()['CPU'] == 1
    wait_for_condition(_node_removed, timeout=3)

def setup_monitor(address):
    if False:
        print('Hello World!')
    monitor = Monitor(address, None)
    return monitor

def assert_correct_pg(pg_response_data, pg_demands, strategy):
    if False:
        while True:
            i = 10
    assert len(pg_response_data) == 1
    pg_response_data = pg_response_data[0]
    strategy_mapping_dict_protobuf = {'PACK': 0, 'SPREAD': 1, 'STRICT_PACK': 2, 'STRICT_SPREAD': 3}
    assert pg_response_data.strategy == strategy_mapping_dict_protobuf[strategy]
    assert pg_response_data.creator_job_id
    assert pg_response_data.creator_actor_id
    assert pg_response_data.creator_actor_dead
    assert pg_response_data.placement_group_id
    for (i, bundle) in enumerate(pg_demands):
        assert pg_response_data.bundles[i].unit_resources == bundle
        assert pg_response_data.bundles[i].bundle_id.placement_group_id

def verify_load_metrics(monitor, expected_resource_usage=None, timeout=30):
    if False:
        for i in range(10):
            print('nop')
    request_resources(num_cpus=42)
    pg_demands = [{'GPU': 2}, {'extra_resource': 2}]
    strategy = 'STRICT_PACK'
    pg = placement_group(pg_demands, strategy=strategy)
    pg.ready()
    time.sleep(2)
    monitor.event_summarizer.clear = lambda *a: None
    visited_atleast_once = [set(), set()]
    while True:
        monitor.update_load_metrics()
        monitor.update_resource_requests()
        monitor.update_event_summary()
        resource_usage = monitor.load_metrics._get_resource_usage()
        req = monitor.load_metrics.resource_requests
        assert req == [{'CPU': 1}] * 42, req
        pg_response_data = monitor.load_metrics.pending_placement_groups
        assert_correct_pg(pg_response_data, pg_demands, strategy)
        if 'memory' in resource_usage[0]:
            del resource_usage[0]['memory']
            visited_atleast_once[0].add('memory')
        if 'object_store_memory' in resource_usage[0]:
            del resource_usage[0]['object_store_memory']
            visited_atleast_once[0].add('object_store_memory')
        if 'memory' in resource_usage[1]:
            del resource_usage[1]['memory']
            visited_atleast_once[1].add('memory')
        if 'object_store_memory' in resource_usage[1]:
            del resource_usage[1]['object_store_memory']
            visited_atleast_once[1].add('object_store_memory')
        for key in list(resource_usage[0].keys()):
            if key.startswith('node:'):
                del resource_usage[0][key]
                visited_atleast_once[0].add('node:')
        for key in list(resource_usage[1].keys()):
            if key.startswith('node:'):
                del resource_usage[1][key]
                visited_atleast_once[1].add('node:')
        if expected_resource_usage is None:
            if all((x for x in resource_usage[0:])):
                break
        elif all((x == y for (x, y) in zip(resource_usage, expected_resource_usage))):
            break
        else:
            timeout -= 1
            time.sleep(1)
        if timeout <= 0:
            raise ValueError('Timeout. {} != {}'.format(resource_usage, expected_resource_usage))
    assert any(('Resized to' in x for x in monitor.event_summarizer.summary()))
    assert visited_atleast_once[0] == {'memory', 'object_store_memory', 'node:'}
    assert visited_atleast_once[0] == visited_atleast_once[1]
    remove_placement_group(pg)
    return resource_usage

@pytest.mark.parametrize('ray_start_cluster_head', [{'num_cpus': 1}, {'num_cpus': 2}], indirect=True)
def test_heartbeats_single(ray_start_cluster_head):
    if False:
        for i in range(10):
            print('nop')
    'Unit test for `Cluster.wait_for_nodes`.\n\n    Test proper metrics.\n    '
    cluster = ray_start_cluster_head
    monitor = setup_monitor(cluster.gcs_address)
    total_cpus = ray._private.state.cluster_resources()['CPU']
    verify_load_metrics(monitor, ({'CPU': 0.0}, {'CPU': total_cpus}))

    @ray.remote
    def work(signal):
        if False:
            while True:
                i = 10
        wait_signal = signal.wait.remote()
        while True:
            (ready, not_ready) = ray.wait([wait_signal], timeout=0)
            if len(ready) == 1:
                break
            time.sleep(1)
    signal = SignalActor.remote()
    work_handle = work.remote(signal)
    verify_load_metrics(monitor, ({'CPU': 1.0}, {'CPU': total_cpus}))
    ray.get(signal.send.remote())
    ray.get(work_handle)

    @ray.remote(num_cpus=1)
    class Actor:

        def work(self, signal):
            if False:
                for i in range(10):
                    print('nop')
            wait_signal = signal.wait.remote()
            while True:
                (ready, not_ready) = ray.wait([wait_signal], timeout=0)
                if len(ready) == 1:
                    break
                time.sleep(1)
    signal = SignalActor.remote()
    test_actor = Actor.remote()
    work_handle = test_actor.work.remote(signal)
    time.sleep(1)
    verify_load_metrics(monitor, ({'CPU': 1.0}, {'CPU': total_cpus}))
    ray.get(signal.send.remote())
    ray.get(work_handle)
    del monitor

def test_wait_for_nodes(ray_start_cluster_head):
    if False:
        return 10
    'Unit test for `Cluster.wait_for_nodes`.\n\n    Adds 4 workers, waits, then removes 4 workers, waits,\n    then adds 1 worker, waits, and removes 1 worker, waits.\n    '
    cluster = ray_start_cluster_head
    workers = [cluster.add_node() for i in range(4)]
    cluster.wait_for_nodes()
    [cluster.remove_node(w) for w in workers]
    cluster.wait_for_nodes()
    assert ray.cluster_resources()['CPU'] == 1
    worker2 = cluster.add_node()
    cluster.wait_for_nodes()
    cluster.remove_node(worker2)
    cluster.wait_for_nodes()
    assert ray.cluster_resources()['CPU'] == 1

@pytest.mark.parametrize('call_ray_start', ['ray start --head --ray-client-server-port 20000 ' + '--min-worker-port=0 --max-worker-port=0 --port 0'], indirect=True)
def test_ray_client(call_ray_start):
    if False:
        while True:
            i = 10
    from ray.util.client import ray as ray_client
    ray.client('localhost:20000').connect()

    @ray.remote
    def f():
        if False:
            i = 10
            return i + 15
        return 'hello client'
    assert ray_client.get(f.remote()) == 'hello client'

def test_detached_actor_autoscaling(ray_start_cluster_head):
    if False:
        return 10
    'Make sure that a detached actor, which belongs to a dead job, can start\n    workers on nodes that were added after the job ended.\n    '
    cluster = ray_start_cluster_head
    cluster.add_node(num_cpus=2)
    cluster.wait_for_nodes(2)

    @ray.remote(num_cpus=1)
    class Actor:

        def __init__(self):
            if False:
                while True:
                    i = 10
            self.handles = []

        def start_actors(self, n):
            if False:
                print('Hello World!')
            self.handles.extend([Actor.remote() for _ in range(n)])

        def get_children(self):
            if False:
                while True:
                    i = 10
            return self.handles

        def ping(self):
            if False:
                for i in range(10):
                    print('nop')
            pass
    main_actor = Actor.options(lifetime='detached', name='main').remote()
    ray.get(main_actor.ping.remote())
    ray.shutdown()
    ray.init(address=cluster.address, namespace='default_test_namespace')
    main_actor = ray.get_actor('main')
    num_to_start = int(ray.available_resources().get('CPU', 0) + 1)
    print(f'Starting {num_to_start} actors')
    ray.get(main_actor.start_actors.remote(num_to_start))
    actor_handles = ray.get(main_actor.get_children.remote())
    (up, down) = ray.wait([actor.ping.remote() for actor in actor_handles], timeout=5, num_returns=len(actor_handles))
    assert len(up) == len(actor_handles) - 1
    assert len(down) == 1
    cluster.add_node(num_cpus=1)
    cluster.wait_for_nodes(3)
    (up, down) = ray.wait([actor.ping.remote() for actor in actor_handles], timeout=5, num_returns=len(actor_handles))
    assert len(up) == len(actor_handles)
    assert len(down) == 0

def test_multi_node_pgs(ray_start_cluster):
    if False:
        for i in range(10):
            print('nop')
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=2)
    cluster.wait_for_nodes(2)
    ray.init(address=cluster.address)
    pgs = [ray.util.placement_group([{'CPU': 1}]) for _ in range(4)]
    (ready, not_ready) = ray.wait([pg.ready() for pg in pgs], timeout=5, num_returns=4)
    assert len(ready) == 2
    assert len(not_ready) == 2
    cluster.add_node(num_cpus=2)
    cluster.wait_for_nodes(3)
    (ready, not_ready) = ray.wait([pg.ready() for pg in pgs], timeout=5, num_returns=4)
    assert len(ready) == 4
    assert len(not_ready) == 0
    for i in range(4, 10):
        cluster.add_node(num_cpus=2)
        cluster.wait_for_nodes(i)
        print('.')
        more_pgs = [ray.util.placement_group([{'CPU': 1}]) for _ in range(2)]
        (ready, not_ready) = ray.wait([pg.ready() for pg in more_pgs], timeout=5, num_returns=2)
        assert len(ready) == 2
if __name__ == '__main__':
    import pytest
    import os
    import sys
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))