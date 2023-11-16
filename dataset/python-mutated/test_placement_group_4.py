import pytest
import os
import sys
import ray
import ray.cluster_utils
from ray._private.test_utils import get_other_nodes, wait_for_condition, is_placement_group_removed, placement_group_assert_no_leak
from ray._raylet import PlacementGroupID
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.client.ray_client_helpers import connect_to_client_or_not
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.plugin import RuntimeEnvPlugin
MOCK_WORKER_STARTUP_SLOWLY_PLUGIN_CLASS_PATH = 'ray.tests.test_placement_group_4.MockWorkerStartupSlowlyPlugin'
MOCK_WORKER_STARTUP_SLOWLY_PLUGIN_NAME = 'MockWorkerStartupSlowlyPlugin'

class MockWorkerStartupSlowlyPlugin(RuntimeEnvPlugin):
    name = MOCK_WORKER_STARTUP_SLOWLY_PLUGIN_NAME

    def validate(runtime_env_dict: dict) -> str:
        if False:
            print('Hello World!')
        return 'success'

    @staticmethod
    def create(uri: str, runtime_env_dict: dict, ctx: RuntimeEnvContext) -> float:
        if False:
            i = 10
            return i + 15
        import time
        time.sleep(15)
        return 0

@pytest.mark.parametrize('connect_to_client', [False, True])
def test_remove_placement_group(ray_start_cluster, connect_to_client):
    if False:
        i = 10
        return i + 15
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=4)
    ray.init(address=cluster.address)

    @ray.remote
    def warmup():
        if False:
            return 10
        pass
    ray.get([warmup.remote() for _ in range(4)])
    with connect_to_client_or_not(connect_to_client):
        random_group_id = PlacementGroupID.from_random()
        random_placement_group = PlacementGroup(random_group_id)
        for _ in range(3):
            ray.util.remove_placement_group(random_placement_group)
        placement_group = ray.util.placement_group([{'CPU': 2}, {'CPU': 2}])
        assert placement_group.wait(10)
        ray.util.remove_placement_group(placement_group)
        wait_for_condition(lambda : is_placement_group_removed(placement_group))
        placement_group = ray.util.placement_group([{'CPU': 2}, {'CPU': 2}])
        assert placement_group.wait(10)

        @ray.remote(num_cpus=2)
        class A:

            def f(self):
                if False:
                    for i in range(10):
                        print('nop')
                return 3

        @ray.remote(num_cpus=2, max_retries=0)
        def long_running_task():
            if False:
                return 10
            print(os.getpid())
            import time
            time.sleep(50)
        task_ref = long_running_task.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=placement_group)).remote()
        a = A.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=placement_group)).remote()
        assert ray.get(a.f.remote()) == 3
        ray.util.remove_placement_group(placement_group)
        for _ in range(3):
            ray.util.remove_placement_group(placement_group)

        @ray.remote(num_cpus=4)
        def f():
            if False:
                for i in range(10):
                    print('nop')
            return 3
        assert ray.get(f.remote()) == 3
        with pytest.raises(ray.exceptions.RayActorError, match='actor died'):
            ray.get(a.f.remote(), timeout=3.0)
        with pytest.raises(ray.exceptions.WorkerCrashedError):
            ray.get(task_ref)

@pytest.mark.parametrize('set_runtime_env_plugins', ['[{"class":"' + MOCK_WORKER_STARTUP_SLOWLY_PLUGIN_CLASS_PATH + '"}]'], indirect=True)
def test_remove_placement_group_worker_startup_slowly(set_runtime_env_plugins, ray_start_cluster):
    if False:
        i = 10
        return i + 15
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=4)
    ray.init(address=cluster.address)
    placement_group = ray.util.placement_group([{'CPU': 2}, {'CPU': 2}])
    assert placement_group.wait(10)

    @ray.remote(num_cpus=2)
    class A:

        def f(self):
            if False:
                i = 10
                return i + 15
            return 3

    @ray.remote(num_cpus=2, max_retries=0)
    def long_running_task():
        if False:
            while True:
                i = 10
        print(os.getpid())
        import time
        time.sleep(60)
    task_ref = long_running_task.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=placement_group), runtime_env={MOCK_WORKER_STARTUP_SLOWLY_PLUGIN_NAME: {}}).remote()
    a = A.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=placement_group)).remote()
    assert ray.get(a.f.remote()) == 3
    ray.util.remove_placement_group(placement_group)
    with pytest.raises(ray.exceptions.RayActorError, match='actor died'):
        ray.get(a.f.remote(), timeout=3.0)
    with pytest.raises(ray.exceptions.TaskPlacementGroupRemoved):
        ray.get(task_ref)

@pytest.mark.parametrize('connect_to_client', [False, True])
def test_remove_pending_placement_group(ray_start_cluster, connect_to_client):
    if False:
        i = 10
        return i + 15
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=4)
    ray.init(address=cluster.address)
    with connect_to_client_or_not(connect_to_client):
        placement_group = ray.util.placement_group([{'GPU': 2}, {'CPU': 2}])
        ray.util.remove_placement_group(placement_group)

        @ray.remote(num_cpus=4)
        def f():
            if False:
                for i in range(10):
                    print('nop')
            return 3
        assert ray.get(f.remote()) == 3
        placement_group_assert_no_leak([placement_group])

@pytest.mark.parametrize('connect_to_client', [False, True])
def test_placement_group_table(ray_start_cluster, connect_to_client):
    if False:
        while True:
            i = 10

    @ray.remote(num_cpus=2)
    class Actor(object):

        def __init__(self):
            if False:
                return 10
            self.n = 0

        def value(self):
            if False:
                i = 10
                return i + 15
            return self.n
    cluster = ray_start_cluster
    num_nodes = 2
    for _ in range(num_nodes):
        cluster.add_node(num_cpus=4)
    ray.init(address=cluster.address)
    pgs_created = []
    with connect_to_client_or_not(connect_to_client):
        name = 'name'
        strategy = 'PACK'
        bundles = [{'CPU': 2, 'GPU': 1}, {'CPU': 2}]
        placement_group = ray.util.placement_group(name=name, strategy=strategy, bundles=bundles)
        pgs_created.append(placement_group)
        result = ray.util.placement_group_table(placement_group)
        assert result['name'] == name
        assert result['strategy'] == strategy
        for i in range(len(bundles)):
            assert bundles[i] == result['bundles'][i]
        assert result['state'] == 'PENDING'
        cluster.add_node(num_cpus=5, num_gpus=1)
        cluster.wait_for_nodes()
        actor_1 = Actor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=placement_group, placement_group_bundle_index=0)).remote()
        ray.get(actor_1.value.remote())
        result = ray.util.placement_group_table(placement_group)
        assert result['state'] == 'CREATED'
        second_strategy = 'SPREAD'
        pgs_created.append(ray.util.placement_group(name='second_placement_group', strategy=second_strategy, bundles=bundles))
        pgs_created.append(ray.util.placement_group(name='third_placement_group', strategy=second_strategy, bundles=bundles))
        placement_group_table = ray.util.placement_group_table()
        assert len(placement_group_table) == 3
        true_name_set = {'name', 'second_placement_group', 'third_placement_group'}
        get_name_set = set()
        for (_, placement_group_data) in placement_group_table.items():
            get_name_set.add(placement_group_data['name'])
        assert true_name_set == get_name_set
        placement_group_assert_no_leak(pgs_created)

def test_placement_group_stats(ray_start_cluster):
    if False:
        i = 10
        return i + 15
    cluster = ray_start_cluster
    num_nodes = 1
    for _ in range(num_nodes):
        cluster.add_node(num_cpus=4, num_gpus=1)
    ray.init(address=cluster.address)
    pg = ray.util.placement_group(bundles=[{'CPU': 4, 'GPU': 1}])
    ray.get(pg.ready())
    stats = ray.util.placement_group_table(pg)['stats']
    assert stats['scheduling_attempt'] == 1
    assert stats['scheduling_state'] == 'FINISHED'
    assert stats['end_to_end_creation_latency_ms'] != 0
    pg2 = ray.util.placement_group(bundles=[{'CPU': 4, 'GPU': 1}])

    def assert_scheduling_state():
        if False:
            print('Hello World!')
        stats = ray.util.placement_group_table(pg2)['stats']
        if stats['scheduling_attempt'] != 1:
            return False
        if stats['scheduling_state'] != 'NO_RESOURCES':
            return False
        if stats['end_to_end_creation_latency_ms'] != 0:
            return False
        return True
    wait_for_condition(assert_scheduling_state)
    ray.util.remove_placement_group(pg)

    def assert_scheduling_state():
        if False:
            print('Hello World!')
        stats = ray.util.placement_group_table(pg2)['stats']
        if stats['scheduling_state'] != 'FINISHED':
            return False
        if stats['end_to_end_creation_latency_ms'] == 0:
            return False
        return True
    wait_for_condition(assert_scheduling_state)
    pg3 = ray.util.placement_group(bundles=[{'CPU': 4, 'a': 1}])
    ray.util.remove_placement_group(pg3)

    def assert_scheduling_state():
        if False:
            for i in range(10):
                print('nop')
        stats = ray.util.placement_group_table(pg3)['stats']
        if stats['scheduling_state'] != 'REMOVED':
            return False
        return True
    wait_for_condition(assert_scheduling_state)
    placement_group_assert_no_leak([pg2])

@pytest.mark.parametrize('connect_to_client', [False, True])
def test_cuda_visible_devices(ray_start_cluster, connect_to_client):
    if False:
        return 10

    @ray.remote(num_gpus=1)
    def f():
        if False:
            i = 10
            return i + 15
        return os.environ['CUDA_VISIBLE_DEVICES']
    cluster = ray_start_cluster
    num_nodes = 1
    for _ in range(num_nodes):
        cluster.add_node(num_gpus=1)
    ray.init(address=cluster.address)
    with connect_to_client_or_not(connect_to_client):
        g1 = ray.util.placement_group([{'CPU': 1, 'GPU': 1}])
        o1 = f.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=g1)).remote()
        devices = ray.get(o1)
        assert devices == '0', devices
        placement_group_assert_no_leak([g1])

@pytest.mark.parametrize('connect_to_client', [False, True])
def test_placement_group_reschedule_when_node_dead(ray_start_cluster, connect_to_client):
    if False:
        for i in range(10):
            print('nop')

    @ray.remote(num_cpus=1)
    class Actor(object):

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self.n = 0

        def value(self):
            if False:
                for i in range(10):
                    print('nop')
            return self.n
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=4)
    cluster.add_node(num_cpus=4)
    cluster.add_node(num_cpus=4)
    cluster.wait_for_nodes()
    ray.init(address=cluster.address, namespace='default_test_namespace')
    nodes = ray.nodes()
    assert len(nodes) == 3
    assert nodes[0]['alive'] and nodes[1]['alive'] and nodes[2]['alive']
    with connect_to_client_or_not(connect_to_client):
        placement_group = ray.util.placement_group(name='name', strategy='SPREAD', bundles=[{'CPU': 2}, {'CPU': 2}, {'CPU': 2}])
        actor_1 = Actor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=placement_group, placement_group_bundle_index=0), lifetime='detached').remote()
        actor_2 = Actor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=placement_group, placement_group_bundle_index=1), lifetime='detached').remote()
        actor_3 = Actor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=placement_group, placement_group_bundle_index=2), lifetime='detached').remote()
        ray.get(actor_1.value.remote())
        ray.get(actor_2.value.remote())
        ray.get(actor_3.value.remote())
        cluster.remove_node(get_other_nodes(cluster, exclude_head=True)[-1])
        cluster.wait_for_nodes()
        actor_4 = Actor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=placement_group, placement_group_bundle_index=0), lifetime='detached').remote()
        actor_5 = Actor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=placement_group, placement_group_bundle_index=1), lifetime='detached').remote()
        actor_6 = Actor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=placement_group, placement_group_bundle_index=2), lifetime='detached').remote()
        ray.get(actor_4.value.remote())
        ray.get(actor_5.value.remote())
        ray.get(actor_6.value.remote())
        placement_group_assert_no_leak([placement_group])
        ray.shutdown()

def test_infeasible_pg(ray_start_cluster):
    if False:
        return 10
    'Test infeasible pgs are scheduled after new nodes are added.'
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=2)
    ray.init('auto')
    bundle = {'CPU': 4, 'GPU': 1}
    pg = ray.util.placement_group([bundle], name='worker_1', strategy='STRICT_PACK')
    with pytest.raises(ray.exceptions.GetTimeoutError):
        ray.get(pg.ready(), timeout=3)
    state = ray.util.placement_group_table()[pg.id.hex()]['stats']['scheduling_state']
    assert state == 'INFEASIBLE'
    cluster.add_node(num_cpus=4, num_gpus=1)
    assert ray.get(pg.ready(), timeout=10)
if __name__ == '__main__':
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))