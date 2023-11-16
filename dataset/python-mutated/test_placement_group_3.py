import sys
import time
import pytest
import ray
import ray._private.gcs_utils as gcs_utils
import ray.cluster_utils
import ray.experimental.internal_kv as internal_kv
from ray._private.ray_constants import DEBUG_AUTOSCALING_ERROR, DEBUG_AUTOSCALING_STATUS
from ray.autoscaler._private.constants import AUTOSCALER_UPDATE_INTERVAL_S
from ray._private.test_utils import convert_actor_state, generate_system_config_map, is_placement_group_removed, kill_actor_and_wait_for_failure, reset_autoscaler_v2_enabled_cache, run_string_as_driver, wait_for_condition
from ray.autoscaler._private.commands import debug_status
from ray.exceptions import RaySystemError
from ray.util.client.ray_client_helpers import connect_to_client_or_not
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
try:
    import pytest_timeout
except ImportError:
    pytest_timeout = None

def get_ray_status_output(address):
    if False:
        print('Hello World!')
    gcs_client = ray._raylet.GcsClient(address=address)
    internal_kv._initialize_internal_kv(gcs_client)
    status = internal_kv._internal_kv_get(DEBUG_AUTOSCALING_STATUS)
    error = internal_kv._internal_kv_get(DEBUG_AUTOSCALING_ERROR)
    return {'demand': debug_status(status, error, address=address).split('Demands:')[1].strip('\n').strip(' '), 'usage': debug_status(status, error, address=address).split('Demands:')[0].split('Usage:')[1].strip('\n').strip(' ')}

@pytest.mark.parametrize('ray_start_cluster_head_with_external_redis', [generate_system_config_map(health_check_initial_delay_ms=0, health_check_failure_threshold=10, gcs_rpc_server_reconnect_timeout_s=60)], indirect=True)
def test_create_placement_group_during_gcs_server_restart(ray_start_cluster_head_with_external_redis):
    if False:
        return 10
    cluster = ray_start_cluster_head_with_external_redis
    cluster.add_node(num_cpus=200)
    cluster.wait_for_nodes()
    placement_groups = []
    for i in range(0, 100):
        placement_group = ray.util.placement_group([{'CPU': 1}, {'CPU': 1}])
        placement_groups.append(placement_group)
    cluster.head_node.kill_gcs_server()
    cluster.head_node.start_gcs_server()
    for i in range(0, 100):
        ray.get(placement_groups[i].ready())

@pytest.mark.parametrize('ray_start_cluster_head_with_external_redis', [generate_system_config_map(health_check_initial_delay_ms=0, health_check_failure_threshold=10, gcs_rpc_server_reconnect_timeout_s=60)], indirect=True)
def test_placement_group_wait_api(ray_start_cluster_head_with_external_redis):
    if False:
        print('Hello World!')
    cluster = ray_start_cluster_head_with_external_redis
    cluster.add_node(num_cpus=2)
    cluster.add_node(num_cpus=2)
    cluster.wait_for_nodes()
    placement_group1 = ray.util.placement_group([{'CPU': 1}, {'CPU': 1}])
    assert placement_group1.wait(10)
    cluster.head_node.kill_gcs_server()
    cluster.head_node.start_gcs_server()
    placement_group2 = ray.util.placement_group([{'CPU': 1}, {'CPU': 1}])
    assert placement_group2.wait(10)
    ray.util.remove_placement_group(placement_group1)
    with pytest.raises(Exception):
        placement_group1.wait(10)

def test_placement_group_wait_api_timeout(shutdown_only):
    if False:
        while True:
            i = 10
    'Make sure the wait API timeout works\n\n    https://github.com/ray-project/ray/issues/27287\n    '
    ray.init(num_cpus=1)
    pg = ray.util.placement_group(bundles=[{'CPU': 2}])
    start = time.time()
    assert not pg.wait(5)
    assert 5 <= time.time() - start

@pytest.mark.parametrize('connect_to_client', [False, True])
def test_schedule_placement_groups_at_the_same_time(connect_to_client):
    if False:
        for i in range(10):
            print('nop')
    ray.init(num_cpus=4)
    with connect_to_client_or_not(connect_to_client):
        pgs = [placement_group([{'CPU': 2}]) for _ in range(6)]
        wait_pgs = {pg.ready(): pg for pg in pgs}

        def is_all_placement_group_removed():
            if False:
                i = 10
                return i + 15
            (ready, _) = ray.wait(list(wait_pgs.keys()), timeout=0.5)
            if ready:
                ready_pg = wait_pgs[ready[0]]
                remove_placement_group(ready_pg)
                del wait_pgs[ready[0]]
            if len(wait_pgs) == 0:
                return True
            return False
        wait_for_condition(is_all_placement_group_removed)
    ray.shutdown()

def test_detached_placement_group(ray_start_cluster):
    if False:
        return 10
    cluster = ray_start_cluster
    for _ in range(2):
        cluster.add_node(num_cpus=3)
    cluster.wait_for_nodes()
    info = ray.init(address=cluster.address)
    driver_code = f'''\nimport ray\nfrom ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy\n\nray.init(address="{info['address']}")\n\npg = ray.util.placement_group(\n        [{{"CPU": 1}} for _ in range(2)],\n        strategy="STRICT_SPREAD", lifetime="detached")\nray.get(pg.ready())\n\n@ray.remote(num_cpus=1)\nclass Actor:\n    def ready(self):\n        return True\n\nfor bundle_index in range(2):\n    actor = Actor.options(lifetime="detached",\n        scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg,\n                placement_group_bundle_index=bundle_index)).remote()\n    ray.get(actor.ready.remote())\n\nray.shutdown()\n    '''
    run_string_as_driver(driver_code)

    def is_job_done():
        if False:
            print('Hello World!')
        jobs = ray._private.state.jobs()
        for job in jobs:
            if job['IsDead']:
                return True
        return False

    def assert_alive_num_pg(expected_num_pg):
        if False:
            for i in range(10):
                print('nop')
        alive_num_pg = 0
        for (_, placement_group_info) in ray.util.placement_group_table().items():
            if placement_group_info['state'] == 'CREATED':
                alive_num_pg += 1
        return alive_num_pg == expected_num_pg

    def assert_alive_num_actor(expected_num_actor):
        if False:
            print('Hello World!')
        alive_num_actor = 0
        for actor_info in ray._private.state.actors().values():
            if actor_info['State'] == convert_actor_state(gcs_utils.ActorTableData.ALIVE):
                alive_num_actor += 1
        return alive_num_actor == expected_num_actor
    wait_for_condition(is_job_done)
    assert assert_alive_num_pg(1)
    assert assert_alive_num_actor(2)

    @ray.remote(num_cpus=1)
    class NestedActor:

        def ready(self):
            if False:
                for i in range(10):
                    print('nop')
            return True

    @ray.remote(num_cpus=1)
    class Actor:

        def __init__(self):
            if False:
                while True:
                    i = 10
            self.actors = []

        def ready(self):
            if False:
                while True:
                    i = 10
            return True

        def schedule_nested_actor_with_detached_pg(self):
            if False:
                print('Hello World!')
            pg = ray.util.placement_group([{'CPU': 1} for _ in range(2)], strategy='STRICT_SPREAD', lifetime='detached', name='detached_pg')
            ray.get(pg.ready())
            for bundle_index in range(2):
                actor = NestedActor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=bundle_index), lifetime='detached').remote()
                ray.get(actor.ready.remote())
                self.actors.append(actor)
    a = Actor.options(lifetime='detached').remote()
    ray.get(a.ready.remote())
    ray.get(a.schedule_nested_actor_with_detached_pg.remote())
    kill_actor_and_wait_for_failure(a)
    with pytest.raises(ray.exceptions.RayActorError):
        ray.get(a.ready.remote())
    assert assert_alive_num_pg(2)
    assert assert_alive_num_actor(4)

def test_named_placement_group(ray_start_cluster):
    if False:
        return 10
    cluster = ray_start_cluster
    for _ in range(2):
        cluster.add_node(num_cpus=3)
    cluster.wait_for_nodes()
    info = ray.init(address=cluster.address, namespace='default_test_namespace')
    global_placement_group_name = 'named_placement_group'
    driver_code = f'''\nimport ray\n\nray.init(address="{info['address']}", namespace="default_test_namespace")\n\npg = ray.util.placement_group(\n        [{{"CPU": 1}} for _ in range(2)],\n        strategy="STRICT_SPREAD",\n        name="{global_placement_group_name}",\n        lifetime="detached")\nray.get(pg.ready())\n\nray.shutdown()\n    '''
    run_string_as_driver(driver_code)

    def is_job_done():
        if False:
            return 10
        jobs = ray._private.state.jobs()
        for job in jobs:
            if job['IsDead']:
                return True
        return False
    wait_for_condition(is_job_done)

    @ray.remote(num_cpus=1)
    class Actor:

        def ping(self):
            if False:
                while True:
                    i = 10
            return 'pong'
    placement_group = ray.util.get_placement_group(global_placement_group_name)
    assert placement_group is not None
    assert placement_group.wait(5)
    actor = Actor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=placement_group, placement_group_bundle_index=0)).remote()
    ray.get(actor.ping.remote())
    error_creation_count = 0
    try:
        ray.util.placement_group([{'CPU': 1} for _ in range(2)], strategy='STRICT_SPREAD', name=global_placement_group_name)
    except RaySystemError:
        error_creation_count += 1
    assert error_creation_count == 1
    ray.util.remove_placement_group(placement_group)
    same_name_pg = ray.util.placement_group([{'CPU': 1} for _ in range(2)], strategy='STRICT_SPREAD', name=global_placement_group_name)
    assert same_name_pg.wait(10)
    error_count = 0
    try:
        ray.util.get_placement_group('inexistent_pg')
    except ValueError:
        error_count = error_count + 1
    assert error_count == 1

@pytest.mark.parametrize('connect_to_client', [False, True])
def test_placement_group_synchronous_registration(ray_start_cluster, connect_to_client):
    if False:
        i = 10
        return i + 15
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=1)
    cluster.wait_for_nodes()
    ray.init(address=cluster.address)
    with connect_to_client_or_not(connect_to_client):
        placement_group = ray.util.placement_group(name='name', strategy='STRICT_PACK', bundles=[{'CPU': 1}, {'CPU': 1}])
        ray.util.remove_placement_group(placement_group)
        wait_for_condition(lambda : is_placement_group_removed(placement_group))

@pytest.mark.parametrize('connect_to_client', [False, True])
def test_placement_group_gpu_set(ray_start_cluster, connect_to_client):
    if False:
        return 10
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=1, num_gpus=1)
    cluster.add_node(num_cpus=1, num_gpus=1)
    cluster.wait_for_nodes()
    ray.init(address=cluster.address)
    with connect_to_client_or_not(connect_to_client):
        placement_group = ray.util.placement_group(name='name', strategy='PACK', bundles=[{'CPU': 1, 'GPU': 1}, {'CPU': 1, 'GPU': 1}])

        @ray.remote(num_gpus=1)
        def get_gpus():
            if False:
                print('Hello World!')
            return ray.get_gpu_ids()
        result = get_gpus.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=placement_group, placement_group_bundle_index=0)).remote()
        result = ray.get(result)
        assert result == [0]
        result = get_gpus.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=placement_group, placement_group_bundle_index=1)).remote()
        result = ray.get(result)
        assert result == [0]

@pytest.mark.parametrize('connect_to_client', [False, True])
def test_placement_group_gpu_assigned(ray_start_cluster, connect_to_client):
    if False:
        i = 10
        return i + 15
    cluster = ray_start_cluster
    cluster.add_node(num_gpus=2)
    ray.init(address=cluster.address)
    gpu_ids_res = set()

    @ray.remote(num_gpus=1, num_cpus=0)
    def f():
        if False:
            print('Hello World!')
        import os
        return os.environ['CUDA_VISIBLE_DEVICES']
    with connect_to_client_or_not(connect_to_client):
        pg1 = ray.util.placement_group([{'GPU': 1}])
        pg2 = ray.util.placement_group([{'GPU': 1}])
        assert pg1.wait(10)
        assert pg2.wait(10)
        gpu_ids_res.add(ray.get(f.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg1)).remote()))
        gpu_ids_res.add(ray.get(f.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg2)).remote()))
        assert len(gpu_ids_res) == 2

@pytest.mark.repeat(3)
def test_actor_scheduling_not_block_with_placement_group(ray_start_cluster):
    if False:
        i = 10
        return i + 15
    'Tests the scheduling of lots of actors will not be blocked\n    when using placement groups.\n\n    For more detailed information please refer to:\n    https://github.com/ray-project/ray/issues/15801.\n    '
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=1)
    ray.init(address=cluster.address)

    @ray.remote(num_cpus=1)
    class A:

        def ready(self):
            if False:
                while True:
                    i = 10
            pass
    actor_num = 1000
    pgs = [ray.util.placement_group([{'CPU': 1}]) for _ in range(actor_num)]
    actors = [A.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)).remote() for pg in pgs]
    refs = [actor.ready.remote() for actor in actors]
    expected_created_num = 1

    def is_actor_created_number_correct():
        if False:
            for i in range(10):
                print('nop')
        (ready, not_ready) = ray.wait(refs, num_returns=len(refs), timeout=1)
        return len(ready) == expected_created_num

    def is_pg_created_number_correct():
        if False:
            for i in range(10):
                print('nop')
        created_pgs = [pg for (_, pg) in ray.util.placement_group_table().items() if pg['state'] == 'CREATED']
        return len(created_pgs) == expected_created_num
    wait_for_condition(is_pg_created_number_correct, timeout=3)
    wait_for_condition(is_actor_created_number_correct, timeout=30, retry_interval_ms=0)
    for _ in range(20):
        expected_created_num += 1
        cluster.add_node(num_cpus=1)
        wait_for_condition(is_pg_created_number_correct, timeout=10)
        wait_for_condition(is_actor_created_number_correct, timeout=30, retry_interval_ms=0)

@pytest.mark.parametrize('connect_to_client', [False, True])
def test_placement_group_gpu_unique_assigned(ray_start_cluster, connect_to_client):
    if False:
        i = 10
        return i + 15
    cluster = ray_start_cluster
    cluster.add_node(num_gpus=4, num_cpus=4)
    ray.init(address=cluster.address)
    gpu_ids_res = set()
    num_gpus = 4
    bundles = [{'GPU': 1, 'CPU': 1} for _ in range(num_gpus)]
    pg = placement_group(bundles)
    ray.get(pg.ready())

    @ray.remote(num_gpus=1, num_cpus=1)
    class Actor:

        def get_gpu(self):
            if False:
                i = 10
                return i + 15
            import os
            return os.environ['CUDA_VISIBLE_DEVICES']
    actors = []
    actors.append(Actor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=0)).remote())
    actors.append(Actor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=3)).remote())
    actors.append(Actor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=2)).remote())
    actors.append(Actor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=1)).remote())
    for actor in actors:
        gpu_ids = ray.get(actor.get_gpu.remote())
        assert len(gpu_ids) == 1
        gpu_ids_res.add(gpu_ids)
    assert len(gpu_ids_res) == 4

@pytest.mark.parametrize('enable_v2', [True, False])
def test_placement_group_status_no_bundle_demand(ray_start_cluster, enable_v2):
    if False:
        while True:
            i = 10
    reset_autoscaler_v2_enabled_cache()
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=4, _system_config={'enable_autoscaler_v2': enable_v2})
    ray.init(address=cluster.address)

    @ray.remote
    def f():
        if False:
            print('Hello World!')
        pass
    pg = ray.util.placement_group([{'CPU': 1}])
    ray.get(pg.ready())
    ray.util.remove_placement_group(pg)
    wait_for_condition(lambda : is_placement_group_removed(pg))
    r = pg.ready()

    def is_usage_updated():
        if False:
            for i in range(10):
                print('nop')
        demand_output = get_ray_status_output(cluster.address)
        return demand_output['usage'] != ''
    wait_for_condition(is_usage_updated)
    demand_output = get_ray_status_output(cluster.address)
    assert demand_output['demand'] == '(no resource demands)'

@pytest.mark.parametrize('enable_v2', [True, False])
def test_placement_group_status(ray_start_cluster, enable_v2):
    if False:
        i = 10
        return i + 15
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=4, _system_config={'enable_autoscaler_v2': enable_v2})
    ray.init(cluster.address)

    @ray.remote(num_cpus=1)
    class A:

        def ready(self):
            if False:
                for i in range(10):
                    print('nop')
            pass
    pg = ray.util.placement_group([{'CPU': 1}])
    ray.get(pg.ready())

    def is_usage_updated():
        if False:
            i = 10
            return i + 15
        demand_output = get_ray_status_output(cluster.address)
        cpu_usage = demand_output['usage']
        if cpu_usage == '':
            return False
        cpu_usage = cpu_usage.split('\n')[0]
        expected = '0.0/4.0 CPU (0.0 used of 1.0 reserved in placement groups)'
        if cpu_usage != expected:
            assert cpu_usage == '0.0/4.0 CPU'
            return False
        return True
    wait_for_condition(is_usage_updated, AUTOSCALER_UPDATE_INTERVAL_S)
    actors = [A.remote() for _ in range(2)]
    actors_in_pg = [A.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)).remote() for _ in range(1)]
    ray.get([actor.ready.remote() for actor in actors])
    ray.get([actor.ready.remote() for actor in actors_in_pg])
    time.sleep(AUTOSCALER_UPDATE_INTERVAL_S)
    demand_output = get_ray_status_output(cluster.address)
    cpu_usage = demand_output['usage'].split('\n')[0]
    expected = '3.0/4.0 CPU (1.0 used of 1.0 reserved in placement groups)'
    assert cpu_usage == expected

def test_placement_group_removal_leak_regression(ray_start_cluster):
    if False:
        while True:
            i = 10
    'Related issue:\n    https://github.com/ray-project/ray/issues/19131\n    '
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=5)
    ray.init(address=cluster.address)
    TOTAL_CPUS = 8
    bundles = [{'CPU': 1, 'GPU': 1}]
    bundles += [{'CPU': 1} for _ in range(TOTAL_CPUS - 1)]
    pg = placement_group(bundles, strategy='PACK')
    o = pg.ready()
    time.sleep(3)
    cluster.add_node(num_cpus=5, num_gpus=1)
    ray.get(o)
    bundle_resource_name = f'bundle_group_{pg.id.hex()}'
    expected_bundle_wildcard_val = TOTAL_CPUS * 1000

    def check_bundle_leaks():
        if False:
            return 10
        bundle_resources = ray.available_resources()[bundle_resource_name]
        return expected_bundle_wildcard_val == bundle_resources
    wait_for_condition(check_bundle_leaks)

def test_placement_group_local_resource_view(monkeypatch, ray_start_cluster):
    if False:
        return 10
    'Please refer to https://github.com/ray-project/ray/pull/19911\n    for more details.\n    '
    with monkeypatch.context() as m:
        m.setenv('RAY_raylet_report_resources_period_milliseconds', '2000')
        cluster = ray_start_cluster
        cluster.add_node(num_cpus=16, object_store_memory=1000000000.0)
        cluster.wait_for_nodes()
        ray.init(address='auto')
        cluster.add_node(num_cpus=16, num_gpus=1)
        cluster.wait_for_nodes()
        NUM_CPU_BUNDLES = 30

        @ray.remote(num_cpus=1)
        class Worker(object):

            def __init__(self, i):
                if False:
                    for i in range(10):
                        print('nop')
                self.i = i

            def work(self):
                if False:
                    for i in range(10):
                        print('nop')
                time.sleep(0.1)
                print('work ', self.i)

        @ray.remote(num_cpus=1, num_gpus=1)
        class Trainer(object):

            def __init__(self, i):
                if False:
                    return 10
                self.i = i

            def train(self):
                if False:
                    while True:
                        i = 10
                time.sleep(0.2)
                print('train ', self.i)
        bundles = [{'CPU': 1, 'GPU': 1}]
        bundles += [{'CPU': 1} for _ in range(NUM_CPU_BUNDLES)]
        pg = placement_group(bundles, strategy='PACK')
        ray.get(pg.ready())
        workers = [Worker.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)).remote(i) for i in range(NUM_CPU_BUNDLES)]
        trainer = Trainer.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)).remote(0)
        ray.get([workers[i].work.remote() for i in range(NUM_CPU_BUNDLES)])
        ray.get(trainer.train.remote())

def test_fractional_resources_handle_correct(ray_start_cluster):
    if False:
        for i in range(10):
            print('nop')
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=1000)
    ray.init(address=cluster.address)
    bundles = [{'CPU': 0.01} for _ in range(5)]
    pg = placement_group(bundles, strategy='SPREAD')
    ray.get(pg.ready(), timeout=10)
if __name__ == '__main__':
    import os
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))