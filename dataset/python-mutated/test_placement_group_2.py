import sys
import time
import pytest
import ray
import ray._private.gcs_utils as gcs_utils
import ray.cluster_utils
from ray._private.test_utils import convert_actor_state, generate_system_config_map, get_error_message, get_other_nodes, kill_actor_and_wait_for_failure, placement_group_assert_no_leak, run_string_as_driver, wait_for_condition
from ray.util.client.ray_client_helpers import connect_to_client_or_not
from ray.util.placement_group import get_current_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

@ray.remote
class Increase:

    def method(self, x):
        if False:
            i = 10
            return i + 15
        return x + 2

@pytest.mark.parametrize('connect_to_client', [False, True])
def test_check_bundle_index(ray_start_cluster, connect_to_client):
    if False:
        for i in range(10):
            print('nop')

    @ray.remote(num_cpus=2)
    class Actor(object):

        def __init__(self):
            if False:
                return 10
            self.n = 0

        def value(self):
            if False:
                while True:
                    i = 10
            return self.n
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=4)
    ray.init(address=cluster.address)
    with connect_to_client_or_not(connect_to_client):
        placement_group = ray.util.placement_group(name='name', strategy='SPREAD', bundles=[{'CPU': 2}, {'CPU': 2}])
        with pytest.raises(ValueError, match='bundle index 3 is invalid'):
            Actor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=placement_group, placement_group_bundle_index=3)).remote()
        with pytest.raises(ValueError, match='bundle index -2 is invalid'):
            Actor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=placement_group, placement_group_bundle_index=-2)).remote()
        with pytest.raises(ValueError, match='bundle index must be -1'):
            Actor.options(placement_group_bundle_index=0).remote()
        placement_group_assert_no_leak([placement_group])

@pytest.mark.parametrize('connect_to_client', [False, True])
def test_pending_placement_group_wait(ray_start_cluster, connect_to_client):
    if False:
        for i in range(10):
            print('nop')
    cluster = ray_start_cluster
    [cluster.add_node(num_cpus=2) for _ in range(1)]
    ray.init(address=cluster.address)
    cluster.wait_for_nodes()
    with connect_to_client_or_not(connect_to_client):
        placement_group = ray.util.placement_group(name='name', strategy='SPREAD', bundles=[{'CPU': 2}, {'CPU': 2}, {'GPU': 2}])
        (ready, unready) = ray.wait([placement_group.ready()], timeout=0.1)
        assert len(unready) == 1
        assert len(ready) == 0
        table = ray.util.placement_group_table(placement_group)
        assert table['state'] == 'PENDING'
        for i in range(3):
            assert len(table['bundles_to_node_id'][i]) == 0
        with pytest.raises(ray.exceptions.GetTimeoutError):
            ray.get(placement_group.ready(), timeout=0.1)

@pytest.mark.parametrize('connect_to_client', [False, True])
def test_placement_group_wait(ray_start_cluster, connect_to_client):
    if False:
        while True:
            i = 10
    cluster = ray_start_cluster
    [cluster.add_node(num_cpus=2) for _ in range(2)]
    ray.init(address=cluster.address)
    cluster.wait_for_nodes()
    with connect_to_client_or_not(connect_to_client):
        placement_group = ray.util.placement_group(name='name', strategy='SPREAD', bundles=[{'CPU': 2}, {'CPU': 2}])
        (ready, unready) = ray.wait([placement_group.ready()])
        assert len(unready) == 0
        assert len(ready) == 1
        table = ray.util.placement_group_table(placement_group)
        assert table['state'] == 'CREATED'
        pg = ray.get(placement_group.ready())
        assert pg.bundle_specs == placement_group.bundle_specs
        assert pg.id.binary() == placement_group.id.binary()

        @ray.remote
        def get_node_id():
            if False:
                i = 10
                return i + 15
            return ray.get_runtime_context().get_node_id()
        for i in range(2):
            scheduling_strategy = PlacementGroupSchedulingStrategy(placement_group=placement_group, placement_group_bundle_index=i)
            node_id = ray.get(get_node_id.options(scheduling_strategy=scheduling_strategy).remote())
            assert node_id == table['bundles_to_node_id'][i]

@pytest.mark.parametrize('connect_to_client', [False, True])
def test_schedule_placement_group_when_node_add(ray_start_cluster, connect_to_client):
    if False:
        while True:
            i = 10
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=4)
    ray.init(address=cluster.address)
    with connect_to_client_or_not(connect_to_client):
        placement_group = ray.util.placement_group([{'GPU': 2}, {'CPU': 2}])

        def is_placement_group_created():
            if False:
                for i in range(10):
                    print('nop')
            table = ray.util.placement_group_table(placement_group)
            if 'state' not in table:
                return False
            return table['state'] == 'CREATED'
        cluster.add_node(num_cpus=4, num_gpus=4)
        wait_for_condition(is_placement_group_created)

@pytest.mark.parametrize('connect_to_client', [False, True])
def test_atomic_creation(ray_start_cluster, connect_to_client):
    if False:
        print('Hello World!')
    cluster = ray_start_cluster
    bundle_cpu_size = 2
    bundle_per_node = 2
    num_nodes = 2
    [cluster.add_node(num_cpus=bundle_cpu_size * bundle_per_node) for _ in range(num_nodes)]
    ray.init(address=cluster.address)

    @ray.remote(num_cpus=1)
    class NormalActor:

        def ping(self):
            if False:
                return 10
            pass

    @ray.remote(num_cpus=3)
    def bothering_task():
        if False:
            return 10
        time.sleep(6)
        return True
    with connect_to_client_or_not(connect_to_client):
        tasks = [bothering_task.remote() for _ in range(2)]

        def tasks_scheduled():
            if False:
                while True:
                    i = 10
            return ray.available_resources()['CPU'] == 2.0
        wait_for_condition(tasks_scheduled)
        pg = ray.util.placement_group(name='name', strategy='SPREAD', bundles=[{'CPU': bundle_cpu_size} for _ in range(num_nodes * bundle_per_node)])
        pg_actor = NormalActor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=num_nodes * bundle_per_node - 1)).remote()
        (ready, unready) = ray.wait([pg.ready()], timeout=0.5)
        assert len(ready) == 0
        assert len(unready) == 1
        assert all(ray.get(tasks))
        (ready, unready) = ray.wait([pg.ready()])
        assert len(ready) == 1
        assert len(unready) == 0
        ray.get(pg_actor.ping.remote(), timeout=3.0)
        ray.kill(pg_actor)

        @ray.remote(num_cpus=bundle_cpu_size)
        def resource_check():
            if False:
                i = 10
                return i + 15
            return True
        check_without_pg = [resource_check.remote() for _ in range(bundle_per_node * num_nodes)]
        check_with_pg = [resource_check.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=i)).remote() for i in range(bundle_per_node * num_nodes)]
        (ready, unready) = ray.wait(check_without_pg, timeout=0)
        assert len(ready) == 0
        assert len(unready) == bundle_per_node * num_nodes
        assert all(ray.get(check_with_pg))
        ray.util.remove_placement_group(pg)

        def pg_removed():
            if False:
                for i in range(10):
                    print('nop')
            return ray.util.placement_group_table(pg)['state'] == 'REMOVED'
        wait_for_condition(pg_removed)
        assert all(ray.get(check_without_pg))

@pytest.mark.parametrize('connect_to_client', [False, True])
def test_mini_integration(ray_start_cluster, connect_to_client):
    if False:
        print('Hello World!')
    cluster = ray_start_cluster
    num_nodes = 5
    per_bundle_gpus = 2
    gpu_per_node = 4
    total_gpus = num_nodes * per_bundle_gpus * gpu_per_node
    per_node_gpus = per_bundle_gpus * gpu_per_node
    bundles_per_pg = 2
    total_num_pg = total_gpus // (bundles_per_pg * per_bundle_gpus)
    [cluster.add_node(num_cpus=2, num_gpus=per_bundle_gpus * gpu_per_node) for _ in range(num_nodes)]
    cluster.wait_for_nodes()
    ray.init(address=cluster.address)
    with connect_to_client_or_not(connect_to_client):

        @ray.remote(num_cpus=0, num_gpus=1)
        def random_tasks():
            if False:
                print('Hello World!')
            import random
            import time
            sleep_time = random.uniform(0.1, 0.2)
            time.sleep(sleep_time)
            return True
        pgs = []
        pg_tasks = []
        for index in range(total_num_pg):
            pgs.append(ray.util.placement_group(name=f'name{index}', strategy='PACK', bundles=[{'GPU': per_bundle_gpus} for _ in range(bundles_per_pg)]))
        for i in range(total_num_pg):
            pg = pgs[i]
            pg_tasks.append([random_tasks.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=bundle_index)).remote() for bundle_index in range(bundles_per_pg)])
        num_removed_pg = 0
        pg_indexes = [2, 3, 1, 7, 8, 9, 0, 6, 4, 5]
        while num_removed_pg < total_num_pg:
            index = pg_indexes[num_removed_pg]
            pg = pgs[index]
            assert all(ray.get(pg_tasks[index]))
            ray.util.remove_placement_group(pg)
            num_removed_pg += 1

        @ray.remote(num_cpus=2, num_gpus=per_node_gpus)
        class A:

            def ping(self):
                if False:
                    i = 10
                    return i + 15
                return True
        actors = [A.remote() for _ in range(num_nodes)]
        assert all(ray.get([a.ping.remote() for a in actors]))

@pytest.mark.parametrize('connect_to_client', [False, True])
def test_capture_child_actors(ray_start_cluster, connect_to_client):
    if False:
        print('Hello World!')
    cluster = ray_start_cluster
    total_num_actors = 4
    for _ in range(2):
        cluster.add_node(num_cpus=total_num_actors)
    ray.init(address=cluster.address)
    with connect_to_client_or_not(connect_to_client):
        pg = ray.util.placement_group([{'CPU': 2}, {'CPU': 2}], strategy='STRICT_PACK')
        ray.get(pg.ready())
        assert get_current_placement_group() is None

        @ray.remote(num_cpus=1)
        class NestedActor:

            def ready(self):
                if False:
                    print('Hello World!')
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
                    for i in range(10):
                        print('nop')
                return True

            def schedule_nested_actor(self):
                if False:
                    i = 10
                    return i + 15
                assert get_current_placement_group() is not None
                actor = NestedActor.remote()
                ray.get(actor.ready.remote())
                self.actors.append(actor)

            def schedule_nested_actor_outside_pg(self):
                if False:
                    for i in range(10):
                        print('nop')
                actor = NestedActor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=None)).remote()
                ray.get(actor.ready.remote())
                self.actors.append(actor)
        a = Actor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_capture_child_tasks=True)).remote()
        ray.get(a.ready.remote())
        for _ in range(total_num_actors - 1):
            ray.get(a.schedule_nested_actor.remote())
        node_id_set = set()
        for actor_info in ray._private.state.actors().values():
            if actor_info['State'] == convert_actor_state(gcs_utils.ActorTableData.ALIVE):
                node_id = actor_info['Address']['NodeID']
                node_id_set.add(node_id)
        assert len(node_id_set) == 1
        kill_actor_and_wait_for_failure(a)
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(a.ready.remote())
        a = Actor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)).remote()
        ray.get(a.ready.remote())
        for _ in range(total_num_actors - 1):
            ray.get(a.schedule_nested_actor.remote())
        node_id_set = set()
        for actor_info in ray._private.state.actors().values():
            if actor_info['State'] == convert_actor_state(gcs_utils.ActorTableData.ALIVE):
                node_id = actor_info['Address']['NodeID']
                node_id_set.add(node_id)
        assert len(node_id_set) == 2
        kill_actor_and_wait_for_failure(a)
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(a.ready.remote())
        a = Actor.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)).remote()
        ray.get(a.ready.remote())
        for _ in range(total_num_actors - 1):
            ray.get(a.schedule_nested_actor_outside_pg.remote())
        node_id_set = set()
        for actor_info in ray._private.state.actors().values():
            if actor_info['State'] == convert_actor_state(gcs_utils.ActorTableData.ALIVE):
                node_id = actor_info['Address']['NodeID']
                node_id_set.add(node_id)
        assert len(node_id_set) == 2

@pytest.mark.parametrize('connect_to_client', [False, True])
def test_capture_child_tasks(ray_start_cluster, connect_to_client):
    if False:
        i = 10
        return i + 15
    cluster = ray_start_cluster
    total_num_tasks = 4
    for _ in range(2):
        cluster.add_node(num_cpus=total_num_tasks, num_gpus=total_num_tasks)
    ray.init(address=cluster.address)
    with connect_to_client_or_not(connect_to_client):
        pg = ray.util.placement_group([{'CPU': 2, 'GPU': 2}, {'CPU': 2, 'GPU': 2}], strategy='STRICT_PACK')
        ray.get(pg.ready())
        assert get_current_placement_group() is None

        @ray.remote
        def task():
            if False:
                print('Hello World!')
            return get_current_placement_group()

        @ray.remote
        def create_nested_task(child_cpu, child_gpu, set_none=False):
            if False:
                while True:
                    i = 10
            assert get_current_placement_group() is not None
            kwargs = {'num_cpus': child_cpu, 'num_gpus': child_gpu}
            if set_none:
                kwargs['placement_group'] = None
            return ray.get([task.options(**kwargs).remote() for _ in range(3)])
        t = create_nested_task.options(num_cpus=1, num_gpus=0, scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_capture_child_tasks=True)).remote(1, 0)
        pgs = ray.get(t)
        assert None not in pgs
        t1 = create_nested_task.options(num_cpus=1, num_gpus=0, scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_capture_child_tasks=True)).remote(1, 0, True)
        pgs = ray.get(t1)
        assert set(pgs) == {None}
        t2 = create_nested_task.options(num_cpus=0, num_gpus=1, scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)).remote(0, 1)
        pgs = ray.get(t2)
        assert not all(pgs)

def test_ready_warning_suppressed(ray_start_regular, error_pubsub):
    if False:
        i = 10
        return i + 15
    p = error_pubsub
    pg = ray.util.placement_group([{'CPU': 2}] * 2, strategy='STRICT_PACK')
    with pytest.raises(ray.exceptions.GetTimeoutError):
        ray.get(pg.ready(), timeout=0.5)
    errors = get_error_message(p, 1, ray._private.ray_constants.INFEASIBLE_TASK_ERROR, timeout=0.1)
    assert len(errors) == 0

def test_automatic_cleanup_job(ray_start_cluster):
    if False:
        while True:
            i = 10
    cluster = ray_start_cluster
    num_nodes = 3
    num_cpu_per_node = 4
    for _ in range(num_nodes):
        cluster.add_node(num_cpus=num_cpu_per_node)
    cluster.wait_for_nodes()
    info = ray.init(address=cluster.address)
    available_cpus = ray.available_resources()['CPU']
    assert available_cpus == num_nodes * num_cpu_per_node
    driver_code = f'''\nimport ray\n\nray.init(address="{info['address']}")\n\ndef create_pg():\n    pg = ray.util.placement_group(\n            [{{"CPU": 1}} for _ in range(3)],\n            strategy="STRICT_SPREAD")\n    ray.get(pg.ready())\n    return pg\n\n@ray.remote(num_cpus=0)\ndef f():\n    create_pg()\n\n@ray.remote(num_cpus=0)\nclass A:\n    def create_pg(self):\n        create_pg()\n\nray.get(f.remote())\na = A.remote()\nray.get(a.create_pg.remote())\n# Create 2 pgs to make sure multiple placement groups that belong\n# to a single job will be properly cleaned.\ncreate_pg()\ncreate_pg()\n\nray.shutdown()\n    '''
    run_string_as_driver(driver_code)

    def is_job_done():
        if False:
            while True:
                i = 10
        jobs = ray._private.state.jobs()
        for job in jobs:
            if job['IsDead']:
                return True
        return False

    def assert_num_cpus(expected_num_cpus):
        if False:
            while True:
                i = 10
        if expected_num_cpus == 0:
            return 'CPU' not in ray.available_resources()
        return ray.available_resources()['CPU'] == expected_num_cpus
    wait_for_condition(is_job_done)
    available_cpus = ray.available_resources()['CPU']
    wait_for_condition(lambda : assert_num_cpus(num_nodes * num_cpu_per_node))

def test_automatic_cleanup_detached_actors(ray_start_cluster):
    if False:
        i = 10
        return i + 15
    cluster = ray_start_cluster
    num_nodes = 3
    num_cpu_per_node = 2
    for _ in range(num_nodes):
        cluster.add_node(num_cpus=num_cpu_per_node)
    cluster.wait_for_nodes()
    info = ray.init(address=cluster.address, namespace='default_test_namespace')
    available_cpus = ray.available_resources()['CPU']
    assert available_cpus == num_nodes * num_cpu_per_node
    driver_code = f'''\nimport ray\n\nray.init(address="{info['address']}", namespace="default_test_namespace")\n\ndef create_pg():\n    pg = ray.util.placement_group(\n            [{{"CPU": 1}} for _ in range(3)],\n            strategy="STRICT_SPREAD")\n    ray.get(pg.ready())\n    return pg\n\n# TODO(sang): Placement groups created by tasks launched by detached actor\n# is not cleaned with the current protocol.\n# @ray.remote(num_cpus=0)\n# def f():\n#     create_pg()\n\n@ray.remote(num_cpus=0, max_restarts=1, max_task_retries=-1)\nclass A:\n    def create_pg(self):\n        create_pg()\n    def create_child_pg(self):\n        self.a = A.options(name="B").remote()\n        ray.get(self.a.create_pg.remote())\n    def kill_child_actor(self):\n        ray.kill(self.a)\n        try:\n            ray.get(self.a.create_pg.remote())\n        except Exception:\n            pass\n\na = A.options(lifetime="detached", name="A").remote()\nray.get(a.create_pg.remote())\n# TODO(sang): Currently, child tasks are cleaned when a detached actor\n# is dead. We cannot test this scenario until it is fixed.\n# ray.get(a.create_child_pg.remote())\n\nray.shutdown()\n    '''
    run_string_as_driver(driver_code)

    def is_job_done():
        if False:
            while True:
                i = 10
        jobs = ray._private.state.jobs()
        for job in jobs:
            if job['IsDead']:
                return True
        return False

    def assert_num_cpus(expected_num_cpus):
        if False:
            i = 10
            return i + 15
        if expected_num_cpus == 0:
            return 'CPU' not in ray.available_resources()
        return ray.available_resources()['CPU'] == expected_num_cpus
    wait_for_condition(is_job_done)
    wait_for_condition(lambda : assert_num_cpus(num_nodes))
    a = ray.get_actor('A')
    ray.kill(a, no_restart=False)
    wait_for_condition(lambda : assert_num_cpus(num_nodes * num_cpu_per_node))
    ray.get(a.create_pg.remote())
    wait_for_condition(lambda : assert_num_cpus(num_nodes))
    ray.kill(a, no_restart=False)
    wait_for_condition(lambda : assert_num_cpus(num_nodes * num_cpu_per_node))

@pytest.mark.parametrize('ray_start_cluster_head_with_external_redis', [generate_system_config_map(gcs_failover_worker_reconnect_timeout=10, gcs_rpc_server_reconnect_timeout_s=60)], indirect=True)
def test_create_placement_group_after_gcs_server_restart(ray_start_cluster_head_with_external_redis):
    if False:
        while True:
            i = 10
    cluster = ray_start_cluster_head_with_external_redis
    cluster.add_node(num_cpus=2)
    cluster.add_node(num_cpus=2)
    cluster.wait_for_nodes()
    placement_group1 = ray.util.placement_group([{'CPU': 1}, {'CPU': 1}])
    ray.get(placement_group1.ready(), timeout=10)
    table = ray.util.placement_group_table(placement_group1)
    assert table['state'] == 'CREATED'
    cluster.head_node.kill_gcs_server()
    cluster.head_node.start_gcs_server()
    placement_group2 = ray.util.placement_group([{'CPU': 1}, {'CPU': 1}])
    ray.get(placement_group2.ready(), timeout=10)
    table = ray.util.placement_group_table(placement_group2)
    assert table['state'] == 'CREATED'
    placement_group3 = ray.util.placement_group([{'CPU': 1}, {'CPU': 1}])
    with pytest.raises(ray.exceptions.GetTimeoutError):
        ray.get(placement_group3.ready(), timeout=2)
    table = ray.util.placement_group_table(placement_group3)
    assert table['state'] == 'PENDING'

@pytest.mark.parametrize('ray_start_cluster_head_with_external_redis', [generate_system_config_map(gcs_failover_worker_reconnect_timeout=10, gcs_rpc_server_reconnect_timeout_s=60)], indirect=True)
def test_create_actor_with_placement_group_after_gcs_server_restart(ray_start_cluster_head_with_external_redis):
    if False:
        print('Hello World!')
    cluster = ray_start_cluster_head_with_external_redis
    cluster.add_node(num_cpus=2)
    cluster.wait_for_nodes()
    placement_group = ray.util.placement_group([{'CPU': 1}, {'CPU': 1}])
    cluster.head_node.kill_gcs_server()
    cluster.head_node.start_gcs_server()
    actor_2 = Increase.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=placement_group, placement_group_bundle_index=1)).remote()
    assert ray.get(actor_2.method.remote(1)) == 3

@pytest.mark.parametrize('ray_start_cluster_head_with_external_redis', [generate_system_config_map(gcs_failover_worker_reconnect_timeout=10, gcs_rpc_server_reconnect_timeout_s=60)], indirect=True)
def test_bundle_recreated_when_raylet_fo_after_gcs_server_restart(ray_start_cluster_head_with_external_redis):
    if False:
        for i in range(10):
            print('nop')
    cluster = ray_start_cluster_head_with_external_redis
    cluster.add_node(num_cpus=2)
    cluster.wait_for_nodes()
    placement_group = ray.util.placement_group([{'CPU': 2}])
    ray.get(placement_group.ready(), timeout=10)
    table = ray.util.placement_group_table(placement_group)
    assert table['state'] == 'CREATED'
    cluster.head_node.kill_gcs_server()
    cluster.head_node.start_gcs_server()
    cluster.remove_node(get_other_nodes(cluster, exclude_head=True)[-1])
    cluster.add_node(num_cpus=2)
    cluster.wait_for_nodes()
    actor = Increase.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=placement_group, placement_group_bundle_index=0)).remote()
    assert ray.get(actor.method.remote(1), timeout=5) == 3
if __name__ == '__main__':
    import os
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))