import sys
import os
import threading
from time import sleep
import pytest
import ray
from ray._private.utils import get_or_create_event_loop
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import ray._private.gcs_utils as gcs_utils
from ray._private import ray_constants
from ray._private.test_utils import convert_actor_state, enable_external_redis, generate_system_config_map, wait_for_condition, wait_for_pid_to_exit, run_string_as_driver
from ray.job_submission import JobSubmissionClient, JobStatus
from ray._raylet import GcsClient
import psutil

@ray.remote
class Increase:

    def method(self, x):
        if False:
            for i in range(10):
                print('nop')
        return x + 2

@ray.remote
def increase(x):
    if False:
        return 10
    return x + 1

@pytest.mark.parametrize('ray_start_regular_with_external_redis', [generate_system_config_map(gcs_failover_worker_reconnect_timeout=20, gcs_rpc_server_reconnect_timeout_s=60)], indirect=True)
def test_gcs_server_restart(ray_start_regular_with_external_redis):
    if False:
        for i in range(10):
            print('nop')
    actor1 = Increase.remote()
    result = ray.get(actor1.method.remote(1))
    assert result == 3
    ray._private.worker._global_node.kill_gcs_server()
    ray._private.worker._global_node.start_gcs_server()
    actor2 = Increase.remote()
    result = ray.get(actor2.method.remote(2))
    assert result == 4
    result = ray.get(increase.remote(1))
    assert result == 2
    result = ray.get(actor1.method.remote(7))
    assert result == 9

@pytest.mark.parametrize('ray_start_regular_with_external_redis', [generate_system_config_map(gcs_failover_worker_reconnect_timeout=20, gcs_rpc_server_reconnect_timeout_s=60)], indirect=True)
@pytest.mark.skip(reason='GCS pubsub may lose messages after GCS restarts. Need to implement re-fetching state in GCS client.')
def test_gcs_server_restart_during_actor_creation(ray_start_regular_with_external_redis):
    if False:
        print('Hello World!')
    ids = []
    for i in range(0, 20):
        actor = Increase.remote()
        ids.append(actor.method.remote(1))
    ray._private.worker._global_node.kill_gcs_server()
    ray._private.worker._global_node.start_gcs_server()
    (ready, unready) = ray.wait(ids, num_returns=20, timeout=240)
    print('Ready objects is {}.'.format(ready))
    print('Unready objects is {}.'.format(unready))
    assert len(unready) == 0

@pytest.mark.parametrize('ray_start_cluster_head_with_external_redis', [generate_system_config_map(gcs_failover_worker_reconnect_timeout=2, gcs_rpc_server_reconnect_timeout_s=60, health_check_initial_delay_ms=0, health_check_period_ms=1000, health_check_failure_threshold=3, enable_autoscaler_v2=True)], indirect=True)
def test_autoscaler_init(ray_start_cluster_head_with_external_redis):
    if False:
        while True:
            i = 10
    '\n    Checks that autoscaler initializes properly after GCS restarts.\n    '
    cluster = ray_start_cluster_head_with_external_redis
    cluster.add_node()
    cluster.wait_for_nodes()
    nodes = ray.nodes()
    assert len(nodes) == 2
    assert nodes[0]['alive'] and nodes[1]['alive']
    head_node = cluster.head_node
    gcs_server_process = head_node.all_processes['gcs_server'][0].process
    gcs_server_pid = gcs_server_process.pid
    cluster.head_node.kill_gcs_server()
    gcs_server_process.wait()
    wait_for_pid_to_exit(gcs_server_pid, 300)
    cluster.head_node.start_gcs_server()
    from ray.autoscaler.v2.sdk import get_cluster_status
    status = get_cluster_status(ray.get_runtime_context().gcs_address)
    assert len(status.idle_nodes) == 2

@pytest.mark.parametrize('ray_start_cluster_head_with_external_redis', [generate_system_config_map(gcs_failover_worker_reconnect_timeout=2, gcs_rpc_server_reconnect_timeout_s=60, health_check_initial_delay_ms=0, health_check_period_ms=1000, health_check_failure_threshold=3)], indirect=True)
def test_node_failure_detector_when_gcs_server_restart(ray_start_cluster_head_with_external_redis):
    if False:
        print('Hello World!')
    'Checks that the node failure detector is correct when gcs server restart.\n\n    We set the cluster to timeout nodes after 2 seconds of heartbeats. We then\n    kill gcs server and remove the worker node and restart gcs server again to\n    check that the removed node will die finally.\n    '
    cluster = ray_start_cluster_head_with_external_redis
    worker = cluster.add_node()
    cluster.wait_for_nodes()
    nodes = ray.nodes()
    assert len(nodes) == 2
    assert nodes[0]['alive'] and nodes[1]['alive']
    to_be_removed_node = None
    for node in nodes:
        if node['RayletSocketName'] == worker.raylet_socket_name:
            to_be_removed_node = node
    assert to_be_removed_node is not None
    head_node = cluster.head_node
    gcs_server_process = head_node.all_processes['gcs_server'][0].process
    gcs_server_pid = gcs_server_process.pid
    cluster.head_node.kill_gcs_server()
    gcs_server_process.wait()
    wait_for_pid_to_exit(gcs_server_pid, 1000)
    raylet_process = worker.all_processes['raylet'][0].process
    raylet_pid = raylet_process.pid
    cluster.remove_node(worker, allow_graceful=False)
    raylet_process.wait()
    wait_for_pid_to_exit(raylet_pid)
    cluster.head_node.start_gcs_server()

    def condition():
        if False:
            return 10
        nodes = ray.nodes()
        assert len(nodes) == 2
        for node in nodes:
            if node['NodeID'] == to_be_removed_node['NodeID']:
                return not node['alive']
        return False
    wait_for_condition(condition, timeout=10)

@pytest.mark.parametrize('ray_start_regular_with_external_redis', [generate_system_config_map(gcs_failover_worker_reconnect_timeout=20, gcs_rpc_server_reconnect_timeout_s=60)], indirect=True)
def test_actor_raylet_resubscription(ray_start_regular_with_external_redis):
    if False:
        print('Hello World!')

    @ray.remote
    class A:

        def ready(self):
            if False:
                print('Hello World!')
            import os
            return os.getpid()
    actor = A.options(name='abc', max_restarts=0).remote()
    pid = ray.get(actor.ready.remote())
    print('actor is ready and kill gcs')
    ray._private.worker._global_node.kill_gcs_server()
    print('make actor exit')
    import psutil
    p = psutil.Process(pid)
    p.kill()
    from time import sleep
    sleep(1)
    print('start gcs')
    ray._private.worker._global_node.start_gcs_server()
    print('try actor method again')
    with pytest.raises(ray.exceptions.RayActorError):
        ray.get(actor.ready.remote())

@pytest.mark.parametrize('ray_start_regular_with_external_redis', [generate_system_config_map(gcs_failover_worker_reconnect_timeout=20, gcs_rpc_server_reconnect_timeout_s=60)], indirect=True)
def test_del_actor_after_gcs_server_restart(ray_start_regular_with_external_redis):
    if False:
        print('Hello World!')
    actor = Increase.options(name='abc').remote()
    result = ray.get(actor.method.remote(1))
    assert result == 3
    ray._private.worker._global_node.kill_gcs_server()
    ray._private.worker._global_node.start_gcs_server()
    actor_id = actor._actor_id.hex()
    del actor

    def condition():
        if False:
            return 10
        actor_status = ray._private.state.actors(actor_id=actor_id)
        if actor_status['State'] == convert_actor_state(gcs_utils.ActorTableData.DEAD):
            return True
        else:
            return False
    wait_for_condition(condition, timeout=10)
    with pytest.raises(ValueError):
        ray.get_actor('abc')

@pytest.mark.parametrize('ray_start_regular_with_external_redis', [generate_system_config_map(gcs_failover_worker_reconnect_timeout=20, gcs_rpc_server_reconnect_timeout_s=60)], indirect=True)
def test_worker_raylet_resubscription(tmp_path, ray_start_regular_with_external_redis):
    if False:
        while True:
            i = 10

    @ray.remote
    def long_run():
        if False:
            for i in range(10):
                print('nop')
        from time import sleep
        print('LONG_RUN')
        import os
        (tmp_path / 'long_run.pid').write_text(str(os.getpid()))
        sleep(10000)

    @ray.remote
    def bar():
        if False:
            for i in range(10):
                print('nop')
        import os
        return (os.getpid(), long_run.options(runtime_env={'env_vars': {'P': ''}}).remote())
    (pid, obj_ref) = ray.get(bar.remote())
    long_run_pid = None

    def condition():
        if False:
            i = 10
            return i + 15
        nonlocal long_run_pid
        long_run_pid = int((tmp_path / 'long_run.pid').read_text())
        return True
    wait_for_condition(condition, timeout=5)
    ray._private.worker._global_node.kill_gcs_server()
    ray._private.worker._global_node.start_gcs_server()
    sleep(4)
    p = psutil.Process(pid)
    p.kill()
    wait_for_pid_to_exit(long_run_pid, 5)

@pytest.mark.parametrize('ray_start_regular_with_external_redis', [generate_system_config_map(gcs_failover_worker_reconnect_timeout=20, gcs_rpc_server_reconnect_timeout_s=60)], indirect=True)
def test_core_worker_resubscription(tmp_path, ray_start_regular_with_external_redis):
    if False:
        print('Hello World!')
    from filelock import FileLock
    lock_file = str(tmp_path / 'lock')
    lock = FileLock(lock_file)
    lock.acquire()

    @ray.remote
    class Actor:

        def __init__(self):
            if False:
                while True:
                    i = 10
            lock = FileLock(lock_file)
            lock.acquire()

        def ready(self):
            if False:
                while True:
                    i = 10
            return
    a = Actor.remote()
    r = a.ready.remote()
    ray._private.worker._global_node.kill_gcs_server()
    lock.release()
    ray._private.worker._global_node.start_gcs_server()
    ray.get(r, timeout=5)

@pytest.mark.parametrize('ray_start_regular_with_external_redis', [generate_system_config_map(gcs_failover_worker_reconnect_timeout=20, gcs_rpc_server_reconnect_timeout_s=60)], indirect=True)
def test_detached_actor_restarts(ray_start_regular_with_external_redis):
    if False:
        print('Hello World!')

    @ray.remote
    class A:

        def ready(self):
            if False:
                i = 10
                return i + 15
            import os
            return os.getpid()
    a = A.options(name='a', lifetime='detached', max_restarts=-1).remote()
    pid = ray.get(a.ready.remote())
    ray._private.worker._global_node.kill_gcs_server()
    p = psutil.Process(pid)
    p.kill()
    ray._private.worker._global_node.start_gcs_server()
    while True:
        try:
            assert ray.get(a.ready.remote()) != pid
            break
        except ray.exceptions.RayActorError:
            continue

@pytest.mark.parametrize('auto_reconnect', [True, False])
def test_gcs_client_reconnect(ray_start_regular_with_external_redis, auto_reconnect):
    if False:
        while True:
            i = 10
    gcs_address = ray._private.worker.global_worker.gcs_client.address
    gcs_client = ray._raylet.GcsClient(address=gcs_address, nums_reconnect_retry=20 if auto_reconnect else 0)
    gcs_client.internal_kv_put(b'a', b'b', True, None)
    assert gcs_client.internal_kv_get(b'a', None) == b'b'
    passed = [False]

    def kv_get():
        if False:
            i = 10
            return i + 15
        if not auto_reconnect:
            with pytest.raises(Exception):
                gcs_client.internal_kv_get(b'a', None)
        else:
            assert gcs_client.internal_kv_get(b'a', None) == b'b'
        passed[0] = True
    ray._private.worker._global_node.kill_gcs_server()
    t = threading.Thread(target=kv_get)
    t.start()
    sleep(5)
    ray._private.worker._global_node.start_gcs_server()
    t.join()
    assert passed[0]

@pytest.mark.parametrize('auto_reconnect', [True, False])
def test_gcs_aio_client_reconnect(ray_start_regular_with_external_redis, auto_reconnect):
    if False:
        print('Hello World!')
    gcs_address = ray._private.worker.global_worker.gcs_client.address
    gcs_client = ray._raylet.GcsClient(address=gcs_address)
    gcs_client.internal_kv_put(b'a', b'b', True, None)
    assert gcs_client.internal_kv_get(b'a', None) == b'b'
    passed = [False]

    async def async_kv_get():
        if not auto_reconnect:
            with pytest.raises(Exception):
                gcs_aio_client = gcs_utils.GcsAioClient(address=gcs_address, nums_reconnect_retry=0)
                await gcs_aio_client.internal_kv_get(b'a', None)
        else:
            gcs_aio_client = gcs_utils.GcsAioClient(address=gcs_address, nums_reconnect_retry=20)
            assert await gcs_aio_client.internal_kv_get(b'a', None) == b'b'
        return True

    def kv_get():
        if False:
            print('Hello World!')
        import asyncio
        asyncio.set_event_loop(asyncio.new_event_loop())
        passed[0] = get_or_create_event_loop().run_until_complete(async_kv_get())
    ray._private.worker._global_node.kill_gcs_server()
    t = threading.Thread(target=kv_get)
    t.start()
    sleep(5)
    ray._private.worker._global_node.start_gcs_server()
    t.join()
    assert passed[0]

@pytest.mark.parametrize('ray_start_regular_with_external_redis', [{**generate_system_config_map(gcs_failover_worker_reconnect_timeout=20, gcs_rpc_server_reconnect_timeout_s=3600), 'namespace': 'actor'}], indirect=True)
def test_actor_workloads(ray_start_regular_with_external_redis):
    if False:
        i = 10
        return i + 15
    'This test cover the case to create actor while gcs is down\n    and also make sure existing actor continue to work even when\n    GCS is down.\n    '

    @ray.remote
    class Counter:

        def r(self, v):
            if False:
                i = 10
                return i + 15
            return v
    c = Counter.remote()
    r = ray.get(c.r.remote(10))
    assert r == 10
    print('GCS is killed')
    ray._private.worker._global_node.kill_gcs_server()
    print('Start to create a new actor')
    cc = Counter.remote()
    with pytest.raises(ray.exceptions.GetTimeoutError):
        ray.get(cc.r.remote(10), timeout=5)
    assert ray.get(c.r.remote(10)) == 10
    ray._private.worker._global_node.start_gcs_server()

    def f():
        if False:
            while True:
                i = 10
        assert ray.get(cc.r.remote(10)) == 10
    t = threading.Thread(target=f)
    t.start()
    t.join()
    c = Counter.options(lifetime='detached', name='C').remote()
    assert ray.get(c.r.remote(10)) == 10
    ray._private.worker._global_node.kill_gcs_server()
    sleep(2)
    assert ray.get(c.r.remote(10)) == 10
    ray._private.worker._global_node.start_gcs_server()
    from ray._private.test_utils import run_string_as_driver
    run_string_as_driver('\nimport ray\nray.init(\'auto\', namespace=\'actor\')\na = ray.get_actor("C")\nassert ray.get(a.r.remote(10)) == 10\n')

@pytest.mark.parametrize('ray_start_regular_with_external_redis', [{**generate_system_config_map(gcs_failover_worker_reconnect_timeout=20, gcs_rpc_server_reconnect_timeout_s=3600, gcs_server_request_timeout_seconds=10), 'namespace': 'actor'}], indirect=True)
def test_named_actor_workloads(ray_start_regular_with_external_redis):
    if False:
        for i in range(10):
            print('nop')
    'This test cover the case to create actor while gcs is down\n    and also make sure existing actor continue to work even when\n    GCS is down.\n    '

    @ray.remote
    class Counter:

        def r(self, v):
            if False:
                i = 10
                return i + 15
            return v
    c = Counter.options(name='c', lifetime='detached').remote()
    r = ray.get(c.r.remote(10))
    assert r == 10
    print('GCS is killed')
    ray.worker._global_node.kill_gcs_server()
    print('Start to create a new actor')
    with pytest.raises(ray.exceptions.GetTimeoutError):
        cc = Counter.options(name='cc', lifetime='detached').remote()
    assert ray.get(c.r.remote(10)) == 10
    ray.worker._global_node.start_gcs_server()
    cc = Counter.options(name='cc', lifetime='detached').remote()
    assert ray.get(cc.r.remote(10)) == 10

@pytest.mark.parametrize('ray_start_regular_with_external_redis', [{**generate_system_config_map(gcs_failover_worker_reconnect_timeout=20, gcs_rpc_server_reconnect_timeout_s=3600), 'namespace': 'actor'}], indirect=True)
def test_pg_actor_workloads(ray_start_regular_with_external_redis):
    if False:
        i = 10
        return i + 15
    from ray.util.placement_group import placement_group
    bundle1 = {'CPU': 1}
    pg = placement_group([bundle1], strategy='STRICT_PACK')
    ray.get(pg.ready())

    @ray.remote
    class Counter:

        def r(self, v):
            if False:
                i = 10
                return i + 15
            return v

        def pid(self):
            if False:
                while True:
                    i = 10
            import os
            return os.getpid()
    c = Counter.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)).remote()
    r = ray.get(c.r.remote(10))
    assert r == 10
    print('GCS is killed')
    pid = ray.get(c.pid.remote())
    ray.worker._global_node.kill_gcs_server()
    assert ray.get(c.r.remote(10)) == 10
    ray.worker._global_node.start_gcs_server()
    for _ in range(100):
        assert pid == ray.get(c.pid.remote())

@pytest.mark.parametrize('ray_start_regular_with_external_redis', [generate_system_config_map(gcs_failover_worker_reconnect_timeout=20, gcs_rpc_server_reconnect_timeout_s=60, gcs_server_request_timeout_seconds=10)], indirect=True)
def test_get_actor_when_gcs_is_down(ray_start_regular_with_external_redis):
    if False:
        print('Hello World!')

    @ray.remote
    def create_actor():
        if False:
            while True:
                i = 10

        @ray.remote
        class A:

            def pid(self):
                if False:
                    for i in range(10):
                        print('nop')
                return os.getpid()
        a = A.options(lifetime='detached', name='A').remote()
        ray.get(a.pid.remote())
    ray.get(create_actor.remote())
    ray._private.worker._global_node.kill_gcs_server()
    with pytest.raises(ray.exceptions.GetTimeoutError):
        ray.get_actor('A')

@pytest.mark.parametrize('ray_start_regular_with_external_redis', [generate_system_config_map(gcs_failover_worker_reconnect_timeout=20, gcs_rpc_server_reconnect_timeout_s=60, gcs_server_request_timeout_seconds=10)], indirect=True)
@pytest.mark.skip(reason="python publisher and subscriber doesn't handle gcs server failover")
def test_publish_and_subscribe_error_info(ray_start_regular_with_external_redis):
    if False:
        print('Hello World!')
    address_info = ray_start_regular_with_external_redis
    gcs_server_addr = address_info['gcs_address']
    subscriber = ray._raylet.GcsErrorSubscriber(address=gcs_server_addr)
    subscriber.subscribe()
    publisher = ray._raylet.GcsPublisher(address=gcs_server_addr)
    print('sending error message 1')
    publisher.publish_error(b'aaa_id', '', 'test error message 1')
    ray._private.worker._global_node.kill_gcs_server()
    ray._private.worker._global_node.start_gcs_server()
    print('sending error message 2')
    publisher.publish_error(b'bbb_id', '', 'test error message 2')
    print('done')
    (key_id, err) = subscriber.poll()
    assert key_id == b'bbb_id'
    assert err['error_message'] == 'test error message 2'
    subscriber.close()

@pytest.fixture
def redis_replicas(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.setenv('TEST_EXTERNAL_REDIS_REPLICAS', '3')

@pytest.mark.parametrize('ray_start_cluster_head_with_external_redis', [generate_system_config_map(gcs_failover_worker_reconnect_timeout=20, gcs_rpc_server_reconnect_timeout_s=60, gcs_server_request_timeout_seconds=10, redis_db_connect_retries=50)], indirect=True)
def test_redis_failureover(redis_replicas, ray_start_cluster_head_with_external_redis):
    if False:
        print('Hello World!')
    "This test is to cover ray cluster's behavior when Redis master failed.\n    The management of the Redis cluster is not covered by Ray, but Ray should handle\n    the failure correctly.\n    For this test we ensure:\n    - When Redis master failed, Ray should crash (TODO: make ray automatically switch to\n      new master).\n    - After Redis recovered, Ray should be able to use the new Master.\n    - When the master becomes slaves, Ray should crash.\n    "
    cluster = ray_start_cluster_head_with_external_redis
    import redis
    redis_addr = os.environ.get('RAY_REDIS_ADDRESS')
    (ip, port) = redis_addr.split(':')
    redis_cli = redis.Redis(ip, port)

    def get_connected_nodes():
        if False:
            print('Hello World!')
        return [(k, v) for (k, v) in redis_cli.cluster('nodes').items() if v['connected']]
    wait_for_condition(lambda : len(get_connected_nodes()) == int(os.environ.get('TEST_EXTERNAL_REDIS_REPLICAS')))
    nodes = redis_cli.cluster('nodes')
    leader_cli = None
    follower_cli = []
    for addr in nodes:
        (ip, port) = addr.split(':')
        cli = redis.Redis(ip, port)
        meta = nodes[addr]
        flags = meta['flags'].split(',')
        if 'master' in flags:
            leader_cli = cli
            print('LEADER', addr, redis_addr)
        else:
            follower_cli.append(cli)
    leader_pid = leader_cli.info()['process_id']

    @ray.remote(max_restarts=-1)
    class Counter:

        def r(self, v):
            if False:
                while True:
                    i = 10
            return v

        def pid(self):
            if False:
                while True:
                    i = 10
            import os
            return os.getpid()
    c = Counter.options(name='c', namespace='test', lifetime='detached').remote()
    c_pid = ray.get(c.pid.remote())
    c_process = psutil.Process(pid=c_pid)
    r = ray.get(c.r.remote(10))
    assert r == 10
    head_node = cluster.head_node
    gcs_server_process = head_node.all_processes['gcs_server'][0].process
    gcs_server_pid = gcs_server_process.pid
    leader_cli.set('_hole', '0')
    wait_for_condition(lambda : all([b'_hole' in f.keys('*') for f in follower_cli]))
    leader_process = psutil.Process(pid=leader_pid)
    leader_process.kill()
    print('>>> Waiting gcs server to exit', gcs_server_pid)
    wait_for_pid_to_exit(gcs_server_pid, 1000)
    print('GCS killed')
    follower_cli[0].cluster('failover', 'takeover')
    wait_for_condition(lambda : len(get_connected_nodes()) == int(os.environ.get('TEST_EXTERNAL_REDIS_REPLICAS')) - 1)
    c_process.kill()
    cluster.head_node.kill_gcs_server(False)
    print('Start gcs')
    sleep(2)
    cluster.head_node.start_gcs_server()
    assert len(ray.nodes()) == 1
    assert ray.nodes()[0]['alive']
    driver_script = f"""\nimport ray\nray.init('{cluster.address}')\n@ray.remote\ndef f():\n    return 10\nassert ray.get(f.remote()) == 10\n\nc = ray.get_actor("c", namespace="test")\nv = ray.get(c.r.remote(10))\nassert v == 10\nprint("DONE")\n"""
    wait_for_condition(lambda : 'DONE' in run_string_as_driver(driver_script))
    follower_cli[1].cluster('failover', 'takeover')
    head_node = cluster.head_node
    gcs_server_process = head_node.all_processes['gcs_server'][0].process
    gcs_server_pid = gcs_server_process.pid
    print('>>> Waiting gcs server to exit', gcs_server_pid)
    wait_for_pid_to_exit(gcs_server_pid, 10000)

@pytest.mark.parametrize('ray_start_regular', [generate_system_config_map(enable_cluster_auth=True)], indirect=True)
def test_cluster_id(ray_start_regular):
    if False:
        return 10
    raylet_proc = ray._private.worker._global_node.all_processes[ray_constants.PROCESS_TYPE_RAYLET][0].process

    def check_raylet_healthy():
        if False:
            while True:
                i = 10
        return raylet_proc.poll() is None
    wait_for_condition(lambda : check_raylet_healthy())
    for i in range(10):
        assert check_raylet_healthy()
        sleep(1)
    ray._private.worker._global_node.kill_gcs_server()
    ray._private.worker._global_node.start_gcs_server()
    if not enable_external_redis():
        wait_for_condition(lambda : not check_raylet_healthy())
    else:
        for i in range(10):
            assert check_raylet_healthy()
            sleep(1)

def test_session_name(ray_start_cluster):
    if False:
        while True:
            i = 10
    cluster = ray_start_cluster
    cluster.add_node()
    cluster.wait_for_nodes()
    head_node = cluster.head_node
    session_dir = head_node.get_session_dir_path()
    gcs_server_process = head_node.all_processes['gcs_server'][0].process
    gcs_server_pid = gcs_server_process.pid
    cluster.remove_node(head_node, allow_graceful=False)
    gcs_server_process.wait()
    wait_for_pid_to_exit(gcs_server_pid, 1000)
    cluster.add_node()
    head_node = cluster.head_node
    new_session_dir = head_node.get_session_dir_path()
    if not enable_external_redis():
        assert session_dir != new_session_dir
    else:
        assert session_dir == new_session_dir

@pytest.mark.parametrize('ray_start_regular_with_external_redis', [generate_system_config_map(gcs_failover_worker_reconnect_timeout=20, gcs_rpc_server_reconnect_timeout_s=60, gcs_server_request_timeout_seconds=10, raylet_liveness_self_check_interval_ms=3000)], indirect=True)
def test_redis_data_loss_no_leak(ray_start_regular_with_external_redis):
    if False:
        while True:
            i = 10

    @ray.remote
    def create_actor():
        if False:
            i = 10
            return i + 15

        @ray.remote
        class A:

            def pid(self):
                if False:
                    print('Hello World!')
                return os.getpid()
        a = A.options(lifetime='detached', name='A').remote()
        ray.get(a.pid.remote())
    ray.get(create_actor.remote())
    ray._private.worker._global_node.kill_gcs_server()
    redis_addr = os.environ.get('RAY_REDIS_ADDRESS')
    import redis
    (ip, port) = redis_addr.split(':')
    cli = redis.Redis(ip, port)
    cli.flushall()
    raylet_proc = ray._private.worker._global_node.all_processes[ray_constants.PROCESS_TYPE_RAYLET][0].process

    def check_raylet_healthy():
        if False:
            for i in range(10):
                print('nop')
        return raylet_proc.poll() is None
    wait_for_condition(lambda : check_raylet_healthy())
    ray._private.worker._global_node.start_gcs_server()
    wait_for_condition(lambda : not check_raylet_healthy())

def test_redis_logs(external_redis):
    if False:
        i = 10
        return i + 15
    try:
        import subprocess
        process = subprocess.Popen(['ray', 'start', '--head'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (stdout, stderr) = process.communicate(timeout=30)
        print(stdout.decode())
        print(stderr.decode())
        assert 'redis_context.cc' not in stderr.decode()
        assert 'redis_context.cc' not in stdout.decode()
        assert 'Resolve Redis address' not in stderr.decode()
        assert 'Resolve Redis address' not in stdout.decode()
    finally:
        from click.testing import CliRunner
        import ray.scripts.scripts as scripts
        runner = CliRunner(env={'RAY_USAGE_STATS_PROMPT_ENABLED': '0'})
        runner.invoke(scripts.stop, ['--force'])

@pytest.mark.parametrize('ray_start_cluster_head_with_external_redis', [generate_system_config_map(gcs_failover_worker_reconnect_timeout=20, gcs_rpc_server_reconnect_timeout_s=2)], indirect=True)
def test_job_finished_after_head_node_restart(ray_start_cluster_head_with_external_redis):
    if False:
        print('Hello World!')
    cluster = ray_start_cluster_head_with_external_redis
    head_node = cluster.head_node
    client = JobSubmissionClient(head_node.address)
    submission_id = client.submit_job(entrypoint="python -c 'import ray; ray.init(); print(ray.cluster_resources());             import time; time.sleep(1000)'")

    def get_job_info(submission_id):
        if False:
            print('Hello World!')
        gcs_client = GcsClient(cluster.address)
        all_job_info = gcs_client.get_all_job_info()
        return list(filter(lambda job_info: 'job_submission_id' in job_info.config.metadata and job_info.config.metadata['job_submission_id'] == submission_id, list(all_job_info.values())))

    def _check_job_running(submission_id: str) -> bool:
        if False:
            print('Hello World!')
        job_infos = get_job_info(submission_id)
        if len(job_infos) == 0:
            return False
        job_info = job_infos[0].job_info
        return job_info.status == JobStatus.RUNNING
    wait_for_condition(_check_job_running, submission_id=submission_id, timeout=10)
    ray.shutdown()
    gcs_server_process = head_node.all_processes['gcs_server'][0].process
    gcs_server_pid = gcs_server_process.pid
    cluster.remove_node(head_node)
    gcs_server_process.wait()
    wait_for_pid_to_exit(gcs_server_pid, 1000)
    cluster.add_node()
    ray.init(cluster.address)

    def _check_job_is_dead(submission_id: str) -> bool:
        if False:
            return 10
        job_infos = get_job_info(submission_id)
        if len(job_infos) == 0:
            return False
        job_info = job_infos[0]
        return job_info.is_dead
    wait_for_condition(_check_job_is_dead, submission_id=submission_id, timeout=10)
if __name__ == '__main__':
    import pytest
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))