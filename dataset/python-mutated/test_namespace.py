import sys
import time
import pytest
import ray
from ray._private import ray_constants
from ray._private.test_utils import get_error_message, init_error_pubsub, run_string_as_driver
from ray.cluster_utils import Cluster

def test_isolation(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    info = ray.init(namespace='namespace')
    address = info['address']
    driver_template = '\nimport ray\n\nray.init(address="{}", namespace="{}")\n\n@ray.remote\nclass DetachedActor:\n    def ping(self):\n        return "pong from other job"\n\nactor = DetachedActor.options(name="Pinger", lifetime="detached").remote()\nray.get(actor.ping.remote())\n    '
    run_string_as_driver(driver_template.format(address, 'different'))

    @ray.remote
    class Actor:

        def ping(self):
            if False:
                for i in range(10):
                    print('nop')
            return 'pong'
    probe = Actor.options(name='Pinger').remote()
    assert ray.get(probe.ping.remote()) == 'pong'
    del probe
    actor_removed = False
    for _ in range(50):
        try:
            ray.get_actor('Pinger')
        except ValueError:
            actor_removed = True
            break
        else:
            time.sleep(0.1)
    assert actor_removed, 'This is an anti-flakey test measure'
    with pytest.raises(ValueError, match='Failed to look up actor with name'):
        ray.get_actor('Pinger')
    run_string_as_driver(driver_template.format(address, 'namespace'))
    detached_actor = ray.get_actor('Pinger')
    assert ray.get(detached_actor.ping.remote()) == 'pong from other job'
    with pytest.raises(ValueError, match='The name .* is already taken'):
        Actor.options(name='Pinger', lifetime='detached').remote()

def test_placement_groups(shutdown_only):
    if False:
        while True:
            i = 10
    info = ray.init(namespace='namespace')
    address = info['address']
    driver_template = '\nimport ray\n\nray.init(address="{}", namespace="{}")\n\npg = ray.util.placement_group(bundles=[dict(CPU=1)], name="hello",\n    lifetime="detached")\nray.get(pg.ready())\n    '
    run_string_as_driver(driver_template.format(address, 'different'))
    probe = ray.util.placement_group(bundles=[{'CPU': 1}], name='hello')
    ray.get(probe.ready())
    ray.util.remove_placement_group(probe)
    removed = False
    for _ in range(50):
        try:
            ray.util.get_placement_group('hello')
        except ValueError:
            removed = True
            break
        else:
            time.sleep(0.1)
    assert removed, 'This is an anti-flakey test measure'
    run_string_as_driver(driver_template.format(address, 'namespace'))

def test_default_namespace(shutdown_only):
    if False:
        return 10
    info = ray.init(namespace='namespace')
    address = info['address']
    driver_template = '\nimport ray\n\nray.init(address="{}")\n\n@ray.remote\nclass DetachedActor:\n    def ping(self):\n        return "pong from other job"\n\nactor = DetachedActor.options(name="Pinger", lifetime="detached").remote()\nray.get(actor.ping.remote())\n    '
    run_string_as_driver(driver_template.format(address))
    run_string_as_driver(driver_template.format(address))

def test_namespace_in_job_config(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    job_config = ray.job_config.JobConfig(ray_namespace='namespace')
    info = ray.init(job_config=job_config)
    address = info['address']
    driver_template = '\nimport ray\n\nray.init(address="{}", namespace="namespace")\n\n@ray.remote\nclass DetachedActor:\n    def ping(self):\n        return "pong from other job"\n\nactor = DetachedActor.options(name="Pinger", lifetime="detached").remote()\nray.get(actor.ping.remote())\n    '
    run_string_as_driver(driver_template.format(address))
    act = ray.get_actor('Pinger')
    assert ray.get(act.ping.remote()) == 'pong from other job'

def test_detached_warning(shutdown_only):
    if False:
        return 10
    ray.init()

    @ray.remote
    class DetachedActor:

        def ping(self):
            if False:
                for i in range(10):
                    print('nop')
            return 'pong'
    error_pubsub = init_error_pubsub()
    actor = DetachedActor.options(name='Pinger', lifetime='detached').remote()
    errors = get_error_message(error_pubsub, 1, None)
    error = errors.pop()
    assert error['type'] == ray_constants.DETACHED_ACTOR_ANONYMOUS_NAMESPACE_ERROR

def test_namespace_client():
    if False:
        for i in range(10):
            print('nop')
    cluster = Cluster()
    cluster.add_node(num_cpus=4, ray_client_server_port=8080)
    cluster.wait_for_nodes(1)
    template = '\nimport ray\nray.util.connect("{address}", namespace="{namespace}")\n\n@ray.remote\nclass DetachedActor:\n    def ping(self):\n        return "pong from other job"\n\nactor = DetachedActor.options(name="Pinger", lifetime="detached").remote()\nray.get(actor.ping.remote())\nprint("Done!!!")\n    '
    print(run_string_as_driver(template.format(address='localhost:8080', namespace='test')))
    ray.util.connect('localhost:8080', namespace='test')
    pinger = ray.get_actor('Pinger')
    assert ray.get(pinger.ping.remote()) == 'pong from other job'
    ray.util.disconnect()
    cluster.shutdown()
    ray._private.client_mode_hook._explicitly_disable_client_mode()

def test_runtime_context(shutdown_only):
    if False:
        while True:
            i = 10
    ray.init(namespace='abc')
    namespace = ray.get_runtime_context().namespace
    assert namespace == 'abc'
    assert namespace == ray.get_runtime_context().get()['namespace']

def test_namespace_validation(shutdown_only):
    if False:
        print('Hello World!')
    with pytest.raises(TypeError):
        ray.init(namespace=123)
    ray.shutdown()
    with pytest.raises(ValueError):
        ray.init(namespace='')
    ray.shutdown()
    ray.init(namespace='abc')

    @ray.remote
    class A:
        pass
    with pytest.raises(TypeError):
        A.options(namespace=123).remote()
    with pytest.raises(ValueError):
        A.options(namespace='').remote()
    A.options(name='a', namespace='test', lifetime='detached').remote()
    with pytest.raises(TypeError):
        ray.get_actor('a', namespace=123)
    with pytest.raises(ValueError):
        ray.get_actor('a', namespace='')
    ray.get_actor('a', namespace='test')
if __name__ == '__main__':
    import os
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))