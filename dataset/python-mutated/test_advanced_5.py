import logging
import sys
import time
import numpy as np
import pytest
import ray.cluster_utils
from ray._private.test_utils import SignalActor, client_test_enabled
if client_test_enabled():
    from ray.util.client import ray
else:
    import ray
logger = logging.getLogger(__name__)

def test_task_arguments_inline_bytes_limit(ray_start_cluster_enabled):
    if False:
        i = 10
        return i + 15
    cluster = ray_start_cluster_enabled
    cluster.add_node(num_cpus=1, resources={'pin_head': 1}, _system_config={'max_direct_call_object_size': 100 * 1024, 'task_rpc_inlined_bytes_limit': 18 * 1024, 'max_grpc_message_size': 20 * 1024})
    cluster.add_node(num_cpus=1, resources={'pin_worker': 1})
    ray.init(address=cluster.address)

    @ray.remote(resources={'pin_worker': 1})
    def foo(ref1, ref2, ref3):
        if False:
            i = 10
            return i + 15
        return ref1 == ref2 + ref3

    @ray.remote(resources={'pin_head': 1})
    def bar():
        if False:
            while True:
                i = 10
        return ray.get(foo.remote(np.random.rand(1024), np.random.rand(1024), np.random.rand(1024)))
    ray.get(bar.remote())

def test_schedule_actor_and_normal_task(ray_start_cluster_enabled):
    if False:
        for i in range(10):
            print('nop')
    cluster = ray_start_cluster_enabled
    cluster.add_node(memory=1024 ** 3, _system_config={'gcs_actor_scheduling_enabled': True})
    ray.init(address=cluster.address)
    cluster.wait_for_nodes()

    @ray.remote(memory=600 * 1024 ** 2, num_cpus=0.01)
    class Foo:

        def method(self):
            if False:
                print('Hello World!')
            return 2

    @ray.remote(memory=600 * 1024 ** 2, num_cpus=0.01)
    def fun(singal1, signal_actor2):
        if False:
            return 10
        signal_actor2.send.remote()
        ray.get(singal1.wait.remote())
        return 1
    singal1 = SignalActor.remote()
    signal2 = SignalActor.remote()
    o1 = fun.remote(singal1, signal2)
    ray.get(signal2.wait.remote())
    foo = Foo.remote()
    o2 = foo.method.remote()
    (ready_list, remaining_list) = ray.wait([o2], timeout=2)
    assert len(ready_list) == 0 and len(remaining_list) == 1
    ray.get(singal1.send.remote())
    assert ray.get(o1) == 1
    assert ray.get(o2) == 2

def test_schedule_many_actors_and_normal_tasks(ray_start_cluster):
    if False:
        return 10
    cluster = ray_start_cluster
    node_count = 10
    actor_count = 50
    each_actor_task_count = 50
    normal_task_count = 1000
    node_memory = 2 * 1024 ** 3
    for i in range(node_count):
        cluster.add_node(memory=node_memory, _system_config={'gcs_actor_scheduling_enabled': True} if i == 0 else {})
    ray.init(address=cluster.address)
    cluster.wait_for_nodes()

    @ray.remote(memory=100 * 1024 ** 2, num_cpus=0.01)
    class Foo:

        def method(self):
            if False:
                while True:
                    i = 10
            return 2

    @ray.remote(memory=100 * 1024 ** 2, num_cpus=0.01)
    def fun():
        if False:
            print('Hello World!')
        return 1
    normal_task_object_list = [fun.remote() for _ in range(normal_task_count)]
    actor_list = [Foo.remote() for _ in range(actor_count)]
    actor_object_list = [actor.method.remote() for _ in range(each_actor_task_count) for actor in actor_list]
    for object in ray.get(actor_object_list):
        assert object == 2
    for object in ray.get(normal_task_object_list):
        assert object == 1

@pytest.mark.parametrize('args', [[5, 20], [5, 3]])
def test_actor_distribution_balance(ray_start_cluster_enabled, args):
    if False:
        while True:
            i = 10
    cluster = ray_start_cluster_enabled
    node_count = args[0]
    actor_count = args[1]
    for i in range(node_count):
        cluster.add_node(memory=1024 ** 3, _system_config={'gcs_actor_scheduling_enabled': True} if i == 0 else {})
    ray.init(address=cluster.address)
    cluster.wait_for_nodes()

    @ray.remote(memory=100 * 1024 ** 2, num_cpus=0.01, scheduling_strategy='SPREAD')
    class Foo:

        def method(self):
            if False:
                for i in range(10):
                    print('nop')
            return ray._private.worker.global_worker.node.unique_id
    actor_distribution = {}
    actor_list = [Foo.remote() for _ in range(actor_count)]
    for actor in actor_list:
        node_id = ray.get(actor.method.remote())
        if node_id not in actor_distribution.keys():
            actor_distribution[node_id] = []
        actor_distribution[node_id].append(actor)
    if node_count >= actor_count:
        assert len(actor_distribution) == actor_count
        for (node_id, actors) in actor_distribution.items():
            assert len(actors) == 1
    else:
        assert len(actor_distribution) == node_count
        for (node_id, actors) in actor_distribution.items():
            assert len(actors) <= int(actor_count / node_count)

def test_worker_lease_reply_with_resources(ray_start_cluster_enabled):
    if False:
        i = 10
        return i + 15
    cluster = ray_start_cluster_enabled
    cluster.add_node(memory=2000 * 1024 ** 2, num_cpus=1, _system_config={'gcs_resource_report_poll_period_ms': 1000000, 'gcs_actor_scheduling_enabled': True})
    node2 = cluster.add_node(memory=1000 * 1024 ** 2, num_cpus=1)
    ray.init(address=cluster.address)
    cluster.wait_for_nodes()

    @ray.remote(memory=1500 * 1024 ** 2, num_cpus=0.01)
    def fun(signal):
        if False:
            print('Hello World!')
        signal.send.remote()
        time.sleep(30)
        return 0
    signal = SignalActor.remote()
    fun.remote(signal)
    ray.get(signal.wait.remote())

    @ray.remote(memory=800 * 1024 ** 2, num_cpus=0.01)
    class Foo:

        def method(self):
            if False:
                while True:
                    i = 10
            return ray._private.worker.global_worker.node.unique_id
    foo1 = Foo.remote()
    o1 = foo1.method.remote()
    (ready_list, remaining_list) = ray.wait([o1], timeout=10)
    assert len(ready_list) == 1 and len(remaining_list) == 0
    assert ray.get(o1) == node2.unique_id
if __name__ == '__main__':
    import os
    import pytest
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))