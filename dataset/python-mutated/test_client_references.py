from concurrent.futures import Future
import pytest
import ray as real_ray
from ray._private.test_utils import object_memory_usage, wait_for_condition
from ray._raylet import ActorID, ObjectRef
from ray.util.client import _ClientContext
from ray.util.client.common import ClientActorRef, ClientObjectRef
from ray.util.client.ray_client_helpers import ray_start_client_server, ray_start_client_server_pair, ray_start_cluster_client_server_pair

def test_client_object_ref_basics(ray_start_regular):
    if False:
        for i in range(10):
            print('nop')
    with ray_start_client_server_pair() as pair:
        (ray, server) = pair
        ref = ray.put('Hello World')
        assert isinstance(ref, ClientObjectRef)
        assert isinstance(ref, ObjectRef)
        with pytest.raises(Exception):
            ClientObjectRef(b'\x00')
        obj_id = b'\x00' * 28
        fut = Future()
        fut.set_result(obj_id)
        server_ref = ObjectRef(obj_id)
        for client_ref in [ClientObjectRef(obj_id), ClientObjectRef(fut)]:
            client_members = set(client_ref.__dir__())
            server_members = set(server_ref.__dir__())
            client_members = {m for m in client_ref.__dir__() if not m.startswith('_')}
            server_members = {m for m in server_ref.__dir__() if not m.startswith('_')}
            assert client_members.difference(server_members) == {'id'}
            assert server_members.difference(client_members) == set()
            assert client_ref == ClientObjectRef(obj_id)
            assert client_ref != ref
            assert client_ref != server_ref
            assert client_ref.__repr__() == f'ClientObjectRef({obj_id.hex()})'
            assert client_ref.binary() == obj_id
            assert client_ref.hex() == obj_id.hex()
            assert not client_ref.is_nil()
            assert client_ref.task_id() == server_ref.task_id()
            assert client_ref.job_id() == server_ref.job_id()

def test_client_actor_ref_basics(ray_start_regular):
    if False:
        while True:
            i = 10
    with ray_start_client_server_pair() as pair:
        (ray, server) = pair

        @ray.remote
        class Counter:

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.acc = 0

            def inc(self):
                if False:
                    return 10
                self.acc += 1

            def get(self):
                if False:
                    return 10
                return self.acc
        counter = Counter.remote()
        ref = counter.actor_ref
        assert isinstance(ref, ClientActorRef)
        assert isinstance(ref, ActorID)
        with pytest.raises(Exception):
            ClientActorRef(b'\x00')
        actor_id = b'\x00' * 16
        fut = Future()
        fut.set_result(actor_id)
        server_ref = ActorID(actor_id)
        for client_ref in [ClientActorRef(actor_id), ClientActorRef(fut)]:
            client_members = {m for m in client_ref.__dir__() if not m.startswith('_')}
            server_members = {m for m in server_ref.__dir__() if not m.startswith('_')}
            assert client_members.difference(server_members) == {'id'}
            assert server_members.difference(client_members) == set()
            assert client_ref == ClientActorRef(actor_id)
            assert client_ref != ref
            assert client_ref != server_ref
            assert client_ref.__repr__() == f'ClientActorRef({actor_id.hex()})'
            assert client_ref.binary() == actor_id
            assert client_ref.hex() == actor_id.hex()
            assert not client_ref.is_nil()

def server_object_ref_count(server, n):
    if False:
        while True:
            i = 10
    assert server is not None

    def test_cond():
        if False:
            i = 10
            return i + 15
        if len(server.task_servicer.object_refs) == 0:
            return n == 0
        client_id = list(server.task_servicer.object_refs.keys())[0]
        return len(server.task_servicer.object_refs[client_id]) == n
    return test_cond

def server_actor_ref_count(server, n):
    if False:
        while True:
            i = 10
    assert server is not None

    def test_cond():
        if False:
            return 10
        if len(server.task_servicer.actor_refs) == 0:
            return n == 0
        return len(server.task_servicer.actor_refs) == n
    return test_cond

@pytest.mark.parametrize('ray_start_cluster', [{'num_nodes': 1, 'do_init': False}], indirect=True)
def test_delete_refs_on_disconnect(ray_start_cluster):
    if False:
        print('Hello World!')
    cluster = ray_start_cluster
    with ray_start_cluster_client_server_pair(cluster.address) as pair:
        (ray, server) = pair

        @ray.remote
        def f(x):
            if False:
                i = 10
                return i + 15
            return x + 2
        thing1 = f.remote(6)
        thing2 = ray.put('Hello World')
        assert server_object_ref_count(server, 3)()
        assert ray.get(thing1) == 8
        ray.close()
        wait_for_condition(server_object_ref_count(server, 0), timeout=5)
        real_ray.init(address=cluster.address, namespace='default_test_namespace')

        def test_cond():
            if False:
                for i in range(10):
                    print('nop')
            return object_memory_usage() == 0
        wait_for_condition(test_cond, timeout=5)

def test_delete_ref_on_object_deletion(ray_start_regular):
    if False:
        for i in range(10):
            print('nop')
    with ray_start_client_server_pair() as pair:
        (ray, server) = pair
        vals = {'ref': ray.put('Hello World'), 'ref2': ray.put('This value stays')}
        del vals['ref']
        wait_for_condition(server_object_ref_count(server, 1), timeout=5)

@pytest.mark.parametrize('ray_start_cluster', [{'num_nodes': 1, 'do_init': False}], indirect=True)
def test_delete_actor_on_disconnect(ray_start_cluster):
    if False:
        for i in range(10):
            print('nop')
    cluster = ray_start_cluster
    with ray_start_cluster_client_server_pair(cluster.address) as pair:
        (ray, server) = pair

        @ray.remote
        class Accumulator:

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.acc = 0

            def inc(self):
                if False:
                    while True:
                        i = 10
                self.acc += 1

            def get(self):
                if False:
                    return 10
                return self.acc
        actor = Accumulator.remote()
        actor.inc.remote()
        assert server_actor_ref_count(server, 1)()
        assert ray.get(actor.get.remote()) == 1
        ray.close()
        wait_for_condition(server_actor_ref_count(server, 0), timeout=5)

        def test_cond():
            if False:
                return 10
            alive_actors = [v for v in real_ray._private.state.actors().values() if v['State'] != 'DEAD']
            return len(alive_actors) == 0
        real_ray.init(address=cluster.address, namespace='default_test_namespace')
        wait_for_condition(test_cond, timeout=10)

def test_delete_actor(ray_start_regular):
    if False:
        i = 10
        return i + 15
    with ray_start_client_server_pair() as pair:
        (ray, server) = pair

        @ray.remote
        class Accumulator:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.acc = 0

            def inc(self):
                if False:
                    return 10
                self.acc += 1
        actor = Accumulator.remote()
        actor.inc.remote()
        actor2 = Accumulator.remote()
        actor2.inc.remote()
        assert server_actor_ref_count(server, 2)()
        del actor
        wait_for_condition(server_actor_ref_count(server, 1), timeout=5)

def test_simple_multiple_references(ray_start_regular):
    if False:
        i = 10
        return i + 15
    with ray_start_client_server() as ray:

        @ray.remote
        class A:

            def __init__(self):
                if False:
                    print('Hello World!')
                self.x = ray.put('hi')

            def get(self):
                if False:
                    print('Hello World!')
                return [self.x]
        a = A.remote()
        ref1 = ray.get(a.get.remote())[0]
        ref2 = ray.get(a.get.remote())[0]
        del a
        assert ray.get(ref1) == 'hi'
        del ref1
        assert ray.get(ref2) == 'hi'
        del ref2

def test_named_actor_refcount(ray_start_regular):
    if False:
        return 10
    with ray_start_client_server_pair() as (ray, server):

        @ray.remote
        class ActorTest:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self._counter = 0

            def bump(self):
                if False:
                    while True:
                        i = 10
                self._counter += 1

            def check(self):
                if False:
                    i = 10
                    return i + 15
                return self._counter
        ActorTest.options(name='actor', lifetime='detached').remote()

        def connect_api():
            if False:
                return 10
            api = _ClientContext()
            api.connect('localhost:50051', namespace='default_test_namespace')
            api.get_actor('actor')
            return api

        def check_owners(size):
            if False:
                print('Hello World!')
            return size == sum((len(x) for x in server.task_servicer.actor_owners.values()))
        apis = [connect_api() for i in range(3)]
        assert check_owners(3)
        assert len(server.task_servicer.actor_refs) == 1
        assert len(server.task_servicer.named_actors) == 1
        apis.pop(0).disconnect()
        assert check_owners(2)
        assert len(server.task_servicer.actor_refs) == 1
        assert len(server.task_servicer.named_actors) == 1
        apis.pop(0).disconnect()
        assert check_owners(1)
        assert len(server.task_servicer.actor_refs) == 1
        assert len(server.task_servicer.named_actors) == 1
        apis.pop(0).disconnect()
        assert check_owners(0)
        assert len(server.task_servicer.actor_refs) == 1
        assert len(server.task_servicer.named_actors) == 1
if __name__ == '__main__':
    import os
    import sys
    import pytest
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))