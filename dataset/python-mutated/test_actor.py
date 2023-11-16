import datetime
import os
import random
import sys
import tempfile
import numpy as np
import pytest
import ray
from ray import cloudpickle as pickle
from ray._private import ray_constants
from ray._private.test_utils import client_test_enabled, wait_for_condition, wait_for_pid_to_exit
from ray.actor import ActorClassInheritanceException
from ray.tests.client_test_utils import create_remote_signal_actor
from ray._private.test_utils import SignalActor
import setproctitle
try:
    import pytest_timeout
except ImportError:
    pytest_timeout = None

@pytest.mark.parametrize('set_enable_auto_connect', ['1', '0'], indirect=True)
def test_caching_actors(shutdown_only, set_enable_auto_connect):
    if False:
        print('Hello World!')

    @ray.remote
    class Foo:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def get_val(self):
            if False:
                while True:
                    i = 10
            return 3
    if set_enable_auto_connect == '0':
        with pytest.raises(Exception):
            f = Foo.remote()
        ray.init(num_cpus=1)
    else:
        f = Foo.remote()
    f = Foo.remote()
    assert ray.get(f.get_val.remote()) == 3

def test_not_reusing_task_workers(shutdown_only):
    if False:
        for i in range(10):
            print('nop')

    @ray.remote
    def create_ref():
        if False:
            return 10
        ref = ray.put(np.zeros(100000000))
        return ref

    @ray.remote
    class Actor:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            return

        def foo(self):
            if False:
                for i in range(10):
                    print('nop')
            return
    ray.init(num_cpus=1, object_store_memory=1000000000)
    wrapped_ref = create_ref.remote()
    print(ray.get(ray.get(wrapped_ref)))
    a = Actor.remote()
    ray.get(a.foo.remote())
    del a
    for _ in range(10):
        ray.put(np.zeros(100000000))
    print(ray.get(ray.get(wrapped_ref)))

def test_remote_function_within_actor(ray_start_10_cpus):
    if False:
        for i in range(10):
            print('nop')
    val1 = 1
    val2 = 2

    @ray.remote
    def f(x):
        if False:
            i = 10
            return i + 15
        return val1 + x

    @ray.remote
    def g(x):
        if False:
            print('Hello World!')
        return ray.get(f.remote(x))

    @ray.remote
    class Actor:

        def __init__(self, x):
            if False:
                for i in range(10):
                    print('nop')
            self.x = x
            self.y = val2
            self.object_refs = [f.remote(i) for i in range(5)]
            self.values2 = ray.get([f.remote(i) for i in range(5)])

        def get_values(self):
            if False:
                i = 10
                return i + 15
            return (self.x, self.y, self.object_refs, self.values2)

        def f(self):
            if False:
                while True:
                    i = 10
            return [f.remote(i) for i in range(5)]

        def g(self):
            if False:
                i = 10
                return i + 15
            return ray.get([g.remote(i) for i in range(5)])

        def h(self, object_refs):
            if False:
                while True:
                    i = 10
            return ray.get(object_refs)
    actor = Actor.remote(1)
    values = ray.get(actor.get_values.remote())
    assert values[0] == 1
    assert values[1] == val2
    assert ray.get(values[2]) == list(range(1, 6))
    assert values[3] == list(range(1, 6))
    assert ray.get(ray.get(actor.f.remote())) == list(range(1, 6))
    assert ray.get(actor.g.remote()) == list(range(1, 6))
    assert ray.get(actor.h.remote([f.remote(i) for i in range(5)])) == list(range(1, 6))

def test_define_actor_within_actor(ray_start_10_cpus):
    if False:
        return 10

    @ray.remote
    class Actor1:

        def __init__(self, x):
            if False:
                print('Hello World!')
            self.x = x

        def new_actor(self, z):
            if False:
                print('Hello World!')

            @ray.remote
            class Actor2:

                def __init__(self, x):
                    if False:
                        while True:
                            i = 10
                    self.x = x

                def get_value(self):
                    if False:
                        print('Hello World!')
                    return self.x
            self.actor2 = Actor2.remote(z)

        def get_values(self, z):
            if False:
                return 10
            self.new_actor(z)
            return (self.x, ray.get(self.actor2.get_value.remote()))
    actor1 = Actor1.remote(3)
    assert ray.get(actor1.get_values.remote(5)) == (3, 5)

def test_use_actor_within_actor(ray_start_10_cpus):
    if False:
        while True:
            i = 10

    @ray.remote
    class Actor1:

        def __init__(self, x):
            if False:
                i = 10
                return i + 15
            self.x = x

        def get_val(self):
            if False:
                while True:
                    i = 10
            return self.x

    @ray.remote
    class Actor2:

        def __init__(self, x, y):
            if False:
                return 10
            self.x = x
            self.actor1 = Actor1.remote(y)

        def get_values(self, z):
            if False:
                print('Hello World!')
            return (self.x, ray.get(self.actor1.get_val.remote()))
    actor2 = Actor2.remote(3, 4)
    assert ray.get(actor2.get_values.remote(5)) == (3, 4)

def test_use_actor_twice(ray_start_10_cpus):
    if False:
        return 10

    @ray.remote
    class Actor1:

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self.count = 0

        def inc(self):
            if False:
                i = 10
                return i + 15
            self.count += 1
            return self.count

    @ray.remote
    class Actor2:

        def __init__(self):
            if False:
                return 10
            pass

        def inc(self, handle):
            if False:
                return 10
            return ray.get(handle.inc.remote())
    a = Actor1.remote()
    a2 = Actor2.remote()
    assert ray.get(a2.inc.remote(a)) == 1
    assert ray.get(a2.inc.remote(a)) == 2

def test_define_actor_within_remote_function(ray_start_10_cpus):
    if False:
        for i in range(10):
            print('nop')

    @ray.remote
    def f(x, n):
        if False:
            i = 10
            return i + 15

        @ray.remote
        class Actor1:

            def __init__(self, x):
                if False:
                    return 10
                self.x = x

            def get_value(self):
                if False:
                    while True:
                        i = 10
                return self.x
        actor = Actor1.remote(x)
        return ray.get([actor.get_value.remote() for _ in range(n)])
    assert ray.get(f.remote(3, 1)) == [3]
    assert ray.get([f.remote(i, 20) for i in range(10)]) == [20 * [i] for i in range(10)]

def test_use_actor_within_remote_function(ray_start_10_cpus):
    if False:
        while True:
            i = 10

    @ray.remote
    class Actor1:

        def __init__(self, x):
            if False:
                for i in range(10):
                    print('nop')
            self.x = x

        def get_values(self):
            if False:
                print('Hello World!')
            return self.x

    @ray.remote
    def f(x):
        if False:
            while True:
                i = 10
        actor = Actor1.remote(x)
        return ray.get(actor.get_values.remote())
    assert ray.get(f.remote(3)) == 3

def test_actor_import_counter(ray_start_10_cpus):
    if False:
        i = 10
        return i + 15
    num_remote_functions = 50
    for i in range(num_remote_functions):

        @ray.remote
        def f():
            if False:
                print('Hello World!')
            return i

    @ray.remote
    def g():
        if False:
            print('Hello World!')

        @ray.remote
        class Actor:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.x = ray.get(f.remote())

            def get_val(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.x
        actor = Actor.remote()
        return ray.get(actor.get_val.remote())
    assert ray.get(g.remote()) == num_remote_functions - 1

@pytest.mark.skipif(client_test_enabled(), reason='internal api')
def test_actor_method_metadata_cache(ray_start_regular):
    if False:
        while True:
            i = 10

    class Actor(object):
        pass
    cache = ray.actor._ActorClassMethodMetadata._cache
    cache.clear()
    A1 = ray.remote(Actor)
    a = A1.remote()
    assert len(cache) == 1
    cached_data_id = [id(x) for x in list(cache.items())[0]]
    for x in range(10):
        a = pickle.loads(pickle.dumps(a))
    assert len(ray.actor._ActorClassMethodMetadata._cache) == 1
    assert [id(x) for x in list(cache.items())[0]] == cached_data_id

@pytest.mark.skipif(client_test_enabled(), reason='internal api')
def test_actor_class_name(ray_start_regular):
    if False:
        while True:
            i = 10

    @ray.remote
    class Foo:

        def __init__(self):
            if False:
                i = 10
                return i + 15
            pass
    Foo.remote()
    g = ray._private.worker.global_worker.gcs_client
    actor_keys = g.internal_kv_keys(b'ActorClass', ray_constants.KV_NAMESPACE_FUNCTION_TABLE)
    assert len(actor_keys) == 1
    actor_class_info = pickle.loads(g.internal_kv_get(actor_keys[0], ray_constants.KV_NAMESPACE_FUNCTION_TABLE))
    assert actor_class_info['class_name'] == 'Foo'
    assert 'test_actor' in actor_class_info['module']

def test_actor_exit_from_task(ray_start_regular_shared):
    if False:
        while True:
            i = 10

    @ray.remote
    class Actor:

        def __init__(self):
            if False:
                while True:
                    i = 10
            print('Actor created')

        def f(self):
            if False:
                for i in range(10):
                    print('nop')
            return 0

    @ray.remote
    def f():
        if False:
            while True:
                i = 10
        a = Actor.remote()
        x_id = a.f.remote()
        return [x_id]
    x_id = ray.get(f.remote())[0]
    print(ray.get(x_id))

def test_actor_init_error_propagated(ray_start_regular_shared):
    if False:
        return 10

    @ray.remote
    class Actor:

        def __init__(self, error=False):
            if False:
                while True:
                    i = 10
            if error:
                raise Exception('oops')

        def foo(self):
            if False:
                return 10
            return 'OK'
    actor = Actor.remote(error=False)
    ray.get(actor.foo.remote())
    actor = Actor.remote(error=True)
    with pytest.raises(Exception, match='.*oops.*'):
        ray.get(actor.foo.remote())

def test_keyword_args(ray_start_regular_shared):
    if False:
        for i in range(10):
            print('nop')

    @ray.remote
    class Actor:

        def __init__(self, arg0, arg1=1, arg2='a'):
            if False:
                i = 10
                return i + 15
            self.arg0 = arg0
            self.arg1 = arg1
            self.arg2 = arg2

        def get_values(self, arg0, arg1=2, arg2='b'):
            if False:
                for i in range(10):
                    print('nop')
            return (self.arg0 + arg0, self.arg1 + arg1, self.arg2 + arg2)
    actor = Actor.remote(0)
    assert ray.get(actor.get_values.remote(1)) == (1, 3, 'ab')
    actor = Actor.remote(1, 2)
    assert ray.get(actor.get_values.remote(2, 3)) == (3, 5, 'ab')
    actor = Actor.remote(1, 2, 'c')
    assert ray.get(actor.get_values.remote(2, 3, 'd')) == (3, 5, 'cd')
    actor = Actor.remote(1, arg2='c')
    assert ray.get(actor.get_values.remote(0, arg2='d')) == (1, 3, 'cd')
    assert ray.get(actor.get_values.remote(0, arg2='d', arg1=0)) == (1, 1, 'cd')
    actor = Actor.remote(1, arg2='c', arg1=2)
    assert ray.get(actor.get_values.remote(0, arg2='d')) == (1, 4, 'cd')
    assert ray.get(actor.get_values.remote(0, arg2='d', arg1=0)) == (1, 2, 'cd')
    assert ray.get(actor.get_values.remote(arg2='d', arg1=0, arg0=2)) == (3, 2, 'cd')
    with pytest.raises(TypeError):
        actor = Actor.remote()
    with pytest.raises(TypeError):
        actor = Actor.remote(0, 1, 2, arg3=3)
    with pytest.raises(TypeError):
        actor = Actor.remote(0, arg0=1)
    actor = Actor.remote(1)
    with pytest.raises(Exception):
        ray.get(actor.get_values.remote())

def test_actor_name_conflict(ray_start_regular_shared):
    if False:
        i = 10
        return i + 15

    @ray.remote
    class A(object):

        def foo(self):
            if False:
                for i in range(10):
                    print('nop')
            return 100000
    a = A.remote()
    r = a.foo.remote()
    results = [r]
    for x in range(10):

        @ray.remote
        class A(object):

            def foo(self):
                if False:
                    i = 10
                    return i + 15
                return x
        a = A.remote()
        r = a.foo.remote()
        results.append(r)
    assert ray.get(results) == [100000, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def test_variable_number_of_args(ray_start_regular_shared):
    if False:
        for i in range(10):
            print('nop')

    @ray.remote
    class Actor:

        def __init__(self, arg0, arg1=1, *args):
            if False:
                while True:
                    i = 10
            self.arg0 = arg0
            self.arg1 = arg1
            self.args = args

        def get_values(self, arg0, arg1=2, *args):
            if False:
                for i in range(10):
                    print('nop')
            return (self.arg0 + arg0, self.arg1 + arg1, self.args, args)
    actor = Actor.remote(0)
    assert ray.get(actor.get_values.remote(1)) == (1, 3, (), ())
    actor = Actor.remote(1, 2)
    assert ray.get(actor.get_values.remote(2, 3)) == (3, 5, (), ())
    actor = Actor.remote(1, 2, 'c')
    assert ray.get(actor.get_values.remote(2, 3, 'd')) == (3, 5, ('c',), ('d',))
    actor = Actor.remote(1, 2, 'a', 'b', 'c', 'd')
    assert ray.get(actor.get_values.remote(2, 3, 1, 2, 3, 4)) == (3, 5, ('a', 'b', 'c', 'd'), (1, 2, 3, 4))

    @ray.remote
    class Actor:

        def __init__(self, *args):
            if False:
                print('Hello World!')
            self.args = args

        def get_values(self, *args):
            if False:
                while True:
                    i = 10
            return (self.args, args)
    a = Actor.remote()
    assert ray.get(a.get_values.remote()) == ((), ())
    a = Actor.remote(1)
    assert ray.get(a.get_values.remote(2)) == ((1,), (2,))
    a = Actor.remote(1, 2)
    assert ray.get(a.get_values.remote(3, 4)) == ((1, 2), (3, 4))

def test_no_args(ray_start_regular_shared):
    if False:
        i = 10
        return i + 15

    @ray.remote
    class Actor:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def get_values(self):
            if False:
                while True:
                    i = 10
            pass
    actor = Actor.remote()
    assert ray.get(actor.get_values.remote()) is None

def test_no_constructor(ray_start_regular_shared):
    if False:
        return 10

    @ray.remote
    class Actor:

        def get_values(self):
            if False:
                for i in range(10):
                    print('nop')
            pass
    actor = Actor.remote()
    assert ray.get(actor.get_values.remote()) is None

def test_custom_classes(ray_start_regular_shared):
    if False:
        while True:
            i = 10

    class Foo:

        def __init__(self, x):
            if False:
                print('Hello World!')
            self.x = x

    @ray.remote
    class Actor:

        def __init__(self, f2):
            if False:
                print('Hello World!')
            self.f1 = Foo(1)
            self.f2 = f2

        def get_values1(self):
            if False:
                return 10
            return (self.f1, self.f2)

        def get_values2(self, f3):
            if False:
                while True:
                    i = 10
            return (self.f1, self.f2, f3)
    actor = Actor.remote(Foo(2))
    results1 = ray.get(actor.get_values1.remote())
    assert results1[0].x == 1
    assert results1[1].x == 2
    results2 = ray.get(actor.get_values2.remote(Foo(3)))
    assert results2[0].x == 1
    assert results2[1].x == 2
    assert results2[2].x == 3

def test_actor_class_attributes(ray_start_regular_shared):
    if False:
        for i in range(10):
            print('nop')

    class Grandparent:
        GRANDPARENT = 2

    class Parent1(Grandparent):
        PARENT1 = 6

    class Parent2:
        PARENT2 = 7

    @ray.remote
    class TestActor(Parent1, Parent2):
        X = 3

        @classmethod
        def f(cls):
            if False:
                print('Hello World!')
            assert TestActor.GRANDPARENT == 2
            assert TestActor.PARENT1 == 6
            assert TestActor.PARENT2 == 7
            assert TestActor.X == 3
            return 4

        def g(self):
            if False:
                i = 10
                return i + 15
            assert TestActor.GRANDPARENT == 2
            assert TestActor.PARENT1 == 6
            assert TestActor.PARENT2 == 7
            assert TestActor.f() == 4
            return TestActor.X
    t = TestActor.remote()
    assert ray.get(t.g.remote()) == 3

def test_actor_static_attributes(ray_start_regular_shared):
    if False:
        i = 10
        return i + 15

    class Grandparent:
        GRANDPARENT = 2

        @staticmethod
        def grandparent_static():
            if False:
                print('Hello World!')
            assert Grandparent.GRANDPARENT == 2
            return 1

    class Parent1(Grandparent):
        PARENT1 = 6

        @staticmethod
        def parent1_static():
            if False:
                return 10
            assert Parent1.PARENT1 == 6
            return 2

        def parent1(self):
            if False:
                return 10
            assert Parent1.PARENT1 == 6

    class Parent2:
        PARENT2 = 7

        def parent2(self):
            if False:
                print('Hello World!')
            assert Parent2.PARENT2 == 7

    @ray.remote
    class TestActor(Parent1, Parent2):
        X = 3

        @staticmethod
        def f():
            if False:
                for i in range(10):
                    print('nop')
            assert TestActor.GRANDPARENT == 2
            assert TestActor.PARENT1 == 6
            assert TestActor.PARENT2 == 7
            assert TestActor.X == 3
            return 4

        def g(self):
            if False:
                while True:
                    i = 10
            assert TestActor.GRANDPARENT == 2
            assert TestActor.PARENT1 == 6
            assert TestActor.PARENT2 == 7
            assert TestActor.f() == 4
            return TestActor.X
    t = TestActor.remote()
    assert ray.get(t.g.remote()) == 3

def test_decorator_args(ray_start_regular_shared):
    if False:
        return 10
    with pytest.raises(Exception):

        @ray.remote()
        class Actor:

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                pass
    with pytest.raises(Exception):

        @ray.remote(invalid_kwarg=0)
        class Actor:

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass
    with pytest.raises(Exception):

        @ray.remote(num_cpus=0, invalid_kwarg=0)
        class Actor:

            def __init__(self):
                if False:
                    return 10
                pass

    @ray.remote(num_cpus=1)
    class Actor:

        def __init__(self):
            if False:
                return 10
            pass

    @ray.remote(num_gpus=1)
    class Actor:

        def __init__(self):
            if False:
                while True:
                    i = 10
            pass

    @ray.remote(num_cpus=1, num_gpus=1)
    class Actor:

        def __init__(self):
            if False:
                print('Hello World!')
            pass

def test_random_id_generation(ray_start_regular_shared):
    if False:
        print('Hello World!')

    @ray.remote
    class Foo:

        def __init__(self):
            if False:
                print('Hello World!')
            pass
    np.random.seed(1234)
    random.seed(1234)
    f1 = Foo.remote()
    np.random.seed(1234)
    random.seed(1234)
    f2 = Foo.remote()
    assert f1._actor_id != f2._actor_id

@pytest.mark.skipif(client_test_enabled(), reason='differing inheritence structure')
def test_actor_inheritance(ray_start_regular_shared):
    if False:
        return 10

    class NonActorBase:

        def __init__(self):
            if False:
                while True:
                    i = 10
            pass

    @ray.remote
    class ActorBase(NonActorBase):

        def __init__(self):
            if False:
                while True:
                    i = 10
            pass
    with pytest.raises(Exception, match='cannot be instantiated directly'):
        ActorBase()
    with pytest.raises(ActorClassInheritanceException, match='Inheriting from actor classes is not currently supported.'):

        class Derived(ActorBase):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass

def test_multiple_return_values(ray_start_regular_shared):
    if False:
        return 10

    @ray.remote
    class Foo:

        def method0(self):
            if False:
                return 10
            return 1

        @ray.method(num_returns=1)
        def method1(self):
            if False:
                while True:
                    i = 10
            return 1

        @ray.method(num_returns=2)
        def method2(self):
            if False:
                print('Hello World!')
            return (1, 2)

        @ray.method(num_returns=3)
        def method3(self):
            if False:
                return 10
            return (1, 2, 3)
    f = Foo.remote()
    id0 = f.method0.remote()
    assert ray.get(id0) == 1
    id1 = f.method1.remote()
    assert ray.get(id1) == 1
    (id2a, id2b) = f.method2.remote()
    assert ray.get([id2a, id2b]) == [1, 2]
    (id3a, id3b, id3c) = f.method3.remote()
    assert ray.get([id3a, id3b, id3c]) == [1, 2, 3]

def test_options_num_returns(ray_start_regular_shared):
    if False:
        for i in range(10):
            print('nop')

    @ray.remote
    class Foo:

        def method(self):
            if False:
                for i in range(10):
                    print('nop')
            return (1, 2)
    f = Foo.remote()
    obj = f.method.remote()
    assert ray.get(obj) == (1, 2)
    (obj1, obj2) = f.method.options(num_returns=2).remote()
    assert ray.get([obj1, obj2]) == [1, 2]

def test_options_name(ray_start_regular_shared):
    if False:
        while True:
            i = 10

    @ray.remote
    class Foo:

        def method(self, name):
            if False:
                return 10
            assert setproctitle.getproctitle() == f'ray::{name}'
    f = Foo.remote()
    ray.get(f.method.options(name='foo').remote('foo'))
    ray.get(f.method.options(name='bar').remote('bar'))

def test_define_actor(ray_start_regular_shared):
    if False:
        return 10

    @ray.remote
    class Test:

        def __init__(self, x):
            if False:
                i = 10
                return i + 15
            self.x = x

        def f(self, y):
            if False:
                for i in range(10):
                    print('nop')
            return self.x + y
    t = Test.remote(2)
    assert ray.get(t.f.remote(1)) == 3
    with pytest.raises(Exception):
        t.f(1)

def test_actor_deletion(ray_start_regular_shared):
    if False:
        print('Hello World!')

    @ray.remote
    class Actor:

        def getpid(self):
            if False:
                while True:
                    i = 10
            return os.getpid()
    a = Actor.remote()
    pid = ray.get(a.getpid.remote())
    a = None
    wait_for_pid_to_exit(pid)
    actors = [Actor.remote() for _ in range(10)]
    pids = ray.get([a.getpid.remote() for a in actors])
    a = None
    actors = None
    [wait_for_pid_to_exit(pid) for pid in pids]

def test_actor_method_deletion(ray_start_regular_shared):
    if False:
        print('Hello World!')

    @ray.remote
    class Actor:

        def method(self):
            if False:
                return 10
            return 1
    assert ray.get(Actor.remote().method.remote()) == 1

def test_distributed_actor_handle_deletion(ray_start_regular_shared):
    if False:
        for i in range(10):
            print('nop')

    @ray.remote
    class Actor:

        def method(self):
            if False:
                print('Hello World!')
            return 1

        def getpid(self):
            if False:
                print('Hello World!')
            return os.getpid()

    @ray.remote
    def f(actor, signal):
        if False:
            print('Hello World!')
        ray.get(signal.wait.remote())
        return ray.get(actor.method.remote())
    SignalActor = create_remote_signal_actor(ray)
    signal = SignalActor.remote()
    a = Actor.remote()
    pid = ray.get(a.getpid.remote())
    x_id = f.remote(a, signal)
    del a
    ray.get(signal.send.remote())
    assert ray.get(x_id) == 1
    wait_for_pid_to_exit(pid)

def test_multiple_actors(ray_start_regular_shared):
    if False:
        return 10

    @ray.remote
    class Counter:

        def __init__(self, value):
            if False:
                while True:
                    i = 10
            self.value = value

        def increase(self):
            if False:
                return 10
            self.value += 1
            return self.value

        def reset(self):
            if False:
                while True:
                    i = 10
            self.value = 0
    num_actors = 5
    num_increases = 50
    actors = [Counter.remote(i) for i in range(num_actors)]
    results = []
    for i in range(num_actors):
        results += [actors[i].increase.remote() for _ in range(num_increases)]
    result_values = ray.get(results)
    for i in range(num_actors):
        v = result_values[num_increases * i:num_increases * (i + 1)]
        assert v == list(range(i + 1, num_increases + i + 1))
    [actor.reset.remote() for actor in actors]
    results = []
    for j in range(num_increases):
        results += [actor.increase.remote() for actor in actors]
    result_values = ray.get(results)
    for j in range(num_increases):
        v = result_values[num_actors * j:num_actors * (j + 1)]
        assert v == num_actors * [j + 1]

def test_inherit_actor_from_class(ray_start_regular_shared):
    if False:
        while True:
            i = 10

    class Foo:

        def __init__(self, x):
            if False:
                while True:
                    i = 10
            self.x = x

        def f(self):
            if False:
                for i in range(10):
                    print('nop')
            return self.x

        def g(self, y):
            if False:
                print('Hello World!')
            return self.x + y

    @ray.remote
    class Actor(Foo):

        def __init__(self, x):
            if False:
                print('Hello World!')
            Foo.__init__(self, x)

        def get_value(self):
            if False:
                while True:
                    i = 10
            return self.f()
    actor = Actor.remote(1)
    assert ray.get(actor.get_value.remote()) == 1
    assert ray.get(actor.g.remote(5)) == 6

def test_get_non_existing_named_actor(ray_start_regular_shared):
    if False:
        print('Hello World!')
    with pytest.raises(ValueError):
        _ = ray.get_actor('non_existing_actor')

def test_actor_namespace(ray_start_regular_shared):
    if False:
        for i in range(10):
            print('nop')

    @ray.remote
    class Actor:

        def f(self):
            if False:
                i = 10
                return i + 15
            return 'ok'
    a = Actor.options(name='foo', namespace='f1').remote()
    with pytest.raises(ValueError):
        ray.get_actor(name='foo', namespace='f2')
    a1 = ray.get_actor(name='foo', namespace='f1')
    assert ray.get(a1.f.remote()) == 'ok'
    del a

def test_named_actor_cache(ray_start_regular_shared):
    if False:
        for i in range(10):
            print('nop')
    'Verify that named actor cache works well.'

    @ray.remote(max_restarts=-1)
    class Counter:

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self.count = 0

        def inc_and_get(self):
            if False:
                return 10
            self.count += 1
            return self.count
    a = Counter.options(name='hi').remote()
    first_get = ray.get_actor('hi')
    assert ray.get(first_get.inc_and_get.remote()) == 1
    second_get = ray.get_actor('hi')
    assert ray.get(second_get.inc_and_get.remote()) == 2
    ray.kill(a, no_restart=True)

    def actor_removed():
        if False:
            print('Hello World!')
        try:
            ray.get_actor('hi')
            return False
        except ValueError:
            return True
    wait_for_condition(actor_removed)
    get_after_restart = Counter.options(name='hi').remote()
    assert ray.get(get_after_restart.inc_and_get.remote()) == 1
    get_by_name = ray.get_actor('hi')
    assert ray.get(get_by_name.inc_and_get.remote()) == 2

def test_named_actor_cache_via_another_actor(ray_start_regular_shared):
    if False:
        print('Hello World!')
    'Verify that named actor cache works well with another actor.'

    @ray.remote(max_restarts=0)
    class Counter:

        def __init__(self):
            if False:
                return 10
            self.count = 0

        def inc_and_get(self):
            if False:
                for i in range(10):
                    print('nop')
            self.count += 1
            return self.count

    @ray.remote(max_restarts=0)
    class ActorGetter:

        def get_actor_count(self, name):
            if False:
                while True:
                    i = 10
            actor = ray.get_actor(name)
            return ray.get(actor.inc_and_get.remote())
    a = Counter.options(name='foo').remote()
    first_get = ray.get_actor('foo')
    assert ray.get(first_get.inc_and_get.remote()) == 1
    actor_getter = ActorGetter.remote()
    assert ray.get(actor_getter.get_actor_count.remote('foo')) == 2
    ray.kill(a, no_restart=True)

    def actor_removed():
        if False:
            for i in range(10):
                print('nop')
        try:
            ray.get_actor('foo')
            return False
        except ValueError:
            return True
    wait_for_condition(actor_removed)
    get_after_restart = Counter.options(name='foo').remote()
    assert ray.get(get_after_restart.inc_and_get.remote()) == 1
    assert ray.get(actor_getter.get_actor_count.remote('foo')) == 2
    get_by_name = ray.get_actor('foo')
    assert ray.get(get_by_name.inc_and_get.remote()) == 3

def test_wrapped_actor_handle(ray_start_regular_shared):
    if False:
        while True:
            i = 10

    @ray.remote
    class B:

        def doit(self):
            if False:
                print('Hello World!')
            return 2

    @ray.remote
    class A:

        def __init__(self):
            if False:
                print('Hello World!')
            self.b = B.remote()

        def get_actor_ref(self):
            if False:
                return 10
            return [self.b]
    a = A.remote()
    b_list = ray.get(a.get_actor_ref.remote())
    assert ray.get(b_list[0].doit.remote()) == 2

@pytest.mark.skip('This test is just used to print the latency of creating 100 actors.')
def test_actor_creation_latency(ray_start_regular_shared):
    if False:
        while True:
            i = 10

    @ray.remote
    class Actor:

        def get_value(self):
            if False:
                for i in range(10):
                    print('nop')
            return 1
    start = datetime.datetime.now()
    actor_handles = [Actor.remote() for _ in range(100)]
    actor_create_time = datetime.datetime.now()
    for actor_handle in actor_handles:
        ray.get(actor_handle.get_value.remote())
    end = datetime.datetime.now()
    print('actor_create_time_consume = {}, total_time_consume = {}'.format(actor_create_time - start, end - start))

@pytest.mark.parametrize('exit_condition', ['__ray_terminate__', 'ray.actor.exit_actor', 'ray.kill'])
def test_atexit_handler(ray_start_regular_shared, exit_condition):
    if False:
        while True:
            i = 10

    @ray.remote
    class A:

        def __init__(self, tmpfile, data):
            if False:
                for i in range(10):
                    print('nop')
            import atexit

            def f(*args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                with open(tmpfile, 'w') as f:
                    f.write(data)
                    f.flush()
            atexit.register(f)

        def ready(self):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def exit(self):
            if False:
                print('Hello World!')
            ray.actor.exit_actor()
    data = 'hello'
    tmpfile = tempfile.NamedTemporaryFile('w+', suffix='.tmp', delete=False)
    tmpfile.close()
    a = A.remote(tmpfile.name, data)
    ray.get(a.ready.remote())
    if exit_condition == 'out_of_scope':
        del a
    elif exit_condition == '__ray_terminate__':
        ray.wait([a.__ray_terminate__.remote()])
    elif exit_condition == 'ray.actor.exit_actor':
        ray.wait([a.exit.remote()])
    elif exit_condition == 'ray.kill':
        ray.kill(a)
    else:
        assert False, 'Unrecognized condition'

    def check_file_written():
        if False:
            print('Hello World!')
        with open(tmpfile.name, 'r') as f:
            if f.read() == data:
                return True
            return False
    if exit_condition == 'ray.kill':
        assert not check_file_written()
    else:
        wait_for_condition(check_file_written)
    os.unlink(tmpfile.name)

def test_actor_ready(ray_start_regular_shared):
    if False:
        while True:
            i = 10

    @ray.remote
    class Actor:
        pass
    actor = Actor.remote()
    with pytest.raises(TypeError):
        actor.__ray_ready__()
    assert ray.get(actor.__ray_ready__.remote())

def test_return_actor_handle_from_actor(ray_start_regular_shared):
    if False:
        i = 10
        return i + 15

    @ray.remote
    class Inner:

        def ping(self):
            if False:
                while True:
                    i = 10
            return 'pong'

    @ray.remote
    class Outer:

        def __init__(self):
            if False:
                while True:
                    i = 10
            self.inner = Inner.remote()

        def get_ref(self):
            if False:
                return 10
            return self.inner
    outer = Outer.remote()
    inner = ray.get(outer.get_ref.remote())
    assert ray.get(inner.ping.remote()) == 'pong'

def test_actor_autocomplete(ray_start_regular_shared):
    if False:
        return 10
    '\n    Test that autocomplete works with actors by checking that the builtin dir()\n    function works as expected.\n    '

    @ray.remote
    class Foo:

        def method_one(self) -> None:
            if False:
                i = 10
                return i + 15
            pass
    class_calls = [fn for fn in dir(Foo) if not fn.startswith('_')]
    assert set(class_calls) == {'method_one', 'options', 'remote', 'bind'}
    f = Foo.remote()
    methods = [fn for fn in dir(f) if not fn.startswith('_')]
    assert methods == ['method_one']
    all_methods = set(dir(f))
    assert all_methods == {'__init__', 'method_one', '__ray_ready__', '__ray_terminate__'}
    method_options = [fn for fn in dir(f.method_one) if not fn.startswith('_')]
    assert set(method_options) == {'options', 'remote'}

def test_actor_mro(ray_start_regular_shared):
    if False:
        i = 10
        return i + 15

    @ray.remote
    class Foo:

        def __init__(self, x):
            if False:
                return 10
            self.x = x

        @classmethod
        def factory_f(cls, x):
            if False:
                print('Hello World!')
            return cls(x)

        def get_x(self):
            if False:
                print('Hello World!')
            return self.x
    obj = Foo.factory_f(1)
    assert obj.get_x() == 1

@pytest.mark.skipif(client_test_enabled(), reason='differing deletion behaviors')
def test_keep_calling_get_actor(ray_start_regular_shared):
    if False:
        return 10
    '\n    Test keep calling get_actor.\n    '

    @ray.remote
    class Actor:

        def hello(self):
            if False:
                for i in range(10):
                    print('nop')
            return 'hello'
    actor = Actor.options(name='ABC').remote()
    assert ray.get(actor.hello.remote()) == 'hello'
    for _ in range(10):
        actor = ray.get_actor('ABC')
        assert ray.get(actor.hello.remote()) == 'hello'
    del actor

    def actor_removed():
        if False:
            while True:
                i = 10
        try:
            ray.get_actor('ABC')
            return False
        except ValueError:
            return True
    wait_for_condition(actor_removed)

@pytest.mark.skipif(client_test_enabled(), reason='internal api')
@pytest.mark.parametrize('actor_type', ['actor', 'threaded_actor', 'async_actor'])
def test_actor_parent_task_correct(shutdown_only, actor_type):
    if False:
        i = 10
        return i + 15
    'Verify the parent task id is correct for all actors.'

    @ray.remote
    def child():
        if False:
            while True:
                i = 10
        pass

    @ray.remote
    class ChildActor:

        def child(self):
            if False:
                i = 10
                return i + 15
            pass

    def parent_func(child_actor):
        if False:
            i = 10
            return i + 15
        core_worker = ray._private.worker.global_worker.core_worker
        refs = [child_actor.child.remote(), child.remote()]
        expected = {ref.task_id().hex() for ref in refs}
        task_id = ray.get_runtime_context().task_id
        children_task_ids = core_worker.get_pending_children_task_ids(task_id)
        actual = {task_id.hex() for task_id in children_task_ids}
        ray.get(refs)
        return (expected, actual)
    if actor_type == 'actor':

        @ray.remote
        class Actor:

            def parent(self, child_actor):
                if False:
                    return 10
                return parent_func(child_actor)

        @ray.remote
        class GeneratorActor:

            def parent(self, child_actor):
                if False:
                    i = 10
                    return i + 15
                yield parent_func(child_actor)
    if actor_type == 'threaded_actor':

        @ray.remote(max_concurrency=5)
        class Actor:

            def parent(self, child_actor):
                if False:
                    while True:
                        i = 10
                return parent_func(child_actor)

        @ray.remote(max_concurrency=5)
        class GeneratorActor:

            def parent(self, child_actor):
                if False:
                    for i in range(10):
                        print('nop')
                yield parent_func(child_actor)
    if actor_type == 'async_actor':

        @ray.remote
        class Actor:

            async def parent(self, child_actor):
                return parent_func(child_actor)

        @ray.remote
        class GeneratorActor:

            async def parent(self, child_actor):
                yield parent_func(child_actor)
    actor = Actor.remote()
    child_actor = ChildActor.remote()
    (actual, expected) = ray.get(actor.parent.remote(child_actor))
    assert actual == expected
    actor = GeneratorActor.remote()
    child_actor = ChildActor.remote()
    gen = actor.parent.options(num_returns='streaming').remote(child_actor)
    for ref in gen:
        result = ray.get(ref)
    (actual, expected) = result
    assert actual == expected

@pytest.mark.skipif(client_test_enabled(), reason='internal api')
def test_parent_task_correct_concurrent_async_actor(shutdown_only):
    if False:
        for i in range(10):
            print('nop')
    'Make sure when there are concurrent async tasks\n    the parent -> children task ids are properly mapped.\n    '
    sig = SignalActor.remote()

    @ray.remote
    def child(sig):
        if False:
            return 10
        ray.get(sig.wait.remote())

    @ray.remote
    class AsyncActor:

        async def f(self, sig):
            refs = [child.remote(sig) for _ in range(2)]
            core_worker = ray._private.worker.global_worker.core_worker
            expected = {ref.task_id().hex() for ref in refs}
            task_id = ray.get_runtime_context().task_id
            children_task_ids = core_worker.get_pending_children_task_ids(task_id)
            actual = {task_id.hex() for task_id in children_task_ids}
            await sig.wait.remote()
            ray.get(refs)
            return (actual, expected)
    a = AsyncActor.remote()
    refs = [a.f.remote(sig) for _ in range(20)]
    ray.get(sig.send.remote())
    result = ray.get(refs)
    for (actual, expected) in result:
        assert actual, expected
if __name__ == '__main__':
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))