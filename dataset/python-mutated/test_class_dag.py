import pytest
import ray
from ray.dag import PARENT_CLASS_NODE_KEY, PREV_CLASS_METHOD_CALL_KEY

@ray.remote
class Counter:

    def __init__(self, init_value=0):
        if False:
            return 10
        self.i = init_value

    def inc(self):
        if False:
            while True:
                i = 10
        self.i += 1

    def get(self):
        if False:
            i = 10
            return i + 15
        return self.i

@ray.remote
class Actor:

    def __init__(self, init_value):
        if False:
            for i in range(10):
                print('nop')
        self.i = init_value

    def inc(self, x):
        if False:
            return 10
        self.i += x

    def get(self):
        if False:
            for i in range(10):
                print('nop')
        return self.i

def test_basic_actor_dag(shared_ray_instance):
    if False:
        for i in range(10):
            print('nop')

    @ray.remote
    def combine(x, y):
        if False:
            return 10
        return x + y
    a1 = Actor.bind(10)
    res = a1.get.bind()
    print(res)
    assert ray.get(res.execute()) == 10
    a2 = Actor.bind(10)
    a1.inc.bind(2)
    a1.inc.bind(4)
    a2.inc.bind(6)
    dag = combine.bind(a1.get.bind(), a2.get.bind())
    print(dag)
    assert ray.get(dag.execute()) == 32

def test_class_as_class_constructor_arg(shared_ray_instance):
    if False:
        for i in range(10):
            print('nop')

    @ray.remote
    class OuterActor:

        def __init__(self, inner_actor):
            if False:
                print('Hello World!')
            self.inner_actor = inner_actor

        def inc(self, x):
            if False:
                print('Hello World!')
            self.inner_actor.inc.remote(x)

        def get(self):
            if False:
                for i in range(10):
                    print('nop')
            return ray.get(self.inner_actor.get.remote())
    outer = OuterActor.bind(Actor.bind(10))
    outer.inc.bind(2)
    dag = outer.get.bind()
    print(dag)
    assert ray.get(dag.execute()) == 12

def test_class_as_function_constructor_arg(shared_ray_instance):
    if False:
        return 10

    @ray.remote
    def f(actor_handle):
        if False:
            i = 10
            return i + 15
        return ray.get(actor_handle.get.remote())
    dag = f.bind(Actor.bind(10))
    print(dag)
    assert ray.get(dag.execute()) == 10

def test_basic_actor_dag_constructor_options(shared_ray_instance):
    if False:
        while True:
            i = 10
    a1 = Actor.bind(10)
    dag = a1.get.bind()
    print(dag)
    assert ray.get(dag.execute()) == 10
    a1 = Actor.options(name='Actor', namespace='test', max_pending_calls=10).bind(10)
    dag = a1.get.bind()
    print(dag)
    assert ray.get(dag.execute()) == 10
    assert a1.get_options().get('name') == 'Actor'
    assert a1.get_options().get('namespace') == 'test'
    assert a1.get_options().get('max_pending_calls') == 10

def test_actor_method_options(shared_ray_instance):
    if False:
        while True:
            i = 10
    a1 = Actor.bind(10)
    dag = a1.get.options(name='actor_method_options').bind()
    print(dag)
    assert ray.get(dag.execute()) == 10
    assert dag.get_options().get('name') == 'actor_method_options'

def test_basic_actor_dag_constructor_invalid_options(shared_ray_instance):
    if False:
        return 10
    with pytest.raises(ValueError, match='.*quantity of resource num_cpus cannot be negative.*'):
        a1 = Actor.options(num_cpus=-1).bind(10)
        invalid_dag = a1.get.bind()
        ray.get(invalid_dag.execute())

def test_actor_options_complicated(shared_ray_instance):
    if False:
        i = 10
        return i + 15
    'Test a more complicated setup where we apply .options() in both\n    constructor and method call with overlapping keys, and ensure end to end\n    options correctness.\n    '

    @ray.remote
    def combine(x, y):
        if False:
            return 10
        return x + y
    a1 = Actor.options(name='a1_v0').bind(10)
    res = a1.get.options(name='v1').bind()
    print(res)
    assert ray.get(res.execute()) == 10
    assert a1.get_options().get('name') == 'a1_v0'
    assert res.get_options().get('name') == 'v1'
    a1 = Actor.options(name='a1_v1').bind(10)
    a2 = Actor.options(name='a2_v0').bind(10)
    a1.inc.options(name='v1').bind(2)
    a1.inc.options(name='v2').bind(4)
    a2.inc.options(name='v3').bind(6)
    dag = combine.options(name='v4').bind(a1.get.bind(), a2.get.bind())
    print(dag)
    assert ray.get(dag.execute()) == 32
    test_a1 = dag.get_args()[0]
    test_a2 = dag.get_args()[1]
    assert test_a2.get_options() == {}
    assert test_a2.get_other_args_to_resolve()[PARENT_CLASS_NODE_KEY].get_options().get('name') == 'a2_v0'
    assert test_a2.get_other_args_to_resolve()[PREV_CLASS_METHOD_CALL_KEY].get_options().get('name') == 'v3'
    assert test_a1.get_other_args_to_resolve()[PARENT_CLASS_NODE_KEY].get_options().get('name') == 'a1_v1'
    assert test_a1.get_other_args_to_resolve()[PREV_CLASS_METHOD_CALL_KEY].get_options().get('name') == 'v2'
    assert test_a1.get_other_args_to_resolve()[PREV_CLASS_METHOD_CALL_KEY].get_other_args_to_resolve()[PREV_CLASS_METHOD_CALL_KEY].get_options().get('name') == 'v1'

def test_pass_actor_handle(shared_ray_instance):
    if False:
        while True:
            i = 10

    @ray.remote
    class Actor:

        def ping(self):
            if False:
                return 10
            return 'hello'

    @ray.remote
    def caller(handle):
        if False:
            i = 10
            return i + 15
        assert isinstance(handle, ray.actor.ActorHandle), handle
        return ray.get(handle.ping.remote())
    a1 = Actor.bind()
    dag = caller.bind(a1)
    print(dag)
    assert ray.get(dag.execute()) == 'hello'

def test_dynamic_pipeline(shared_ray_instance):
    if False:
        while True:
            i = 10

    @ray.remote
    class Model:

        def __init__(self, arg):
            if False:
                print('Hello World!')
            self.arg = arg

        def forward(self, x):
            if False:
                return 10
            return self.arg + str(x)

    @ray.remote
    class ModelSelection:

        def is_even(self, x):
            if False:
                for i in range(10):
                    print('nop')
            return x % 2 == 0

    @ray.remote
    def pipeline(x, m1, m2, selection):
        if False:
            return 10
        sel = selection.is_even.remote(x)
        if ray.get(sel):
            result = m1.forward.remote(x)
        else:
            result = m2.forward.remote(x)
        return ray.get(result)
    m1 = Model.bind('Even: ')
    m2 = Model.bind('Odd: ')
    selection = ModelSelection.bind()
    even_input = pipeline.bind(20, m1, m2, selection)
    print(even_input)
    assert ray.get(even_input.execute()) == 'Even: 20'
    odd_input = pipeline.bind(21, m1, m2, selection)
    print(odd_input)
    assert ray.get(odd_input.execute()) == 'Odd: 21'

def test_unsupported_bind():
    if False:
        for i in range(10):
            print('nop')

    @ray.remote
    class Actor:

        def ping(self):
            if False:
                i = 10
                return i + 15
            return 'hello'
    with pytest.raises(AttributeError, match='\\.bind\\(\\) cannot be used again on'):
        actor = Actor.bind()
        _ = actor.bind()
    with pytest.raises(AttributeError, match='\\.remote\\(\\) cannot be used on ClassMethodNodes'):
        actor = Actor.bind()
        _ = actor.ping.remote()

def test_unsupported_remote():
    if False:
        for i in range(10):
            print('nop')

    @ray.remote
    class Actor:

        def ping(self):
            if False:
                while True:
                    i = 10
            return 'hello'
    with pytest.raises(AttributeError, match="'Actor' has no attribute 'remote'"):
        _ = Actor.bind().remote()

    @ray.remote
    def func():
        if False:
            return 10
        return 1
    with pytest.raises(AttributeError, match='\\.remote\\(\\) cannot be used on'):
        _ = func.bind().remote()
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))