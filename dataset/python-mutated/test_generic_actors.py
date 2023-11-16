from unittest.mock import Mock
import pytest
import dramatiq

def test_generic_actors_can_be_defined(stub_broker):
    if False:
        print('Hello World!')

    class Add(dramatiq.GenericActor):

        def perform(self, x, y):
            if False:
                return 10
            return x + y
    assert isinstance(Add.__actor__, dramatiq.Actor)
    assert Add(1, 2) == 3

def test_generic_actors_can_be_assigned_options(stub_broker):
    if False:
        return 10

    class Add(dramatiq.GenericActor):

        class Meta:
            max_retries = 32

        def perform(self, x, y):
            if False:
                for i in range(10):
                    print('nop')
            return x + y
    assert Add.options['max_retries'] == 32

def test_generic_actors_raise_not_implemented_if_perform_is_missing(stub_broker):
    if False:
        i = 10
        return i + 15

    class Foo(dramatiq.GenericActor):
        pass
    with pytest.raises(NotImplementedError):
        Foo()

def test_generic_actors_can_be_abstract(stub_broker, stub_worker):
    if False:
        i = 10
        return i + 15
    calls = set()

    class BaseTask(dramatiq.GenericActor):

        class Meta:
            abstract = True
            queue_name = 'tasks'

        def get_task_name(self):
            if False:
                for i in range(10):
                    print('nop')
            raise NotImplementedError

        def perform(self):
            if False:
                for i in range(10):
                    print('nop')
            calls.add(self.get_task_name())
    assert not isinstance(BaseTask, dramatiq.Actor)

    class FooTask(BaseTask):

        def get_task_name(self):
            if False:
                i = 10
                return i + 15
            return 'Foo'

    class BarTask(BaseTask):

        def get_task_name(self):
            if False:
                i = 10
                return i + 15
            return 'Bar'
    assert isinstance(FooTask.__actor__, dramatiq.Actor)
    assert isinstance(BarTask.__actor__, dramatiq.Actor)
    assert FooTask.queue_name == BarTask.queue_name == 'tasks'
    FooTask.send()
    BarTask.send()
    stub_broker.join(queue_name=BaseTask.Meta.queue_name)
    stub_worker.join()
    assert calls == {'Foo', 'Bar'}

def test_generic_actors_can_have_class_attributes(stub_broker):
    if False:
        print('Hello World!')

    class DoSomething(dramatiq.GenericActor):
        STATUS_RUNNING = 'running'
        STATUS_DONE = 'done'
    assert DoSomething.STATUS_DONE == 'done'

def test_generic_actors_can_accept_custom_actor_registry(stub_broker):
    if False:
        for i in range(10):
            print('nop')
    actor_instance = Mock()
    actor_registry = Mock(return_value=actor_instance)

    class CustomActor(dramatiq.GenericActor):

        class Meta:
            actor = actor_registry

        def perform(self):
            if False:
                return 10
            pass
    assert CustomActor.__actor__ is actor_instance
    actor_registry.assert_called_once_with(CustomActor)