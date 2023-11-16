"""
Test the cleanup-funcs functionality provided via
:meth:`behave.runner.Context.add_cleanup()` method.

OPEN ISSUES:
* Should cleanup_func use context args, like: cleanup_func(context) ?
* Should formatters be somehow informed about cleanup-errors ?
* Should context._pop() calls be protected against raised exceptions
  from on_cleanup_error() implementations ?
"""
from __future__ import print_function
from behave.runner import Context, scoped_context_layer
from contextlib import contextmanager
from mock import Mock, NonCallableMock
import pytest

def cleanup_func():
    if False:
        return 10
    pass

def cleanup_func_with_args(*args, **kwargs):
    if False:
        return 10
    pass

class CleanupFunction(object):

    def __init__(self, name='CLEANUP-FUNC', listener=None):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.listener = listener

    def __call__(self, *args, **kwargs):
        if False:
            return 10
        if self.listener:
            message = 'called:%s' % self.name
            self.listener(message)

class CallListener(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.collected = []

    def __call__(self, message):
        if False:
            while True:
                i = 10
        self.collected.append(message)

class TestContextCleanup(object):

    def test_cleanup_func_is_called_when_context_frame_is_popped(self):
        if False:
            print('Hello World!')
        my_cleanup = Mock(spec=cleanup_func)
        context = Context(runner=Mock())
        with scoped_context_layer(context):
            context.add_cleanup(my_cleanup)
            my_cleanup.assert_not_called()
        my_cleanup.assert_called_once()

    def test_cleanup_funcs_are_called_when_context_frame_is_popped(self):
        if False:
            print('Hello World!')
        my_cleanup1 = Mock(spec=cleanup_func)
        my_cleanup2 = Mock(spec=cleanup_func)
        context = Context(runner=Mock())
        with scoped_context_layer(context):
            context.add_cleanup(my_cleanup1)
            context.add_cleanup(my_cleanup2)
            my_cleanup1.assert_not_called()
            my_cleanup2.assert_not_called()
        my_cleanup1.assert_called_once()
        my_cleanup2.assert_called_once()

    def test_cleanup_funcs_are_called_in_reversed_order(self):
        if False:
            print('Hello World!')
        call_listener = CallListener()
        my_cleanup1A = CleanupFunction('CLEANUP1', listener=call_listener)
        my_cleanup2A = CleanupFunction('CLEANUP2', listener=call_listener)
        my_cleanup1 = Mock(side_effect=my_cleanup1A)
        my_cleanup2 = Mock(side_effect=my_cleanup2A)
        context = Context(runner=Mock())
        with scoped_context_layer(context):
            context.add_cleanup(my_cleanup1)
            context.add_cleanup(my_cleanup2)
            my_cleanup1.assert_not_called()
            my_cleanup2.assert_not_called()
        expected_call_order = ['called:CLEANUP2', 'called:CLEANUP1']
        assert call_listener.collected == expected_call_order
        my_cleanup1.assert_called_once()
        my_cleanup2.assert_called_once()

    def test_cleanup_funcs_on_two_context_frames(self):
        if False:
            for i in range(10):
                print('nop')
        call_listener = CallListener()
        my_cleanup_A1 = CleanupFunction('CLEANUP_A1', listener=call_listener)
        my_cleanup_A2 = CleanupFunction('CLEANUP_A2', listener=call_listener)
        my_cleanup_B1 = CleanupFunction('CLEANUP_B1', listener=call_listener)
        my_cleanup_B2 = CleanupFunction('CLEANUP_B2', listener=call_listener)
        my_cleanup_B3 = CleanupFunction('CLEANUP_B3', listener=call_listener)
        my_cleanup_A1M = Mock(side_effect=my_cleanup_A1)
        my_cleanup_A2M = Mock(side_effect=my_cleanup_A2)
        my_cleanup_B1M = Mock(side_effect=my_cleanup_B1)
        my_cleanup_B2M = Mock(side_effect=my_cleanup_B2)
        my_cleanup_B3M = Mock(side_effect=my_cleanup_B3)
        context = Context(runner=Mock())
        with scoped_context_layer(context):
            context.add_cleanup(my_cleanup_A1M)
            context.add_cleanup(my_cleanup_A2M)
            with scoped_context_layer(context):
                context.add_cleanup(my_cleanup_B1M)
                context.add_cleanup(my_cleanup_B2M)
                context.add_cleanup(my_cleanup_B3M)
                my_cleanup_B1M.assert_not_called()
                my_cleanup_B2M.assert_not_called()
                my_cleanup_B3M.assert_not_called()
            expected_call_order = ['called:CLEANUP_B3', 'called:CLEANUP_B2', 'called:CLEANUP_B1']
            assert call_listener.collected == expected_call_order
            my_cleanup_A1M.assert_not_called()
            my_cleanup_A2M.assert_not_called()
            my_cleanup_B1M.assert_called_once()
            my_cleanup_B2M.assert_called_once()
            my_cleanup_B3M.assert_called_once()
        expected_call_order = ['called:CLEANUP_B3', 'called:CLEANUP_B2', 'called:CLEANUP_B1', 'called:CLEANUP_A2', 'called:CLEANUP_A1']
        assert call_listener.collected == expected_call_order
        my_cleanup_A1M.assert_called_once()
        my_cleanup_A2M.assert_called_once()
        my_cleanup_B1M.assert_called_once()
        my_cleanup_B2M.assert_called_once()
        my_cleanup_B3M.assert_called_once()

    def test_add_cleanup_with_args(self):
        if False:
            for i in range(10):
                print('nop')
        my_cleanup = Mock(spec=cleanup_func_with_args)
        context = Context(runner=Mock())
        with scoped_context_layer(context):
            context.add_cleanup(my_cleanup, 1, 2, 3)
            my_cleanup.assert_not_called()
        my_cleanup.assert_called_once_with(1, 2, 3)

    def test_add_cleanup_with_args_and_kwargs(self):
        if False:
            i = 10
            return i + 15
        my_cleanup = Mock(spec=cleanup_func_with_args)
        context = Context(runner=Mock())
        with scoped_context_layer(context):
            context.add_cleanup(my_cleanup, 1, 2, 3, name='alice')
            my_cleanup.assert_not_called()
        my_cleanup.assert_called_once_with(1, 2, 3, name='alice')

    def test_add_cleanup__rejects_noncallable_cleanup_func(self):
        if False:
            i = 10
            return i + 15

        class NonCallable(object):
            pass
        non_callable = NonCallable()
        context = Context(runner=Mock())
        with pytest.raises(AssertionError) as e:
            with scoped_context_layer(context):
                context.add_cleanup(non_callable)
        assert 'REQUIRES: callable(cleanup_func)' in str(e.value)

    def test_on_cleanup_error__prints_error_by_default(self, capsys):
        if False:
            for i in range(10):
                print('nop')

        def bad_cleanup_func():
            if False:
                for i in range(10):
                    print('nop')
            raise RuntimeError('in CLEANUP call')
        bad_cleanup = Mock(side_effect=bad_cleanup_func)
        context = Context(runner=Mock())
        with pytest.raises(RuntimeError):
            with scoped_context_layer(context):
                context.add_cleanup(bad_cleanup)
        (captured_output, _) = capsys.readouterr()
        bad_cleanup.assert_called()
        assert 'CLEANUP-ERROR in ' in captured_output
        assert 'RuntimeError: in CLEANUP call' in captured_output
        print(captured_output)

    def test_on_cleanup_error__is_called_if_defined(self):
        if False:
            while True:
                i = 10

        def bad_cleanup():
            if False:
                return 10
            raise RuntimeError('in CLEANUP call')

        def handle_cleanup_error(context, cleanup_func, exception):
            if False:
                print('Hello World!')
            print('CALLED: handle_cleanup_error')
        context = Context(runner=Mock())
        handle_cleanup_error_func = Mock(spec=handle_cleanup_error)
        with pytest.raises(RuntimeError):
            with scoped_context_layer(context):
                context.on_cleanup_error = handle_cleanup_error_func
                context.add_cleanup(bad_cleanup)
        handle_cleanup_error_func.assert_called_once()

    def test_on_cleanup_error__may_be_called_several_times_per_cleanup(self):
        if False:
            print('Hello World!')

        def bad_cleanup1():
            if False:
                while True:
                    i = 10
            raise RuntimeError('CLEANUP_1')

        def bad_cleanup2():
            if False:
                print('Hello World!')
            raise RuntimeError('CLEANUP_2')

        class CleanupErrorCollector(object):

            def __init__(self):
                if False:
                    print('Hello World!')
                self.collected = []

            def __call__(self, context, cleanup_func, exception):
                if False:
                    i = 10
                    return i + 15
                self.collected.append((context, cleanup_func, exception))
        context = Context(runner=Mock())
        collect_cleanup_error = CleanupErrorCollector()
        with pytest.raises(RuntimeError):
            with scoped_context_layer(context):
                context.on_cleanup_error = collect_cleanup_error
                context.add_cleanup(bad_cleanup1)
                context.add_cleanup(bad_cleanup2)
        expected = [(context, bad_cleanup2, RuntimeError('CLEANUP_2')), (context, bad_cleanup1, RuntimeError('CLEANUP_1'))]
        assert len(collect_cleanup_error.collected) == 2
        assert collect_cleanup_error.collected[0][:-1] == expected[0][:-1]
        assert collect_cleanup_error.collected[1][:-1] == expected[1][:-1]

class TestContextCleanupWithLayer(object):
    """Tests :meth:`behave.runner.Context.add_cleanup()`
    with layer parameter.

    :meth:`cleanup_func()` is called when Context layer is removed/popped.
    """

    def test_add_cleanup_with_known_layer(self):
        if False:
            return 10
        my_cleanup = Mock(spec=cleanup_func)
        context = Context(runner=Mock())
        with scoped_context_layer(context, layer='scenario'):
            context.add_cleanup(my_cleanup, layer='scenario')
            my_cleanup.assert_not_called()
        my_cleanup.assert_called_once()

    def test_add_cleanup_with_known_layer_and_args(self):
        if False:
            return 10
        my_cleanup = Mock(spec=cleanup_func_with_args)
        context = Context(runner=Mock())
        with scoped_context_layer(context, layer='scenario'):
            context.add_cleanup(my_cleanup, 1, 2, 3, layer='scenario')
            my_cleanup.assert_not_called()
        my_cleanup.assert_called_once_with(1, 2, 3)

    def test_add_cleanup_with_known_layer_and_kwargs(self):
        if False:
            i = 10
            return i + 15
        my_cleanup = Mock(spec=cleanup_func_with_args)
        context = Context(runner=Mock())
        with scoped_context_layer(context, layer='scenario'):
            context.add_cleanup(my_cleanup, layer='scenario', name='alice')
            my_cleanup.assert_not_called()
        my_cleanup.assert_called_once_with(name='alice')

    def test_add_cleanup_with_known_deeper_layer2(self):
        if False:
            for i in range(10):
                print('nop')
        my_cleanup = Mock(spec=cleanup_func)
        context = Context(runner=Mock())
        with scoped_context_layer(context, layer='feature'):
            with scoped_context_layer(context, layer='scenario'):
                context.add_cleanup(my_cleanup, layer='feature')
            my_cleanup.assert_not_called()
        my_cleanup.assert_called_once()

    def test_add_cleanup_with_known_deeper_layer3(self):
        if False:
            print('Hello World!')
        my_cleanup = Mock(spec=cleanup_func)
        context = Context(runner=Mock())
        with scoped_context_layer(context, layer='testrun'):
            with scoped_context_layer(context, layer='feature'):
                with scoped_context_layer(context, layer='scenario'):
                    context.add_cleanup(my_cleanup, layer='feature')
                my_cleanup.assert_not_called()
            my_cleanup.assert_called_once()
        my_cleanup.assert_called_once()

    def test_add_cleanup_with_unknown_layer_raises_lookup_error(self):
        if False:
            i = 10
            return i + 15
        'Cleanup function is not registered'
        my_cleanup = Mock(spec=cleanup_func)
        context = Context(runner=Mock())
        with scoped_context_layer(context):
            with pytest.raises(LookupError) as error:
                context.add_cleanup(my_cleanup, layer='other')
        my_cleanup.assert_not_called()