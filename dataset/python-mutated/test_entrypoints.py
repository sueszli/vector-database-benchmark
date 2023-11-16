from collections import defaultdict
import pytest
from mock import Mock, call
from nameko.extensions import DependencyProvider, Entrypoint
from nameko.testing.services import dummy, entrypoint_hook, once
from nameko.testing.utils import get_extension
from nameko.utils import REDACTED, get_redacted_args

@pytest.fixture
def tracker():
    if False:
        while True:
            i = 10
    return Mock()

class TestDecorator(object):

    def test_decorator_without_args(self, container_factory, tracker):
        if False:
            print('Hello World!')

        class Service(object):
            name = 'service'

            @once
            def method(self, a='a', b='b'):
                if False:
                    i = 10
                    return i + 15
                tracker(a, b)
        container = container_factory(Service, config={})
        container.start()
        container.stop()
        assert tracker.call_args == call('a', 'b')

    def test_decorator_with_args(self, container_factory, tracker):
        if False:
            return 10

        class Service(object):
            name = 'service'

            @once('x')
            def method(self, a, b='b'):
                if False:
                    i = 10
                    return i + 15
                tracker(a, b)
        container = container_factory(Service, config={})
        container.start()
        container.stop()
        assert tracker.call_args == call('x', 'b')

    def test_decorator_with_kwargs(self, container_factory, tracker):
        if False:
            while True:
                i = 10

        class Service(object):
            name = 'service'

            @once(b='x')
            def method(self, a='a', b='b'):
                if False:
                    while True:
                        i = 10
                tracker(a, b)
        container = container_factory(Service, config={})
        container.start()
        container.stop()
        assert tracker.call_args == call('a', 'x')

class TestExpectedExceptions(object):

    def test_expected_exceptions(self, container_factory):
        if False:
            while True:
                i = 10
        exceptions = defaultdict(list)

        class CustomException(Exception):
            pass

        class Logger(DependencyProvider):
            """ Example DependencyProvider that interprets
            ``expected_exceptions`` on an entrypoint
            """

            def worker_result(self, worker_ctx, result=None, exc_info=None):
                if False:
                    while True:
                        i = 10
                if exc_info is None:
                    return
                exc = exc_info[1]
                expected = worker_ctx.entrypoint.expected_exceptions
                if isinstance(exc, expected):
                    exceptions['expected'].append(exc)
                else:
                    exceptions['unexpected'].append(exc)

        class Service(object):
            name = 'service'
            logger = Logger()

            @dummy(expected_exceptions=CustomException)
            def expected(self):
                if False:
                    print('Hello World!')
                raise CustomException()

            @dummy
            def unexpected(self):
                if False:
                    return 10
                raise CustomException()
        container = container_factory(Service, {})
        container.start()
        with entrypoint_hook(container, 'expected') as hook:
            with pytest.raises(CustomException) as expected_exc:
                hook()
        assert expected_exc.value in exceptions['expected']
        with entrypoint_hook(container, 'unexpected') as hook:
            with pytest.raises(CustomException) as unexpected_exc:
                hook()
        assert unexpected_exc.value in exceptions['unexpected']

class TestSensitiveArguments(object):

    def test_sensitive_arguments(self, container_factory):
        if False:
            for i in range(10):
                print('nop')
        redacted = {}

        class Logger(DependencyProvider):
            """ Example DependencyProvider that makes use of
            ``get_redacted_args`` to redact ``sensitive_arguments``
            on entrypoints.
            """

            def worker_setup(self, worker_ctx):
                if False:
                    i = 10
                    return i + 15
                entrypoint = worker_ctx.entrypoint
                args = worker_ctx.args
                kwargs = worker_ctx.kwargs
                redacted.update(get_redacted_args(entrypoint, *args, **kwargs))

        class Service(object):
            name = 'service'
            logger = Logger()

            @dummy(sensitive_arguments=('a', 'b.x[0]', 'b.x[2]'))
            def method(self, a, b, c):
                if False:
                    while True:
                        i = 10
                return [a, b, c]
        container = container_factory(Service, {})
        entrypoint = get_extension(container, Entrypoint)
        assert entrypoint.sensitive_arguments == ('a', 'b.x[0]', 'b.x[2]')
        a = 'A'
        b = {'x': [1, 2, 3], 'y': [4, 5, 6]}
        c = 'C'
        with entrypoint_hook(container, 'method') as method:
            assert method(a, b, c) == [a, b, c]
        assert redacted == {'a': REDACTED, 'b': {'x': [REDACTED, 2, REDACTED], 'y': [4, 5, 6]}, 'c': 'C'}

    @pytest.mark.filterwarnings('ignore:The `sensitive_variables`:DeprecationWarning')
    def test_sensitive_variables_backwards_compat(self, container_factory):
        if False:
            while True:
                i = 10

        class Service(object):
            name = 'service'

            @dummy(sensitive_variables=('a', 'b.x[0]', 'b.x[2]'))
            def method(self, a, b, c):
                if False:
                    while True:
                        i = 10
                pass
        with pytest.deprecated_call():
            container = container_factory(Service, {})
        entrypoint = get_extension(container, Entrypoint)
        assert entrypoint.sensitive_arguments == ('a', 'b.x[0]', 'b.x[2]')