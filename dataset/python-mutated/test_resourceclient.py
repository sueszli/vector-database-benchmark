import time
from unittest import TestCase
from unittest.mock import Mock
import requests
from twisted.internet.defer import Deferred
from twisted.python.failure import Failure
from golem.core import golem_async
from golem.resource.client import ClientHandler, ClientError, ClientOptions, ClientConfig
from golem.tools.testwithreactor import TestWithReactor

class TestClientHandler(TestCase):

    class State:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.counter = 0
            self.failure = None

        def verify(self, test_case):
            if False:
                return 10
            if self.failure:
                test_case.fail(self.failure)

        def increment(self):
            if False:
                for i in range(10):
                    print('nop')
            self.counter += 1

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        config = ClientConfig(max_retries=3)
        self.handler = ClientHandler(config)
        self.state = self.State()

    def test_retry(self):
        if False:
            i = 10
            return i + 15
        valid_exceptions = filter(lambda t: t is not Failure, ClientHandler.retry_exceptions)
        valid_exceptions = list(valid_exceptions)
        value_exc = valid_exceptions[0]()

        def func(e):
            if False:
                return 10
            self.state.increment()
            raise e
        for exc_class in valid_exceptions:
            self.state.counter = 0
            self.handler._retry(func, exc_class(value_exc), raise_exc=False)
            assert self.state.counter == self.handler.config.max_retries

    def test_retry_unsupported_exception(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(ArithmeticError):

            def func():
                if False:
                    return 10
                self.state.increment()
                raise ArithmeticError()
            self.handler._retry(func, raise_exc=False)
        assert self.state.counter == 1

    def test_retry_max_tries_exception(self):
        if False:
            return 10
        self.handler.config.max_retries = 0
        raised_exc = ClientHandler.retry_exceptions[0]
        with self.assertRaises(raised_exc):

            def func():
                if False:
                    for i in range(10):
                        print('nop')
                self.state.increment()
                raise raised_exc()
            self.handler._retry(func, raise_exc=True)
        assert self.state.counter == 1

    def test_retry_async(self):
        if False:
            for i in range(10):
                print('nop')

        def func():
            if False:
                while True:
                    i = 10
            self.state.increment()
            deferred = Deferred()
            deferred.callback(42)
            return deferred

        def success(*_a, **_k):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def error(err):
            if False:
                print('Hello World!')
            self.state.failure = 'Error encountered: ' + str(err)
        self._run_and_verify_state(func, success, error)

    def test_retry_async_unsupported_exception(self):
        if False:
            return 10

        def func():
            if False:
                while True:
                    i = 10
            self.state.increment()
            deferred = Deferred()
            deferred.errback(ArithmeticError())
            return deferred

        def success(*_a, **_k):
            if False:
                print('Hello World!')
            self.state.failure = "Success shouldn't have fired"

        def error(*_a):
            if False:
                while True:
                    i = 10
            if self.state.counter != 1:
                self.state.failure = 'Counter error: {} != 1'.format(self.state.counter)
        self._run_and_verify_state(func, success, error)

    def test_retry_async_unwrap_failure(self):
        if False:
            print('Hello World!')

        def func():
            if False:
                for i in range(10):
                    print('nop')
            self.state.increment()
            deferred = Deferred()
            timeout = requests.exceptions.Timeout()
            deferred.errback(Failure(timeout))
            return deferred

        def success(*_a, **_k):
            if False:
                i = 10
                return i + 15
            self.state.failure = "Success shouldn't have fired"

        def error(err):
            if False:
                print('Hello World!')
            if isinstance(err, Failure) and isinstance(err.value, requests.exceptions.Timeout):
                return
            instance = str(type(err))
            self.state.failure = 'Invalid error instance: ' + instance
        self._run_and_verify_state(func, success, error)

    def test_retry_async_max_retries(self):
        if False:
            while True:
                i = 10

        def func():
            if False:
                for i in range(10):
                    print('nop')
            self.state.increment()
            deferred = Deferred()
            deferred.errback(requests.exceptions.Timeout())
            return deferred

        def success(*_a, **_k):
            if False:
                i = 10
                return i + 15
            self.state.failure = "Success shouldn't have fired"

        def error(*_a):
            if False:
                for i in range(10):
                    print('nop')
            if self.state.counter != self.handler.config.max_retries:
                self.state.failure = 'Invalid retry counter'
        self._run_and_verify_state(func, success, error)

    def _run_and_verify_state(self, func, success, error):
        if False:
            print('Hello World!')
        self.handler._retry_async(func).addCallbacks(success, error)
        self.state.verify(self)

class TestClientOptions(TestCase):

    def test_init(self):
        if False:
            print('Hello World!')
        with self.assertRaises(AssertionError):
            ClientOptions(None, 1.0)
        with self.assertRaises(AssertionError):
            ClientOptions('client_id', None)

    def test_get(self):
        if False:
            print('Hello World!')
        option = 'test_option'
        options = ClientOptions('valid_id', 1.0, {})
        options.options[option] = True
        with self.assertRaises(ClientError):
            options.get('valid_id', 0.5, option)
        with self.assertRaises(ClientError):
            options.get('invalid_id', 1.0, option)
        assert options.get('valid_id', 1.0, option)

    def test_clone(self):
        if False:
            while True:
                i = 10
        dict_options = dict(key='val')
        options = ClientOptions('client_id', 1.0, options=dict_options)
        cloned = options.clone()
        assert isinstance(cloned, ClientOptions)
        assert cloned.options == dict_options
        assert cloned.options is not dict_options

    def test_filtered(self):
        if False:
            print('Hello World!')
        dict_options = dict(key='val')
        options = ClientOptions('client_id', 1.0, options=dict_options)
        filtered = options.filtered('client_id', 1.0)
        assert isinstance(filtered, ClientOptions)
        assert filtered is not options
        assert filtered.client_id == options.client_id
        assert filtered.version == options.version
        assert filtered.options == options.options
        assert filtered.options is not options.options
        filtered = options.filtered(None, 1.0)
        assert filtered is None
        filtered = options.filtered('client_id', None)
        assert isinstance(filtered, ClientOptions)

class TestAsyncRequest(TestWithReactor):

    @staticmethod
    def test_initialization():
        if False:
            print('Hello World!')
        request = golem_async.AsyncRequest(lambda x: x)
        assert request.args == []
        assert request.kwargs == {}
        request = golem_async.AsyncRequest(lambda x: x, 'arg', kwarg='kwarg')
        assert request.args == ('arg',)
        assert request.kwargs == {'kwarg': 'kwarg'}

    def test_callbacks(self):
        if False:
            print('Hello World!')
        method = Mock()
        request = golem_async.AsyncRequest(method)
        result = Mock(value=None)

        def success(*_):
            if False:
                return 10
            result.value = True

        def error(*_):
            if False:
                for i in range(10):
                    print('nop')
            result.value = False
        golem_async.async_run(request)
        time.sleep(0.5)
        assert method.call_count == 1
        assert result.value is None
        golem_async.async_run(request, success)
        time.sleep(0.5)
        assert method.call_count == 2
        assert result.value is True
        method.side_effect = Exception
        golem_async.async_run(request, success, error)
        time.sleep(0.5)
        assert method.call_count == 3
        assert result.value is False