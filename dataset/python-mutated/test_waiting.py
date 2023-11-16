import sys
import time
from threading import Thread
import pytest
from mock import ANY, Mock, call, patch
from nameko.testing.waiting import WaitResult, wait_for_call

@pytest.fixture
def forever():
    if False:
        i = 10
        return i + 15
    value = [True]
    yield value
    value.pop()

class TestPatchWaitUseCases(object):

    def test_wait_for_specific_result(self, forever):
        if False:
            print('Hello World!')

        class Counter(object):
            value = 0

            def count(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.value += 1
                return self.value
        counter = Counter()

        def count_forever():
            if False:
                i = 10
                return i + 15
            while forever:
                counter.count()
                time.sleep(0)

        def cb(args, kwargs, res, exc_info):
            if False:
                i = 10
                return i + 15
            return res == 10
        with wait_for_call(counter, 'count', callback=cb) as result:
            Thread(target=count_forever).start()
        assert result.get() == 10

    def test_wait_until_called_with_argument(self, forever):
        if False:
            for i in range(10):
                print('nop')

        class CounterWithSet(object):
            value = 0

            def set(self, value):
                if False:
                    print('Hello World!')
                self.value = value
                return self.value
        counter = CounterWithSet()

        def increment_forever_via_set():
            if False:
                print('Hello World!')
            while forever:
                counter.set(counter.value + 1)
                time.sleep(0)

        def cb(args, kwargs, res, exc_info):
            if False:
                print('Hello World!')
            return args == (10,)
        with wait_for_call(counter, 'set', callback=cb) as result:
            Thread(target=increment_forever_via_set).start()
        assert result.get() == 10

    def test_wait_until_raises(self, forever):
        if False:
            print('Hello World!')

        class LimitExceeded(Exception):
            pass

        class CounterWithLimit(object):

            def __init__(self, limit):
                if False:
                    return 10
                self.value = 0
                self.limit = limit

            def count(self):
                if False:
                    return 10
                self.value += 1
                if self.value >= self.limit:
                    raise LimitExceeded(self.limit)
                return self.value
        limit = 10
        counter = CounterWithLimit(limit)

        def count_forever():
            if False:
                for i in range(10):
                    print('nop')
            while forever:
                counter.count()
                time.sleep(0)

        def cb(args, kwargs, res, exc_info):
            if False:
                print('Hello World!')
            return exc_info is not None
        with wait_for_call(counter, 'count', callback=cb) as result:
            Thread(target=count_forever).start()
        with pytest.raises(LimitExceeded):
            result.get()

    def test_wait_until_stops_raising(self, forever):
        if False:
            print('Hello World!')

        class ThresholdNotReached(Exception):
            pass

        class CounterWithThreshold(object):

            def __init__(self, threshold):
                if False:
                    print('Hello World!')
                self.value = 0
                self.threshold = threshold

            def count(self):
                if False:
                    print('Hello World!')
                self.value += 1
                if self.value < self.threshold:
                    raise ThresholdNotReached(self.threshold)
                return self.value
        threshold = 10
        counter = CounterWithThreshold(threshold)

        def count_forever():
            if False:
                while True:
                    i = 10
            while forever:
                try:
                    counter.count()
                except ThresholdNotReached:
                    pass
                time.sleep(0)

        def cb(args, kwargs, res, exc_info):
            if False:
                i = 10
                return i + 15
            return exc_info is None
        with wait_for_call(counter, 'count', callback=cb) as result:
            Thread(target=count_forever).start()
        assert result.get() == threshold

class TestPatchWait(object):

    def test_direct(self):
        if False:
            return 10

        class Echo(object):

            def upper(self, arg):
                if False:
                    return 10
                return arg.upper()
        echo = Echo()
        arg = 'hello'
        with wait_for_call(echo, 'upper'):
            res = echo.upper(arg)
            assert res == 'HELLO'

    def test_indirect(self):
        if False:
            i = 10
            return i + 15

        class Echo(object):

            def proxy(self, arg):
                if False:
                    i = 10
                    return i + 15
                return self.upper(arg)

            def upper(self, arg):
                if False:
                    return 10
                return arg.upper()
        echo = Echo()
        arg = 'hello'
        with wait_for_call(echo, 'upper'):
            assert echo.proxy(arg) == 'HELLO'

    def test_patch_class(self):
        if False:
            while True:
                i = 10

        class Echo(object):

            def upper(self, arg):
                if False:
                    return 10
                return arg.upper()
        echo = Echo()
        arg = 'hello'
        with wait_for_call(Echo, 'upper'):
            res = echo.upper(arg)
            assert res == 'HELLO'

    def test_result(self):
        if False:
            print('Hello World!')

        class Echo(object):

            def upper(self, arg):
                if False:
                    i = 10
                    return i + 15
                return arg.upper()
        echo = Echo()
        arg = 'hello'
        with wait_for_call(echo, 'upper') as result:
            res = echo.upper(arg)
        assert result.get() == res

    def test_result_not_ready(self):
        if False:
            i = 10
            return i + 15

        class Echo(object):

            def upper(self, arg):
                if False:
                    while True:
                        i = 10
                return arg.upper()
        echo = Echo()
        arg = 'hello'
        with wait_for_call(echo, 'upper') as result:
            with pytest.raises(result.NotReady):
                result.get()
            res = echo.upper(arg)
        assert result.get() == res

    def test_result_is_none(self):
        if False:
            for i in range(10):
                print('nop')

        class Echo(object):

            def nothing(self):
                if False:
                    while True:
                        i = 10
                return None
        echo = Echo()
        with wait_for_call(echo, 'nothing') as result:
            res = echo.nothing()
        assert res is None
        assert result.get() is None
        assert result.has_result is True

    def test_wrapped_method_raises(self):
        if False:
            return 10

        class EchoException(Exception):
            pass

        class Echo(object):

            def error(self):
                if False:
                    while True:
                        i = 10
                raise EchoException('error!')
        echo = Echo()
        with wait_for_call(echo, 'error'):
            with pytest.raises(EchoException):
                echo.error()

    def test_result_get_raises(self):
        if False:
            i = 10
            return i + 15

        class EchoException(Exception):
            pass

        class Echo(object):

            def error(self):
                if False:
                    while True:
                        i = 10
                raise EchoException('error!')
        echo = Echo()
        with wait_for_call(echo, 'error') as result:
            with pytest.raises(EchoException):
                echo.error()
            with pytest.raises(EchoException):
                result.get()

    def test_callback(self):
        if False:
            print('Hello World!')

        class Echo(object):

            def upper(self, arg):
                if False:
                    while True:
                        i = 10
                return arg.upper()
        echo = Echo()
        arg = 'hello'
        callback = Mock()
        callback.return_value = True
        with wait_for_call(echo, 'upper', callback):
            res = echo.upper(arg)
            assert res == 'HELLO'
        assert callback.called
        assert callback.call_args_list == [call((arg,), {}, res, None)]

    def test_callback_multiple_calls(self):
        if False:
            for i in range(10):
                print('nop')

        class Echo(object):
            count = 0

            def upper(self, arg):
                if False:
                    print('Hello World!')
                self.count += 1
                return '{}-{}'.format(arg.upper(), self.count)
        echo = Echo()
        arg = 'hello'
        callback = Mock()
        callback.side_effect = [False, True]
        with wait_for_call(echo, 'upper', callback):
            res1 = echo.upper(arg)
            assert res1 == 'HELLO-1'
            res2 = echo.upper(arg)
            assert res2 == 'HELLO-2'
        assert callback.called
        assert callback.call_args_list == [call((arg,), {}, res1, None), call((arg,), {}, res2, None)]

    def test_callback_with_exception(self):
        if False:
            for i in range(10):
                print('nop')

        class EchoException(Exception):
            pass

        class Echo(object):

            def error(self):
                if False:
                    i = 10
                    return i + 15
                raise exc
        echo = Echo()
        exc = EchoException('error!')
        callback = Mock()
        callback.return_value = True
        with wait_for_call(echo, 'error', callback):
            with pytest.raises(EchoException):
                echo.error()
        assert callback.called
        assert callback.call_args_list == [call((), {}, None, (EchoException, exc, ANY))]

    def test_callback_with_exception_multiple_calls(self):
        if False:
            i = 10
            return i + 15

        class EchoException(Exception):
            pass

        class Echo(object):

            def error(self):
                if False:
                    while True:
                        i = 10
                raise exc
        echo = Echo()
        exc = EchoException('error!')
        callback = Mock()
        callback.side_effect = [False, True]
        with wait_for_call(echo, 'error', callback):
            with pytest.raises(EchoException):
                echo.error()
            with pytest.raises(EchoException):
                echo.error()
        assert callback.called
        assert callback.call_args_list == [call((), {}, None, (EchoException, exc, ANY)), call((), {}, None, (EchoException, exc, ANY))]

    def test_with_new_thread(self):
        if False:
            while True:
                i = 10

        class Echo(object):

            def proxy(self, arg):
                if False:
                    print('Hello World!')
                Thread(target=self.upper, args=(arg,)).start()

            def upper(self, arg):
                if False:
                    return 10
                return arg.upper()
        echo = Echo()
        arg = 'hello'
        callback = Mock()
        callback.return_value = True
        with wait_for_call(echo, 'upper', callback):
            res = echo.proxy(arg)
            assert res is None
        assert callback.called
        assert callback.call_args_list == [call((arg,), {}, 'HELLO', None)]

    def test_target_as_mock(self):
        if False:
            return 10

        class Klass(object):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.attr = 'value'

            def method(self):
                if False:
                    return 10
                return self.attr.upper()
        instance = Klass()
        with patch.object(instance, 'attr') as patched_attr:
            with wait_for_call(patched_attr, 'upper'):
                instance.method()
            assert patched_attr.upper.called
            assert instance.attr.upper.called

class TestWaitResult(object):

    class CustomError(Exception):
        pass

    @pytest.fixture
    def exc_info(self):
        if False:
            return 10
        try:
            raise self.CustomError('whoops')
        except:
            exc_info = sys.exc_info()
        return exc_info

    def test_has_result(self):
        if False:
            return 10
        result = WaitResult()
        assert result.has_result is False
        result.send('ok', None)
        assert result.has_result is True

    def test_has_exception(self, exc_info):
        if False:
            return 10
        result = WaitResult()
        assert result.has_result is False
        result.send(None, exc_info)
        assert result.has_result is True

    def test_send_multiple_times(self):
        if False:
            i = 10
            return i + 15
        result = WaitResult()
        result.send(1, None)
        result.send(2, None)
        assert result.get() == 1

    def test_get_result_multiple_times(self):
        if False:
            return 10
        result = WaitResult()
        result.send(1, None)
        assert result.get() == 1
        assert result.get() == 1

    def test_get_raises(self, exc_info):
        if False:
            i = 10
            return i + 15
        result = WaitResult()
        result.send(1, exc_info)
        with pytest.raises(self.CustomError):
            result.get()