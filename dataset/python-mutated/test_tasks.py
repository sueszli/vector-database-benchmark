import socket
import tempfile
from datetime import datetime, timedelta
from unittest.mock import ANY, MagicMock, Mock, patch, sentinel
import pytest
from kombu import Queue
from kombu.exceptions import EncodeError
from celery import Task, chain, group, uuid
from celery.app.task import _reprtask
from celery.canvas import StampingVisitor, signature
from celery.contrib.testing.mocks import ContextMock
from celery.exceptions import Ignore, ImproperlyConfigured, Retry
from celery.result import AsyncResult, EagerResult
from celery.utils.serialization import UnpickleableExceptionWrapper
try:
    from urllib.error import HTTPError
except ImportError:
    from urllib2 import HTTPError

def return_True(*args, **kwargs):
    if False:
        return 10
    return True

class MockApplyTask(Task):
    abstract = True
    applied = 0

    def run(self, x, y):
        if False:
            while True:
                i = 10
        return x * y

    def apply_async(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.applied += 1

class TaskWithPriority(Task):
    priority = 10

class TaskWithRetry(Task):
    autoretry_for = (TypeError,)
    retry_kwargs = {'max_retries': 5}
    retry_backoff = True
    retry_backoff_max = 700
    retry_jitter = False

class TaskWithRetryButForTypeError(Task):
    autoretry_for = (Exception,)
    dont_autoretry_for = (TypeError,)
    retry_kwargs = {'max_retries': 5}
    retry_backoff = True
    retry_backoff_max = 700
    retry_jitter = False

class TasksCase:

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        self.mytask = self.app.task(shared=False)(return_True)

        @self.app.task(bind=True, count=0, shared=False)
        def increment_counter(self, increment_by=1):
            if False:
                print('Hello World!')
            self.count += increment_by or 1
            return self.count
        self.increment_counter = increment_counter

        @self.app.task(shared=False)
        def raising():
            if False:
                i = 10
                return i + 15
            raise KeyError('foo')
        self.raising = raising

        @self.app.task(bind=True, max_retries=3, iterations=0, shared=False)
        def retry_task(self, arg1, arg2, kwarg=1, max_retries=None, care=True):
            if False:
                print('Hello World!')
            self.iterations += 1
            rmax = self.max_retries if max_retries is None else max_retries
            assert repr(self.request)
            retries = self.request.retries
            if care and retries >= rmax:
                return arg1
            else:
                raise self.retry(countdown=0, max_retries=rmax)
        self.retry_task = retry_task

        @self.app.task(bind=True, max_retries=3, iterations=0, shared=False)
        def retry_task_noargs(self, **kwargs):
            if False:
                print('Hello World!')
            self.iterations += 1
            if self.request.retries >= 3:
                return 42
            else:
                raise self.retry(countdown=0)
        self.retry_task_noargs = retry_task_noargs

        @self.app.task(bind=True, max_retries=3, iterations=0, shared=False)
        def retry_task_return_without_throw(self, **kwargs):
            if False:
                i = 10
                return i + 15
            self.iterations += 1
            try:
                if self.request.retries >= 3:
                    return 42
                else:
                    raise Exception('random code exception')
            except Exception as exc:
                return self.retry(exc=exc, throw=False)
        self.retry_task_return_without_throw = retry_task_return_without_throw

        @self.app.task(bind=True, max_retries=3, iterations=0, shared=False)
        def retry_task_return_with_throw(self, **kwargs):
            if False:
                i = 10
                return i + 15
            self.iterations += 1
            try:
                if self.request.retries >= 3:
                    return 42
                else:
                    raise Exception('random code exception')
            except Exception as exc:
                return self.retry(exc=exc, throw=True)
        self.retry_task_return_with_throw = retry_task_return_with_throw

        @self.app.task(bind=True, max_retries=3, iterations=0, shared=False, autoretry_for=(Exception,))
        def retry_task_auto_retry_with_single_new_arg(self, ret=None, **kwargs):
            if False:
                print('Hello World!')
            if ret is None:
                return self.retry(exc=Exception('I have filled now'), args=['test'], kwargs=kwargs)
            else:
                return ret
        self.retry_task_auto_retry_with_single_new_arg = retry_task_auto_retry_with_single_new_arg

        @self.app.task(bind=True, max_retries=3, iterations=0, shared=False)
        def retry_task_auto_retry_with_new_args(self, ret=None, place_holder=None, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            if ret is None:
                return self.retry(args=[place_holder, place_holder], kwargs=kwargs)
            else:
                return ret
        self.retry_task_auto_retry_with_new_args = retry_task_auto_retry_with_new_args

        @self.app.task(bind=True, max_retries=3, iterations=0, shared=False, autoretry_for=(Exception,))
        def retry_task_auto_retry_exception_with_new_args(self, ret=None, place_holder=None, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            if ret is None:
                return self.retry(exc=Exception('I have filled'), args=[place_holder, place_holder], kwargs=kwargs)
            else:
                return ret
        self.retry_task_auto_retry_exception_with_new_args = retry_task_auto_retry_exception_with_new_args

        @self.app.task(bind=True, max_retries=10, iterations=0, shared=False, autoretry_for=(Exception,))
        def retry_task_max_retries_override(self, **kwargs):
            if False:
                print('Hello World!')
            self.iterations += 1
            if self.iterations == 3:
                self.retry(exc=MyCustomException, max_retries=0)
            self.retry(exc=MyCustomException)
        self.retry_task_max_retries_override = retry_task_max_retries_override

        @self.app.task(bind=True, max_retries=0, iterations=0, shared=False, autoretry_for=(Exception,))
        def retry_task_explicit_exception(self, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            self.iterations += 1
            raise MyCustomException()
        self.retry_task_explicit_exception = retry_task_explicit_exception

        @self.app.task(bind=True, max_retries=3, iterations=0, shared=False)
        def retry_task_raise_without_throw(self, **kwargs):
            if False:
                i = 10
                return i + 15
            self.iterations += 1
            try:
                if self.request.retries >= 3:
                    return 42
                else:
                    raise Exception('random code exception')
            except Exception as exc:
                raise self.retry(exc=exc, throw=False)
        self.retry_task_raise_without_throw = retry_task_raise_without_throw

        @self.app.task(bind=True, max_retries=3, iterations=0, base=MockApplyTask, shared=False)
        def retry_task_mockapply(self, arg1, arg2, kwarg=1):
            if False:
                return 10
            self.iterations += 1
            retries = self.request.retries
            if retries >= 3:
                return arg1
            raise self.retry(countdown=0)
        self.retry_task_mockapply = retry_task_mockapply

        @self.app.task(bind=True, max_retries=3, iterations=0, shared=False)
        def retry_task_customexc(self, arg1, arg2, kwarg=1, **kwargs):
            if False:
                while True:
                    i = 10
            self.iterations += 1
            retries = self.request.retries
            if retries >= 3:
                return arg1 + kwarg
            else:
                try:
                    raise MyCustomException('Elaine Marie Benes')
                except MyCustomException as exc:
                    kwargs.update(kwarg=kwarg)
                    raise self.retry(countdown=0, exc=exc)
        self.retry_task_customexc = retry_task_customexc

        @self.app.task(bind=True, max_retries=3, iterations=0, shared=False)
        def retry_task_unpickleable_exc(self, foo, bar):
            if False:
                return 10
            self.iterations += 1
            raise self.retry(countdown=0, exc=UnpickleableException(foo, bar))
        self.retry_task_unpickleable_exc = retry_task_unpickleable_exc

        @self.app.task(bind=True, autoretry_for=(ZeroDivisionError,), shared=False)
        def autoretry_task_no_kwargs(self, a, b):
            if False:
                print('Hello World!')
            self.iterations += 1
            return a / b
        self.autoretry_task_no_kwargs = autoretry_task_no_kwargs

        @self.app.task(bind=True, autoretry_for=(ZeroDivisionError,), retry_kwargs={'max_retries': 5}, shared=False)
        def autoretry_task(self, a, b):
            if False:
                for i in range(10):
                    print('nop')
            self.iterations += 1
            return a / b
        self.autoretry_task = autoretry_task

        @self.app.task(bind=True, autoretry_for=(ArithmeticError,), dont_autoretry_for=(ZeroDivisionError,), retry_kwargs={'max_retries': 5}, shared=False)
        def autoretry_arith_task(self, a, b):
            if False:
                i = 10
                return i + 15
            self.iterations += 1
            return a / b
        self.autoretry_arith_task = autoretry_arith_task

        @self.app.task(bind=True, base=TaskWithRetry, shared=False)
        def autoretry_for_from_base_task(self, a, b):
            if False:
                return 10
            self.iterations += 1
            return a + b
        self.autoretry_for_from_base_task = autoretry_for_from_base_task

        @self.app.task(bind=True, base=TaskWithRetry, autoretry_for=(ZeroDivisionError,), shared=False)
        def override_autoretry_for_from_base_task(self, a, b):
            if False:
                i = 10
                return i + 15
            self.iterations += 1
            return a / b
        self.override_autoretry_for = override_autoretry_for_from_base_task

        @self.app.task(bind=True, base=TaskWithRetry, shared=False)
        def retry_kwargs_from_base_task(self, a, b):
            if False:
                i = 10
                return i + 15
            self.iterations += 1
            return a + b
        self.retry_kwargs_from_base_task = retry_kwargs_from_base_task

        @self.app.task(bind=True, base=TaskWithRetry, retry_kwargs={'max_retries': 2}, shared=False)
        def override_retry_kwargs_from_base_task(self, a, b):
            if False:
                return 10
            self.iterations += 1
            return a + b
        self.override_retry_kwargs = override_retry_kwargs_from_base_task

        @self.app.task(bind=True, base=TaskWithRetry, shared=False)
        def retry_backoff_from_base_task(self, a, b):
            if False:
                i = 10
                return i + 15
            self.iterations += 1
            return a + b
        self.retry_backoff_from_base_task = retry_backoff_from_base_task

        @self.app.task(bind=True, base=TaskWithRetry, retry_backoff=False, shared=False)
        def override_retry_backoff_from_base_task(self, a, b):
            if False:
                for i in range(10):
                    print('nop')
            self.iterations += 1
            return a + b
        self.override_retry_backoff = override_retry_backoff_from_base_task

        @self.app.task(bind=True, base=TaskWithRetry, shared=False)
        def retry_backoff_max_from_base_task(self, a, b):
            if False:
                while True:
                    i = 10
            self.iterations += 1
            return a + b
        self.retry_backoff_max_from_base_task = retry_backoff_max_from_base_task

        @self.app.task(bind=True, base=TaskWithRetry, retry_backoff_max=16, shared=False)
        def override_retry_backoff_max_from_base_task(self, a, b):
            if False:
                i = 10
                return i + 15
            self.iterations += 1
            return a + b
        self.override_backoff_max = override_retry_backoff_max_from_base_task

        @self.app.task(bind=True, base=TaskWithRetry, shared=False)
        def retry_backoff_jitter_from_base_task(self, a, b):
            if False:
                return 10
            self.iterations += 1
            return a + b
        self.retry_backoff_jitter_from_base = retry_backoff_jitter_from_base_task

        @self.app.task(bind=True, base=TaskWithRetry, retry_jitter=True, shared=False)
        def override_backoff_jitter_from_base_task(self, a, b):
            if False:
                for i in range(10):
                    print('nop')
            self.iterations += 1
            return a + b
        self.override_backoff_jitter = override_backoff_jitter_from_base_task

        @self.app.task(bind=True)
        def task_check_request_context(self):
            if False:
                return 10
            assert self.request.hostname == socket.gethostname()
        self.task_check_request_context = task_check_request_context

        @self.app.task(ignore_result=True)
        def task_with_ignored_result():
            if False:
                i = 10
                return i + 15
            pass
        self.task_with_ignored_result = task_with_ignored_result

        @self.app.task(bind=True)
        def task_called_by_other_task(self):
            if False:
                print('Hello World!')
            pass

        @self.app.task(bind=True)
        def task_which_calls_other_task(self):
            if False:
                print('Hello World!')
            self.request.delivery_info['priority'] = 5
            task_called_by_other_task.delay()
        self.task_which_calls_other_task = task_which_calls_other_task

        @self.app.task(bind=True)
        def task_replacing_another_task(self):
            if False:
                for i in range(10):
                    print('nop')
            return 'replaced'
        self.task_replacing_another_task = task_replacing_another_task

        @self.app.task(bind=True)
        def task_replaced_by_other_task(self):
            if False:
                i = 10
                return i + 15
            return self.replace(task_replacing_another_task.si())

        @self.app.task(bind=True, autoretry_for=(Exception,))
        def task_replaced_by_other_task_with_autoretry(self):
            if False:
                print('Hello World!')
            return self.replace(task_replacing_another_task.si())
        self.task_replaced_by_other_task = task_replaced_by_other_task
        self.task_replaced_by_other_task_with_autoretry = task_replaced_by_other_task_with_autoretry
        from kombu.transport.memory import Channel
        Channel.queues.clear()

class MyCustomException(Exception):
    """Random custom exception."""

class UnpickleableException(Exception):
    """Exception that doesn't survive a pickling roundtrip (dump + load)."""

    def __init__(self, foo, bar):
        if False:
            return 10
        super().__init__(foo)
        self.bar = bar

class test_task_retries(TasksCase):

    def test_retry(self):
        if False:
            print('Hello World!')
        self.retry_task.max_retries = 3
        self.retry_task.iterations = 0
        self.retry_task.apply([255, 65535])
        assert self.retry_task.iterations == 4
        self.retry_task.max_retries = 3
        self.retry_task.iterations = 0
        self.retry_task.apply([255, 65535], {'max_retries': 10})
        assert self.retry_task.iterations == 11

    def test_retry_priority(self):
        if False:
            while True:
                i = 10
        priority = 7
        self.retry_task.priority = priority
        self.retry_task.push_request()
        self.retry_task.request.delivery_info = {'priority': priority}
        sig = self.retry_task.signature_from_request()
        assert sig.options['priority'] == priority

    def test_retry_no_args(self):
        if False:
            while True:
                i = 10
        self.retry_task_noargs.max_retries = 3
        self.retry_task_noargs.iterations = 0
        self.retry_task_noargs.apply(propagate=True).get()
        assert self.retry_task_noargs.iterations == 4

    def test_signature_from_request__passes_headers(self):
        if False:
            return 10
        self.retry_task.push_request()
        self.retry_task.request.headers = {'custom': 10.1}
        sig = self.retry_task.signature_from_request()
        assert sig.options['headers']['custom'] == 10.1

    def test_signature_from_request__delivery_info(self):
        if False:
            return 10
        self.retry_task.push_request()
        self.retry_task.request.delivery_info = {'exchange': 'testex', 'routing_key': 'testrk'}
        sig = self.retry_task.signature_from_request()
        assert sig.options['exchange'] == 'testex'
        assert sig.options['routing_key'] == 'testrk'

    def test_signature_from_request__shadow_name(self):
        if False:
            return 10
        self.retry_task.push_request()
        self.retry_task.request.shadow = 'test'
        sig = self.retry_task.signature_from_request()
        assert sig.options['shadow'] == 'test'

    def test_retry_kwargs_can_be_empty(self):
        if False:
            for i in range(10):
                print('nop')
        self.retry_task_mockapply.push_request()
        try:
            with pytest.raises(Retry):
                import sys
                try:
                    sys.exc_clear()
                except AttributeError:
                    pass
                self.retry_task_mockapply.retry(args=[4, 4], kwargs=None)
        finally:
            self.retry_task_mockapply.pop_request()

    def test_retry_without_throw_eager(self):
        if False:
            return 10
        assert self.retry_task_return_without_throw.apply().get() == 42

    def test_raise_without_throw_eager(self):
        if False:
            i = 10
            return i + 15
        assert self.retry_task_raise_without_throw.apply().get() == 42

    def test_return_with_throw_eager(self):
        if False:
            return 10
        assert self.retry_task_return_with_throw.apply().get() == 42

    def test_eager_retry_with_single_new_params(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.retry_task_auto_retry_with_single_new_arg.apply().get() == 'test'

    def test_eager_retry_with_new_params(self):
        if False:
            i = 10
            return i + 15
        assert self.retry_task_auto_retry_with_new_args.si(place_holder='test').apply().get() == 'test'

    def test_eager_retry_with_autoretry_for_exception(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.retry_task_auto_retry_exception_with_new_args.si(place_holder='test').apply().get() == 'test'

    def test_retry_task_max_retries_override(self):
        if False:
            for i in range(10):
                print('nop')
        self.retry_task_max_retries_override.max_retries = 10
        self.retry_task_max_retries_override.iterations = 0
        result = self.retry_task_max_retries_override.apply()
        with pytest.raises(MyCustomException):
            result.get()
        assert self.retry_task_max_retries_override.iterations == 3

    def test_retry_task_explicit_exception(self):
        if False:
            while True:
                i = 10
        self.retry_task_explicit_exception.max_retries = 0
        self.retry_task_explicit_exception.iterations = 0
        result = self.retry_task_explicit_exception.apply()
        with pytest.raises(MyCustomException):
            result.get()
        assert self.retry_task_explicit_exception.iterations == 1

    def test_retry_eager_should_return_value(self):
        if False:
            print('Hello World!')
        self.retry_task.max_retries = 3
        self.retry_task.iterations = 0
        assert self.retry_task.apply([255, 65535]).get() == 255
        assert self.retry_task.iterations == 4

    def test_retry_not_eager(self):
        if False:
            return 10
        self.retry_task_mockapply.push_request()
        try:
            self.retry_task_mockapply.request.called_directly = False
            exc = Exception('baz')
            try:
                self.retry_task_mockapply.retry(args=[4, 4], kwargs={'task_retries': 0}, exc=exc, throw=False)
                assert self.retry_task_mockapply.applied
            finally:
                self.retry_task_mockapply.applied = 0
            try:
                with pytest.raises(Retry):
                    self.retry_task_mockapply.retry(args=[4, 4], kwargs={'task_retries': 0}, exc=exc, throw=True)
                assert self.retry_task_mockapply.applied
            finally:
                self.retry_task_mockapply.applied = 0
        finally:
            self.retry_task_mockapply.pop_request()

    def test_retry_with_kwargs(self):
        if False:
            return 10
        self.retry_task_customexc.max_retries = 3
        self.retry_task_customexc.iterations = 0
        self.retry_task_customexc.apply([255, 65535], {'kwarg': 15})
        assert self.retry_task_customexc.iterations == 4

    def test_retry_with_custom_exception(self):
        if False:
            print('Hello World!')
        self.retry_task_customexc.max_retries = 2
        self.retry_task_customexc.iterations = 0
        result = self.retry_task_customexc.apply([255, 65535], {'kwarg': 15})
        with pytest.raises(MyCustomException):
            result.get()
        assert self.retry_task_customexc.iterations == 3

    def test_retry_with_unpickleable_exception(self):
        if False:
            i = 10
            return i + 15
        self.retry_task_unpickleable_exc.max_retries = 2
        self.retry_task_unpickleable_exc.iterations = 0
        result = self.retry_task_unpickleable_exc.apply(['foo', 'bar'])
        with pytest.raises(UnpickleableExceptionWrapper) as exc_info:
            result.get()
        assert self.retry_task_unpickleable_exc.iterations == 3
        exc_wrapper = exc_info.value
        assert exc_wrapper.exc_cls_name == 'UnpickleableException'
        assert exc_wrapper.exc_args == ('foo',)

    def test_max_retries_exceeded(self):
        if False:
            for i in range(10):
                print('nop')
        self.retry_task.max_retries = 2
        self.retry_task.iterations = 0
        result = self.retry_task.apply([255, 65535], {'care': False})
        with pytest.raises(self.retry_task.MaxRetriesExceededError):
            result.get()
        assert self.retry_task.iterations == 3
        self.retry_task.max_retries = 1
        self.retry_task.iterations = 0
        result = self.retry_task.apply([255, 65535], {'care': False})
        with pytest.raises(self.retry_task.MaxRetriesExceededError):
            result.get()
        assert self.retry_task.iterations == 2

    def test_max_retries_exceeded_task_args(self):
        if False:
            return 10
        self.retry_task.max_retries = 2
        self.retry_task.iterations = 0
        args = (255, 65535)
        kwargs = {'care': False}
        result = self.retry_task.apply(args, kwargs)
        with pytest.raises(self.retry_task.MaxRetriesExceededError) as e:
            result.get()
        assert e.value.task_args == args
        assert e.value.task_kwargs == kwargs

    def test_autoretry_no_kwargs(self):
        if False:
            while True:
                i = 10
        self.autoretry_task_no_kwargs.max_retries = 3
        self.autoretry_task_no_kwargs.iterations = 0
        self.autoretry_task_no_kwargs.apply((1, 0))
        assert self.autoretry_task_no_kwargs.iterations == 4

    def test_autoretry(self):
        if False:
            print('Hello World!')
        self.autoretry_task.max_retries = 3
        self.autoretry_task.iterations = 0
        self.autoretry_task.apply((1, 0))
        assert self.autoretry_task.iterations == 6

    def test_autoretry_arith(self):
        if False:
            for i in range(10):
                print('nop')
        self.autoretry_arith_task.max_retries = 3
        self.autoretry_arith_task.iterations = 0
        self.autoretry_arith_task.apply((1, 0))
        assert self.autoretry_arith_task.iterations == 1

    @pytest.mark.parametrize('retry_backoff, expected_countdowns', [(False, [None, None, None, None]), (0, [None, None, None, None]), (0.0, [None, None, None, None]), (True, [1, 2, 4, 8]), (-1, [1, 2, 4, 8]), (0.1, [1, 2, 4, 8]), (1, [1, 2, 4, 8]), (1.9, [1, 2, 4, 8]), (2, [2, 4, 8, 16])])
    def test_autoretry_backoff(self, retry_backoff, expected_countdowns):
        if False:
            i = 10
            return i + 15

        @self.app.task(bind=True, shared=False, autoretry_for=(ZeroDivisionError,), retry_backoff=retry_backoff, retry_jitter=False, max_retries=3)
        def task(self_, x, y):
            if False:
                return 10
            self_.iterations += 1
            return x / y
        task.iterations = 0
        with patch.object(task, 'retry', wraps=task.retry) as fake_retry:
            task.apply((1, 0))
        assert task.iterations == 4
        retry_call_countdowns = [call_[1].get('countdown') for call_ in fake_retry.call_args_list]
        assert retry_call_countdowns == expected_countdowns

    @pytest.mark.parametrize('retry_backoff, expected_countdowns', [(False, [None, None, None, None]), (0, [None, None, None, None]), (0.0, [None, None, None, None]), (True, [0, 1, 3, 7]), (-1, [0, 1, 3, 7]), (0.1, [0, 1, 3, 7]), (1, [0, 1, 3, 7]), (1.9, [0, 1, 3, 7]), (2, [1, 3, 7, 15])])
    @patch('random.randrange', side_effect=lambda i: i - 2)
    def test_autoretry_backoff_jitter(self, randrange, retry_backoff, expected_countdowns):
        if False:
            print('Hello World!')

        @self.app.task(bind=True, shared=False, autoretry_for=(HTTPError,), retry_backoff=retry_backoff, retry_jitter=True, max_retries=3)
        def task(self_, url):
            if False:
                return 10
            self_.iterations += 1
            if 'error' in url:
                fp = tempfile.TemporaryFile()
                raise HTTPError(url, '500', 'Error', '', fp)
        task.iterations = 0
        with patch.object(task, 'retry', wraps=task.retry) as fake_retry:
            task.apply(('http://httpbin.org/error',))
        assert task.iterations == 4
        retry_call_countdowns = [call_[1].get('countdown') for call_ in fake_retry.call_args_list]
        assert retry_call_countdowns == expected_countdowns

    def test_autoretry_for_from_base(self):
        if False:
            for i in range(10):
                print('nop')
        self.autoretry_for_from_base_task.iterations = 0
        self.autoretry_for_from_base_task.apply((1, 'a'))
        assert self.autoretry_for_from_base_task.iterations == 6

    def test_override_autoretry_for_from_base(self):
        if False:
            i = 10
            return i + 15
        self.override_autoretry_for.iterations = 0
        self.override_autoretry_for.apply((1, 0))
        assert self.override_autoretry_for.iterations == 6

    def test_retry_kwargs_from_base(self):
        if False:
            print('Hello World!')
        self.retry_kwargs_from_base_task.iterations = 0
        self.retry_kwargs_from_base_task.apply((1, 'a'))
        assert self.retry_kwargs_from_base_task.iterations == 6

    def test_override_retry_kwargs_from_base(self):
        if False:
            while True:
                i = 10
        self.override_retry_kwargs.iterations = 0
        self.override_retry_kwargs.apply((1, 'a'))
        assert self.override_retry_kwargs.iterations == 3

    def test_retry_backoff_from_base(self):
        if False:
            print('Hello World!')
        task = self.retry_backoff_from_base_task
        task.iterations = 0
        with patch.object(task, 'retry', wraps=task.retry) as fake_retry:
            task.apply((1, 'a'))
        assert task.iterations == 6
        retry_call_countdowns = [call_[1]['countdown'] for call_ in fake_retry.call_args_list]
        assert retry_call_countdowns == [1, 2, 4, 8, 16, 32]

    @patch('celery.app.autoretry.get_exponential_backoff_interval')
    def test_override_retry_backoff_from_base(self, backoff):
        if False:
            for i in range(10):
                print('nop')
        self.override_retry_backoff.iterations = 0
        self.override_retry_backoff.apply((1, 'a'))
        assert self.override_retry_backoff.iterations == 6
        assert backoff.call_count == 0

    def test_retry_backoff_max_from_base(self):
        if False:
            for i in range(10):
                print('nop')
        task = self.retry_backoff_max_from_base_task
        task.iterations = 0
        with patch.object(task, 'retry', wraps=task.retry) as fake_retry:
            task.apply((1, 'a'))
        assert task.iterations == 6
        retry_call_countdowns = [call_[1]['countdown'] for call_ in fake_retry.call_args_list]
        assert retry_call_countdowns == [1, 2, 4, 8, 16, 32]

    def test_override_retry_backoff_max_from_base(self):
        if False:
            for i in range(10):
                print('nop')
        task = self.override_backoff_max
        task.iterations = 0
        with patch.object(task, 'retry', wraps=task.retry) as fake_retry:
            task.apply((1, 'a'))
        assert task.iterations == 6
        retry_call_countdowns = [call_[1]['countdown'] for call_ in fake_retry.call_args_list]
        assert retry_call_countdowns == [1, 2, 4, 8, 16, 16]

    def test_retry_backoff_jitter_from_base(self):
        if False:
            for i in range(10):
                print('nop')
        task = self.retry_backoff_jitter_from_base
        task.iterations = 0
        with patch.object(task, 'retry', wraps=task.retry) as fake_retry:
            task.apply((1, 'a'))
        assert task.iterations == 6
        retry_call_countdowns = [call_[1]['countdown'] for call_ in fake_retry.call_args_list]
        assert retry_call_countdowns == [1, 2, 4, 8, 16, 32]

    @patch('random.randrange', side_effect=lambda i: i - 2)
    def test_override_backoff_jitter_from_base(self, randrange):
        if False:
            i = 10
            return i + 15
        task = self.override_backoff_jitter
        task.iterations = 0
        with patch.object(task, 'retry', wraps=task.retry) as fake_retry:
            task.apply((1, 'a'))
        assert task.iterations == 6
        retry_call_countdowns = [call_[1]['countdown'] for call_ in fake_retry.call_args_list]
        assert retry_call_countdowns == [0, 1, 3, 7, 15, 31]

    def test_retry_wrong_eta_when_not_enable_utc(self):
        if False:
            while True:
                i = 10
        'Issue #3753'
        self.app.conf.enable_utc = False
        self.app.conf.timezone = 'US/Eastern'
        self.autoretry_task.iterations = 0
        self.autoretry_task.default_retry_delay = 2
        self.autoretry_task.apply((1, 0))
        assert self.autoretry_task.iterations == 6

    @pytest.mark.parametrize('backoff_value, expected_countdowns', [(False, [None, None, None]), (0, [None, None, None]), (0.0, [None, None, None]), (True, [1, 2, 4]), (-1, [1, 2, 4]), (0.1, [1, 2, 4]), (1, [1, 2, 4]), (1.9, [1, 2, 4]), (2, [2, 4, 8])])
    def test_autoretry_class_based_task(self, backoff_value, expected_countdowns):
        if False:
            while True:
                i = 10

        class ClassBasedAutoRetryTask(Task):
            name = 'ClassBasedAutoRetryTask'
            autoretry_for = (ZeroDivisionError,)
            retry_kwargs = {'max_retries': 2}
            retry_backoff = backoff_value
            retry_backoff_max = 700
            retry_jitter = False
            iterations = 0
            _app = self.app

            def run(self, x, y):
                if False:
                    return 10
                self.iterations += 1
                return x / y
        task = ClassBasedAutoRetryTask()
        self.app.tasks.register(task)
        task.iterations = 0
        with patch.object(task, 'retry', wraps=task.retry) as fake_retry:
            task.apply((1, 0))
        assert task.iterations == 3
        retry_call_countdowns = [call_[1].get('countdown') for call_ in fake_retry.call_args_list]
        assert retry_call_countdowns == expected_countdowns

class test_canvas_utils(TasksCase):

    def test_si(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.retry_task.si()
        assert self.retry_task.si().immutable

    def test_chunks(self):
        if False:
            print('Hello World!')
        assert self.retry_task.chunks(range(100), 10)

    def test_map(self):
        if False:
            print('Hello World!')
        assert self.retry_task.map(range(100))

    def test_starmap(self):
        if False:
            i = 10
            return i + 15
        assert self.retry_task.starmap(range(100))

    def test_on_success(self):
        if False:
            i = 10
            return i + 15
        self.retry_task.on_success(1, 1, (), {})

class test_tasks(TasksCase):

    def now(self):
        if False:
            while True:
                i = 10
        return self.app.now()

    def test_typing(self):
        if False:
            while True:
                i = 10

        @self.app.task()
        def add(x, y, kw=1):
            if False:
                for i in range(10):
                    print('nop')
            pass
        with pytest.raises(TypeError):
            add.delay(1)
        with pytest.raises(TypeError):
            add.delay(1, kw=2)
        with pytest.raises(TypeError):
            add.delay(1, 2, foobar=3)
        add.delay(2, 2)

    def test_shadow_name(self):
        if False:
            for i in range(10):
                print('nop')

        def shadow_name(task, args, kwargs, options):
            if False:
                for i in range(10):
                    print('nop')
            return 'fooxyz'

        @self.app.task(shadow_name=shadow_name)
        def shadowed():
            if False:
                for i in range(10):
                    print('nop')
            pass
        old_send_task = self.app.send_task
        self.app.send_task = Mock()
        shadowed.delay()
        self.app.send_task.assert_called_once_with(ANY, ANY, ANY, compression=ANY, delivery_mode=ANY, exchange=ANY, expires=ANY, immediate=ANY, link=ANY, link_error=ANY, mandatory=ANY, priority=ANY, producer=ANY, queue=ANY, result_cls=ANY, routing_key=ANY, serializer=ANY, soft_time_limit=ANY, task_id=ANY, task_type=ANY, time_limit=ANY, shadow='fooxyz', ignore_result=False)
        self.app.send_task = old_send_task

    def test_inherit_parent_priority_child_task(self):
        if False:
            print('Hello World!')
        self.app.conf.task_inherit_parent_priority = True
        self.app.producer_or_acquire = Mock()
        self.app.producer_or_acquire.attach_mock(ContextMock(serializer='json'), 'return_value')
        self.app.amqp.send_task_message = Mock(name='send_task_message')
        self.task_which_calls_other_task.apply(args=[])
        self.app.amqp.send_task_message.assert_called_with(ANY, 't.unit.tasks.test_tasks.task_called_by_other_task', ANY, priority=5, queue=ANY, serializer=ANY)

    def test_typing__disabled(self):
        if False:
            while True:
                i = 10

        @self.app.task(typing=False)
        def add(x, y, kw=1):
            if False:
                i = 10
                return i + 15
            pass
        add.delay(1)
        add.delay(1, kw=2)
        add.delay(1, 2, foobar=3)

    def test_typing__disabled_by_app(self):
        if False:
            print('Hello World!')
        with self.Celery(set_as_current=False, strict_typing=False) as app:

            @app.task()
            def add(x, y, kw=1):
                if False:
                    print('Hello World!')
                pass
            assert not add.typing
            add.delay(1)
            add.delay(1, kw=2)
            add.delay(1, 2, foobar=3)

    @pytest.mark.usefixtures('depends_on_current_app')
    def test_unpickle_task(self):
        if False:
            while True:
                i = 10
        import pickle

        @self.app.task(shared=True)
        def xxx():
            if False:
                print('Hello World!')
            pass
        assert pickle.loads(pickle.dumps(xxx)) is xxx.app.tasks[xxx.name]

    @patch('celery.app.task.current_app')
    @pytest.mark.usefixtures('depends_on_current_app')
    def test_bind__no_app(self, current_app):
        if False:
            for i in range(10):
                print('nop')

        class XTask(Task):
            _app = None
        XTask._app = None
        XTask.__bound__ = False
        XTask.bind = Mock(name='bind')
        assert XTask.app is current_app
        XTask.bind.assert_called_with(current_app)

    def test_reprtask__no_fmt(self):
        if False:
            return 10
        assert _reprtask(self.mytask)

    def test_AsyncResult(self):
        if False:
            for i in range(10):
                print('nop')
        task_id = uuid()
        result = self.retry_task.AsyncResult(task_id)
        assert result.backend == self.retry_task.backend
        assert result.id == task_id

    def assert_next_task_data_equal(self, consumer, presult, task_name, test_eta=False, test_expires=False, properties=None, headers=None, **kwargs):
        if False:
            print('Hello World!')
        next_task = consumer.queues[0].get(accept=['pickle', 'json'])
        task_properties = next_task.properties
        task_headers = next_task.headers
        task_body = next_task.decode()
        (task_args, task_kwargs, embed) = task_body
        assert task_headers['id'] == presult.id
        assert task_headers['task'] == task_name
        if test_eta:
            assert isinstance(task_headers.get('eta'), str)
            to_datetime = datetime.fromisoformat(task_headers.get('eta'))
            assert isinstance(to_datetime, datetime)
        if test_expires:
            assert isinstance(task_headers.get('expires'), str)
            to_datetime = datetime.fromisoformat(task_headers.get('expires'))
            assert isinstance(to_datetime, datetime)
        properties = properties or {}
        for (arg_name, arg_value) in properties.items():
            assert task_properties.get(arg_name) == arg_value
        headers = headers or {}
        for (arg_name, arg_value) in headers.items():
            assert task_headers.get(arg_name) == arg_value
        for (arg_name, arg_value) in kwargs.items():
            assert task_kwargs.get(arg_name) == arg_value

    def test_incomplete_task_cls(self):
        if False:
            return 10

        class IncompleteTask(Task):
            app = self.app
            name = 'c.unittest.t.itask'
        with pytest.raises(NotImplementedError):
            IncompleteTask().run()

    def test_task_kwargs_must_be_dictionary(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(TypeError):
            self.increment_counter.apply_async([], 'str')

    def test_task_args_must_be_list(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(TypeError):
            self.increment_counter.apply_async('s', {})

    def test_regular_task(self):
        if False:
            while True:
                i = 10
        assert isinstance(self.mytask, Task)
        assert self.mytask.run()
        assert callable(self.mytask)
        assert self.mytask(), 'Task class runs run() when called'
        with self.app.connection_or_acquire() as conn:
            consumer = self.app.amqp.TaskConsumer(conn)
            with pytest.raises(NotImplementedError):
                consumer.receive('foo', 'foo')
            consumer.purge()
            assert consumer.queues[0].get() is None
            self.app.amqp.TaskConsumer(conn, queues=[Queue('foo')])
            presult = self.mytask.delay()
            self.assert_next_task_data_equal(consumer, presult, self.mytask.name)
            presult2 = self.mytask.apply_async(kwargs={'name': 'George Costanza'})
            self.assert_next_task_data_equal(consumer, presult2, self.mytask.name, name='George Costanza')
            sresult = self.app.send_task(self.mytask.name, kwargs={'name': 'Elaine M. Benes'})
            self.assert_next_task_data_equal(consumer, sresult, self.mytask.name, name='Elaine M. Benes')
            presult2 = self.mytask.apply_async(kwargs={'name': 'George Costanza'}, eta=self.now() + timedelta(days=1), expires=self.now() + timedelta(days=2))
            self.assert_next_task_data_equal(consumer, presult2, self.mytask.name, name='George Costanza', test_eta=True, test_expires=True)
            presult2 = self.mytask.apply_async(kwargs={'name': 'George Constanza'}, eta=self.now() + timedelta(days=1), expires=(self.now() + timedelta(hours=2)).replace(tzinfo=None))
            self.assert_next_task_data_equal(consumer, presult2, self.mytask.name, name='George Constanza', test_eta=True, test_expires=True)
            presult2 = self.mytask.apply_async(kwargs={'name': 'George Costanza'}, eta=self.now() + timedelta(days=1), expires=self.now() - timedelta(days=2))
            self.assert_next_task_data_equal(consumer, presult2, self.mytask.name, name='George Costanza', test_eta=True, test_expires=True)
            presult2 = self.mytask.apply_async(kwargs={'name': 'George Costanza'}, eta=self.now() + timedelta(days=1), expires=2 * 24 * 60 * 60)
            self.assert_next_task_data_equal(consumer, presult2, self.mytask.name, name='George Costanza', test_eta=True, test_expires=True)
            presult2 = self.mytask.apply_async(kwargs={'name': 'George Costanza'}, countdown=10, expires=12)
            self.assert_next_task_data_equal(consumer, presult2, self.mytask.name, name='George Costanza', test_eta=True, test_expires=True)
            presult2 = self.mytask.apply_async(kwargs={'name': 'George Costanza'}, eta=self.now() + timedelta(days=1), expires=self.now() - timedelta(days=2))
            self.assert_next_task_data_equal(consumer, presult2, self.mytask.name, name='George Costanza', test_eta=True, test_expires=True)
            presult2 = self.mytask.apply_async(args=('spam',), kwargs={'name': 'Jerry Seinfeld'})
            self.assert_next_task_data_equal(consumer, presult2, self.mytask.name, headers={'argsrepr': "('spam',)", 'kwargsrepr': "{'name': 'Jerry Seinfeld'}"})
            presult2 = self.mytask.apply_async(args=('secret',), argsrepr="'***'", kwargs={'password': 'foo'}, kwargsrepr="{'password': '***'}")
            self.assert_next_task_data_equal(consumer, presult2, self.mytask.name, headers={'argsrepr': "'***'", 'kwargsrepr': "{'password': '***'}"})
            consumer.purge()
            self.mytask.apply_async()
            assert consumer.purge() == 1
            assert consumer.queues[0].get() is None
            assert not presult.successful()
            self.mytask.backend.mark_as_done(presult.id, result=None)
            assert presult.successful()

    def test_send_event(self):
        if False:
            for i in range(10):
                print('nop')
        mytask = self.mytask._get_current_object()
        mytask.app.events = Mock(name='events')
        mytask.app.events.attach_mock(ContextMock(), 'default_dispatcher')
        mytask.request.id = 'fb'
        mytask.send_event('task-foo', id=3122)
        mytask.app.events.default_dispatcher().send.assert_called_with('task-foo', uuid='fb', id=3122, retry=True, retry_policy=self.app.conf.task_publish_retry_policy)

    @pytest.mark.usefixtures('depends_on_current_app')
    def test_on_replace(self):
        if False:
            return 10

        class CustomStampingVisitor(StampingVisitor):

            def on_signature(self, sig, **headers) -> dict:
                if False:
                    i = 10
                    return i + 15
                return {'header': 'value'}

        class MyTask(Task):

            def on_replace(self, sig):
                if False:
                    i = 10
                    return i + 15
                sig.stamp(CustomStampingVisitor())
                return super().on_replace(sig)
        mytask = self.app.task(shared=False, base=MyTask)(return_True)
        sig1 = signature('sig1')
        with pytest.raises(Ignore):
            mytask.replace(sig1)
        assert sig1.options['header'] == 'value'

    def test_replace(self):
        if False:
            for i in range(10):
                print('nop')
        sig1 = MagicMock(name='sig1')
        sig1.options = {}
        self.mytask.request.id = sentinel.request_id
        with pytest.raises(Ignore):
            self.mytask.replace(sig1)
        sig1.freeze.assert_called_once_with(self.mytask.request.id)
        sig1.set.assert_called_once_with(replaced_task_nesting=1, chord=ANY, group_id=ANY, group_index=ANY, root_id=ANY)

    def test_replace_with_chord(self):
        if False:
            while True:
                i = 10
        sig1 = Mock(name='sig1')
        sig1.options = {'chord': None}
        with pytest.raises(ImproperlyConfigured):
            self.mytask.replace(sig1)

    def test_replace_callback(self):
        if False:
            i = 10
            return i + 15
        c = group([self.mytask.s()], app=self.app)
        c.freeze = Mock(name='freeze')
        c.delay = Mock(name='delay')
        self.mytask.request.id = 'id'
        self.mytask.request.group = 'group'
        self.mytask.request.root_id = 'root_id'
        self.mytask.request.callbacks = callbacks = 'callbacks'
        self.mytask.request.errbacks = errbacks = 'errbacks'
        with patch('celery.canvas.chord.link') as mock_chord_link, patch('celery.canvas.chord.link_error') as mock_chord_link_error:
            with pytest.raises(Ignore):
                self.mytask.replace(c)
        mock_chord_link.assert_called_once_with(callbacks)
        mock_chord_link_error.assert_called_once_with(errbacks)

    def test_replace_group(self):
        if False:
            i = 10
            return i + 15
        c = group([self.mytask.s()], app=self.app)
        c.freeze = Mock(name='freeze')
        c.delay = Mock(name='delay')
        self.mytask.request.id = 'id'
        self.mytask.request.group = 'group'
        self.mytask.request.root_id = ('root_id',)
        with pytest.raises(Ignore):
            self.mytask.replace(c)

    def test_replace_chain(self):
        if False:
            return 10
        c = chain([self.mytask.si(), self.mytask.si()], app=self.app)
        c.freeze = Mock(name='freeze')
        c.delay = Mock(name='delay')
        self.mytask.request.id = 'id'
        self.mytask.request.chain = c
        with pytest.raises(Ignore):
            self.mytask.replace(c)

    def test_replace_run(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(Ignore):
            self.task_replaced_by_other_task.run()

    def test_replace_run_with_autoretry(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(Ignore):
            self.task_replaced_by_other_task_with_autoretry.run()

    def test_replace_delay(self):
        if False:
            for i in range(10):
                print('nop')
        res = self.task_replaced_by_other_task.delay()
        assert isinstance(res, AsyncResult)

    def test_replace_apply(self):
        if False:
            print('Hello World!')
        res = self.task_replaced_by_other_task.apply()
        assert isinstance(res, EagerResult)
        assert res.get() == 'replaced'

    def test_add_trail__no_trail(self):
        if False:
            while True:
                i = 10
        mytask = self.increment_counter._get_current_object()
        mytask.trail = False
        mytask.add_trail('foo')

    def test_repr_v2_compat(self):
        if False:
            for i in range(10):
                print('nop')
        self.mytask.__v2_compat__ = True
        assert 'v2 compatible' in repr(self.mytask)

    def test_context_get(self):
        if False:
            i = 10
            return i + 15
        self.mytask.push_request()
        try:
            request = self.mytask.request
            request.foo = 32
            assert request.get('foo') == 32
            assert request.get('bar', 36) == 36
            request.clear()
        finally:
            self.mytask.pop_request()

    def test_annotate(self):
        if False:
            for i in range(10):
                print('nop')
        with patch('celery.app.task.resolve_all_annotations') as anno:
            anno.return_value = [{'FOO': 'BAR'}]

            @self.app.task(shared=False)
            def task():
                if False:
                    i = 10
                    return i + 15
                pass
            task.annotate()
            assert task.FOO == 'BAR'

    def test_after_return(self):
        if False:
            return 10
        self.mytask.push_request()
        try:
            self.mytask.request.chord = self.mytask.s()
            self.mytask.after_return('SUCCESS', 1.0, 'foobar', (), {}, None)
            self.mytask.request.clear()
        finally:
            self.mytask.pop_request()

    def test_update_state(self):
        if False:
            i = 10
            return i + 15

        @self.app.task(shared=False)
        def yyy():
            if False:
                for i in range(10):
                    print('nop')
            pass
        yyy.push_request()
        try:
            tid = uuid()
            yyy.update_state(tid, 'FROBULATING', {'fooz': 'baaz'}, arbitrary_kwarg=None)
            assert yyy.AsyncResult(tid).status == 'FROBULATING'
            assert yyy.AsyncResult(tid).result == {'fooz': 'baaz'}
            yyy.request.id = tid
            yyy.update_state(state='FROBUZATING', meta={'fooz': 'baaz'})
            assert yyy.AsyncResult(tid).status == 'FROBUZATING'
            assert yyy.AsyncResult(tid).result == {'fooz': 'baaz'}
        finally:
            yyy.pop_request()

    def test_update_state_passes_request_to_backend(self):
        if False:
            return 10
        backend = Mock()

        @self.app.task(shared=False, backend=backend)
        def ttt():
            if False:
                return 10
            pass
        ttt.push_request()
        tid = uuid()
        ttt.update_state(tid, 'SHRIMMING', {'foo': 'bar'})
        backend.store_result.assert_called_once_with(tid, {'foo': 'bar'}, 'SHRIMMING', request=ttt.request)

    def test_repr(self):
        if False:
            i = 10
            return i + 15

        @self.app.task(shared=False)
        def task_test_repr():
            if False:
                while True:
                    i = 10
            pass
        assert 'task_test_repr' in repr(task_test_repr)

    def test_has___name__(self):
        if False:
            for i in range(10):
                print('nop')

        @self.app.task(shared=False)
        def yyy2():
            if False:
                while True:
                    i = 10
            pass
        assert yyy2.__name__

    def test_default_priority(self):
        if False:
            for i in range(10):
                print('nop')

        @self.app.task(shared=False)
        def yyy3():
            if False:
                print('Hello World!')
            pass

        @self.app.task(shared=False, priority=66)
        def yyy4():
            if False:
                return 10
            pass

        @self.app.task(shared=False, bind=True, base=TaskWithPriority)
        def yyy5(self):
            if False:
                return 10
            pass
        self.app.conf.task_default_priority = 42
        old_send_task = self.app.send_task
        self.app.send_task = Mock()
        yyy3.delay()
        self.app.send_task.assert_called_once_with(ANY, ANY, ANY, compression=ANY, delivery_mode=ANY, exchange=ANY, expires=ANY, immediate=ANY, link=ANY, link_error=ANY, mandatory=ANY, priority=42, producer=ANY, queue=ANY, result_cls=ANY, routing_key=ANY, serializer=ANY, soft_time_limit=ANY, task_id=ANY, task_type=ANY, time_limit=ANY, shadow=None, ignore_result=False)
        self.app.send_task = Mock()
        yyy4.delay()
        self.app.send_task.assert_called_once_with(ANY, ANY, ANY, compression=ANY, delivery_mode=ANY, exchange=ANY, expires=ANY, immediate=ANY, link=ANY, link_error=ANY, mandatory=ANY, priority=66, producer=ANY, queue=ANY, result_cls=ANY, routing_key=ANY, serializer=ANY, soft_time_limit=ANY, task_id=ANY, task_type=ANY, time_limit=ANY, shadow=None, ignore_result=False)
        self.app.send_task = Mock()
        yyy5.delay()
        self.app.send_task.assert_called_once_with(ANY, ANY, ANY, compression=ANY, delivery_mode=ANY, exchange=ANY, expires=ANY, immediate=ANY, link=ANY, link_error=ANY, mandatory=ANY, priority=10, producer=ANY, queue=ANY, result_cls=ANY, routing_key=ANY, serializer=ANY, soft_time_limit=ANY, task_id=ANY, task_type=ANY, time_limit=ANY, shadow=None, ignore_result=False)
        self.app.send_task = old_send_task

class test_apply_task(TasksCase):

    def test_apply_throw(self):
        if False:
            while True:
                i = 10
        with pytest.raises(KeyError):
            self.raising.apply(throw=True)

    def test_apply_with_task_eager_propagates(self):
        if False:
            i = 10
            return i + 15
        self.app.conf.task_eager_propagates = True
        with pytest.raises(KeyError):
            self.raising.apply()

    def test_apply_request_context_is_ok(self):
        if False:
            i = 10
            return i + 15
        self.app.conf.task_eager_propagates = True
        self.task_check_request_context.apply()

    def test_apply(self):
        if False:
            i = 10
            return i + 15
        self.increment_counter.count = 0
        e = self.increment_counter.apply()
        assert isinstance(e, EagerResult)
        assert e.get() == 1
        e = self.increment_counter.apply(args=[1])
        assert e.get() == 2
        e = self.increment_counter.apply(kwargs={'increment_by': 4})
        assert e.get() == 6
        assert e.successful()
        assert e.ready()
        assert e.name == 't.unit.tasks.test_tasks.increment_counter'
        assert repr(e).startswith('<EagerResult:')
        f = self.raising.apply()
        assert f.ready()
        assert not f.successful()
        assert f.traceback
        with pytest.raises(KeyError):
            f.get()

    def test_apply_eager_populates_request_task(self):
        if False:
            print('Hello World!')
        task_to_apply = self.task_check_request_context
        with patch.object(task_to_apply.request_stack, 'push', wraps=task_to_apply.request_stack.push) as mock_push:
            task_to_apply.apply()
        mock_push.assert_called_once()
        request = mock_push.call_args[0][0]
        assert request.is_eager is True
        assert request.task == 't.unit.tasks.test_tasks.task_check_request_context'

    def test_apply_simulates_delivery_info(self):
        if False:
            for i in range(10):
                print('nop')
        task_to_apply = self.task_check_request_context
        with patch.object(task_to_apply.request_stack, 'push', wraps=task_to_apply.request_stack.push) as mock_push:
            task_to_apply.apply(priority=4, routing_key='myroutingkey', exchange='myexchange')
        mock_push.assert_called_once()
        request = mock_push.call_args[0][0]
        assert request.delivery_info == {'is_eager': True, 'exchange': 'myexchange', 'routing_key': 'myroutingkey', 'priority': 4}

class test_apply_async(TasksCase):

    def common_send_task_arguments(self):
        if False:
            print('Hello World!')
        return ((ANY, ANY, ANY), dict(compression=ANY, delivery_mode=ANY, exchange=ANY, expires=ANY, immediate=ANY, link=ANY, link_error=ANY, mandatory=ANY, priority=ANY, producer=ANY, queue=ANY, result_cls=ANY, routing_key=ANY, serializer=ANY, soft_time_limit=ANY, task_id=ANY, task_type=ANY, time_limit=ANY, shadow=None, ignore_result=False))

    def test_eager_serialization_failure(self):
        if False:
            i = 10
            return i + 15

        @self.app.task
        def task(*args, **kwargs):
            if False:
                print('Hello World!')
            pass
        with pytest.raises(EncodeError):
            task.apply_async((1, 2, 3, 4, {1}))

    def test_eager_serialization_uses_task_serializer_setting(self):
        if False:
            print('Hello World!')

        @self.app.task
        def task(*args, **kwargs):
            if False:
                print('Hello World!')
            pass
        with pytest.raises(EncodeError):
            task.apply_async((1, 2, 3, 4, {1}))
        self.app.conf.task_serializer = 'pickle'

        @self.app.task
        def task2(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            pass
        task2.apply_async((1, 2, 3, 4, {1}))

    def test_always_eager_with_task_serializer_option(self):
        if False:
            print('Hello World!')
        self.app.conf.task_always_eager = True
        self.app.conf.task_serializer = 'pickle'

        @self.app.task
        def task(*args, **kwargs):
            if False:
                return 10
            pass
        task.apply_async((1, 2, 3, 4, {1}))

    def test_always_eager_uses_task_serializer_setting(self):
        if False:
            for i in range(10):
                print('nop')
        self.app.conf.task_always_eager = True

        @self.app.task(serializer='pickle')
        def task(*args, **kwargs):
            if False:
                return 10
            pass
        task.apply_async((1, 2, 3, 4, {1}))

    def test_task_with_ignored_result(self):
        if False:
            for i in range(10):
                print('nop')
        with patch.object(self.app, 'send_task') as send_task:
            self.task_with_ignored_result.apply_async()
            (expected_args, expected_kwargs) = self.common_send_task_arguments()
            expected_kwargs['ignore_result'] = True
            send_task.assert_called_once_with(*expected_args, **expected_kwargs)

    def test_task_with_result(self):
        if False:
            while True:
                i = 10
        with patch.object(self.app, 'send_task') as send_task:
            self.mytask.apply_async()
            (expected_args, expected_kwargs) = self.common_send_task_arguments()
            send_task.assert_called_once_with(*expected_args, **expected_kwargs)

    def test_task_with_result_ignoring_on_call(self):
        if False:
            print('Hello World!')
        with patch.object(self.app, 'send_task') as send_task:
            self.mytask.apply_async(ignore_result=True)
            (expected_args, expected_kwargs) = self.common_send_task_arguments()
            expected_kwargs['ignore_result'] = True
            send_task.assert_called_once_with(*expected_args, **expected_kwargs)