from typing import Callable
from functools import wraps
from contextlib import contextmanager
from rqalpha.const import EXECUTION_PHASE
from rqalpha.utils.i18n import gettext as _
from rqalpha.utils.exception import CustomException, patch_user_exc
from rqalpha.environment import Environment

class ContextStack(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.stack = []

    def push(self, obj):
        if False:
            while True:
                i = 10
        self.stack.append(obj)

    def pop(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self.stack.pop()
        except IndexError:
            raise RuntimeError('stack is empty')

    @contextmanager
    def pushed(self, obj):
        if False:
            i = 10
            return i + 15
        self.push(obj)
        try:
            yield self
        finally:
            self.pop()

    @property
    def top(self):
        if False:
            return 10
        try:
            return self.stack[-1]
        except IndexError:
            raise RuntimeError('stack is empty')

class ExecutionContext(object):
    stack = ContextStack()

    def __init__(self, phase):
        if False:
            print('Hello World!')
        self.phase = phase

    def _push(self):
        if False:
            return 10
        self.stack.push(self)

    def _pop(self):
        if False:
            return 10
        popped = self.stack.pop()
        if popped is not self:
            raise RuntimeError('Popped wrong context')
        return self

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        self._push()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            while True:
                i = 10
        '\n        Restore the algo instance stored in __enter__.\n        '
        if exc_type is None:
            self._pop()
            return False
        last_exc_val = exc_val
        while isinstance(exc_val, CustomException):
            last_exc_val = exc_val
            if exc_val.error.exc_val is not None:
                exc_val = exc_val.error.exc_val
            else:
                break
        if isinstance(last_exc_val, CustomException):
            raise last_exc_val
        from rqalpha.utils import create_custom_exception
        strategy_file = Environment.get_instance().config.base.strategy_file
        user_exc = create_custom_exception(exc_type, exc_val, exc_tb, strategy_file)
        raise user_exc

    @classmethod
    def enforce_phase(cls, *phases):
        if False:
            print('Hello World!')

        def decorator(func):
            if False:
                return 10

            @wraps(func)
            def wrapper(*args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                phase = cls.stack.top.phase
                if phase not in phases:
                    raise patch_user_exc(RuntimeError(_(u'You cannot call %s when executing %s') % (func.__name__, phase.value)))
                return func(*args, **kwargs)
            return wrapper
        return decorator

    @classmethod
    def phase(cls):
        if False:
            for i in range(10):
                print('nop')
        return cls.stack.top.phase