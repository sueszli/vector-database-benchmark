from __future__ import print_function
from functools import wraps

def wrap_error_fatal(method):
    if False:
        print('Hello World!')
    from gevent._hub_local import get_hub_class
    system_error = get_hub_class().SYSTEM_ERROR

    @wraps(method)
    def fatal_error_wrapper(self, *args, **kwargs):
        if False:
            return 10
        get_hub_class().SYSTEM_ERROR = object
        try:
            return method(self, *args, **kwargs)
        finally:
            get_hub_class().SYSTEM_ERROR = system_error
    return fatal_error_wrapper

def wrap_restore_handle_error(method):
    if False:
        for i in range(10):
            print('nop')
    from gevent._hub_local import get_hub_if_exists
    from gevent import getcurrent

    @wraps(method)
    def restore_fatal_error_wrapper(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        try:
            return method(self, *args, **kwargs)
        finally:
            try:
                del get_hub_if_exists().handle_error
            except AttributeError:
                pass
        if self.peek_error()[0] is not None:
            getcurrent().throw(*self.peek_error()[1:])
    return restore_fatal_error_wrapper