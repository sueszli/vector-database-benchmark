import platform
import sys
import threading
if platform.system() != 'Windows':
    import resource

class AlternativeRecursionLimit:
    """A reentrant context manager for setting global recursion limits."""

    def __init__(self, new_py_limit):
        if False:
            while True:
                i = 10
        self.new_py_limit = new_py_limit
        self.count = 0
        self.lock = threading.Lock()
        self.orig_py_limit = 0
        self.orig_rlim_stack_soft = 0
        self.orig_rlim_stack_hard = 0

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        with self.lock:
            if self.count == 0:
                self.orig_py_limit = sys.getrecursionlimit()
            if platform.system() != 'Windows':
                (self.orig_rlim_stack_soft, self.orig_rlim_stack_hard) = resource.getrlimit(resource.RLIMIT_STACK)
                try:
                    resource.setrlimit(resource.RLIMIT_STACK, (self.orig_rlim_stack_hard, self.orig_rlim_stack_hard))
                except ValueError as exc:
                    if platform.system() != 'Darwin':
                        raise exc
            sys.setrecursionlimit(self.new_py_limit)
            self.count += 1

    def __exit__(self, type, value, traceback):
        if False:
            return 10
        with self.lock:
            self.count -= 1
            if self.count == 0:
                sys.setrecursionlimit(self.orig_py_limit)
            if platform.system() != 'Windows':
                try:
                    resource.setrlimit(resource.RLIMIT_STACK, (self.orig_rlim_stack_soft, self.orig_rlim_stack_hard))
                except ValueError as exc:
                    if platform.system() != 'Darwin':
                        raise exc
_max_recursion_limit_context_manager = AlternativeRecursionLimit(2 ** 31 - 1)

def max_recursion_limit():
    if False:
        print('Hello World!')
    'Sets recursion limit to the max possible value.'
    return _max_recursion_limit_context_manager