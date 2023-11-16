import multiprocessing
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
try:
    from cinderjit import force_compile, is_jit_compiled, jit_frame_mode
    CINDERJIT_ENABLED = True
except ImportError:

    def is_jit_compiled(f):
        if False:
            i = 10
            return i + 15
        return False

    def force_compile(f):
        if False:
            print('Hello World!')
        return False
    CINDERJIT_ENABLED = False
try:
    import cinder

    def hasCinderX():
        if False:
            i = 10
            return i + 15
        return True
except ImportError:

    def hasCinderX():
        if False:
            return 10
        return False

def get_await_stack(coro):
    if False:
        print('Hello World!')
    'Return the chain of coroutines reachable from coro via its awaiter'
    stack = []
    awaiter = cinder._get_coro_awaiter(coro)
    while awaiter is not None:
        stack.append(awaiter)
        awaiter = cinder._get_coro_awaiter(awaiter)
    return stack

def verify_stack(testcase, stack, expected):
    if False:
        i = 10
        return i + 15
    n = len(expected)
    frames = stack[-n:]
    testcase.assertEqual(len(frames), n, 'Callstack had less frames than expected')
    for (actual, expected) in zip(frames, expected):
        testcase.assertTrue(actual.endswith(expected), f"The actual frame {actual} doesn't refer to the expected function {expected}")

def skipUnderJIT(reason):
    if False:
        print('Hello World!')
    if CINDERJIT_ENABLED:
        return unittest.skip(reason)
    return unittest.case._id

def skipUnlessJITEnabled(reason):
    if False:
        while True:
            i = 10
    if not CINDERJIT_ENABLED:
        return unittest.skip(reason)
    return unittest.case._id

def failUnlessJITCompiled(func):
    if False:
        return 10
    "\n    Fail a test if the JIT is enabled but the test body wasn't JIT-compiled.\n    "
    if not CINDERJIT_ENABLED:
        return func
    try:
        force_compile(func)
    except RuntimeError as re:
        if re.args == ('PYJIT_RESULT_NOT_ON_JITLIST',):
            return func
        exc = re

        def wrapper(*args):
            if False:
                for i in range(10):
                    print('nop')
            raise RuntimeError(f'JIT compilation of {func.__qualname__} failed with {exc}')
        return wrapper
    return func
SUBPROCESS_TIMEOUT_SEC = 5

@contextmanager
def temp_sys_path():
    if False:
        print('Hello World!')
    with tempfile.TemporaryDirectory() as tmpdir:
        _orig_sys_modules = sys.modules
        sys.modules = _orig_sys_modules.copy()
        _orig_sys_path = sys.path[:]
        sys.path.insert(0, tmpdir)
        try:
            yield Path(tmpdir)
        finally:
            sys.path[:] = _orig_sys_path
            sys.modules = _orig_sys_modules

def runInSubprocess(func):
    if False:
        return 10
    queue = multiprocessing.Queue()

    def wrapper(queue, *args):
        if False:
            for i in range(10):
                print('nop')
        result = func(*args)
        queue.put(result, timeout=SUBPROCESS_TIMEOUT_SEC)

    def wrapped(*args):
        if False:
            print('Hello World!')
        p = multiprocessing.Process(target=wrapper, args=(queue, *args))
        p.start()
        value = queue.get(timeout=SUBPROCESS_TIMEOUT_SEC)
        p.join(timeout=SUBPROCESS_TIMEOUT_SEC)
        return value
    return wrapped