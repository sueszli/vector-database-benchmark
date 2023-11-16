"""Utils for memory tests."""
import collections
import gc
import time
from tensorflow.python.eager import context
try:
    import memory_profiler
except ImportError:
    memory_profiler = None

def _instance_count_by_class():
    if False:
        return 10
    counter = collections.Counter()
    for obj in gc.get_objects():
        try:
            counter[obj.__class__.__name__] += 1
        except Exception:
            pass
    return counter

def assert_no_leak(f, num_iters=100000, increase_threshold_absolute_mb=25):
    if False:
        i = 10
        return i + 15
    "Assert memory usage doesn't increase beyond given threshold for f."
    with context.eager_mode():
        f()
        time.sleep(4)
        gc.collect()
        initial = memory_profiler.memory_usage(-1)[0]
        instance_count_by_class_before = _instance_count_by_class()
        for _ in range(num_iters):
            f()
        gc.collect()
        increase = memory_profiler.memory_usage(-1)[0] - initial
        assert increase < increase_threshold_absolute_mb, 'Increase is too high. Initial memory usage: %f MB. Increase: %f MB. Maximum allowed increase: %f MB. Instance count diff before/after: %s' % (initial, increase, increase_threshold_absolute_mb, _instance_count_by_class() - instance_count_by_class_before)

def memory_profiler_is_available():
    if False:
        print('Hello World!')
    return memory_profiler is not None