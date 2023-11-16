"""Memory leak detection utility."""
from tensorflow.python.framework.python_memory_checker import _PythonMemoryChecker
from tensorflow.python.profiler import trace
from tensorflow.python.util import tf_inspect
try:
    from tensorflow.python.platform.cpp_memory_checker import _CppMemoryChecker as CppMemoryChecker
except ImportError:
    CppMemoryChecker = None

def _get_test_name_best_effort():
    if False:
        i = 10
        return i + 15
    'If available, return the current test name. Otherwise, `None`.'
    for stack in tf_inspect.stack():
        function_name = stack[3]
        if function_name.startswith('test'):
            try:
                class_name = stack[0].f_locals['self'].__class__.__name__
                return class_name + '.' + function_name
            except:
                pass
    return None

class MemoryChecker(object):
    """Memory leak detection class.

  This is a utility class to detect Python and C++ memory leaks. It's intended
  for both testing and debugging. Basic usage:

  >>> # MemoryChecker() context manager tracks memory status inside its scope.
  >>> with MemoryChecker() as memory_checker:
  >>>   tensors = []
  >>>   for _ in range(10):
  >>>     # Simulating `tf.constant(1)` object leak every iteration.
  >>>     tensors.append(tf.constant(1))
  >>>
  >>>     # Take a memory snapshot for later analysis.
  >>>     memory_checker.record_snapshot()
  >>>
  >>> # `report()` generates a html graph file showing allocations over
  >>> # snapshots per every stack trace.
  >>> memory_checker.report()
  >>>
  >>> # This assertion will detect `tf.constant(1)` object leak.
  >>> memory_checker.assert_no_leak_if_all_possibly_except_one()

  `record_snapshot()` must be called once every iteration at the same location.
  This is because the detection algorithm relies on the assumption that if there
  is a leak, it's happening similarly on every snapshot.
  """

    @trace.trace_wrapper
    def __enter__(self):
        if False:
            return 10
        self._python_memory_checker = _PythonMemoryChecker()
        if CppMemoryChecker:
            self._cpp_memory_checker = CppMemoryChecker(_get_test_name_best_effort())
        return self

    @trace.trace_wrapper
    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            print('Hello World!')
        if CppMemoryChecker:
            self._cpp_memory_checker.stop()

    def record_snapshot(self):
        if False:
            for i in range(10):
                print('nop')
        "Take a memory snapshot for later analysis.\n\n    `record_snapshot()` must be called once every iteration at the same\n    location. This is because the detection algorithm relies on the assumption\n    that if there is a leak, it's happening similarly on every snapshot.\n\n    The recommended number of `record_snapshot()` call depends on the testing\n    code complexity and the allcoation pattern.\n    "
        self._python_memory_checker.record_snapshot()
        if CppMemoryChecker:
            self._cpp_memory_checker.record_snapshot()

    @trace.trace_wrapper
    def report(self):
        if False:
            print('Hello World!')
        'Generates a html graph file showing allocations over snapshots.\n\n    It create a temporary directory and put all the output files there.\n    If this is running under Google internal testing infra, it will use the\n    directory provided the infra instead.\n    '
        self._python_memory_checker.report()
        if CppMemoryChecker:
            self._cpp_memory_checker.report()

    @trace.trace_wrapper
    def assert_no_leak_if_all_possibly_except_one(self):
        if False:
            while True:
                i = 10
        "Raises an exception if a leak is detected.\n\n    This algorithm classifies a series of allocations as a leak if it's the same\n    type(Python) or it happens at the same stack trace(C++) at every snapshot,\n    but possibly except one snapshot.\n    "
        self._python_memory_checker.assert_no_leak_if_all_possibly_except_one()
        if CppMemoryChecker:
            self._cpp_memory_checker.assert_no_leak_if_all_possibly_except_one()

    @trace.trace_wrapper
    def assert_no_new_python_objects(self, threshold=None):
        if False:
            print('Hello World!')
        "Raises an exception if there are new Python objects created.\n\n    It computes the number of new Python objects per type using the first and\n    the last snapshots.\n\n    Args:\n      threshold: A dictionary of [Type name string], [count] pair. It won't\n        raise an exception if the new Python objects are under this threshold.\n    "
        self._python_memory_checker.assert_no_new_objects(threshold=threshold)