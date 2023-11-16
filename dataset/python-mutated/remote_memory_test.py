"""Tests for memory leaks in remote eager execution."""
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.eager.memory_tests import memory_test_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.training import server_lib

class RemoteWorkerMemoryTest(test.TestCase):

    def __init__(self, method):
        if False:
            i = 10
            return i + 15
        super(RemoteWorkerMemoryTest, self).__init__(method)
        self._cached_server = server_lib.Server.create_local_server()
        self._cached_server_target = self._cached_server.target[len('grpc://'):]

    def testMemoryLeakInLocalCopy(self):
        if False:
            while True:
                i = 10
        if not memory_test_util.memory_profiler_is_available():
            self.skipTest('memory_profiler required to run this test')
        remote.connect_to_remote_host(self._cached_server_target)

        @def_function.function
        def local_func(i):
            if False:
                for i in range(10):
                    print('nop')
            return i

        def func():
            if False:
                i = 10
                return i + 15
            with ops.device('job:worker/replica:0/task:0/device:CPU:0'):
                x = array_ops.zeros([1000, 1000], dtypes.int32)
            local_func(x)
        memory_test_util.assert_no_leak(func, num_iters=100, increase_threshold_absolute_mb=50)
if __name__ == '__main__':
    test.main()