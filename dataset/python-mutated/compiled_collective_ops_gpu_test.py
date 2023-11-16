"""Tests for Collective Operations that require GPU."""
import os
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test

class CompiledCollectiveOpGPUTest(test.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        'Set group_size = num_gpus = 2 for all tests in this class.'
        super(CompiledCollectiveOpGPUTest, cls).setUpClass()
        cls._group_size = 2
        cls._devices = ['/device:GPU:{}'.format(i) for i in range(2)]
        os.environ['NCCL_DEBUG'] = 'INFO'
        os.environ['NCCL_LAUNCH_MODE'] = 'PARALLEL'

    def _setup_context(self, num_gpus=2):
        if False:
            i = 10
            return i + 15
        context._reset_context()
        gpus = config.list_physical_devices('GPU')
        if len(gpus) < num_gpus:
            self.skipTest('Expected at least {} GPUs but found {} GPUs'.format(num_gpus, len(gpus)))
        context.ensure_initialized()

    def testCompiledAllReduce(self):
        if False:
            return 10
        self._setup_context()

        def all_reduce_sum(v):
            if False:
                i = 10
                return i + 15
            return collective_ops.all_reduce_v2(t=v, group_size=2, group_key=1, instance_key=1, merge_op='Add', final_op='Id')
        strategy = mirrored_strategy.MirroredStrategy(['GPU:0', 'GPU:1'])

        @def_function.function(jit_compile=True)
        def f():
            if False:
                return 10
            return while_loop.while_loop(lambda i, _: i < 5, lambda i, t: (i + 1, all_reduce_sum(t)), (array_ops.zeros([]), constant_op.constant(1.0)))

        @def_function.function
        def run():
            if False:
                return 10
            return strategy.run(f)
        (_, reduce) = strategy.experimental_local_results(run())[0]
        self.assertEqual(reduce.numpy(), 32.0)
if __name__ == '__main__':
    ops.enable_eager_execution()
    test.main()