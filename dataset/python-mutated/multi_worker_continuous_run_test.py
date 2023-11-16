"""Tests for continuous runs using cross-worker collective ops."""
import json
import os
from absl.testing import parameterized
import numpy as np
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base as test_base
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import config
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
try:
    import dill
    _REGISTER_DECORATOR = dill.register
except ImportError:
    _REGISTER_DECORATOR = lambda fn, *_: fn
CollectiveAllReduceExtended = collective_all_reduce_strategy.CollectiveAllReduceExtended
CollectiveAllReduceExtended._enable_check_health = False
NUM_WORKERS = 5

class MultiWorkerContinuousRunTest(test.TestCase, parameterized.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(MultiWorkerContinuousRunTest, self).setUp()
        self._maybe_setup_gpus(setup=True)

    def _maybe_setup_gpus(self, setup=False):
        if False:
            print('Hello World!')
        self._gpus = config.list_physical_devices('GPU')
        self._local_device = '/device:GPU:0' if self._gpus else '/device:CPU:0'
        if self._gpus and (not setup):
            config.set_logical_device_configuration(self._gpus[0], [context.LogicalDeviceConfiguration(64)])

    @combinations.generate(combinations.combine(mode=['eager']))
    def testAllReduceContinuousRun(self, mode):
        if False:
            for i in range(10):
                print('nop')
        tensor_shape = [2, 2]

        def worker_step_fn(worker_id):
            if False:
                print('Hello World!')
            strategy = collective_all_reduce_strategy.CollectiveAllReduceStrategy()
            multi_process_runner.get_barrier().wait()

            @def_function.function
            def run_reduce():
                if False:
                    print('Hello World!')
                with ops.device(self._local_device):
                    t_in = array_ops.ones(tensor_shape) * worker_id
                    return strategy.reduce(reduce_util.ReduceOp.MEAN, t_in, axis=None)
            t_out = run_reduce()
            expected_mean = (NUM_WORKERS - 1) / 2
            expected_out = np.ones(tensor_shape) * expected_mean
            self.assertAllClose(t_out, expected_out)

        def worker_fn():
            if False:
                i = 10
                return i + 15
            self._maybe_setup_gpus()
            tf_config = json.loads(os.environ['TF_CONFIG'])
            worker_id = tf_config['task']['index']
            for _ in range(20):
                worker_step_fn(worker_id)
        with test_util.skip_if_error(self, errors_impl.UnavailableError):
            multi_process_runner.run(worker_fn, cluster_spec=test_base.create_cluster_spec(num_workers=NUM_WORKERS))

    @combinations.generate(combinations.combine(mode=['eager']))
    def testVariableInitializationWithChangingShape(self, mode):
        if False:
            i = 10
            return i + 15

        def worker_step_fn(worker_id, num_dims):
            if False:
                for i in range(10):
                    print('nop')
            strategy = collective_all_reduce_strategy.CollectiveAllReduceStrategy()
            multi_process_runner.get_barrier().wait()
            tensor_shape = [2] * num_dims

            def variable_fn():
                if False:
                    print('Hello World!')
                with ops.device(self._local_device):
                    initial_value = array_ops.ones(tensor_shape) if worker_id == 0 else array_ops.zeros(tensor_shape)
                    var = variable_scope.get_variable(name='x', initializer=initial_value)
                    return array_ops.identity(var)
            t_out = strategy.extended.call_for_each_replica(variable_fn)
            expected_out = np.ones(tensor_shape)
            self.assertAllClose(t_out, expected_out)

        def worker_fn():
            if False:
                for i in range(10):
                    print('nop')
            self._maybe_setup_gpus()
            tf_config = json.loads(os.environ['TF_CONFIG'])
            worker_id = tf_config['task']['index']
            for i in range(20):
                worker_step_fn(worker_id, num_dims=i + 1)
        with test_util.skip_if_error(self, errors_impl.UnavailableError):
            multi_process_runner.run(worker_fn, cluster_spec=test_base.create_cluster_spec(num_workers=NUM_WORKERS))

@_REGISTER_DECORATOR(MultiWorkerContinuousRunTest)
def _save_test_case(pickler, obj):
    if False:
        print('Hello World!')

    def reconstruct(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        del args, kwargs
        return MultiWorkerContinuousRunTest()
    return pickler.save_reduce(reconstruct, (), obj=obj)
if __name__ == '__main__':
    multi_process_runner.test_main()