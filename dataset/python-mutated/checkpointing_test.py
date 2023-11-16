import os
from absl.testing import parameterized
from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.eager import test
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables as variables_lib

class TrainingCheckpointTests(test.TestCase, parameterized.TestCase):

    @combinations.generate(combinations.combine(distribution=[strategy_combinations.mirrored_strategy_with_one_cpu, strategy_combinations.mirrored_strategy_with_gpu_and_cpu, strategy_combinations.tpu_strategy, strategy_combinations.tpu_strategy_packed_var, strategy_combinations.central_storage_strategy_with_two_gpus], mode=['eager']))
    def testInitializeFromCheckpoint(self, distribution):
        if False:
            while True:
                i = 10
        variable_shape = [5]
        save_checkpoint = trackable_utils.Checkpoint(v=variables_lib.Variable(array_ops.ones(variable_shape)))
        save_path = save_checkpoint.save(os.path.join(self.get_temp_dir(), 'checkpoint'))
        with distribution.scope():
            restore_checkpoint = trackable_utils.Checkpoint()
            restore_checkpoint.restore(save_path)
            initial_value = restore_checkpoint._preload_simple_restoration('v')
            v = variables_lib.Variable(initial_value)
            self.assertGreater(v._update_uid, 0)
            self.assertAllClose(array_ops.ones(variable_shape), v)
            v.assign(array_ops.zeros(variable_shape))
            restore_checkpoint.v = v
            self.assertAllClose(array_ops.zeros(variable_shape), v)
if __name__ == '__main__':
    test.main()