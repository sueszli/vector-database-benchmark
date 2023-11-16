"""Tests for tensorflow.python.training.saver.py."""
import os
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import saver

class SaverLargePartitionedVariableTest(test.TestCase):

    def testLargePartitionedVariables(self):
        if False:
            return 10
        save_path = os.path.join(self.get_temp_dir(), 'large_variable')
        var_name = 'my_var'
        with session.Session('', graph=ops.Graph()) as sess:
            with ops.device('/cpu:0'):
                init = lambda shape, dtype, partition_info: constant_op.constant(True, dtype, shape)
                partitioned_var = list(variable_scope.get_variable(var_name, shape=[1 << 31], partitioner=partitioned_variables.fixed_size_partitioner(4), initializer=init, dtype=dtypes.bool))
                self.evaluate(variables.global_variables_initializer())
                save = saver.Saver(partitioned_var)
                val = save.save(sess, save_path)
                self.assertEqual(save_path, val)
if __name__ == '__main__':
    test.main()