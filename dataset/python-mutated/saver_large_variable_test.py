"""Tests for tensorflow.python.training.saver.py."""
import os
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import saver

class SaverLargeVariableTest(test.TestCase):

    def testLargeVariable(self):
        if False:
            while True:
                i = 10
        save_path = os.path.join(self.get_temp_dir(), 'large_variable')
        with session.Session('', graph=ops.Graph()) as sess:
            with ops.device('/cpu:0'):
                var = variables.Variable(constant_op.constant(False, shape=[2, 1024, 1024, 1024], dtype=dtypes.bool))
            save = saver.Saver({var.op.name: var}, write_version=saver_pb2.SaverDef.V1)
            var.initializer.run()
            with self.assertRaisesRegex(errors_impl.InvalidArgumentError, 'Tensor slice is too large to serialize'):
                save.save(sess, save_path)
if __name__ == '__main__':
    test.main()