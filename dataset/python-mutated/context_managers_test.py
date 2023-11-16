"""Tests for context_managers module."""
from tensorflow.python.autograph.utils import context_managers
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.platform import test

class ContextManagersTest(test.TestCase):

    def test_control_dependency_on_returns(self):
        if False:
            return 10
        with context_managers.control_dependency_on_returns(None):
            pass
        with context_managers.control_dependency_on_returns(constant_op.constant(1)):
            pass
        with context_managers.control_dependency_on_returns(tensor_array_ops.TensorArray(dtypes.int32, size=1)):
            pass
        with context_managers.control_dependency_on_returns([constant_op.constant(1), constant_op.constant(2)]):
            pass
if __name__ == '__main__':
    test.main()