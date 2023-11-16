"""Tests for decode_proto op."""
from tensorflow.python.kernel_tests.proto import decode_proto_op_test_base as test_base
from tensorflow.python.ops import proto_ops as proto_ops
from tensorflow.python.platform import test

class DecodeProtoOpTest(test_base.DecodeProtoOpTestBase):

    def __init__(self, methodName='runTest'):
        if False:
            return 10
        super(DecodeProtoOpTest, self).__init__(proto_ops, methodName)
if __name__ == '__main__':
    test.main()