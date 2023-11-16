"""Tests for proto ops reading descriptors from other sources."""
from tensorflow.python.kernel_tests.proto import descriptor_source_test_base as test_base
from tensorflow.python.ops import proto_ops
from tensorflow.python.platform import test

class DescriptorSourceTest(test_base.DescriptorSourceTestBase):

    def __init__(self, methodName='runTest'):
        if False:
            while True:
                i = 10
        super(DescriptorSourceTest, self).__init__(decode_module=proto_ops, encode_module=proto_ops, methodName=methodName)
if __name__ == '__main__':
    test.main()