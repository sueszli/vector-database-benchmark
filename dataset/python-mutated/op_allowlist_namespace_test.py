"""# Test that buidling using op_allowlist works with ops with namespaces."""
from tensorflow.python.framework import test_namespace_ops
from tensorflow.python.platform import googletest

class OpAllowlistNamespaceTest(googletest.TestCase):

    def testOpAllowListNamespace(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that the building of the python wrapper worked.'
        op = test_namespace_ops.namespace_test_string_output
        self.assertIsNotNone(op)
if __name__ == '__main__':
    googletest.main()