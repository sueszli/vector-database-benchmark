"""Tests enabling eager execution at process level."""
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.platform import googletest

class OpsEnableAndDisableEagerTest(googletest.TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        ops.enable_eager_execution()
        self.assertTrue(context.executing_eagerly())
        ops.enable_eager_execution()
        self.assertTrue(context.executing_eagerly())

    def testEnableDisableEagerExecution(self):
        if False:
            while True:
                i = 10
        pass

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        super().tearDown()
        ops.disable_eager_execution()
        self.assertFalse(context.executing_eagerly())
        ops.disable_eager_execution()
        self.assertFalse(context.executing_eagerly())
if __name__ == '__main__':
    googletest.main()