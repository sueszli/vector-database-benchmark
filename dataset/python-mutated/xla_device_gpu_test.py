"""Test cases for XLA devices."""
from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

class XlaDeviceGpuTest(test.TestCase):

    def __init__(self, method_name='runTest'):
        if False:
            for i in range(10):
                print('nop')
        super(XlaDeviceGpuTest, self).__init__(method_name)
        context.context().enable_xla_devices()

    def testCopiesToAndFromGpuWork(self):
        if False:
            while True:
                i = 10
        'Tests that copies between GPU and XLA devices work.'
        if not test.is_gpu_available():
            return
        with session_lib.Session() as sess:
            x = array_ops.placeholder(dtypes.float32, [2])
            with ops.device('GPU'):
                y = x * 2
            with ops.device('device:XLA_CPU:0'):
                z = y * y
            with ops.device('GPU'):
                w = y + z
            result = sess.run(w, {x: [1.5, 0.5]})
        self.assertAllClose(result, [12.0, 2.0], rtol=0.001)
if __name__ == '__main__':
    test.main()