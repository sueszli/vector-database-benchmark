"""Tests for tensorflow.ops.array_ops.repeat."""
from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

class ImageOpsTest(xla_test.XLATestCase):

    def testGradImageResize(self):
        if False:
            return 10
        'Tests that the gradient of image.resize is compilable.'
        with ops.device('device:{}:0'.format(self.device)):
            img_width = 2048
            var = variables.Variable(array_ops.ones(1, dtype=dtypes.float32))

            def model(x):
                if False:
                    i = 10
                    return i + 15
                x = var * x
                x = image_ops.resize_images(x, size=[img_width, img_width], method=image_ops.ResizeMethod.BILINEAR)
                return x

            def train(x, y):
                if False:
                    print('Hello World!')
                with backprop.GradientTape() as tape:
                    output = model(x)
                    loss_value = math_ops.reduce_mean((y - output) ** 2)
                grads = tape.gradient(loss_value, [var])
                return grads
            compiled_train = def_function.function(train, jit_compile=True)
            x = array_ops.zeros((1, img_width // 2, img_width // 2, 1), dtype=dtypes.float32)
            y = array_ops.zeros((1, img_width, img_width, 1), dtype=dtypes.float32)
            self.assertAllClose(train(x, y), compiled_train(x, y))
if __name__ == '__main__':
    ops.enable_eager_execution()
    test.main()