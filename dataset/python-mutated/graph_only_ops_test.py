"""Tests for graph_only_ops."""
import numpy as np
from tensorflow.python.eager import graph_only_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

class GraphOnlyOpsTest(test_util.TensorFlowTestCase):

    def testGraphPlaceholder(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            x_tf = graph_only_ops.graph_placeholder(dtypes.int32, shape=(1,))
            y_tf = math_ops.square(x_tf)
            with self.cached_session() as sess:
                x = np.array([42])
                y = sess.run(y_tf, feed_dict={x_tf: np.array([42])})
                self.assertAllClose(np.square(x), y)
if __name__ == '__main__':
    test.main()