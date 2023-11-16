from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

class TFProfLoggerTest(test.TestCase):

    def _BuildSmallPlaceholderlModel(self):
        if False:
            while True:
                i = 10
        a = array_ops.placeholder(dtypes.int32, [2, 2])
        b = array_ops.placeholder(dtypes.int32, [2, 2])
        y = math_ops.matmul(a, b)
        return (a, b, y)

    def _BuildSmallModel(self):
        if False:
            i = 10
            return i + 15
        a = constant_op.constant([[1, 2], [3, 4]])
        b = constant_op.constant([[1, 2], [3, 4]])
        return math_ops.matmul(a, b)
    "# TODO(xpan): This out of core so it doesn't depend on contrib.\n  def testFillMissingShape(self):\n    a, b, y = self._BuildSmallPlaceholderlModel()\n    run_options = config_pb2.RunOptions(\n        trace_level=config_pb2.RunOptions.FULL_TRACE)\n    run_metadata = config_pb2.RunMetadata()\n    sess = session.Session()\n    sess.run(y,\n             options=run_options,\n             run_metadata=run_metadata,\n             feed_dict={a: [[1, 2], [2, 3]],\n                        b: [[1, 2], [2, 3]]})\n\n    graph2 = ops.Graph()\n    # Use copy_op_to_graph to remove shape information.\n    y2 = copy_elements.copy_op_to_graph(y, graph2, [])\n    self.assertEqual('<unknown>', str(y2.get_shape()))\n\n    tfprof_logger._fill_missing_graph_shape(graph2, run_metadata)\n    self.assertEqual('(2, 2)', str(y2.get_shape()))\n\n  def testFailedFillMissingShape(self):\n    y = self._BuildSmallModel()\n    run_options = config_pb2.RunOptions(\n        trace_level=config_pb2.RunOptions.FULL_TRACE)\n    run_metadata = config_pb2.RunMetadata()\n    sess = session.Session()\n    sess.run(y, options=run_options, run_metadata=run_metadata)\n\n    graph2 = ops.Graph()\n    y2 = copy_elements.copy_op_to_graph(y, graph2, [])\n    self.assertEqual('<unknown>', str(y2.get_shape()))\n    # run_metadata has special name for MatMul, hence failed to fill shape.\n    tfprof_logger._fill_missing_graph_shape(graph2, run_metadata)\n    self.assertEqual('<unknown>', str(y2.get_shape()))\n  "
if __name__ == '__main__':
    test.main()