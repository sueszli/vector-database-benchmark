"""Tests for test_util."""
from absl.testing import parameterized
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test
from tensorflow.tools.proto_splitter.python import test_util

class MakeGraphDefTest(test.TestCase, parameterized.TestCase):

    @parameterized.named_parameters(('Float64', dtypes.float64), ('Float32', dtypes.float32))
    def testMakeGraphDef(self, dtype):
        if False:
            return 10
        expected_sizes = [75, 50, 100, 95, 120]
        fn1 = [121, 153, 250, 55]
        fn2 = [552, 45]
        graph_def = test_util.make_graph_def_with_constant_nodes(expected_sizes, dtype=dtype, fn1=fn1, fn2=fn2)
        self.assertAllClose(expected_sizes, [node.ByteSize() for node in graph_def.node], atol=5)
        self.assertAllClose(fn1, [node.ByteSize() for node in graph_def.library.function[0].node_def], atol=10)
        self.assertAllClose(fn2, [node.ByteSize() for node in graph_def.library.function[1].node_def], atol=10)
if __name__ == '__main__':
    test.main()