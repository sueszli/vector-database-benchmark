"""Tests for debugger functionalities in tf.Session."""
import os
import tempfile
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest

def _grappler_enabled_session_config():
    if False:
        while True:
            i = 10
    'Constructs a Session config proto that explicitly enables Grappler.\n\n  Returns:\n    A config proto that obtains extra safety for the unit tests in this\n    file by ensuring that the relevant Grappler rewrites are always enabled.\n  '
    rewriter_config = rewriter_config_pb2.RewriterConfig(disable_model_pruning=False, arithmetic_optimization=rewriter_config_pb2.RewriterConfig.ON)
    graph_options = config_pb2.GraphOptions(rewrite_options=rewriter_config)
    return config_pb2.ConfigProto(graph_options=graph_options)

class SessionDebugGrapplerInteractionTest(test_util.TensorFlowTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(SessionDebugGrapplerInteractionTest, self).setUp()
        self._dump_root = tempfile.mkdtemp()
        self._debug_url = 'file://%s' % self._dump_root

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        ops.reset_default_graph()
        if os.path.isdir(self._dump_root):
            file_io.delete_recursively(self._dump_root)
        super(SessionDebugGrapplerInteractionTest, self).tearDown()

    def testArithmeticOptimizationActive(self):
        if False:
            while True:
                i = 10
        'Tests that tfdbg can dump the tensor from nodes created by Grappler.'
        with session.Session(config=_grappler_enabled_session_config()) as sess:
            u = variable_v1.VariableV1([[1, 2], [3, 4]], name='u', dtype=dtypes.float32)
            x = math_ops.add(u, u)
            x = math_ops.add(x, u)
            y = math_ops.multiply(x, u)
            sess.run(variables.global_variables_initializer())
            run_options = config_pb2.RunOptions(output_partition_graphs=True)
            debug_utils.watch_graph(run_options, sess.graph, debug_ops=['DebugIdentity'], debug_urls=[self._debug_url])
            run_metadata = config_pb2.RunMetadata()
            run_result = sess.run(y, options=run_options, run_metadata=run_metadata)
            self.assertAllClose(run_result, [[3, 12], [27, 48]])
            dump_data = debug_data.DebugDumpDir(self._dump_root, partition_graphs=run_metadata.partition_graphs, validate=True)
            original_node_names = set((op.name for op in sess.graph.get_operations()))
            dumped_node_names = set(dump_data.nodes())
            grappler_created_node_names = dumped_node_names - original_node_names
            grappler_removed_node_names = original_node_names - dumped_node_names
            self.assertTrue(grappler_created_node_names)
            self.assertTrue(grappler_removed_node_names)
            found_optimized_node = False
            for grappler_node_name in grappler_created_node_names:
                node_op_type = dump_data.node_op_type(grappler_node_name)
                if test_util.IsMklEnabled() and node_op_type in ('_MklAddN', 'Mul') or node_op_type in ('AddN', 'Mul'):
                    datum = dump_data.get_tensors(grappler_node_name, 0, 'DebugIdentity')
                    self.assertEqual(1, len(datum))
                    self.assertAllClose(datum[0], [[3, 6], [9, 12]])
                    found_optimized_node = True
                    break
            self.assertTrue(found_optimized_node, "Failed to find optimized node created by Grappler's arithmetic optimization.")
if __name__ == '__main__':
    googletest.main()