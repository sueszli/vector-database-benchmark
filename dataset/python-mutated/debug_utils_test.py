"""Tests for TensorFlow Debugger (tfdbg) Utilities."""
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import googletest

@test_util.run_v1_only('Requires tf.Session')
class DebugUtilsTest(test_util.TensorFlowTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        cls._sess = session.Session()
        with cls._sess:
            cls._a_init_val = np.array([[5.0, 3.0], [-1.0, 0.0]])
            cls._b_init_val = np.array([[2.0], [-1.0]])
            cls._c_val = np.array([[-4.0], [np.nan]])
            cls._a_init = constant_op.constant(cls._a_init_val, shape=[2, 2], name='a1_init')
            cls._b_init = constant_op.constant(cls._b_init_val, shape=[2, 1], name='b_init')
            cls._a = variable_v1.VariableV1(cls._a_init, name='a1')
            cls._b = variable_v1.VariableV1(cls._b_init, name='b')
            cls._c = constant_op.constant(cls._c_val, shape=[2, 1], name='c')
            cls._p = math_ops.matmul(cls._a, cls._b, name='p1')
            cls._s = math_ops.add(cls._p, cls._c, name='s')
        cls._graph = cls._sess.graph
        cls._expected_num_nodes = 4 * 2 + 1 + 1 + 1 + 1

    def setUp(self):
        if False:
            while True:
                i = 10
        self._run_options = config_pb2.RunOptions()

    def _verify_watches(self, watch_opts, expected_output_slot, expected_debug_ops, expected_debug_urls):
        if False:
            i = 10
            return i + 15
        'Verify a list of debug tensor watches.\n\n    This requires all watches in the watch list have exactly the same\n    output_slot, debug_ops and debug_urls.\n\n    Args:\n      watch_opts: Repeated protobuf field of DebugTensorWatch.\n      expected_output_slot: Expected output slot index, as an integer.\n      expected_debug_ops: Expected debug ops, as a list of strings.\n      expected_debug_urls: Expected debug URLs, as a list of strings.\n\n    Returns:\n      List of node names from the list of debug tensor watches.\n    '
        node_names = []
        for watch in watch_opts:
            node_names.append(watch.node_name)
            if watch.node_name == '*':
                self.assertEqual(-1, watch.output_slot)
                self.assertEqual(expected_debug_ops, watch.debug_ops)
                self.assertEqual(expected_debug_urls, watch.debug_urls)
            else:
                self.assertEqual(expected_output_slot, watch.output_slot)
                self.assertEqual(expected_debug_ops, watch.debug_ops)
                self.assertEqual(expected_debug_urls, watch.debug_urls)
        return node_names

    def testAddDebugTensorWatches_defaultDebugOp(self):
        if False:
            i = 10
            return i + 15
        debug_utils.add_debug_tensor_watch(self._run_options, 'foo/node_a', 1, debug_urls='file:///tmp/tfdbg_1')
        debug_utils.add_debug_tensor_watch(self._run_options, 'foo/node_b', 0, debug_urls='file:///tmp/tfdbg_2')
        debug_watch_opts = self._run_options.debug_options.debug_tensor_watch_opts
        self.assertEqual(2, len(debug_watch_opts))
        watch_0 = debug_watch_opts[0]
        watch_1 = debug_watch_opts[1]
        self.assertEqual('foo/node_a', watch_0.node_name)
        self.assertEqual(1, watch_0.output_slot)
        self.assertEqual('foo/node_b', watch_1.node_name)
        self.assertEqual(0, watch_1.output_slot)
        self.assertEqual(['DebugIdentity'], watch_0.debug_ops)
        self.assertEqual(['DebugIdentity'], watch_1.debug_ops)
        self.assertEqual(['file:///tmp/tfdbg_1'], watch_0.debug_urls)
        self.assertEqual(['file:///tmp/tfdbg_2'], watch_1.debug_urls)

    def testAddDebugTensorWatches_explicitDebugOp(self):
        if False:
            while True:
                i = 10
        debug_utils.add_debug_tensor_watch(self._run_options, 'foo/node_a', 0, debug_ops='DebugNanCount', debug_urls='file:///tmp/tfdbg_1')
        debug_watch_opts = self._run_options.debug_options.debug_tensor_watch_opts
        self.assertEqual(1, len(debug_watch_opts))
        watch_0 = debug_watch_opts[0]
        self.assertEqual('foo/node_a', watch_0.node_name)
        self.assertEqual(0, watch_0.output_slot)
        self.assertEqual(['DebugNanCount'], watch_0.debug_ops)
        self.assertEqual(['file:///tmp/tfdbg_1'], watch_0.debug_urls)

    def testAddDebugTensorWatches_multipleDebugOps(self):
        if False:
            print('Hello World!')
        debug_utils.add_debug_tensor_watch(self._run_options, 'foo/node_a', 0, debug_ops=['DebugNanCount', 'DebugIdentity'], debug_urls='file:///tmp/tfdbg_1')
        debug_watch_opts = self._run_options.debug_options.debug_tensor_watch_opts
        self.assertEqual(1, len(debug_watch_opts))
        watch_0 = debug_watch_opts[0]
        self.assertEqual('foo/node_a', watch_0.node_name)
        self.assertEqual(0, watch_0.output_slot)
        self.assertEqual(['DebugNanCount', 'DebugIdentity'], watch_0.debug_ops)
        self.assertEqual(['file:///tmp/tfdbg_1'], watch_0.debug_urls)

    def testAddDebugTensorWatches_multipleURLs(self):
        if False:
            for i in range(10):
                print('nop')
        debug_utils.add_debug_tensor_watch(self._run_options, 'foo/node_a', 0, debug_ops='DebugNanCount', debug_urls=['file:///tmp/tfdbg_1', 'file:///tmp/tfdbg_2'])
        debug_watch_opts = self._run_options.debug_options.debug_tensor_watch_opts
        self.assertEqual(1, len(debug_watch_opts))
        watch_0 = debug_watch_opts[0]
        self.assertEqual('foo/node_a', watch_0.node_name)
        self.assertEqual(0, watch_0.output_slot)
        self.assertEqual(['DebugNanCount'], watch_0.debug_ops)
        self.assertEqual(['file:///tmp/tfdbg_1', 'file:///tmp/tfdbg_2'], watch_0.debug_urls)

    def testWatchGraph_allNodes(self):
        if False:
            for i in range(10):
                print('nop')
        debug_utils.watch_graph(self._run_options, self._graph, debug_ops=['DebugIdentity', 'DebugNanCount'], debug_urls='file:///tmp/tfdbg_1')
        debug_watch_opts = self._run_options.debug_options.debug_tensor_watch_opts
        self.assertEqual(self._expected_num_nodes, len(debug_watch_opts))
        node_names = self._verify_watches(debug_watch_opts, 0, ['DebugIdentity', 'DebugNanCount'], ['file:///tmp/tfdbg_1'])
        self.assertIn('a1_init', node_names)
        self.assertIn('a1', node_names)
        self.assertIn('a1/Assign', node_names)
        self.assertIn('a1/read', node_names)
        self.assertIn('b_init', node_names)
        self.assertIn('b', node_names)
        self.assertIn('b/Assign', node_names)
        self.assertIn('b/read', node_names)
        self.assertIn('c', node_names)
        self.assertIn('p1', node_names)
        self.assertIn('s', node_names)
        self.assertIn('*', node_names)

    def testWatchGraph_nodeNameAllowlist(self):
        if False:
            while True:
                i = 10
        debug_utils.watch_graph(self._run_options, self._graph, debug_urls='file:///tmp/tfdbg_1', node_name_regex_allowlist='(a1$|a1_init$|a1/.*|p1$)')
        node_names = self._verify_watches(self._run_options.debug_options.debug_tensor_watch_opts, 0, ['DebugIdentity'], ['file:///tmp/tfdbg_1'])
        self.assertEqual(sorted(['a1_init', 'a1', 'a1/Assign', 'a1/read', 'p1']), sorted(node_names))

    def testWatchGraph_opTypeAllowlist(self):
        if False:
            i = 10
            return i + 15
        debug_utils.watch_graph(self._run_options, self._graph, debug_urls='file:///tmp/tfdbg_1', op_type_regex_allowlist='(Variable|MatMul)')
        node_names = self._verify_watches(self._run_options.debug_options.debug_tensor_watch_opts, 0, ['DebugIdentity'], ['file:///tmp/tfdbg_1'])
        self.assertEqual(sorted(['a1', 'b', 'p1']), sorted(node_names))

    def testWatchGraph_nodeNameAndOpTypeAllowlists(self):
        if False:
            print('Hello World!')
        debug_utils.watch_graph(self._run_options, self._graph, debug_urls='file:///tmp/tfdbg_1', node_name_regex_allowlist='([a-z]+1$)', op_type_regex_allowlist='(MatMul)')
        node_names = self._verify_watches(self._run_options.debug_options.debug_tensor_watch_opts, 0, ['DebugIdentity'], ['file:///tmp/tfdbg_1'])
        self.assertEqual(['p1'], node_names)

    def testWatchGraph_tensorDTypeAllowlist(self):
        if False:
            for i in range(10):
                print('nop')
        debug_utils.watch_graph(self._run_options, self._graph, debug_urls='file:///tmp/tfdbg_1', tensor_dtype_regex_allowlist='.*_ref')
        node_names = self._verify_watches(self._run_options.debug_options.debug_tensor_watch_opts, 0, ['DebugIdentity'], ['file:///tmp/tfdbg_1'])
        self.assertItemsEqual(['a1', 'a1/Assign', 'b', 'b/Assign'], node_names)

    def testWatchGraph_nodeNameAndTensorDTypeAllowlists(self):
        if False:
            return 10
        debug_utils.watch_graph(self._run_options, self._graph, debug_urls='file:///tmp/tfdbg_1', node_name_regex_allowlist='^a.*', tensor_dtype_regex_allowlist='.*_ref')
        node_names = self._verify_watches(self._run_options.debug_options.debug_tensor_watch_opts, 0, ['DebugIdentity'], ['file:///tmp/tfdbg_1'])
        self.assertItemsEqual(['a1', 'a1/Assign'], node_names)

    def testWatchGraph_nodeNameDenylist(self):
        if False:
            return 10
        debug_utils.watch_graph_with_denylists(self._run_options, self._graph, debug_urls='file:///tmp/tfdbg_1', node_name_regex_denylist='(a1$|a1_init$|a1/.*|p1$)')
        node_names = self._verify_watches(self._run_options.debug_options.debug_tensor_watch_opts, 0, ['DebugIdentity'], ['file:///tmp/tfdbg_1'])
        self.assertEqual(sorted(['b_init', 'b', 'b/Assign', 'b/read', 'c', 's']), sorted(node_names))

    def testWatchGraph_opTypeDenylist(self):
        if False:
            i = 10
            return i + 15
        debug_utils.watch_graph_with_denylists(self._run_options, self._graph, debug_urls='file:///tmp/tfdbg_1', op_type_regex_denylist='(Variable|Identity|Assign|Const)')
        node_names = self._verify_watches(self._run_options.debug_options.debug_tensor_watch_opts, 0, ['DebugIdentity'], ['file:///tmp/tfdbg_1'])
        self.assertEqual(sorted(['p1', 's']), sorted(node_names))

    def testWatchGraph_nodeNameAndOpTypeDenylists(self):
        if False:
            return 10
        debug_utils.watch_graph_with_denylists(self._run_options, self._graph, debug_urls='file:///tmp/tfdbg_1', node_name_regex_denylist='p1$', op_type_regex_denylist='(Variable|Identity|Assign|Const)')
        node_names = self._verify_watches(self._run_options.debug_options.debug_tensor_watch_opts, 0, ['DebugIdentity'], ['file:///tmp/tfdbg_1'])
        self.assertEqual(['s'], node_names)

    def testWatchGraph_tensorDTypeDenylists(self):
        if False:
            print('Hello World!')
        debug_utils.watch_graph_with_denylists(self._run_options, self._graph, debug_urls='file:///tmp/tfdbg_1', tensor_dtype_regex_denylist='.*_ref')
        node_names = self._verify_watches(self._run_options.debug_options.debug_tensor_watch_opts, 0, ['DebugIdentity'], ['file:///tmp/tfdbg_1'])
        self.assertNotIn('a1', node_names)
        self.assertNotIn('a1/Assign', node_names)
        self.assertNotIn('b', node_names)
        self.assertNotIn('b/Assign', node_names)
        self.assertIn('s', node_names)

    def testWatchGraph_nodeNameAndTensorDTypeDenylists(self):
        if False:
            for i in range(10):
                print('nop')
        debug_utils.watch_graph_with_denylists(self._run_options, self._graph, debug_urls='file:///tmp/tfdbg_1', node_name_regex_denylist='^s$', tensor_dtype_regex_denylist='.*_ref')
        node_names = self._verify_watches(self._run_options.debug_options.debug_tensor_watch_opts, 0, ['DebugIdentity'], ['file:///tmp/tfdbg_1'])
        self.assertNotIn('a1', node_names)
        self.assertNotIn('a1/Assign', node_names)
        self.assertNotIn('b', node_names)
        self.assertNotIn('b/Assign', node_names)
        self.assertNotIn('s', node_names)
if __name__ == '__main__':
    googletest.main()