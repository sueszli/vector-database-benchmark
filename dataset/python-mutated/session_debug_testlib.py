"""Tests for debugger functionalities in tf.Session."""
import collections
import functools
import glob
import os
import tempfile
import threading
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
import tensorflow.python.ops.tensor_array_grad
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent

def no_rewrite_session_config():
    if False:
        for i in range(10):
            print('nop')
    rewriter_config = rewriter_config_pb2.RewriterConfig(disable_model_pruning=True, arithmetic_optimization=rewriter_config_pb2.RewriterConfig.OFF, dependency_optimization=rewriter_config_pb2.RewriterConfig.OFF)
    graph_options = config_pb2.GraphOptions(rewrite_options=rewriter_config)
    return config_pb2.ConfigProto(graph_options=graph_options)

class _RNNCellForTest(rnn_cell_impl.RNNCell):
    """RNN cell for testing."""

    def __init__(self, input_output_size, state_size):
        if False:
            while True:
                i = 10
        self._input_output_size = input_output_size
        self._state_size = state_size
        self._w = variable_v1.VariableV1(1.0, dtype=dtypes.float32, name='w')

    @property
    def output_size(self):
        if False:
            return 10
        return self._input_output_size

    @property
    def state_size(self):
        if False:
            i = 10
            return i + 15
        return self._state_size

    def __call__(self, input_, state, scope=None):
        if False:
            for i in range(10):
                print('nop')
        return (math_ops.multiply(self._w, input_), state)

@test_util.run_v1_only('b/120545219')
class SessionDebugTestBase(test_util.TensorFlowTestCase):
    """Base class for unit tests of tfdbg running with tf.Session."""

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        if test.is_gpu_available():
            cls._expected_partition_graph_count = 2
            cls._expected_num_devices = 2
            gpu_name = test_util.gpu_device_name()
            cls._main_device = '/job:localhost/replica:0/task:0' + gpu_name
        else:
            cls._expected_partition_graph_count = 1
            cls._expected_num_devices = 1
            cls._main_device = '/job:localhost/replica:0/task:0/device:CPU:0'

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        pass

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self._dump_root = tempfile.mkdtemp()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        ops.reset_default_graph()
        if os.path.isdir(self._dump_root):
            file_io.delete_recursively(self._dump_root)

    def _debug_urls(self, run_number=None):
        if False:
            while True:
                i = 10
        raise NotImplementedError('_debug_urls() method is not implemented in the base test class.')

    def _debug_dump_dir(self, run_number=None):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('_debug_dump_dir() method is not implemented in the base test class.')

    def _debug_run_and_get_dump(self, sess, fetches, feed_dict=None, debug_ops='DebugIdentity', tolerate_debug_op_creation_failures=False, global_step=-1, validate=True, expected_partition_graph_count=None):
        if False:
            print('Hello World!')
        'Run fetches with debugging and obtain DebugDumpDir.\n\n    Args:\n      sess: the tf.compat.v1.Session to be used.\n      fetches: fetches of the Session.run().\n      feed_dict: feed dict for the Session.run().\n      debug_ops: name(s) of the debug ops to be used.\n      tolerate_debug_op_creation_failures: whether to tolerate debug op\n        creation failures.\n      global_step: Optional global step.\n      validate: whether to validate dumped tensors against graph.\n      expected_partition_graph_count: optional count of partition graphs to\n        assert on.\n\n    Returns:\n      1. Return values of the Session.run().\n      2. The DebugDumpDir object from the debugged run().\n    '
        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        debug_utils.watch_graph(run_options, sess.graph, debug_ops=debug_ops, debug_urls=self._debug_urls(), tolerate_debug_op_creation_failures=tolerate_debug_op_creation_failures, global_step=global_step)
        run_metadata = config_pb2.RunMetadata()
        run_output = sess.run(fetches, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
        if expected_partition_graph_count is not None:
            self.assertEqual(expected_partition_graph_count, len(run_metadata.partition_graphs))
        return (run_output, debug_data.DebugDumpDir(self._dump_root, partition_graphs=run_metadata.partition_graphs, validate=validate))

    def _generate_dump_from_simple_addition_graph(self):
        if False:
            i = 10
            return i + 15
        with session.Session(config=no_rewrite_session_config()) as sess:
            u_init_val = np.array([[5.0, 3.0], [-1.0, 0.0]])
            v_init_val = np.array([[2.0], [-1.0]])
            u_name = 'u'
            v_name = 'v'
            w_name = 'w'
            u_init = constant_op.constant(u_init_val, shape=[2, 2])
            u = variable_v1.VariableV1(u_init, name=u_name)
            v_init = constant_op.constant(v_init_val, shape=[2, 1])
            v = variable_v1.VariableV1(v_init, name=v_name)
            w = math_ops.matmul(u, v, name=w_name)
            u.initializer.run()
            v.initializer.run()
            run_options = config_pb2.RunOptions(output_partition_graphs=True)
            debug_urls = 'file://%s' % self._dump_root
            debug_utils.add_debug_tensor_watch(run_options, '%s/read' % u_name, 0, debug_urls=debug_urls)
            debug_utils.add_debug_tensor_watch(run_options, '%s/read' % v_name, 0, debug_urls=debug_urls)
            run_metadata = config_pb2.RunMetadata()
            sess.run(w, options=run_options, run_metadata=run_metadata)
            self.assertEqual(self._expected_partition_graph_count, len(run_metadata.partition_graphs))
            dump = debug_data.DebugDumpDir(self._dump_root, partition_graphs=run_metadata.partition_graphs)
        simple_add_results = collections.namedtuple('SimpleAddResults', ['u_init_val', 'v_init_val', 'u', 'v', 'w', 'u_name', 'v_name', 'w_name', 'dump'])
        return simple_add_results(u_init_val, v_init_val, u, v, w, u_name, v_name, w_name, dump)

    def testCopyNodesHaveCorrectDebugOpsAndURLsAttributeValues(self):
        if False:
            while True:
                i = 10
        with session.Session() as sess:
            u = variable_v1.VariableV1(2.1, name='u')
            v = variable_v1.VariableV1(20.0, name='v')
            w = math_ops.multiply(u, v, name='w')
            sess.run(variables.global_variables_initializer())
            run_options = config_pb2.RunOptions(output_partition_graphs=True)
            debug_urls = self._debug_urls()
            debug_utils.add_debug_tensor_watch(run_options, 'u', 0, ['DebugNumericSummary(gated_grpc=True)', 'DebugIdentity'], debug_urls=debug_urls)
            debug_utils.add_debug_tensor_watch(run_options, 'v', 0, ['DebugNumericSummary'], debug_urls=debug_urls)
            run_metadata = config_pb2.RunMetadata()
            r = sess.run(w, options=run_options, run_metadata=run_metadata)
            self.assertAllClose(42.0, r)
            u_copy_node_def = None
            v_copy_node_def = None
            for partition_graph in run_metadata.partition_graphs:
                for node_def in partition_graph.node:
                    if debug_graphs.is_copy_node(node_def.name):
                        if node_def.name == '__copy_u_0':
                            u_copy_node_def = node_def
                        elif node_def.name == '__copy_v_0':
                            v_copy_node_def = node_def
            self.assertIsNotNone(u_copy_node_def)
            debug_ops_spec = u_copy_node_def.attr['debug_ops_spec'].list.s
            self.assertEqual(2, len(debug_ops_spec))
            self.assertEqual('DebugNumericSummary;%s;1' % debug_urls[0], debug_ops_spec[0].decode('utf-8'))
            self.assertEqual('DebugIdentity;%s;0' % debug_urls[0], debug_ops_spec[1].decode('utf-8'))
            self.assertIsNotNone(v_copy_node_def)
            debug_ops_spec = v_copy_node_def.attr['debug_ops_spec'].list.s
            self.assertEqual(1, len(debug_ops_spec))
            self.assertEqual('DebugNumericSummary;%s;0' % debug_urls[0], debug_ops_spec[0].decode('utf-8'))

    def testConcurrentDumpingToPathsWithOverlappingParentDirsWorks(self):
        if False:
            i = 10
            return i + 15
        results = self._generate_dump_from_simple_addition_graph()
        self.assertTrue(results.dump.loaded_partition_graphs())
        self.assertEqual(-1, results.dump.core_metadata.global_step)
        self.assertGreaterEqual(results.dump.core_metadata.session_run_index, 0)
        self.assertGreaterEqual(results.dump.core_metadata.executor_step_index, 0)
        self.assertEqual([], results.dump.core_metadata.input_names)
        self.assertEqual([results.w.name], results.dump.core_metadata.output_names)
        self.assertEqual([], results.dump.core_metadata.target_nodes)
        self.assertEqual(2, results.dump.size)
        self.assertAllClose([results.u_init_val], results.dump.get_tensors('%s/read' % results.u_name, 0, 'DebugIdentity'))
        self.assertAllClose([results.v_init_val], results.dump.get_tensors('%s/read' % results.v_name, 0, 'DebugIdentity'))
        self.assertGreaterEqual(results.dump.get_rel_timestamps('%s/read' % results.u_name, 0, 'DebugIdentity')[0], 0)
        self.assertGreaterEqual(results.dump.get_rel_timestamps('%s/read' % results.v_name, 0, 'DebugIdentity')[0], 0)
        self.assertGreater(results.dump.get_dump_sizes_bytes('%s/read' % results.u_name, 0, 'DebugIdentity')[0], 0)
        self.assertGreater(results.dump.get_dump_sizes_bytes('%s/read' % results.v_name, 0, 'DebugIdentity')[0], 0)

    def testGetOpTypeWorks(self):
        if False:
            while True:
                i = 10
        results = self._generate_dump_from_simple_addition_graph()
        self.assertEqual(results.u.op.type, results.dump.node_op_type(results.u_name))
        self.assertIn(results.v.op.type, results.dump.node_op_type(results.v_name))
        self.assertIn(results.w.op.type, results.dump.node_op_type(results.w_name))
        with self.assertRaisesRegexp(ValueError, 'None of the .* device\\(s\\) has a node named '):
            results.dump.node_op_type('foo_bar')

    def testDumpStringTensorsWorks(self):
        if False:
            while True:
                i = 10
        with session.Session(config=no_rewrite_session_config()) as sess:
            str1_init_val = np.array(b'abc')
            str2_init_val = np.array(b'def')
            str1_init = constant_op.constant(str1_init_val)
            str2_init = constant_op.constant(str2_init_val)
            str1_name = 'str1'
            str2_name = 'str2'
            str1 = variable_v1.VariableV1(str1_init, name=str1_name)
            str2 = variable_v1.VariableV1(str2_init, name=str2_name)
            str_concat = math_ops.add(str1, str2, name='str_concat')
            str1.initializer.run()
            str2.initializer.run()
            run_options = config_pb2.RunOptions(output_partition_graphs=True)
            debug_urls = self._debug_urls()
            debug_utils.add_debug_tensor_watch(run_options, '%s/read' % str1_name, 0, debug_urls=debug_urls)
            debug_utils.add_debug_tensor_watch(run_options, '%s/read' % str2_name, 0, debug_urls=debug_urls)
            run_metadata = config_pb2.RunMetadata()
            sess.run(str_concat, options=run_options, run_metadata=run_metadata)
            self.assertEqual(1, len(run_metadata.partition_graphs))
            dump = debug_data.DebugDumpDir(self._dump_root, partition_graphs=run_metadata.partition_graphs)
            self.assertIn(str1_name, dump.nodes())
            self.assertIn(str2_name, dump.nodes())
            self.assertEqual(2, dump.size)
            self.assertEqual([str1_init_val], dump.get_tensors('%s/read' % str1_name, 0, 'DebugIdentity'))
            self.assertEqual([str2_init_val], dump.get_tensors('%s/read' % str2_name, 0, 'DebugIdentity'))
            self.assertGreaterEqual(dump.get_rel_timestamps('%s/read' % str1_name, 0, 'DebugIdentity')[0], 0)
            self.assertGreaterEqual(dump.get_rel_timestamps('%s/read' % str2_name, 0, 'DebugIdentity')[0], 0)
            self.assertGreater(dump.get_dump_sizes_bytes('%s/read' % str1_name, 0, 'DebugIdentity')[0], 0)
            self.assertGreater(dump.get_dump_sizes_bytes('%s/read' % str2_name, 0, 'DebugIdentity')[0], 0)

    def testDumpUninitializedVariable(self):
        if False:
            i = 10
            return i + 15
        op_namespace = 'testDumpUninitializedVariable'
        with session.Session() as sess:
            u_init_val = np.array([[5.0, 3.0], [-1.0, 0.0]])
            s_init_val = b'str1'
            u_name = '%s/u' % op_namespace
            s_name = '%s/s' % op_namespace
            u_init = constant_op.constant(u_init_val, shape=[2, 2])
            u = variable_v1.VariableV1(u_init, name=u_name)
            s_init = constant_op.constant(s_init_val)
            s = variable_v1.VariableV1(s_init, name=s_name)
            run_options = config_pb2.RunOptions(output_partition_graphs=True)
            debug_urls = self._debug_urls()
            debug_utils.add_debug_tensor_watch(run_options, u_name, 0, debug_urls=debug_urls)
            debug_utils.add_debug_tensor_watch(run_options, s_name, 0, debug_urls=debug_urls)
            run_metadata = config_pb2.RunMetadata()
            sess.run(variables.global_variables_initializer(), options=run_options, run_metadata=run_metadata)
            dump = debug_data.DebugDumpDir(self._dump_root, partition_graphs=run_metadata.partition_graphs)
            self.assertEqual(2, dump.size)
            self.assertEqual(self._expected_partition_graph_count, len(run_metadata.partition_graphs))
            u_vals = dump.get_tensors(u_name, 0, 'DebugIdentity')
            s_vals = dump.get_tensors(s_name, 0, 'DebugIdentity')
            self.assertEqual(1, len(u_vals))
            self.assertIsInstance(u_vals[0], debug_data.InconvertibleTensorProto)
            self.assertFalse(u_vals[0].initialized)
            self.assertEqual(1, len(s_vals))
            self.assertIsInstance(s_vals[0], debug_data.InconvertibleTensorProto)
            self.assertFalse(s_vals[0].initialized)
            self.assertAllClose(u_init_val, sess.run(u))
            self.assertEqual(s_init_val, sess.run(s))

    def testDebugWhileLoopGeneratesMultipleDumps(self):
        if False:
            while True:
                i = 10
        with session.Session(config=no_rewrite_session_config()) as sess:
            num_iter = 10
            u_name = 'testDumpToFileWhileLoop/u'
            u_namespace = u_name.split('/')[0]
            u_init_val = np.array(11.0)
            u_init = constant_op.constant(u_init_val)
            u = variable_v1.VariableV1(u_init, name=u_name)
            v_name = 'testDumpToFileWhileLoop/v'
            v_namespace = v_name.split('/')[0]
            v_init_val = np.array(2.0)
            v_init = constant_op.constant(v_init_val)
            v = variable_v1.VariableV1(v_init, name=v_name)
            u.initializer.run()
            v.initializer.run()
            i = constant_op.constant(0, name='testDumpToFileWhileLoop/i')

            def cond(i):
                if False:
                    i = 10
                    return i + 15
                return math_ops.less(i, num_iter)

            def body(i):
                if False:
                    print('Hello World!')
                new_u = state_ops.assign_add(u, v)
                new_i = math_ops.add(i, 1)
                op = control_flow_ops.group(new_u)
                new_i = control_flow_ops.with_dependencies([op], new_i)
                return [new_i]
            loop = while_loop.while_loop(cond, body, [i], parallel_iterations=10)
            run_options = config_pb2.RunOptions(output_partition_graphs=True)
            debug_urls = self._debug_urls()
            debug_utils.add_debug_tensor_watch(run_options, u_name, 0, debug_urls=debug_urls)
            debug_utils.add_debug_tensor_watch(run_options, '%s/read' % v_name, 0, debug_urls=debug_urls)
            debug_utils.add_debug_tensor_watch(run_options, 'while/Identity', 0, debug_urls=debug_urls)
            debug_utils.add_debug_tensor_watch(run_options, 'while/Add/y', 0, debug_urls=debug_urls)
            run_metadata = config_pb2.RunMetadata()
            r = sess.run(loop, options=run_options, run_metadata=run_metadata)
            self.assertEqual(self._expected_partition_graph_count, len(run_metadata.partition_graphs))
            self.assertEqual(num_iter, r)
            u_val_final = sess.run(u)
            self.assertAllClose(u_init_val + num_iter * v_init_val, u_val_final)
            self.assertTrue(os.path.isdir(self._dump_root))
            u_glob_out = glob.glob(os.path.join(self._dump_root, '*', u_namespace))
            v_glob_out = glob.glob(os.path.join(self._dump_root, '*', v_namespace, 'v'))
            self.assertTrue(os.path.isdir(u_glob_out[0]))
            self.assertTrue(os.path.isdir(v_glob_out[0]))
            dump = debug_data.DebugDumpDir(self._dump_root, partition_graphs=run_metadata.partition_graphs)
            self.assertEqual(1 + 1 + num_iter + num_iter, dump.size)
            self.assertAllClose([u_init_val], dump.get_tensors(u_name, 0, 'DebugIdentity'))
            self.assertAllClose([v_init_val], dump.get_tensors('%s/read' % v_name, 0, 'DebugIdentity'))
            while_id_tensors = dump.get_tensors('while/Identity', 0, 'DebugIdentity')
            self.assertEqual(10, len(while_id_tensors))
            for k in range(len(while_id_tensors)):
                self.assertAllClose(np.array(k), while_id_tensors[k])
            while_id_rel_timestamps = dump.get_rel_timestamps('while/Identity', 0, 'DebugIdentity')
            while_id_dump_sizes_bytes = dump.get_dump_sizes_bytes('while/Identity', 0, 'DebugIdentity')
            self.assertEqual(10, len(while_id_rel_timestamps))
            prev_rel_time = 0
            prev_dump_size_bytes = while_id_dump_sizes_bytes[0]
            for (rel_time, dump_size_bytes) in zip(while_id_rel_timestamps, while_id_dump_sizes_bytes):
                self.assertGreaterEqual(rel_time, prev_rel_time)
                self.assertEqual(dump_size_bytes, prev_dump_size_bytes)
                prev_rel_time = rel_time
                prev_dump_size_bytes = dump_size_bytes
            watch_keys = dump.debug_watch_keys('while/Identity')
            self.assertEqual(['while/Identity:0:DebugIdentity'], watch_keys)
            self.assertEqual(10, len(dump.watch_key_to_data(watch_keys[0])))
            self.assertEqual([], dump.watch_key_to_data('foo'))

    def testDebugWhileLoopWatchingWholeGraphWorks(self):
        if False:
            while True:
                i = 10
        with session.Session() as sess:
            loop_body = lambda i: math_ops.add(i, 2)
            loop_cond = lambda i: math_ops.less(i, 16)
            i = constant_op.constant(10, name='i')
            loop = while_loop.while_loop(loop_cond, loop_body, [i])
            (loop_result, dump) = self._debug_run_and_get_dump(sess, loop)
            self.assertEqual(16, loop_result)
            self.assertEqual([[10]], dump.get_tensors('while/Enter', 0, 'DebugIdentity'))
            self.assertEqual([[12], [14], [16]], dump.get_tensors('while/NextIteration', 0, 'DebugIdentity'))

    def testDebugTrainingDynamicRNNWorks(self):
        if False:
            while True:
                i = 10
        with session.Session() as sess:
            input_size = 3
            state_size = 2
            time_steps = 4
            batch_size = 2
            input_values = np.random.randn(time_steps, batch_size, input_size)
            sequence_length = np.random.randint(0, time_steps, size=batch_size)
            concat_inputs = array_ops.placeholder(dtypes.float32, shape=(time_steps, batch_size, input_size))
            (outputs_dynamic, _) = rnn.dynamic_rnn(_RNNCellForTest(input_size, state_size), inputs=concat_inputs, sequence_length=sequence_length, time_major=True, dtype=dtypes.float32)
            toy_loss = math_ops.reduce_sum(outputs_dynamic * outputs_dynamic)
            train_op = gradient_descent.GradientDescentOptimizer(learning_rate=0.1).minimize(toy_loss, name='train_op')
            sess.run(variables.global_variables_initializer())
            run_options = config_pb2.RunOptions(output_partition_graphs=True)
            debug_utils.watch_graph_with_denylists(run_options, sess.graph, node_name_regex_denylist='(.*rnn/while/.*|.*TensorArray.*)', debug_urls=self._debug_urls())
            run_metadata = config_pb2.RunMetadata()
            sess.run(train_op, feed_dict={concat_inputs: input_values}, options=run_options, run_metadata=run_metadata)
            debug_data.DebugDumpDir(self._dump_root, partition_graphs=run_metadata.partition_graphs)

    def testDebugCondWatchingWholeGraphWorks(self):
        if False:
            print('Hello World!')
        with session.Session() as sess:
            x = variable_v1.VariableV1(10.0, name='x')
            y = variable_v1.VariableV1(20.0, name='y')
            cond = tf_cond.cond(x > y, lambda : math_ops.add(x, 1), lambda : math_ops.add(y, 1))
            sess.run(variables.global_variables_initializer())
            (cond_result, dump) = self._debug_run_and_get_dump(sess, cond)
            self.assertEqual(21, cond_result)
            self.assertAllClose([21.0], dump.get_tensors('cond/Merge', 0, 'DebugIdentity'))

    def testFindNodesWithBadTensorValues(self):
        if False:
            while True:
                i = 10
        with session.Session() as sess:
            u_name = 'testFindNodesWithBadTensorValues/u'
            v_name = 'testFindNodesWithBadTensorValues/v'
            w_name = 'testFindNodesWithBadTensorValues/w'
            x_name = 'testFindNodesWithBadTensorValues/x'
            y_name = 'testFindNodesWithBadTensorValues/y'
            z_name = 'testFindNodesWithBadTensorValues/z'
            u_init = constant_op.constant([2.0, 4.0])
            u = variable_v1.VariableV1(u_init, name=u_name)
            v_init = constant_op.constant([2.0, 1.0])
            v = variable_v1.VariableV1(v_init, name=v_name)
            w = math_ops.subtract(u, v, name=w_name)
            x = math_ops.div(u, w, name=x_name)
            y = math_ops.multiply(w, x, name=y_name)
            z = math_ops.multiply(y, y, name=z_name)
            u.initializer.run()
            v.initializer.run()
            (_, dump) = self._debug_run_and_get_dump(sess, z, expected_partition_graph_count=self._expected_partition_graph_count)

            def has_bad_value(_, tensor):
                if False:
                    while True:
                        i = 10
                return np.any(np.isnan(tensor)) or np.any(np.isinf(tensor))
            bad_data = dump.find(has_bad_value)
            self.assertLessEqual(3, len(bad_data))
            node_names = [datum.node_name for datum in bad_data]
            self.assertIn(x_name, node_names)
            self.assertIn(y_name, node_names)
            self.assertIn(z_name, node_names)
            first_bad_datum = dump.find(has_bad_value, first_n=1)
            self.assertEqual(1, len(first_bad_datum))

    def testFindInfOrNanWithOpNameExclusion(self):
        if False:
            print('Hello World!')
        with session.Session() as sess:
            u_name = 'testFindInfOrNanWithOpNameExclusion/u'
            v_name = 'testFindInfOrNanWithOpNameExclusion/v'
            w_name = 'testFindInfOrNanWithOpNameExclusion/w'
            x_name = 'testFindInfOrNanWithOpNameExclusion/x'
            y_name = 'testFindInfOrNanWithOpNameExclusion/y'
            z_name = 'testFindInfOrNanWithOpNameExclusion/z'
            u_init = constant_op.constant([2.0, 4.0])
            u = variable_v1.VariableV1(u_init, name=u_name)
            v_init = constant_op.constant([2.0, 1.0])
            v = variable_v1.VariableV1(v_init, name=v_name)
            w = math_ops.subtract(u, v, name=w_name)
            x = math_ops.div(u, w, name=x_name)
            y = math_ops.multiply(w, x, name=y_name)
            z = math_ops.multiply(y, y, name=z_name)
            u.initializer.run()
            v.initializer.run()
            (_, dump) = self._debug_run_and_get_dump(sess, z, expected_partition_graph_count=self._expected_partition_graph_count)
            bad_data = dump.find(debug_data.has_inf_or_nan, exclude_node_names='.*/x$')
            self.assertLessEqual(2, len(bad_data))
            node_names = [datum.node_name for datum in bad_data]
            self.assertIn(y_name, node_names)
            self.assertIn(z_name, node_names)
            first_bad_datum = dump.find(debug_data.has_inf_or_nan, first_n=1, exclude_node_names='.*/x$')
            self.assertEqual(1, len(first_bad_datum))

    def _session_run_for_graph_structure_lookup(self):
        if False:
            print('Hello World!')
        with session.Session(config=no_rewrite_session_config()) as sess:
            u_name = 'testDumpGraphStructureLookup/u'
            v_name = 'testDumpGraphStructureLookup/v'
            w_name = 'testDumpGraphStructureLookup/w'
            u_init = constant_op.constant([2.0, 4.0])
            u = variable_v1.VariableV1(u_init, name=u_name)
            v = math_ops.add(u, u, name=v_name)
            w = math_ops.add(v, v, name=w_name)
            u.initializer.run()
            (_, dump) = self._debug_run_and_get_dump(sess, w, expected_partition_graph_count=self._expected_partition_graph_count)
        return (u_name, v_name, w_name, dump)

    def testGraphStructureLookupGivesDevicesAndNodesInfo(self):
        if False:
            i = 10
            return i + 15
        (u_name, _, _, dump) = self._session_run_for_graph_structure_lookup()
        self.assertEqual(self._expected_num_devices, len(dump.devices()))
        self.assertEqual(self._main_device, dump.node_device(u_name))
        with self.assertRaisesRegexp(ValueError, 'does not exist in partition graphs'):
            dump.node_device(u_name + 'foo')
        self.assertTrue(dump.node_exists(u_name))
        self.assertTrue(dump.node_exists(u_name + '/read'))
        self.assertFalse(dump.node_exists(u_name + '/read' + '/foo'))

    def testGraphStructureLookupGivesNodesAndAttributes(self):
        if False:
            print('Hello World!')
        (u_name, _, _, dump) = self._session_run_for_graph_structure_lookup()
        u_read_name = u_name + '/read'
        if test_util.gpu_device_name():
            node_names = dump.nodes(device_name='/job:localhost/replica:0/task:0/device:GPU:0')
        else:
            node_names = dump.nodes()
        self.assertTrue(u_name in node_names)
        self.assertTrue(u_read_name in node_names)
        u_attr = dump.node_attributes(u_name)
        self.assertEqual(dtypes.float32, u_attr['dtype'].type)
        self.assertEqual(1, len(u_attr['shape'].shape.dim))
        self.assertEqual(2, u_attr['shape'].shape.dim[0].size)
        with self.assertRaisesRegexp(ValueError, 'None of the .* device\\(s\\) has a node named '):
            dump.node_attributes('foo')

    def testGraphStructureLookupGivesDebugWatchKeys(self):
        if False:
            return 10
        (u_name, v_name, w_name, dump) = self._session_run_for_graph_structure_lookup()
        self.assertEqual(['%s:0:DebugIdentity' % u_name], dump.debug_watch_keys(u_name))
        self.assertEqual(['%s:0:DebugIdentity' % v_name], dump.debug_watch_keys(v_name))
        self.assertEqual(['%s:0:DebugIdentity' % w_name], dump.debug_watch_keys(w_name))
        self.assertEqual([], dump.debug_watch_keys('foo'))
        u_data = dump.watch_key_to_data(dump.debug_watch_keys(u_name)[0])
        self.assertEqual(1, len(u_data))
        self.assertEqual(u_name, u_data[0].node_name)
        self.assertEqual(0, u_data[0].output_slot)
        self.assertEqual('DebugIdentity', u_data[0].debug_op)
        self.assertGreaterEqual(u_data[0].timestamp, 0)
        self.assertEqual([], dump.watch_key_to_data('foo'))

    def testGraphStructureLookupGivesNodeInputsAndRecipients(self):
        if False:
            return 10
        (u_name, v_name, w_name, dump) = self._session_run_for_graph_structure_lookup()
        u_read_name = u_name + '/read'
        self.assertEqual([], dump.node_inputs(u_name))
        self.assertEqual([u_name], dump.node_inputs(u_read_name))
        self.assertEqual([u_read_name] * 2, dump.node_inputs(v_name))
        self.assertEqual([v_name] * 2, dump.node_inputs(w_name))
        self.assertEqual([], dump.node_inputs(u_name, is_control=True))
        self.assertEqual([], dump.node_inputs(u_read_name, is_control=True))
        self.assertEqual([], dump.node_inputs(v_name, is_control=True))
        self.assertEqual([], dump.node_inputs(w_name, is_control=True))
        self.assertTrue(u_read_name in dump.node_recipients(u_name))
        self.assertEqual(2, dump.node_recipients(u_read_name).count(v_name))
        self.assertEqual(2, dump.node_recipients(v_name).count(w_name))
        self.assertEqual([], dump.node_recipients(u_name, is_control=True))
        self.assertEqual([], dump.node_recipients(u_read_name, is_control=True))
        self.assertEqual([], dump.node_recipients(v_name, is_control=True))
        self.assertEqual([], dump.node_recipients(w_name, is_control=True))
        with self.assertRaisesRegexp(ValueError, 'None of the .* device\\(s\\) has a node named '):
            dump.node_inputs(u_name + 'foo')
        with self.assertRaisesRegexp(ValueError, 'None of the .* device\\(s\\) has a node named '):
            dump.node_recipients(u_name + 'foo')
        self.assertEqual([], dump.transitive_inputs(u_name))
        self.assertEqual([u_name], dump.transitive_inputs(u_read_name))
        self.assertEqual(set([u_name, u_read_name]), set(dump.transitive_inputs(v_name)))
        self.assertEqual(set([u_name, u_read_name, v_name]), set(dump.transitive_inputs(w_name)))
        with self.assertRaisesRegexp(ValueError, 'None of the .* device\\(s\\) has a node named '):
            dump.transitive_inputs(u_name + 'foo')

    def testGraphStructureLookupWithoutPartitionGraphsDoesNotErrorOut(self):
        if False:
            for i in range(10):
                print('nop')
        (_, _, _, dump) = self._session_run_for_graph_structure_lookup()
        dump = debug_data.DebugDumpDir(self._dump_root, validate=False)
        self.assertTrue(dump.loaded_partition_graphs())

    def testGraphPathFindingOnControlEdgesWorks(self):
        if False:
            return 10
        with session.Session(config=no_rewrite_session_config()) as sess:
            v1 = variable_v1.VariableV1(1.0, name='v1')
            v2 = variable_v1.VariableV1(2.0, name='v2')
            v3 = variable_v1.VariableV1(3.0, name='v3')
            a = math_ops.add(v1, v2, name='a')
            with ops.control_dependencies([a]):
                c = math_ops.subtract(v3, v3, name='c')
            sess.run(variables.global_variables_initializer())
            (_, dump) = self._debug_run_and_get_dump(sess, c)
            self.assertEqual(['v1', 'v1/read', 'a', 'c'], dump.find_some_path('v1', 'c'))
            self.assertIsNone(dump.find_some_path('v1', 'c', include_control=False))

    def testGraphPathFindingReverseRefEdgeWorks(self):
        if False:
            while True:
                i = 10
        with session.Session(config=no_rewrite_session_config()) as sess:
            v = variable_v1.VariableV1(10.0, name='v')
            delta = variable_v1.VariableV1(1.0, name='delta')
            inc_v = state_ops.assign_add(v, delta, name='inc_v')
            sess.run(variables.global_variables_initializer())
            (_, dump) = self._debug_run_and_get_dump(sess, inc_v)
            self.assertEqual(['delta', 'delta/read', 'inc_v', 'v'], dump.find_some_path('delta', 'v', include_reversed_ref=True))
            self.assertIsNone(dump.find_some_path('delta', 'v'))

    def testCausalityCheckOnDumpsDetectsWrongTemporalOrder(self):
        if False:
            return 10
        with session.Session(config=no_rewrite_session_config()) as sess:
            u_name = 'testDumpCausalityCheck/u'
            v_name = 'testDumpCausalityCheck/v'
            w_name = 'testDumpCausalityCheck/w'
            u_init = constant_op.constant([2.0, 4.0])
            u = variable_v1.VariableV1(u_init, name=u_name)
            v = math_ops.add(u, u, name=v_name)
            w = math_ops.add(v, v, name=w_name)
            u.initializer.run()
            run_options = config_pb2.RunOptions(output_partition_graphs=True)
            debug_utils.watch_graph(run_options, sess.graph, debug_ops=['DebugIdentity'], debug_urls=self._debug_urls())
            run_metadata = config_pb2.RunMetadata()
            sess.run(w, options=run_options, run_metadata=run_metadata)
            self.assertEqual(self._expected_partition_graph_count, len(run_metadata.partition_graphs))
            debug_data.DebugDumpDir(self._dump_root)
            dump = debug_data.DebugDumpDir(self._dump_root, partition_graphs=run_metadata.partition_graphs)
            self.assertEqual(1, len(dump.get_tensor_file_paths(v_name, 0, 'DebugIdentity')))
            v_file_path = dump.get_tensor_file_paths(v_name, 0, 'DebugIdentity')[0]
            self.assertEqual(1, len(dump.get_tensor_file_paths(w_name, 0, 'DebugIdentity')))
            w_file_path = dump.get_tensor_file_paths(w_name, 0, 'DebugIdentity')[0]
            v_timestamp = int(v_file_path[v_file_path.rindex('_') + 1:])
            w_timestamp = int(w_file_path[w_file_path.rindex('_') + 1:])
            v_file_path_1 = v_file_path[:v_file_path.rindex('_')] + '_%d' % w_timestamp
            w_file_path_1 = w_file_path[:w_file_path.rindex('_')] + '_%d' % (v_timestamp - 1)
            os.rename(v_file_path, v_file_path_1)
            os.rename(w_file_path, w_file_path_1)
            with self.assertRaisesRegexp(ValueError, 'Causality violated'):
                dump = debug_data.DebugDumpDir(self._dump_root, partition_graphs=run_metadata.partition_graphs)
            dump = debug_data.DebugDumpDir(self._dump_root, partition_graphs=run_metadata.partition_graphs, validate=False)
            v_file_path_2 = v_file_path[:v_file_path.rindex('_')] + '_%d' % w_timestamp
            w_file_path_2 = w_file_path[:w_file_path.rindex('_')] + '_%d' % w_timestamp
            os.rename(v_file_path_1, v_file_path_2)
            os.rename(w_file_path_1, w_file_path_2)
            debug_data.DebugDumpDir(self._dump_root, partition_graphs=run_metadata.partition_graphs)

    def testWatchingOnlyOneOfTwoOutputSlotsDoesNotLeadToCausalityFailure(self):
        if False:
            while True:
                i = 10
        with session.Session() as sess:
            x_name = 'oneOfTwoSlots/x'
            u_name = 'oneOfTwoSlots/u'
            v_name = 'oneOfTwoSlots/v'
            w_name = 'oneOfTwoSlots/w'
            y_name = 'oneOfTwoSlots/y'
            x = variable_v1.VariableV1([1, 3, 3, 7], dtype=dtypes.int32, name=x_name)
            sess.run(x.initializer)
            (unique_x, indices, _) = array_ops.unique_with_counts(x, name=u_name)
            v = math_ops.add(unique_x, unique_x, name=v_name)
            w = math_ops.add(indices, indices, name=w_name)
            y = math_ops.add(w, w, name=y_name)
            run_options = config_pb2.RunOptions(output_partition_graphs=True)
            debug_utils.add_debug_tensor_watch(run_options, u_name, 0, debug_urls=self._debug_urls())
            debug_utils.add_debug_tensor_watch(run_options, w_name, 0, debug_urls=self._debug_urls())
            debug_utils.add_debug_tensor_watch(run_options, y_name, 0, debug_urls=self._debug_urls())
            run_metadata = config_pb2.RunMetadata()
            sess.run([v, y], options=run_options, run_metadata=run_metadata)
            dump = debug_data.DebugDumpDir(self._dump_root, partition_graphs=run_metadata.partition_graphs, validate=True)
            self.assertAllClose([1, 3, 7], dump.get_tensors(u_name, 0, 'DebugIdentity')[0])

    def testOutputSlotWithoutOutgoingEdgeCanBeWatched(self):
        if False:
            while True:
                i = 10
        'Test watching output slots not attached to any outgoing edges.'
        with session.Session(config=no_rewrite_session_config()) as sess:
            u_init_val = np.array([[5.0, 3.0], [-1.0, 0.0]])
            u = constant_op.constant(u_init_val, shape=[2, 2], name='u')
            with ops.control_dependencies([u]):
                z = control_flow_ops.no_op(name='z')
            (_, dump) = self._debug_run_and_get_dump(sess, z)
            self.assertEqual(1, len(dump.dumped_tensor_data))
            datum = dump.dumped_tensor_data[0]
            self.assertEqual('u', datum.node_name)
            self.assertEqual(0, datum.output_slot)
            self.assertEqual('DebugIdentity', datum.debug_op)
            self.assertAllClose([[5.0, 3.0], [-1.0, 0.0]], datum.get_tensor())

    def testWatchingVariableUpdateOpsSeesUpdatedValues(self):
        if False:
            while True:
                i = 10
        'Watch output slots on Variable-updating ops, with no emitted edges.'
        with session.Session(config=no_rewrite_session_config()) as sess:
            u_init = constant_op.constant(10.0)
            u = variable_v1.VariableV1(u_init, name='gdo/u')
            v_init = constant_op.constant(20.0)
            v = variable_v1.VariableV1(v_init, name='gdo/v')
            w = math_ops.multiply(u, v, name='gdo/w')
            train_op = gradient_descent.GradientDescentOptimizer(learning_rate=0.1).minimize(w, name='gdo/train')
            u.initializer.run()
            v.initializer.run()
            (_, dump) = self._debug_run_and_get_dump(sess, train_op)
            update_u_data = dump.watch_key_to_data('gdo/train/update_gdo/u/ApplyGradientDescent:0:DebugIdentity')
            self.assertEqual(1, len(update_u_data))
            self.assertAllClose(8.0, update_u_data[0].get_tensor())
            update_v_data = dump.watch_key_to_data('gdo/train/update_gdo/v/ApplyGradientDescent:0:DebugIdentity')
            self.assertEqual(1, len(update_v_data))
            self.assertAllClose(19.0, update_v_data[0].get_tensor())
            self.assertAllClose(8.0, sess.run(u))
            self.assertAllClose(19.0, sess.run(v))

    def testAllowsWatchingUnconnectedOutputTensor(self):
        if False:
            return 10
        'Watch an output slot not emitting any edges.\n\n    (Not even control edges from the node.)\n    '
        with session.Session() as sess:
            x_init = constant_op.constant([2, 2, 3, 5, 5])
            x = variable_v1.VariableV1(x_init, name='unconnected/x')
            (unique_x, _) = array_ops.unique(x, name='unconnected/unique_x')
            y = math_ops.add(unique_x, [0, 1, 2], name='unconnected/y')
            x.initializer.run()
            unique_x_slot_0_recipients = []
            unique_x_slot_1_recipients = []
            for op in sess.graph.get_operations():
                for inp in op.inputs:
                    if inp.name == 'unconnected/unique_x:0':
                        unique_x_slot_0_recipients.append(op.name)
                    elif inp.name == 'unconnected/unique_x:1':
                        unique_x_slot_1_recipients.append(op.name)
            self.assertEqual(['unconnected/y'], unique_x_slot_0_recipients)
            self.assertEqual([], unique_x_slot_1_recipients)
            (y_result, dump) = self._debug_run_and_get_dump(sess, y)
            self.assertAllClose([2, 4, 7], y_result)
            unique_x_slot_0_dumps = dump.watch_key_to_data('unconnected/unique_x:0:DebugIdentity')
            self.assertEqual(1, len(unique_x_slot_0_dumps))
            self.assertEqual('unconnected/unique_x', unique_x_slot_0_dumps[0].node_name)
            self.assertEqual(0, unique_x_slot_0_dumps[0].output_slot)
            self.assertAllClose([2, 3, 5], unique_x_slot_0_dumps[0].get_tensor())
            unique_x_slot_1_dumps = dump.watch_key_to_data('unconnected/unique_x:1:DebugIdentity')
            self.assertEqual(1, len(unique_x_slot_1_dumps))
            self.assertEqual('unconnected/unique_x', unique_x_slot_1_dumps[0].node_name)
            self.assertEqual(1, unique_x_slot_1_dumps[0].output_slot)
            self.assertAllClose([0, 0, 1, 2, 2], unique_x_slot_1_dumps[0].get_tensor())

    def testSuccessiveDebuggingRunsIncreasesCounters(self):
        if False:
            while True:
                i = 10
        'Test repeated Session.run() calls with debugger increments counters.'
        with session.Session() as sess:
            ph = array_ops.placeholder(dtypes.float32, name='successive/ph')
            x = array_ops.transpose(ph, name='mismatch/x')
            y = array_ops.squeeze(ph, name='mismatch/y')
            (_, dump1) = self._debug_run_and_get_dump(sess, x, feed_dict={ph: np.array([[7.0, 8.0]])}, global_step=1)
            self.assertEqual(1, dump1.core_metadata.global_step)
            self.assertGreaterEqual(dump1.core_metadata.session_run_index, 0)
            self.assertEqual(0, dump1.core_metadata.executor_step_index)
            self.assertEqual([ph.name], dump1.core_metadata.input_names)
            self.assertEqual([x.name], dump1.core_metadata.output_names)
            self.assertEqual([], dump1.core_metadata.target_nodes)
            file_io.delete_recursively(self._dump_root)
            (_, dump2) = self._debug_run_and_get_dump(sess, x, feed_dict={ph: np.array([[7.0, 8.0]])}, global_step=2)
            self.assertEqual(2, dump2.core_metadata.global_step)
            self.assertEqual(dump1.core_metadata.session_run_index + 1, dump2.core_metadata.session_run_index)
            self.assertEqual(dump1.core_metadata.executor_step_index + 1, dump2.core_metadata.executor_step_index)
            self.assertEqual([ph.name], dump2.core_metadata.input_names)
            self.assertEqual([x.name], dump2.core_metadata.output_names)
            self.assertEqual([], dump2.core_metadata.target_nodes)
            file_io.delete_recursively(self._dump_root)
            run_options = config_pb2.RunOptions(output_partition_graphs=True)
            debug_utils.watch_graph(run_options, sess.graph, debug_urls=self._debug_urls(), global_step=3)
            (_, dump3) = self._debug_run_and_get_dump(sess, y, feed_dict={ph: np.array([[7.0, 8.0]])}, global_step=3)
            self.assertEqual(3, dump3.core_metadata.global_step)
            self.assertEqual(dump2.core_metadata.session_run_index + 1, dump3.core_metadata.session_run_index)
            self.assertEqual(0, dump3.core_metadata.executor_step_index)
            self.assertEqual([ph.name], dump3.core_metadata.input_names)
            self.assertEqual([y.name], dump3.core_metadata.output_names)
            self.assertEqual([], dump3.core_metadata.target_nodes)

    def testDebuggingDuringOpError(self):
        if False:
            return 10
        'Test the debug tensor dumping when error occurs in graph runtime.'
        with session.Session() as sess:
            ph = array_ops.placeholder(dtypes.float32, name='mismatch/ph')
            x = array_ops.transpose(ph, name='mismatch/x')
            m = constant_op.constant(np.array([[1.0, 2.0]], dtype=np.float32), name='mismatch/m')
            y = math_ops.matmul(m, x, name='mismatch/y')
            run_options = config_pb2.RunOptions(output_partition_graphs=True)
            debug_utils.watch_graph(run_options, sess.graph, debug_ops=['DebugIdentity'], debug_urls=self._debug_urls())
            with self.assertRaises(errors.OpError):
                sess.run(y, options=run_options, feed_dict={ph: np.array([[-3.0], [0.0]])})
            dump = debug_data.DebugDumpDir(self._dump_root)
            self.assertGreaterEqual(dump.core_metadata.session_run_index, 0)
            self.assertGreaterEqual(dump.core_metadata.executor_step_index, 0)
            self.assertEqual([ph.name], dump.core_metadata.input_names)
            self.assertEqual([y.name], dump.core_metadata.output_names)
            self.assertEqual([], dump.core_metadata.target_nodes)
            self.assertTrue(dump.loaded_partition_graphs())
            m_dumps = dump.watch_key_to_data('mismatch/m:0:DebugIdentity')
            self.assertEqual(1, len(m_dumps))
            self.assertAllClose(np.array([[1.0, 2.0]]), m_dumps[0].get_tensor())
            x_dumps = dump.watch_key_to_data('mismatch/x:0:DebugIdentity')
            self.assertEqual(1, len(x_dumps))
            self.assertAllClose(np.array([[-3.0, 0.0]]), x_dumps[0].get_tensor())

    def testDebugNumericSummaryOnInitializedTensorGivesCorrectResult(self):
        if False:
            print('Hello World!')
        with session.Session(config=no_rewrite_session_config()) as sess:
            a = variable_v1.VariableV1([np.nan, np.nan, 0.0, 0.0, 0.0, -1.0, -3.0, 3.0, 7.0, -np.inf, -np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.nan, np.nan], dtype=np.float32, name='numeric_summary/a')
            b = variable_v1.VariableV1([0.0] * 18, dtype=np.float32, name='numeric_summary/b')
            c = math_ops.add(a, b, name='numeric_summary/c')
            sess.run(variables.global_variables_initializer())
            (_, dump) = self._debug_run_and_get_dump(sess, c, debug_ops=['DebugNumericSummary'])
            self.assertTrue(dump.loaded_partition_graphs())
            self.assertAllClose([[1.0, 18.0, 4.0, 2.0, 2.0, 3.0, 2.0, 5.0, -3.0, 7.0, 0.85714286, 8.97959184, 1.0, 1.0, 18.0]], dump.get_tensors('numeric_summary/a/read', 0, 'DebugNumericSummary'))

    def testDebugNumericSummaryOnUninitializedTensorGivesCorrectResult(self):
        if False:
            for i in range(10):
                print('nop')
        with session.Session() as sess:
            a = variable_v1.VariableV1([42], dtype=np.float32, name='numeric_summary_uninit/a')
            (_, dump) = self._debug_run_and_get_dump(sess, a.initializer, debug_ops=['DebugNumericSummary'])
            self.assertTrue(dump.loaded_partition_graphs())
            numeric_summary = dump.get_tensors('numeric_summary_uninit/a', 0, 'DebugNumericSummary')[0]
            self.assertAllClose([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], numeric_summary[0:8])
            self.assertAllClose([1.0, 1.0, 1.0], numeric_summary[12:])
            self.assertTrue(np.isinf(numeric_summary[8]))
            self.assertGreater(numeric_summary[8], 0.0)
            self.assertTrue(np.isinf(numeric_summary[9]))
            self.assertLess(numeric_summary[9], 0.0)
            self.assertTrue(np.isnan(numeric_summary[10]))
            self.assertTrue(np.isnan(numeric_summary[11]))

    def testDebugNumericSummaryFailureIsToleratedWhenOrdered(self):
        if False:
            while True:
                i = 10
        with session.Session() as sess:
            a = variable_v1.VariableV1('1', name='a')
            b = variable_v1.VariableV1('3', name='b')
            c = variable_v1.VariableV1('2', name='c')
            d = math_ops.add(a, b, name='d')
            e = math_ops.add(d, c, name='e')
            n = parsing_ops.string_to_number(e, name='n')
            m = math_ops.add(n, n, name='m')
            sess.run(variables.global_variables_initializer())
            run_metadata = config_pb2.RunMetadata()
            run_options = config_pb2.RunOptions(output_partition_graphs=True)
            debug_utils.watch_graph(run_options, sess.graph, debug_ops=['DebugNumericSummary'], debug_urls=self._debug_urls())
            with self.assertRaises(errors.FailedPreconditionError):
                sess.run(m, options=run_options, run_metadata=run_metadata)
            (m_result, dump) = self._debug_run_and_get_dump(sess, m, debug_ops=['DebugNumericSummary'], tolerate_debug_op_creation_failures=True)
            self.assertEqual(264, m_result)
            self.assertIn('n:0:DebugNumericSummary', dump.debug_watch_keys('n'))
            self.assertIn('m:0:DebugNumericSummary', dump.debug_watch_keys('m'))

    def testDebugNumericSummaryInvalidAttributesStringAreCaught(self):
        if False:
            while True:
                i = 10
        with session.Session(config=no_rewrite_session_config()) as sess:
            a = variable_v1.VariableV1(10.0, name='a')
            b = variable_v1.VariableV1(0.0, name='b')
            c = variable_v1.VariableV1(0.0, name='c')
            x = math_ops.divide(a, b, name='x')
            y = math_ops.multiply(x, c, name='y')
            sess.run(variables.global_variables_initializer())
            run_metadata = config_pb2.RunMetadata()
            run_options = config_pb2.RunOptions(output_partition_graphs=True)
            debug_utils.watch_graph(run_options, sess.graph, debug_ops=['DebugNumericSummary(foo=1.0)'], debug_urls=self._debug_urls())
            with self.assertRaisesRegexp(errors.FailedPreconditionError, '1 attribute key\\(s\\) were not valid for debug node __dbg_.:0_0_DebugNumericSummary: foo'):
                sess.run(y, options=run_options, run_metadata=run_metadata)
            run_options = config_pb2.RunOptions(output_partition_graphs=True)
            debug_utils.watch_graph(run_options, sess.graph, debug_ops=['DebugNumericSummary(foo=1.0; bar=false)'], debug_urls=self._debug_urls())
            with self.assertRaisesRegexp(errors.FailedPreconditionError, '2 attribute key\\(s\\) were not valid for debug node __dbg_.:0_0_DebugNumericSummary:'):
                sess.run(y, options=run_options, run_metadata=run_metadata)
            run_options = config_pb2.RunOptions(output_partition_graphs=True)
            debug_utils.watch_graph(run_options, sess.graph, debug_ops=['DebugNumericSummary(foo=1.0; mute_if_healthy=true)'], debug_urls=self._debug_urls())
            with self.assertRaisesRegexp(errors.FailedPreconditionError, '1 attribute key\\(s\\) were not valid for debug node __dbg_.:0_0_DebugNumericSummary: foo'):
                sess.run(y, options=run_options, run_metadata=run_metadata)

    def testDebugNumericSummaryMuteOnHealthyMutesOnlyHealthyTensorDumps(self):
        if False:
            print('Hello World!')
        with session.Session(config=no_rewrite_session_config()) as sess:
            a = variable_v1.VariableV1(10.0, name='a')
            b = variable_v1.VariableV1(0.0, name='b')
            c = variable_v1.VariableV1(0.0, name='c')
            x = math_ops.divide(a, b, name='x')
            y = math_ops.multiply(x, c, name='y')
            sess.run(variables.global_variables_initializer())
            (_, dump) = self._debug_run_and_get_dump(sess, y, debug_ops=['DebugNumericSummary(mute_if_healthy=true)'], validate=False)
            self.assertLessEqual(2, dump.size)
            self.assertAllClose([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, np.inf, -np.inf, np.nan, np.nan, 1.0, 0.0]], dump.get_tensors('x', 0, 'DebugNumericSummary'))
            self.assertAllClose([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.inf, -np.inf, np.nan, np.nan, 1.0, 0.0]], dump.get_tensors('y', 0, 'DebugNumericSummary'))
            file_io.delete_recursively(self._dump_root)
            (_, dump) = self._debug_run_and_get_dump(sess, y, debug_ops=['DebugNumericSummary()'])
            self.assertLessEqual(8, dump.size)

    def testDebugNumericSummaryMuteOnHealthyAndCustomBoundsWork(self):
        if False:
            print('Hello World!')
        with session.Session() as sess:
            a = variable_v1.VariableV1([10.0, 10.0], name='a')
            b = variable_v1.VariableV1([10.0, 2.0], name='b')
            x = math_ops.add(a, b, name='x')
            y = math_ops.divide(x, b, name='y')
            sess.run(variables.global_variables_initializer())
            (_, dump) = self._debug_run_and_get_dump(sess, y, debug_ops=['DebugNumericSummary(mute_if_healthy=true; upper_bound=11.0)'], validate=False)
            self.assertEqual(1, dump.size)
            self.assertAllClose([[1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 12.0, 20.0, 16.0, 16.0, 1.0, 1.0, 2.0]], dump.get_tensors('x', 0, 'DebugNumericSummary'))

    def testDebugQueueOpsDoesNotoErrorOut(self):
        if False:
            while True:
                i = 10
        with session.Session() as sess:
            q = data_flow_ops.FIFOQueue(3, 'float', name='fifo_queue')
            q_init = q.enqueue_many(([101.0, 202.0, 303.0],), name='enqueue_many')
            (_, dump) = self._debug_run_and_get_dump(sess, q_init)
            self.assertTrue(dump.loaded_partition_graphs())
            fifo_queue_tensor = dump.get_tensors('fifo_queue', 0, 'DebugIdentity')[0]
            self.assertIsInstance(fifo_queue_tensor, debug_data.InconvertibleTensorProto)
            self.assertTrue(fifo_queue_tensor.initialized)
            self.assertAllClose([101.0, 202.0, 303.0], dump.get_tensors('enqueue_many/component_0', 0, 'DebugIdentity')[0])

    def testLookUpNodePythonTracebackWorks(self):
        if False:
            print('Hello World!')
        with session.Session() as sess:
            u_init = constant_op.constant(10.0)
            u = variable_v1.VariableV1(u_init, name='traceback/u')
            v_init = constant_op.constant(20.0)
            v = variable_v1.VariableV1(v_init, name='traceback/v')
            w = math_ops.multiply(u, v, name='traceback/w')
            sess.run(variables.global_variables_initializer())
            (_, dump) = self._debug_run_and_get_dump(sess, w)
            with self.assertRaisesRegexp(LookupError, 'Python graph is not available for traceback lookup'):
                dump.node_traceback('traceback/w')
            dump.set_python_graph(sess.graph)
            with self.assertRaisesRegexp(KeyError, 'Cannot find node \\"foo\\" in Python graph'):
                dump.node_traceback('foo')
            traceback = dump.node_traceback('traceback/w')
            self.assertIsInstance(traceback, tuple)
            self.assertGreater(len(traceback), 0)
            for trace in traceback:
                self.assertIsInstance(trace, tuple)
            traceback = dump.node_traceback('traceback/w:0')
            self.assertIsInstance(traceback, tuple)
            self.assertGreater(len(traceback), 0)
            for trace in traceback:
                self.assertIsInstance(trace, tuple)

class DebugConcurrentRunCallsTest(test_util.TensorFlowTestCase):
    """Test for debugging concurrent Session.run() calls."""

    def _get_concurrent_debug_urls(self):
        if False:
            while True:
                i = 10
        'Abstract method to generate debug URLs for concurrent debugged runs.'
        raise NotImplementedError('_get_concurrent_debug_urls is not implemented in the base test class')

    def testDebugConcurrentVariableUpdates(self):
        if False:
            while True:
                i = 10
        if test.is_gpu_available():
            self.skipTest('No testing concurrent runs on a single GPU.')
        with session.Session() as sess:
            v = variable_v1.VariableV1(30.0, name='v')
            constants = []
            for i in range(self._num_concurrent_runs):
                constants.append(constant_op.constant(1.0, name='c%d' % i))
            incs = [state_ops.assign_add(v, c, use_locking=True, name='inc%d' % i) for (i, c) in enumerate(constants)]
            sess.run(v.initializer)
            concurrent_debug_urls = self._get_concurrent_debug_urls()

            def inc_job(index):
                if False:
                    for i in range(10):
                        print('nop')
                run_options = config_pb2.RunOptions(output_partition_graphs=True)
                debug_utils.watch_graph(run_options, sess.graph, debug_urls=concurrent_debug_urls[index])
                for _ in range(100):
                    sess.run(incs[index], options=run_options)
            inc_threads = []
            for index in range(self._num_concurrent_runs):
                inc_thread = threading.Thread(target=functools.partial(inc_job, index))
                inc_thread.start()
                inc_threads.append(inc_thread)
            for inc_thread in inc_threads:
                inc_thread.join()
            self.assertAllClose(30.0 + 1.0 * self._num_concurrent_runs * 100, sess.run(v))
            all_session_run_indices = []
            for index in range(self._num_concurrent_runs):
                dump = debug_data.DebugDumpDir(self._dump_roots[index])
                self.assertTrue(dump.loaded_partition_graphs())
                v_data = dump.get_tensors('v', 0, 'DebugIdentity')
                self.assertEqual(100, len(v_data))
                core_metadata_files = glob.glob(os.path.join(self._dump_roots[index], '_tfdbg_core*'))
                timestamps = []
                session_run_indices = []
                executor_step_indices = []
                for core_metadata_file in core_metadata_files:
                    with open(core_metadata_file, 'rb') as f:
                        event = event_pb2.Event()
                        event.ParseFromString(f.read())
                        core_metadata = debug_data.extract_core_metadata_from_event_proto(event)
                        timestamps.append(event.wall_time)
                        session_run_indices.append(core_metadata.session_run_index)
                        executor_step_indices.append(core_metadata.executor_step_index)
                all_session_run_indices.extend(session_run_indices)
                executor_step_indices = zip(timestamps, executor_step_indices)
                executor_step_indices = sorted(executor_step_indices, key=lambda x: x[0])
                for i in range(len(executor_step_indices) - 1):
                    self.assertEqual(executor_step_indices[i][1] + 1, executor_step_indices[i + 1][1])
                session_run_indices = zip(timestamps, session_run_indices)
                session_run_indices = sorted(session_run_indices, key=lambda x: x[0])
                for i in range(len(session_run_indices) - 1):
                    self.assertGreater(session_run_indices[i + 1][1], session_run_indices[i][1])
            self.assertEqual(len(all_session_run_indices), len(set(all_session_run_indices)))
if __name__ == '__main__':
    googletest.main()