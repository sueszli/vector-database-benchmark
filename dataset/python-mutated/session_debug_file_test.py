"""Tests for debugger functionalities in tf.Session with file:// URLs."""
import os
import tempfile
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.debug.lib import session_debug_testlib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import googletest

@test_util.run_v1_only('b/120545219')
class SessionDebugFileTest(session_debug_testlib.SessionDebugTestBase):

    def _debug_urls(self, run_number=None):
        if False:
            return 10
        return ['file://%s' % self._debug_dump_dir(run_number=run_number)]

    def _debug_dump_dir(self, run_number=None):
        if False:
            return 10
        if run_number is None:
            return self._dump_root
        else:
            return os.path.join(self._dump_root, 'run_%d' % run_number)

    def testAllowsDifferentWatchesOnDifferentRuns(self):
        if False:
            return 10
        'Test watching different tensors on different runs of the same graph.'
        with session.Session(config=session_debug_testlib.no_rewrite_session_config()) as sess:
            u_init_val = [[5.0, 3.0], [-1.0, 0.0]]
            v_init_val = [[2.0], [-1.0]]
            u_name = 'diff_Watch/u'
            v_name = 'diff_Watch/v'
            u_init = constant_op.constant(u_init_val, shape=[2, 2])
            u = variable_v1.VariableV1(u_init, name=u_name)
            v_init = constant_op.constant(v_init_val, shape=[2, 1])
            v = variable_v1.VariableV1(v_init, name=v_name)
            w = math_ops.matmul(u, v, name='diff_Watch/matmul')
            u.initializer.run()
            v.initializer.run()
            for i in range(2):
                run_options = config_pb2.RunOptions(output_partition_graphs=True)
                run_dump_root = self._debug_dump_dir(run_number=i)
                debug_urls = self._debug_urls(run_number=i)
                if i == 0:
                    debug_utils.add_debug_tensor_watch(run_options, '%s/read' % u_name, 0, debug_urls=debug_urls)
                else:
                    debug_utils.add_debug_tensor_watch(run_options, '%s/read' % v_name, 0, debug_urls=debug_urls)
                run_metadata = config_pb2.RunMetadata()
                sess.run(w, options=run_options, run_metadata=run_metadata)
                self.assertEqual(self._expected_partition_graph_count, len(run_metadata.partition_graphs))
                dump = debug_data.DebugDumpDir(run_dump_root, partition_graphs=run_metadata.partition_graphs)
                self.assertTrue(dump.loaded_partition_graphs())
                self.assertEqual(1, dump.size)
                if i == 0:
                    self.assertAllClose([u_init_val], dump.get_tensors('%s/read' % u_name, 0, 'DebugIdentity'))
                    self.assertGreaterEqual(dump.get_rel_timestamps('%s/read' % u_name, 0, 'DebugIdentity')[0], 0)
                else:
                    self.assertAllClose([v_init_val], dump.get_tensors('%s/read' % v_name, 0, 'DebugIdentity'))
                    self.assertGreaterEqual(dump.get_rel_timestamps('%s/read' % v_name, 0, 'DebugIdentity')[0], 0)

class SessionDebugConcurrentTest(session_debug_testlib.DebugConcurrentRunCallsTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self._num_concurrent_runs = 3
        self._dump_roots = []
        for _ in range(self._num_concurrent_runs):
            self._dump_roots.append(tempfile.mkdtemp())

    def tearDown(self):
        if False:
            print('Hello World!')
        ops.reset_default_graph()
        for dump_root in self._dump_roots:
            if os.path.isdir(dump_root):
                file_io.delete_recursively(dump_root)

    def _get_concurrent_debug_urls(self):
        if False:
            while True:
                i = 10
        return ['file://%s' % dump_root for dump_root in self._dump_roots]
if __name__ == '__main__':
    googletest.main()