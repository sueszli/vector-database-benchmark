"""Unit Tests for classes in dumping_wrapper.py."""
import glob
import os
import tempfile
import threading
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.wrappers import dumping_wrapper
from tensorflow.python.debug.wrappers import framework
from tensorflow.python.debug.wrappers import hooks
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import gfile
from tensorflow.python.platform import googletest
from tensorflow.python.training import monitored_session

@test_util.run_v1_only('b/120545219')
class DumpingDebugWrapperSessionTest(test_util.TensorFlowTestCase):

    def setUp(self):
        if False:
            return 10
        self.session_root = tempfile.mkdtemp()
        self.v = variable_v1.VariableV1(10.0, dtype=dtypes.float32, name='v')
        self.delta = constant_op.constant(1.0, dtype=dtypes.float32, name='delta')
        self.eta = constant_op.constant(-1.4, dtype=dtypes.float32, name='eta')
        self.inc_v = state_ops.assign_add(self.v, self.delta, name='inc_v')
        self.dec_v = state_ops.assign_add(self.v, self.eta, name='dec_v')
        self.ph = array_ops.placeholder(dtypes.float32, shape=(), name='ph')
        self.inc_w_ph = state_ops.assign_add(self.v, self.ph, name='inc_w_ph')
        self.sess = session.Session()
        self.sess.run(self.v.initializer)

    def tearDown(self):
        if False:
            print('Hello World!')
        ops.reset_default_graph()
        if os.path.isdir(self.session_root):
            file_io.delete_recursively(self.session_root)

    def _assert_correct_run_subdir_naming(self, run_subdir):
        if False:
            return 10
        self.assertStartsWith(run_subdir, 'run_')
        self.assertEqual(2, run_subdir.count('_'))
        self.assertGreater(int(run_subdir.split('_')[1]), 0)

    def testConstructWrapperWithExistingNonEmptyRootDirRaisesException(self):
        if False:
            i = 10
            return i + 15
        dir_path = os.path.join(self.session_root, 'foo')
        os.mkdir(dir_path)
        self.assertTrue(os.path.isdir(dir_path))
        with self.assertRaisesRegex(ValueError, 'session_root path points to a non-empty directory'):
            dumping_wrapper.DumpingDebugWrapperSession(session.Session(), session_root=self.session_root)

    def testConstructWrapperWithExistingFileDumpRootRaisesException(self):
        if False:
            i = 10
            return i + 15
        file_path = os.path.join(self.session_root, 'foo')
        open(file_path, 'a').close()
        self.assertTrue(gfile.Exists(file_path))
        self.assertFalse(gfile.IsDirectory(file_path))
        with self.assertRaisesRegex(ValueError, 'session_root path points to a file'):
            dumping_wrapper.DumpingDebugWrapperSession(session.Session(), session_root=file_path)

    def testConstructWrapperWithNonexistentSessionRootCreatesDirectory(self):
        if False:
            return 10
        new_dir_path = os.path.join(tempfile.mkdtemp(), 'new_dir')
        dumping_wrapper.DumpingDebugWrapperSession(session.Session(), session_root=new_dir_path)
        self.assertTrue(gfile.IsDirectory(new_dir_path))
        gfile.DeleteRecursively(new_dir_path)

    def testDumpingOnASingleRunWorks(self):
        if False:
            for i in range(10):
                print('nop')
        sess = dumping_wrapper.DumpingDebugWrapperSession(self.sess, session_root=self.session_root)
        sess.run(self.inc_v)
        dump_dirs = glob.glob(os.path.join(self.session_root, 'run_*'))
        self.assertEqual(1, len(dump_dirs))
        self._assert_correct_run_subdir_naming(os.path.basename(dump_dirs[0]))
        dump = debug_data.DebugDumpDir(dump_dirs[0])
        self.assertAllClose([10.0], dump.get_tensors('v', 0, 'DebugIdentity'))
        self.assertEqual(repr(self.inc_v), dump.run_fetches_info)
        self.assertEqual(repr(None), dump.run_feed_keys_info)

    def testDumpingOnASingleRunWorksWithRelativePathForDebugDumpDir(self):
        if False:
            print('Hello World!')
        sess = dumping_wrapper.DumpingDebugWrapperSession(self.sess, session_root=self.session_root)
        sess.run(self.inc_v)
        dump_dirs = glob.glob(os.path.join(self.session_root, 'run_*'))
        cwd = os.getcwd()
        try:
            os.chdir(self.session_root)
            dump = debug_data.DebugDumpDir(os.path.relpath(dump_dirs[0], self.session_root))
            self.assertAllClose([10.0], dump.get_tensors('v', 0, 'DebugIdentity'))
        finally:
            os.chdir(cwd)

    def testDumpingOnASingleRunWithFeedDictWorks(self):
        if False:
            for i in range(10):
                print('nop')
        sess = dumping_wrapper.DumpingDebugWrapperSession(self.sess, session_root=self.session_root)
        feed_dict = {self.ph: 3.2}
        sess.run(self.inc_w_ph, feed_dict=feed_dict)
        dump_dirs = glob.glob(os.path.join(self.session_root, 'run_*'))
        self.assertEqual(1, len(dump_dirs))
        self._assert_correct_run_subdir_naming(os.path.basename(dump_dirs[0]))
        dump = debug_data.DebugDumpDir(dump_dirs[0])
        self.assertAllClose([10.0], dump.get_tensors('v', 0, 'DebugIdentity'))
        self.assertEqual(repr(self.inc_w_ph), dump.run_fetches_info)
        self.assertEqual(repr(feed_dict.keys()), dump.run_feed_keys_info)

    def testDumpingOnMultipleRunsWorks(self):
        if False:
            for i in range(10):
                print('nop')
        sess = dumping_wrapper.DumpingDebugWrapperSession(self.sess, session_root=self.session_root)
        for _ in range(3):
            sess.run(self.inc_v)
        dump_dirs = glob.glob(os.path.join(self.session_root, 'run_*'))
        dump_dirs = sorted(dump_dirs, key=lambda x: int(os.path.basename(x).split('_')[1]))
        self.assertEqual(3, len(dump_dirs))
        for (i, dump_dir) in enumerate(dump_dirs):
            self._assert_correct_run_subdir_naming(os.path.basename(dump_dir))
            dump = debug_data.DebugDumpDir(dump_dir)
            self.assertAllClose([10.0 + 1.0 * i], dump.get_tensors('v', 0, 'DebugIdentity'))
            self.assertEqual(repr(self.inc_v), dump.run_fetches_info)
            self.assertEqual(repr(None), dump.run_feed_keys_info)

    def testUsingNonCallableAsWatchFnRaisesTypeError(self):
        if False:
            for i in range(10):
                print('nop')
        bad_watch_fn = 'bad_watch_fn'
        with self.assertRaisesRegex(TypeError, 'watch_fn is not callable'):
            dumping_wrapper.DumpingDebugWrapperSession(self.sess, session_root=self.session_root, watch_fn=bad_watch_fn)

    def testDumpingWithLegacyWatchFnOnFetchesWorks(self):
        if False:
            print('Hello World!')
        'Use a watch_fn that returns different allowlists for different runs.'

        def watch_fn(fetches, feeds):
            if False:
                print('Hello World!')
            del feeds
            if fetches.name == 'inc_v:0':
                return ('DebugIdentity', '.*', '.*')
            else:
                return ('DebugIdentity', '$^', '$^')
        sess = dumping_wrapper.DumpingDebugWrapperSession(self.sess, session_root=self.session_root, watch_fn=watch_fn)
        for _ in range(3):
            sess.run(self.inc_v)
            sess.run(self.dec_v)
        dump_dirs = glob.glob(os.path.join(self.session_root, 'run_*'))
        dump_dirs = sorted(dump_dirs, key=lambda x: int(os.path.basename(x).split('_')[1]))
        self.assertEqual(6, len(dump_dirs))
        for (i, dump_dir) in enumerate(dump_dirs):
            self._assert_correct_run_subdir_naming(os.path.basename(dump_dir))
            dump = debug_data.DebugDumpDir(dump_dir)
            if i % 2 == 0:
                self.assertGreater(dump.size, 0)
                self.assertAllClose([10.0 - 0.4 * (i / 2)], dump.get_tensors('v', 0, 'DebugIdentity'))
                self.assertEqual(repr(self.inc_v), dump.run_fetches_info)
                self.assertEqual(repr(None), dump.run_feed_keys_info)
            else:
                self.assertEqual(0, dump.size)
                self.assertEqual(repr(self.dec_v), dump.run_fetches_info)
                self.assertEqual(repr(None), dump.run_feed_keys_info)

    def testDumpingWithLegacyWatchFnWithNonDefaultDebugOpsWorks(self):
        if False:
            while True:
                i = 10
        'Use a watch_fn that specifies non-default debug ops.'

        def watch_fn(fetches, feeds):
            if False:
                while True:
                    i = 10
            del fetches, feeds
            return (['DebugIdentity', 'DebugNumericSummary'], '.*', '.*')
        sess = dumping_wrapper.DumpingDebugWrapperSession(self.sess, session_root=self.session_root, watch_fn=watch_fn)
        sess.run(self.inc_v)
        dump_dirs = glob.glob(os.path.join(self.session_root, 'run_*'))
        self.assertEqual(1, len(dump_dirs))
        dump = debug_data.DebugDumpDir(dump_dirs[0])
        self.assertAllClose([10.0], dump.get_tensors('v', 0, 'DebugIdentity'))
        self.assertEqual(14, len(dump.get_tensors('v', 0, 'DebugNumericSummary')[0]))

    def testDumpingWithWatchFnWithNonDefaultDebugOpsWorks(self):
        if False:
            return 10
        'Use a watch_fn that specifies non-default debug ops.'

        def watch_fn(fetches, feeds):
            if False:
                i = 10
                return i + 15
            del fetches, feeds
            return framework.WatchOptions(debug_ops=['DebugIdentity', 'DebugNumericSummary'], node_name_regex_allowlist='^v.*', op_type_regex_allowlist='.*', tensor_dtype_regex_allowlist='.*_ref')
        sess = dumping_wrapper.DumpingDebugWrapperSession(self.sess, session_root=self.session_root, watch_fn=watch_fn)
        sess.run(self.inc_v)
        dump_dirs = glob.glob(os.path.join(self.session_root, 'run_*'))
        self.assertEqual(1, len(dump_dirs))
        dump = debug_data.DebugDumpDir(dump_dirs[0])
        self.assertAllClose([10.0], dump.get_tensors('v', 0, 'DebugIdentity'))
        self.assertEqual(14, len(dump.get_tensors('v', 0, 'DebugNumericSummary')[0]))
        dumped_nodes = [dump.node_name for dump in dump.dumped_tensor_data]
        self.assertNotIn('inc_v', dumped_nodes)
        self.assertNotIn('delta', dumped_nodes)

    def testDumpingDebugHookWithoutWatchFnWorks(self):
        if False:
            print('Hello World!')
        dumping_hook = hooks.DumpingDebugHook(self.session_root)
        mon_sess = monitored_session._HookedSession(self.sess, [dumping_hook])
        mon_sess.run(self.inc_v)
        dump_dirs = glob.glob(os.path.join(self.session_root, 'run_*'))
        self.assertEqual(1, len(dump_dirs))
        self._assert_correct_run_subdir_naming(os.path.basename(dump_dirs[0]))
        dump = debug_data.DebugDumpDir(dump_dirs[0])
        self.assertAllClose([10.0], dump.get_tensors('v', 0, 'DebugIdentity'))
        self.assertEqual(repr(self.inc_v), dump.run_fetches_info)
        self.assertEqual(repr(None), dump.run_feed_keys_info)

    def testDumpingDebugHookWithStatefulWatchFnWorks(self):
        if False:
            print('Hello World!')
        watch_fn_state = {'run_counter': 0}

        def counting_watch_fn(fetches, feed_dict):
            if False:
                return 10
            del fetches, feed_dict
            watch_fn_state['run_counter'] += 1
            if watch_fn_state['run_counter'] % 2 == 1:
                return framework.WatchOptions(debug_ops='DebugIdentity', tensor_dtype_regex_allowlist='.*_ref')
            else:
                return framework.WatchOptions(debug_ops='DebugIdentity', node_name_regex_allowlist='^$', op_type_regex_allowlist='^$')
        dumping_hook = hooks.DumpingDebugHook(self.session_root, watch_fn=counting_watch_fn)
        mon_sess = monitored_session._HookedSession(self.sess, [dumping_hook])
        for _ in range(4):
            mon_sess.run(self.inc_v)
        dump_dirs = glob.glob(os.path.join(self.session_root, 'run_*'))
        dump_dirs = sorted(dump_dirs, key=lambda x: int(os.path.basename(x).split('_')[1]))
        self.assertEqual(4, len(dump_dirs))
        for (i, dump_dir) in enumerate(dump_dirs):
            self._assert_correct_run_subdir_naming(os.path.basename(dump_dir))
            dump = debug_data.DebugDumpDir(dump_dir)
            if i % 2 == 0:
                self.assertAllClose([10.0 + 1.0 * i], dump.get_tensors('v', 0, 'DebugIdentity'))
                self.assertNotIn('delta', [datum.node_name for datum in dump.dumped_tensor_data])
            else:
                self.assertEqual(0, dump.size)
            self.assertEqual(repr(self.inc_v), dump.run_fetches_info)
            self.assertEqual(repr(None), dump.run_feed_keys_info)

    def testDumpingDebugHookWithStatefulLegacyWatchFnWorks(self):
        if False:
            print('Hello World!')
        watch_fn_state = {'run_counter': 0}

        def counting_watch_fn(fetches, feed_dict):
            if False:
                return 10
            del fetches, feed_dict
            watch_fn_state['run_counter'] += 1
            if watch_fn_state['run_counter'] % 2 == 1:
                return ('DebugIdentity', '.*', '.*')
            else:
                return ('DebugIdentity', '$^', '$^')
        dumping_hook = hooks.DumpingDebugHook(self.session_root, watch_fn=counting_watch_fn)
        mon_sess = monitored_session._HookedSession(self.sess, [dumping_hook])
        for _ in range(4):
            mon_sess.run(self.inc_v)
        dump_dirs = glob.glob(os.path.join(self.session_root, 'run_*'))
        dump_dirs = sorted(dump_dirs, key=lambda x: int(os.path.basename(x).split('_')[1]))
        self.assertEqual(4, len(dump_dirs))
        for (i, dump_dir) in enumerate(dump_dirs):
            self._assert_correct_run_subdir_naming(os.path.basename(dump_dir))
            dump = debug_data.DebugDumpDir(dump_dir)
            if i % 2 == 0:
                self.assertAllClose([10.0 + 1.0 * i], dump.get_tensors('v', 0, 'DebugIdentity'))
            else:
                self.assertEqual(0, dump.size)
            self.assertEqual(repr(self.inc_v), dump.run_fetches_info)
            self.assertEqual(repr(None), dump.run_feed_keys_info)

    def testDumpingFromMultipleThreadsObeysThreadNameFilter(self):
        if False:
            return 10
        sess = dumping_wrapper.DumpingDebugWrapperSession(self.sess, session_root=self.session_root, thread_name_filter='MainThread$')
        self.assertAllClose(1.0, sess.run(self.delta))
        child_thread_result = []

        def child_thread_job():
            if False:
                return 10
            child_thread_result.append(sess.run(self.eta))
        thread = threading.Thread(name='ChildThread', target=child_thread_job)
        thread.start()
        thread.join()
        self.assertAllClose([-1.4], child_thread_result)
        dump_dirs = glob.glob(os.path.join(self.session_root, 'run_*'))
        self.assertEqual(1, len(dump_dirs))
        dump = debug_data.DebugDumpDir(dump_dirs[0])
        self.assertEqual(1, dump.size)
        self.assertEqual('delta', dump.dumped_tensor_data[0].node_name)

    def testDumpingWrapperWithEmptyFetchWorks(self):
        if False:
            while True:
                i = 10
        sess = dumping_wrapper.DumpingDebugWrapperSession(self.sess, session_root=self.session_root)
        sess.run([])
if __name__ == '__main__':
    googletest.main()