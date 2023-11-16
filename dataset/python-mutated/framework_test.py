"""Framework of debug-wrapped sessions."""
import os
import tempfile
import threading
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.wrappers import framework
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import monitored_session
from tensorflow.python.util import tf_inspect

class TestDebugWrapperSession(framework.BaseDebugWrapperSession):
    """A concrete implementation of BaseDebugWrapperSession for test."""

    def __init__(self, sess, dump_root, observer, thread_name_filter=None):
        if False:
            i = 10
            return i + 15
        self._dump_root = dump_root
        self._obs = observer
        framework.BaseDebugWrapperSession.__init__(self, sess, thread_name_filter=thread_name_filter)

    def on_session_init(self, request):
        if False:
            for i in range(10):
                print('nop')
        'Override abstract on-session-init callback method.'
        self._obs['sess_init_count'] += 1
        self._obs['request_sess'] = request.session
        return framework.OnSessionInitResponse(framework.OnSessionInitAction.PROCEED)

    def on_run_start(self, request):
        if False:
            print('Hello World!')
        'Override abstract on-run-start callback method.'
        self._obs['on_run_start_count'] += 1
        self._obs['run_fetches'] = request.fetches
        self._obs['run_feed_dict'] = request.feed_dict
        return framework.OnRunStartResponse(framework.OnRunStartAction.DEBUG_RUN, ['file://' + self._dump_root])

    def on_run_end(self, request):
        if False:
            while True:
                i = 10
        'Override abstract on-run-end callback method.'
        self._obs['on_run_end_count'] += 1
        self._obs['performed_action'] = request.performed_action
        self._obs['tf_error'] = request.tf_error
        return framework.OnRunEndResponse()

class TestDebugWrapperSessionBadAction(framework.BaseDebugWrapperSession):
    """A concrete implementation of BaseDebugWrapperSession for test.

  This class intentionally puts a bad action value in OnSessionInitResponse
  and/or in OnRunStartAction to test the handling of such invalid cases.
  """

    def __init__(self, sess, bad_init_action=None, bad_run_start_action=None, bad_debug_urls=None):
        if False:
            i = 10
            return i + 15
        'Constructor.\n\n    Args:\n      sess: The TensorFlow Session object to be wrapped.\n      bad_init_action: (str) bad action value to be returned during the\n        on-session-init callback.\n      bad_run_start_action: (str) bad action value to be returned during the\n        the on-run-start callback.\n      bad_debug_urls: Bad URL values to be returned during the on-run-start\n        callback.\n    '
        self._bad_init_action = bad_init_action
        self._bad_run_start_action = bad_run_start_action
        self._bad_debug_urls = bad_debug_urls
        framework.BaseDebugWrapperSession.__init__(self, sess)

    def on_session_init(self, request):
        if False:
            print('Hello World!')
        if self._bad_init_action:
            return framework.OnSessionInitResponse(self._bad_init_action)
        else:
            return framework.OnSessionInitResponse(framework.OnSessionInitAction.PROCEED)

    def on_run_start(self, request):
        if False:
            print('Hello World!')
        debug_urls = self._bad_debug_urls or []
        if self._bad_run_start_action:
            return framework.OnRunStartResponse(self._bad_run_start_action, debug_urls)
        else:
            return framework.OnRunStartResponse(framework.OnRunStartAction.DEBUG_RUN, debug_urls)

    def on_run_end(self, request):
        if False:
            while True:
                i = 10
        return framework.OnRunEndResponse()

@test_util.run_v1_only('Sessions are not available in TF 2.x')
class DebugWrapperSessionTest(test_util.TensorFlowTestCase):

    def _no_rewrite_session_config(self):
        if False:
            return 10
        rewriter_config = rewriter_config_pb2.RewriterConfig(disable_model_pruning=True)
        graph_options = config_pb2.GraphOptions(rewrite_options=rewriter_config)
        return config_pb2.ConfigProto(graph_options=graph_options)

    def setUp(self):
        if False:
            print('Hello World!')
        self._observer = {'sess_init_count': 0, 'request_sess': None, 'on_run_start_count': 0, 'run_fetches': None, 'run_feed_dict': None, 'on_run_end_count': 0, 'performed_action': None, 'tf_error': None}
        self._dump_root = tempfile.mkdtemp()
        self._sess = session.Session(config=self._no_rewrite_session_config())
        self._a_init_val = np.array([[5.0, 3.0], [-1.0, 0.0]])
        self._b_init_val = np.array([[2.0], [-1.0]])
        self._c_val = np.array([[-4.0], [6.0]])
        self._a_init = constant_op.constant(self._a_init_val, shape=[2, 2], name='a_init')
        self._b_init = constant_op.constant(self._b_init_val, shape=[2, 1], name='b_init')
        self._ph = array_ops.placeholder(dtype=dtypes.float64, name='ph')
        self._a = variables.Variable(self._a_init, name='a1')
        self._b = variables.Variable(self._b_init, name='b')
        self._c = constant_op.constant(self._c_val, shape=[2, 1], name='c')
        self._p = math_ops.matmul(self._a, self._b, name='p1')
        self._q = math_ops.matmul(self._a, self._ph, name='q')
        self._s = math_ops.add(self._p, self._c, name='s')
        self._sess.run(self._a.initializer)
        self._sess.run(self._b.initializer)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        if os.path.isdir(self._dump_root):
            file_io.delete_recursively(self._dump_root)
        ops.reset_default_graph()

    def testSessionInit(self):
        if False:
            return 10
        self.assertEqual(0, self._observer['sess_init_count'])
        wrapper_sess = TestDebugWrapperSession(self._sess, self._dump_root, self._observer)
        self.assertEqual(1, self._observer['sess_init_count'])
        self.assertEqual(self._sess, self._observer['request_sess'])
        self.assertTrue(isinstance(wrapper_sess, session.SessionInterface))
        self.assertEqual(self._sess.sess_str, wrapper_sess.sess_str)
        self.assertEqual(self._sess.graph, wrapper_sess.graph)
        self.assertEqual(self._sess.graph_def, wrapper_sess.graph_def)
        with self.assertRaises(NotImplementedError):
            wrapper_sess.partial_run_setup(self._p)

    def testInteractiveSessionInit(self):
        if False:
            return 10
        'The wrapper should work also on other subclasses of session.Session.'
        TestDebugWrapperSession(session.InteractiveSession(), self._dump_root, self._observer)

    def testSessionRun(self):
        if False:
            while True:
                i = 10
        wrapper = TestDebugWrapperSession(self._sess, self._dump_root, self._observer)
        self.assertEqual(0, self._observer['on_run_start_count'])
        self.assertEqual(0, self._observer['on_run_end_count'])
        s = wrapper.run(self._s)
        self.assertAllClose(np.array([[3.0], [4.0]]), s)
        self.assertEqual(1, self._observer['on_run_start_count'])
        self.assertEqual(self._s, self._observer['run_fetches'])
        self.assertIsNone(self._observer['run_feed_dict'])
        dump = debug_data.DebugDumpDir(self._dump_root)
        self.assertEqual(7, len(dump.dumped_tensor_data))
        self.assertEqual(1, self._observer['on_run_end_count'])
        self.assertEqual(framework.OnRunStartAction.DEBUG_RUN, self._observer['performed_action'])
        self.assertIsNone(self._observer['tf_error'])

    def testSessionInitInvalidSessionType(self):
        if False:
            while True:
                i = 10
        'Attempt to wrap a non-Session-type object should cause an exception.'
        wrapper = TestDebugWrapperSessionBadAction(self._sess)
        with self.assertRaisesRegex(TypeError, 'Expected type .*; got type .*'):
            TestDebugWrapperSessionBadAction(wrapper)

    def testSessionInitBadActionValue(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(ValueError, 'Invalid OnSessionInitAction value: nonsense_action'):
            TestDebugWrapperSessionBadAction(self._sess, bad_init_action='nonsense_action')

    def testRunStartBadActionValue(self):
        if False:
            return 10
        wrapper = TestDebugWrapperSessionBadAction(self._sess, bad_run_start_action='nonsense_action')
        with self.assertRaisesRegex(ValueError, 'Invalid OnRunStartAction value: nonsense_action'):
            wrapper.run(self._s)

    def testRunStartBadURLs(self):
        if False:
            return 10
        wrapper = TestDebugWrapperSessionBadAction(self._sess, bad_debug_urls='file://foo')
        with self.assertRaisesRegex(TypeError, 'Expected type .*; got type .*'):
            wrapper.run(self._s)

    def testErrorDuringRun(self):
        if False:
            for i in range(10):
                print('nop')
        wrapper = TestDebugWrapperSession(self._sess, self._dump_root, self._observer)
        self.assertAllClose(np.array([[11.0], [-1.0]]), wrapper.run(self._q, feed_dict={self._ph: np.array([[1.0], [2.0]])}))
        self.assertEqual(1, self._observer['on_run_end_count'])
        self.assertIsNone(self._observer['tf_error'])
        wrapper.run(self._q, feed_dict={self._ph: np.array([[1.0], [2.0], [3.0]])})
        self.assertEqual(2, self._observer['on_run_end_count'])
        self.assertTrue(isinstance(self._observer['tf_error'], errors.InvalidArgumentError))

    def testUsingWrappedSessionShouldWorkAsContextManager(self):
        if False:
            for i in range(10):
                print('nop')
        wrapper = TestDebugWrapperSession(self._sess, self._dump_root, self._observer)
        with wrapper as sess:
            self.assertAllClose([[3.0], [4.0]], self._s)
            self.assertEqual(1, self._observer['on_run_start_count'])
            self.assertEqual([self._s], self._observer['run_fetches'])
            self.assertEqual(1, self._observer['on_run_end_count'])
            self.assertAllClose([[11.0], [-1.0]], sess.run(self._q, feed_dict={self._ph: np.array([[1.0], [2.0]])}))
            self.assertEqual(2, self._observer['on_run_start_count'])
            self.assertEqual(self._q, self._observer['run_fetches'])
            self.assertEqual(2, self._observer['on_run_end_count'])

    def testUsingWrappedSessionShouldSupportEvalWithAsDefault(self):
        if False:
            while True:
                i = 10
        wrapper = TestDebugWrapperSession(self._sess, self._dump_root, self._observer)
        with wrapper.as_default():
            foo = constant_op.constant(42, name='foo')
            self.assertEqual(42, self.evaluate(foo))
            self.assertEqual([foo], self._observer['run_fetches'])

    def testWrapperShouldSupportSessionClose(self):
        if False:
            i = 10
            return i + 15
        wrapper = TestDebugWrapperSession(self._sess, self._dump_root, self._observer)
        wrapper.close()

    def testWrapperThreadNameFilterMainThread(self):
        if False:
            for i in range(10):
                print('nop')
        wrapper = TestDebugWrapperSession(self._sess, self._dump_root, self._observer, thread_name_filter='MainThread')
        child_run_output = []

        def child_thread_job():
            if False:
                while True:
                    i = 10
            child_run_output.append(wrapper.run(self._b_init))
        thread = threading.Thread(name='ChildThread', target=child_thread_job)
        thread.start()
        self.assertAllClose(self._a_init_val, wrapper.run(self._a_init))
        thread.join()
        self.assertAllClose([self._b_init_val], child_run_output)
        dump = debug_data.DebugDumpDir(self._dump_root)
        self.assertEqual(1, dump.size)
        self.assertEqual('a_init', dump.dumped_tensor_data[0].node_name)

    def testWrapperThreadNameFilterChildThread(self):
        if False:
            print('Hello World!')
        wrapper = TestDebugWrapperSession(self._sess, self._dump_root, self._observer, thread_name_filter='Child.*')
        child_run_output = []

        def child_thread_job():
            if False:
                i = 10
                return i + 15
            child_run_output.append(wrapper.run(self._b_init))
        thread = threading.Thread(name='ChildThread', target=child_thread_job)
        thread.start()
        self.assertAllClose(self._a_init_val, wrapper.run(self._a_init))
        thread.join()
        self.assertAllClose([self._b_init_val], child_run_output)
        dump = debug_data.DebugDumpDir(self._dump_root)
        self.assertEqual(1, dump.size)
        self.assertEqual('b_init', dump.dumped_tensor_data[0].node_name)

    def testWrapperThreadNameFilterBothThreads(self):
        if False:
            for i in range(10):
                print('nop')
        wrapper = TestDebugWrapperSession(self._sess, self._dump_root, self._observer, thread_name_filter=None)
        child_run_output = []

        def child_thread_job():
            if False:
                print('Hello World!')
            child_run_output.append(wrapper.run(self._b_init))
        thread = threading.Thread(name='ChildThread', target=child_thread_job)
        thread.start()
        self.assertAllClose(self._a_init_val, wrapper.run(self._a_init))
        thread.join()
        self.assertAllClose([self._b_init_val], child_run_output)
        dump = debug_data.DebugDumpDir(self._dump_root, validate=False)
        self.assertEqual(2, dump.size)
        self.assertItemsEqual(['a_init', 'b_init'], [datum.node_name for datum in dump.dumped_tensor_data])

def _is_public_method_name(method_name):
    if False:
        return 10
    return method_name.startswith('__') and method_name.endswith('__') or not method_name.startswith('_')

class SessionWrapperPublicMethodParityTest(test_util.TensorFlowTestCase):

    def testWrapperHasAllPublicMethodsOfSession(self):
        if False:
            return 10
        session_public_methods = [method_tuple[0] for method_tuple in tf_inspect.getmembers(session.Session, predicate=tf_inspect.ismethod) if _is_public_method_name(method_tuple[0])]
        wrapper_public_methods = [method_tuple[0] for method_tuple in tf_inspect.getmembers(framework.BaseDebugWrapperSession, predicate=tf_inspect.ismethod) if _is_public_method_name(method_tuple[0])]
        missing_public_methods = [method for method in session_public_methods if method not in wrapper_public_methods]
        self.assertFalse(missing_public_methods)

    def testWrapperHasAllPublicMethodsOfMonitoredSession(self):
        if False:
            print('Hello World!')
        session_public_methods = [method_tuple[0] for method_tuple in tf_inspect.getmembers(monitored_session.MonitoredSession, predicate=tf_inspect.ismethod) if _is_public_method_name(method_tuple[0])]
        wrapper_public_methods = [method_tuple[0] for method_tuple in tf_inspect.getmembers(framework.BaseDebugWrapperSession, predicate=tf_inspect.ismethod) if _is_public_method_name(method_tuple[0])]
        missing_public_methods = [method for method in session_public_methods if method not in wrapper_public_methods]
        self.assertFalse(missing_public_methods)
if __name__ == '__main__':
    googletest.main()