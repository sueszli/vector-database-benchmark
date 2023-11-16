"""Debugger Wrapper Session Consisting of a Local Curses-based CLI."""
import os
import tempfile
from tensorflow.python.client import session
from tensorflow.python.debug.wrappers import dumping_wrapper
from tensorflow.python.debug.wrappers import hooks
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import monitored_session

@test_util.run_v1_only('Sessions are not available in TF 2.x')
class DumpingDebugWrapperDiskUsageLimitTest(test_util.TensorFlowTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        os.environ['TFDBG_DISK_BYTES_LIMIT'] = '10'

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.session_root = tempfile.mkdtemp()
        self.v = variables.Variable(10.0, dtype=dtypes.float32, name='v')
        self.delta = constant_op.constant(1.0, dtype=dtypes.float32, name='delta')
        self.eta = constant_op.constant(-1.4, dtype=dtypes.float32, name='eta')
        self.inc_v = state_ops.assign_add(self.v, self.delta, name='inc_v')
        self.dec_v = state_ops.assign_add(self.v, self.eta, name='dec_v')
        self.sess = session.Session()
        self.sess.run(self.v.initializer)

    def testWrapperSessionNotExceedingLimit(self):
        if False:
            for i in range(10):
                print('nop')

        def _watch_fn(fetches, feeds):
            if False:
                for i in range(10):
                    print('nop')
            del fetches, feeds
            return ('DebugIdentity', '(.*delta.*|.*inc_v.*)', '.*')
        sess = dumping_wrapper.DumpingDebugWrapperSession(self.sess, session_root=self.session_root, watch_fn=_watch_fn)
        sess.run(self.inc_v)

    def testWrapperSessionExceedingLimit(self):
        if False:
            i = 10
            return i + 15

        def _watch_fn(fetches, feeds):
            if False:
                while True:
                    i = 10
            del fetches, feeds
            return ('DebugIdentity', '.*delta.*', '.*')
        sess = dumping_wrapper.DumpingDebugWrapperSession(self.sess, session_root=self.session_root, watch_fn=_watch_fn)
        sess.run(self.inc_v)
        sess.run(self.inc_v)
        with self.assertRaises(ValueError):
            sess.run(self.inc_v)

    def testHookNotExceedingLimit(self):
        if False:
            i = 10
            return i + 15

        def _watch_fn(fetches, feeds):
            if False:
                print('Hello World!')
            del fetches, feeds
            return ('DebugIdentity', '.*delta.*', '.*')
        dumping_hook = hooks.DumpingDebugHook(self.session_root, watch_fn=_watch_fn)
        mon_sess = monitored_session._HookedSession(self.sess, [dumping_hook])
        mon_sess.run(self.inc_v)

    def testHookExceedingLimit(self):
        if False:
            for i in range(10):
                print('nop')

        def _watch_fn(fetches, feeds):
            if False:
                for i in range(10):
                    print('nop')
            del fetches, feeds
            return ('DebugIdentity', '.*delta.*', '.*')
        dumping_hook = hooks.DumpingDebugHook(self.session_root, watch_fn=_watch_fn)
        mon_sess = monitored_session._HookedSession(self.sess, [dumping_hook])
        mon_sess.run(self.inc_v)
        mon_sess.run(self.inc_v)
        with self.assertRaises(ValueError):
            mon_sess.run(self.inc_v)
if __name__ == '__main__':
    googletest.main()