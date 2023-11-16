"""
Tests that on Python 2, if the futures backport of 'thread' is already
imported before we monkey-patch, it gets patched too.
"""
import unittest
try:
    import thread
    import _thread
    HAS_BOTH = True
except ImportError:
    HAS_BOTH = False

class TestMonkey(unittest.TestCase):

    @unittest.skipUnless(HAS_BOTH, 'Python 2, needs future backport installed')
    def test_patches_both(self):
        if False:
            while True:
                i = 10
        thread_lt = thread.LockType
        _thread_lt = _thread.LockType
        self.assertIs(thread_lt, _thread_lt)
        from gevent.thread import LockType as gLockType
        self.assertIsNot(thread_lt, gLockType)
        import gevent.monkey
        gevent.monkey.patch_all()
        thread_lt2 = thread.LockType
        _thread_lt2 = _thread.LockType
        self.assertIs(thread_lt2, gLockType)
        self.assertIs(_thread_lt2, gLockType)
        self.assertIs(thread_lt2, _thread_lt2)
        self.assertIsNot(thread_lt2, thread_lt)
        orig_locktype = gevent.monkey.get_original('thread', 'LockType')
        self.assertIs(orig_locktype, thread_lt)
        orig__locktype = gevent.monkey.get_original('_thread', 'LockType')
        self.assertIs(orig__locktype, thread_lt)
if __name__ == '__main__':
    unittest.main()