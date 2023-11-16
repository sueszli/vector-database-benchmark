from __future__ import print_function
from gevent import monkey
monkey.patch_all()
import os
import unittest
import multiprocessing
import gevent
hub = gevent.get_hub()
pid = os.getpid()
newpid = None

def on_fork():
    if False:
        return 10
    global newpid
    newpid = os.getpid()
fork_watcher = hub.loop.fork(ref=False)
fork_watcher.start(on_fork)

def in_child(q):
    if False:
        while True:
            i = 10
    gevent.sleep(0.001)
    gevent.sleep(0.001)
    q.put(newpid)

class Test(unittest.TestCase):

    def test(self):
        if False:
            while True:
                i = 10
        self.assertEqual(hub.threadpool.size, 0)
        hub.threadpool.apply(lambda : None)
        self.assertEqual(hub.threadpool.size, 1)
        try:
            fork_ctx = multiprocessing.get_context('fork')
        except (AttributeError, ValueError):
            fork_ctx = multiprocessing
        q = fork_ctx.Queue()
        p = fork_ctx.Process(target=in_child, args=(q,))
        p.start()
        p.join()
        p_val = q.get()
        self.assertIsNone(newpid, 'The fork watcher ran in the parent for some reason.')
        self.assertIsNotNone(p_val, "The child process returned nothing, meaning the fork watcher didn't run in the child.")
        self.assertNotEqual(p_val, pid)
        assert p_val != pid
if __name__ == '__main__':
    multiprocessing.freeze_support()
    unittest.main()