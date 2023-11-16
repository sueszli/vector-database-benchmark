from __future__ import absolute_import, print_function
import gevent
import unittest

class TestDestroyHub(unittest.TestCase):

    def test_destroy_hub(self):
        if False:
            return 10
        hub = gevent.get_hub()
        self.assertTrue(hub.loop.default)
        initloop = hub.loop
        tp = hub.threadpool
        self.assertIsNotNone(tp)
        hub.destroy()
        hub = gevent.get_hub()
        self.assertTrue(hub.loop.default)
        self.assertIs(hub.loop, initloop)
        hub.destroy(destroy_loop=True)
        hub = gevent.get_hub()
        self.assertTrue(hub.loop.default)
        self.assertIsNot(hub.loop, initloop)
        self.assertIsNot(hub.loop.ptr, initloop.ptr)
        self.assertNotEqual(hub.loop.ptr, initloop.ptr)
        hub.destroy(destroy_loop=True)
        hub = gevent.get_hub()
        self.assertTrue(hub.loop.default)
        hub.destroy()
if __name__ == '__main__':
    unittest.main()