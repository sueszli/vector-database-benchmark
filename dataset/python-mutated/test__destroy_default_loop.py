from __future__ import print_function
import gevent
import unittest

class TestDestroyDefaultLoop(unittest.TestCase):

    def tearDown(self):
        if False:
            return 10
        self._reset_hub()
        super(TestDestroyDefaultLoop, self).tearDown()

    def _reset_hub(self):
        if False:
            i = 10
            return i + 15
        from gevent._hub_local import set_hub
        from gevent._hub_local import set_loop
        from gevent._hub_local import get_hub_if_exists
        hub = get_hub_if_exists()
        if hub is not None:
            hub.destroy(destroy_loop=True)
        set_hub(None)
        set_loop(None)

    def test_destroy_gc(self):
        if False:
            print('Hello World!')
        gevent.get_hub()
        loop = gevent.config.loop(default=True)
        self.assertTrue(loop.default)
        loop.destroy()
        self.assertFalse(loop.default)
        del loop
        self._reset_hub()

    def test_destroy_two(self):
        if False:
            while True:
                i = 10
        loop1 = gevent.config.loop(default=True)
        loop2 = gevent.config.loop(default=True)
        self.assertTrue(loop1.default)
        self.assertTrue(loop2.default)
        loop1.destroy()
        self.assertFalse(loop1.default)
        loop2.destroy()
        self.assertFalse(loop2.default)
        self.assertFalse(loop2.ptr)
        self._reset_hub()
        self.assertTrue(gevent.get_hub().loop.ptr)
if __name__ == '__main__':
    unittest.main()