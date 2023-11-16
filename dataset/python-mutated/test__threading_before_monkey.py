import threading
from gevent import monkey
monkey.patch_all()
import gevent.testing as greentest

class Test(greentest.TestCase):

    def test_main_thread(self):
        if False:
            i = 10
            return i + 15
        current = threading.current_thread()
        self.assertFalse(isinstance(current, threading._DummyThread))
        self.assertTrue(isinstance(current, monkey.get_original('threading', 'Thread')))
        repr(current)
if __name__ == '__main__':
    greentest.main()