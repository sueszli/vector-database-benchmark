from threading import Thread
from time import sleep
from unittest import TestCase, mock
from mycroft.util.monotonic_event import MonotonicEvent

class MonotonicEventTest(TestCase):

    def test_wait_set(self):
        if False:
            i = 10
            return i + 15
        event = MonotonicEvent()
        event.set()
        self.assertTrue(event.wait())

    def test_wait_timeout(self):
        if False:
            print('Hello World!')
        event = MonotonicEvent()
        self.assertFalse(event.wait(0.1))

    def test_wait_set_with_timeout(self):
        if False:
            print('Hello World!')
        wait_result = False
        event = MonotonicEvent()

        def wait_event():
            if False:
                i = 10
                return i + 15
            nonlocal wait_result
            wait_result = event.wait(30)
        wait_thread = Thread(target=wait_event)
        wait_thread.start()
        sleep(0.1)
        event.set()
        wait_thread.join()
        self.assertTrue(wait_result)