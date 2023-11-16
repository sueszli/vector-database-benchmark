from tests import unittest
from tests.utils import MockClock

class MockClockTestCase(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        self.clock = MockClock()

    def test_advance_time(self) -> None:
        if False:
            while True:
                i = 10
        start_time = self.clock.time()
        self.clock.advance_time(20)
        self.assertEqual(20, self.clock.time() - start_time)

    def test_later(self) -> None:
        if False:
            while True:
                i = 10
        invoked = [0, 0]

        def _cb0() -> None:
            if False:
                while True:
                    i = 10
            invoked[0] = 1
        self.clock.call_later(10, _cb0)

        def _cb1() -> None:
            if False:
                return 10
            invoked[1] = 1
        self.clock.call_later(20, _cb1)
        self.assertFalse(invoked[0])
        self.clock.advance_time(15)
        self.assertTrue(invoked[0])
        self.assertFalse(invoked[1])
        self.clock.advance_time(5)
        self.assertTrue(invoked[1])

    def test_cancel_later(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        invoked = [0, 0]

        def _cb0() -> None:
            if False:
                for i in range(10):
                    print('nop')
            invoked[0] = 1
        t0 = self.clock.call_later(10, _cb0)

        def _cb1() -> None:
            if False:
                print('Hello World!')
            invoked[1] = 1
        self.clock.call_later(20, _cb1)
        self.clock.cancel_call_later(t0)
        self.clock.advance_time(30)
        self.assertFalse(invoked[0])
        self.assertTrue(invoked[1])