import gevent.testing as greentest
import gevent

class ExpectedError(greentest.ExpectedException):
    pass

def f():
    if False:
        while True:
            i = 10
    gevent.sleep(999)

class TestKillWithException(greentest.TestCase):

    def test_kill_without_exception(self):
        if False:
            while True:
                i = 10
        g = gevent.spawn(f)
        g.kill()
        assert g.successful()
        assert isinstance(g.get(), gevent.GreenletExit)

    def test_kill_with_exception(self):
        if False:
            return 10
        g = gevent.spawn(f)
        with gevent.get_hub().ignoring_expected_test_error():
            g.kill(ExpectedError)
        self.assertFalse(g.successful())
        self.assertRaises(ExpectedError, g.get)
        self.assertIsNone(g.value)
        self.assertIsInstance(g.exception, ExpectedError)

    def test_kill_with_exception_after_started(self):
        if False:
            i = 10
            return i + 15
        with gevent.get_hub().ignoring_expected_test_error():
            g = gevent.spawn(f)
            g.join(0)
            g.kill(ExpectedError)
        self.assertFalse(g.successful())
        self.assertRaises(ExpectedError, g.get)
        self.assertIsNone(g.value)
        self.assertIsInstance(g.exception, ExpectedError)
if __name__ == '__main__':
    greentest.main()