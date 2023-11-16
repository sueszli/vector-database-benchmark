import gevent.testing as greentest
import gevent
from gevent import util, socket
DELAY = 0.1

class Test(greentest.TestCase):

    @greentest.skipOnAppVeyor('Timing causes the state to often be [start,finished]')
    def test_killing_dormant(self):
        if False:
            for i in range(10):
                print('nop')
        state = []

        def test():
            if False:
                for i in range(10):
                    print('nop')
            try:
                state.append('start')
                gevent.sleep(DELAY * 3.0)
            except:
                state.append('except')
            state.append('finished')
        g = gevent.spawn(test)
        gevent.sleep(DELAY / 2)
        assert state == ['start'], state
        g.kill()
        self.assertEqual(state, ['start', 'except', 'finished'])

    def test_nested_with_timeout(self):
        if False:
            print('Hello World!')

        def func():
            if False:
                return 10
            return gevent.with_timeout(0.2, gevent.sleep, 2, timeout_value=1)
        self.assertRaises(gevent.Timeout, gevent.with_timeout, 0.1, func)

    def test_sleep_invalid_switch(self):
        if False:
            while True:
                i = 10
        p = gevent.spawn(util.wrap_errors(AssertionError, gevent.sleep), 2)
        gevent.sleep(0)
        switcher = gevent.spawn(p.switch, None)
        result = p.get()
        assert isinstance(result, AssertionError), result
        assert 'Invalid switch' in str(result), repr(str(result))
        switcher.kill()
    if hasattr(socket, 'socketpair'):

        def _test_wait_read_invalid_switch(self, sleep):
            if False:
                print('Hello World!')
            (sock1, sock2) = socket.socketpair()
            try:
                p = gevent.spawn(util.wrap_errors(AssertionError, socket.wait_read), sock1.fileno())
                gevent.get_hub().loop.run_callback(switch_None, p)
                if sleep is not None:
                    gevent.sleep(sleep)
                result = p.get()
                assert isinstance(result, AssertionError), result
                assert 'Invalid switch' in str(result), repr(str(result))
            finally:
                sock1.close()
                sock2.close()

        def test_invalid_switch_None(self):
            if False:
                while True:
                    i = 10
            self._test_wait_read_invalid_switch(None)

        def test_invalid_switch_0(self):
            if False:
                for i in range(10):
                    print('nop')
            self._test_wait_read_invalid_switch(0)

        def test_invalid_switch_1(self):
            if False:
                return 10
            self._test_wait_read_invalid_switch(0.001)

def switch_None(g):
    if False:
        return 10
    g.switch(None)

class TestTimers(greentest.TestCase):

    def test_timer_fired(self):
        if False:
            print('Hello World!')
        lst = [1]

        def func():
            if False:
                while True:
                    i = 10
            gevent.spawn_later(0.01, lst.pop)
            gevent.sleep(0.02)
        gevent.spawn(func)
        self.assertEqual(lst, [1])
        gevent.sleep()
        gevent.sleep(0.1)
        self.assertEqual(lst, [])

    def test_spawn_is_not_cancelled(self):
        if False:
            i = 10
            return i + 15
        lst = [1]

        def func():
            if False:
                i = 10
                return i + 15
            gevent.spawn(lst.pop)
        gevent.spawn(func)
        gevent.sleep(0.1)
        self.assertEqual(lst, [])
if __name__ == '__main__':
    greentest.main()