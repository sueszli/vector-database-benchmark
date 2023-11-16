import sys
import gevent.testing as greentest
import gevent
from gevent.hub import get_hub

def raise_(ex):
    if False:
        i = 10
        return i + 15
    raise ex
MSG = 'should be re-raised and caught'

class Test(greentest.TestCase):
    x = None
    error_fatal = False

    def start(self, *args):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.x = None

    def test_sys_exit(self):
        if False:
            print('Hello World!')
        self.start(sys.exit, MSG)
        try:
            gevent.sleep(0.001)
        except SystemExit as ex:
            assert str(ex) == MSG, repr(str(ex))
        else:
            raise AssertionError('must raise SystemExit')

    def test_keyboard_interrupt(self):
        if False:
            return 10
        self.start(raise_, KeyboardInterrupt)
        try:
            gevent.sleep(0.001)
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError('must raise KeyboardInterrupt')

    def test_keyboard_interrupt_stderr_patched(self):
        if False:
            print('Hello World!')
        from gevent import monkey
        monkey.patch_sys(stdin=False, stdout=False, stderr=True)
        try:
            try:
                self.start(raise_, KeyboardInterrupt)
                while True:
                    gevent.sleep(0.1)
            except KeyboardInterrupt:
                pass
        finally:
            sys.stderr = monkey.get_original('sys', 'stderr')

    def test_system_error(self):
        if False:
            while True:
                i = 10
        self.start(raise_, SystemError(MSG))
        with self.assertRaisesRegex(SystemError, MSG):
            gevent.sleep(0.002)

    def test_exception(self):
        if False:
            for i in range(10):
                print('nop')
        self.start(raise_, Exception('regular exception must not kill the program'))
        gevent.sleep(0.001)

class TestCallback(Test):

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        if self.x is not None:
            assert not self.x.pending, self.x

    def start(self, *args):
        if False:
            for i in range(10):
                print('nop')
        self.x = get_hub().loop.run_callback(*args)
    if greentest.LIBUV:

        def test_exception(self):
            if False:
                for i in range(10):
                    print('nop')
            gevent.sleep(0.001)
            super(TestCallback, self).test_exception()

class TestSpawn(Test):

    def tearDown(self):
        if False:
            while True:
                i = 10
        gevent.sleep(0.0001)
        if self.x is not None:
            assert self.x.dead, self.x

    def start(self, *args):
        if False:
            print('Hello World!')
        self.x = gevent.spawn(*args)
del Test
if __name__ == '__main__':
    greentest.main()