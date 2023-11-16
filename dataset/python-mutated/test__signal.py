from __future__ import print_function
import signal
import gevent.testing as greentest
import gevent
import pkg_resources
try:
    cffi_version = pkg_resources.get_distribution('cffi').parsed_version
except Exception:
    cffi_version = None

class Expected(Exception):
    pass

def raise_Expected():
    if False:
        for i in range(10):
            print('nop')
    raise Expected('TestSignal')

@greentest.skipUnless(hasattr(signal, 'SIGALRM'), 'Uses SIGALRM')
class TestSignal(greentest.TestCase):
    error_fatal = False
    __timeout__ = greentest.LARGE_TIMEOUT

    def test_handler(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(TypeError):
            gevent.signal_handler(signal.SIGALRM, 1)

    def test_alarm(self):
        if False:
            while True:
                i = 10
        sig = gevent.signal_handler(signal.SIGALRM, raise_Expected)
        self.assertFalse(sig.ref)
        sig.ref = True
        self.assertTrue(sig.ref)
        sig.ref = False

        def test():
            if False:
                for i in range(10):
                    print('nop')
            signal.alarm(1)
            with self.assertRaises(Expected) as exc:
                gevent.sleep(2)
            ex = exc.exception
            self.assertEqual(str(ex), 'TestSignal')
        try:
            test()
            test()
        finally:
            sig.cancel()

    @greentest.skipIf(greentest.PY3 and greentest.CFFI_BACKEND and (cffi_version < pkg_resources.parse_version('1.11.3')), 'https://bitbucket.org/cffi/cffi/issues/352/systemerror-returned-a-result-with-an')
    @greentest.ignores_leakcheck
    def test_reload(self):
        if False:
            while True:
                i = 10
        import gevent.signal
        assert gevent.signal
        import site
        if greentest.PY3:
            from importlib import reload as reload_module
        else:
            reload_module = reload
        try:
            reload_module(site)
        except TypeError:
            assert greentest.PY36
            import sys
            for m in set(sys.modules.values()):
                try:
                    if m.__cached__ is None:
                        print('Module has None __cached__', m, file=sys.stderr)
                except AttributeError:
                    continue
if __name__ == '__main__':
    greentest.main()