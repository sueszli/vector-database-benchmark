from twisted.internet import defer
from twisted.trial.unittest import SynchronousTestCase
from buildbot.test.util.patch_delay import patchForDelay

class TestException(Exception):
    pass

def fun_to_patch(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    return defer.succeed((args, kwargs))

def fun_to_patch_exception():
    if False:
        while True:
            i = 10
    raise TestException()
non_callable = 1

class Tests(SynchronousTestCase):

    def test_raises_not_found(self):
        if False:
            return 10
        with self.assertRaises(Exception):
            with patchForDelay(__name__ + '.notfound'):
                pass

    def test_raises_not_callable(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(Exception):
            with patchForDelay(__name__ + '.non_callable'):
                pass

    def test_patches_within_context(self):
        if False:
            i = 10
            return i + 15
        d = fun_to_patch()
        self.assertTrue(d.called)
        with patchForDelay(__name__ + '.fun_to_patch') as delay:
            d = fun_to_patch()
            self.assertEqual(len(delay), 1)
            self.assertFalse(d.called)
            delay.fire()
            self.assertEqual(len(delay), 0)
            self.assertTrue(d.called)
        d = fun_to_patch()
        self.assertTrue(d.called)

    def test_auto_fires_unfired_delay(self):
        if False:
            i = 10
            return i + 15
        with patchForDelay(__name__ + '.fun_to_patch') as delay:
            d = fun_to_patch()
            self.assertEqual(len(delay), 1)
            self.assertFalse(d.called)
        self.assertTrue(d.called)

    def test_auto_fires_unfired_delay_exception(self):
        if False:
            print('Hello World!')
        try:
            with patchForDelay(__name__ + '.fun_to_patch') as delay:
                d = fun_to_patch()
                self.assertEqual(len(delay), 1)
                self.assertFalse(d.called)
                raise TestException()
        except TestException:
            pass
        self.assertTrue(d.called)

    def test_passes_arguments(self):
        if False:
            for i in range(10):
                print('nop')
        with patchForDelay(__name__ + '.fun_to_patch') as delay:
            d = fun_to_patch('arg', kw='kwarg')
            self.assertEqual(len(delay), 1)
            delay.fire()
            args = self.successResultOf(d)
        self.assertEqual(args, (('arg',), {'kw': 'kwarg'}))

    def test_passes_exception(self):
        if False:
            return 10
        with patchForDelay(__name__ + '.fun_to_patch_exception') as delay:
            d = fun_to_patch_exception()
            self.assertEqual(len(delay), 1)
            delay.fire()
            f = self.failureResultOf(d)
            f.check(TestException)