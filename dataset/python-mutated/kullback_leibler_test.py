"""Tests for distributions KL mechanism."""
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.ops.distributions import normal
from tensorflow.python.platform import test
_DIVERGENCES = kullback_leibler._DIVERGENCES
_registered_kl = kullback_leibler._registered_kl

class KLTest(test.TestCase):

    def testRegistration(self):
        if False:
            for i in range(10):
                print('nop')

        class MyDist(normal.Normal):
            pass

        @kullback_leibler.RegisterKL(MyDist, MyDist)
        def _kl(a, b, name=None):
            if False:
                i = 10
                return i + 15
            return name
        a = MyDist(loc=0.0, scale=1.0)
        self.assertEqual('OK', kullback_leibler.kl_divergence(a, a, name='OK'))

    @test_util.run_deprecated_v1
    def testDomainErrorExceptions(self):
        if False:
            for i in range(10):
                print('nop')

        class MyDistException(normal.Normal):
            pass

        @kullback_leibler.RegisterKL(MyDistException, MyDistException)
        def _kl(a, b, name=None):
            if False:
                i = 10
                return i + 15
            return array_ops.identity([float('nan')])
        with self.cached_session():
            a = MyDistException(loc=0.0, scale=1.0, allow_nan_stats=False)
            kl = kullback_leibler.kl_divergence(a, a, allow_nan_stats=False)
            with self.assertRaisesOpError('KL calculation between .* and .* returned NaN values'):
                self.evaluate(kl)
            with self.assertRaisesOpError('KL calculation between .* and .* returned NaN values'):
                a.kl_divergence(a).eval()
            a = MyDistException(loc=0.0, scale=1.0, allow_nan_stats=True)
            kl_ok = kullback_leibler.kl_divergence(a, a)
            self.assertAllEqual([float('nan')], self.evaluate(kl_ok))
            self_kl_ok = a.kl_divergence(a)
            self.assertAllEqual([float('nan')], self.evaluate(self_kl_ok))
            cross_ok = a.cross_entropy(a)
            self.assertAllEqual([float('nan')], self.evaluate(cross_ok))

    def testRegistrationFailures(self):
        if False:
            i = 10
            return i + 15

        class MyDist(normal.Normal):
            pass
        with self.assertRaisesRegex(TypeError, 'must be callable'):
            kullback_leibler.RegisterKL(MyDist, MyDist)('blah')
        kullback_leibler.RegisterKL(MyDist, MyDist)(lambda a, b: None)
        with self.assertRaisesRegex(ValueError, 'has already been registered'):
            kullback_leibler.RegisterKL(MyDist, MyDist)(lambda a, b: None)

    def testExactRegistrationsAllMatch(self):
        if False:
            print('Hello World!')
        for (k, v) in _DIVERGENCES.items():
            self.assertEqual(v, _registered_kl(*k))

    def _testIndirectRegistration(self, fn):
        if False:
            while True:
                i = 10

        class Sub1(normal.Normal):

            def entropy(self):
                if False:
                    return 10
                return ''

        class Sub2(normal.Normal):

            def entropy(self):
                if False:
                    for i in range(10):
                        print('nop')
                return ''

        class Sub11(Sub1):

            def entropy(self):
                if False:
                    i = 10
                    return i + 15
                return ''

        @kullback_leibler.RegisterKL(Sub1, Sub1)
        def _kl11(a, b, name=None):
            if False:
                return 10
            return 'sub1-1'

        @kullback_leibler.RegisterKL(Sub1, Sub2)
        def _kl12(a, b, name=None):
            if False:
                while True:
                    i = 10
            return 'sub1-2'

        @kullback_leibler.RegisterKL(Sub2, Sub1)
        def _kl21(a, b, name=None):
            if False:
                for i in range(10):
                    print('nop')
            return 'sub2-1'
        sub1 = Sub1(loc=0.0, scale=1.0)
        sub2 = Sub2(loc=0.0, scale=1.0)
        sub11 = Sub11(loc=0.0, scale=1.0)
        self.assertEqual('sub1-1', fn(sub1, sub1))
        self.assertEqual('sub1-2', fn(sub1, sub2))
        self.assertEqual('sub2-1', fn(sub2, sub1))
        self.assertEqual('sub1-1', fn(sub11, sub11))
        self.assertEqual('sub1-1', fn(sub11, sub1))
        self.assertEqual('sub1-2', fn(sub11, sub2))
        self.assertEqual('sub1-1', fn(sub11, sub1))
        self.assertEqual('sub1-2', fn(sub11, sub2))
        self.assertEqual('sub2-1', fn(sub2, sub11))
        self.assertEqual('sub1-1', fn(sub1, sub11))

    def testIndirectRegistrationKLFun(self):
        if False:
            i = 10
            return i + 15
        self._testIndirectRegistration(kullback_leibler.kl_divergence)

    def testIndirectRegistrationKLSelf(self):
        if False:
            print('Hello World!')
        self._testIndirectRegistration(lambda p, q: p.kl_divergence(q))

    def testIndirectRegistrationCrossEntropy(self):
        if False:
            i = 10
            return i + 15
        self._testIndirectRegistration(lambda p, q: p.cross_entropy(q))

    def testFunctionCrossEntropy(self):
        if False:
            while True:
                i = 10
        self._testIndirectRegistration(kullback_leibler.cross_entropy)
if __name__ == '__main__':
    test.main()