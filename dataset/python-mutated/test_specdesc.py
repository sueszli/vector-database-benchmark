from numpy.testing import TestCase, assert_equal, assert_almost_equal
from numpy import random, arange, log, zeros
from aubio import specdesc, cvec, float_type
methods = ['default', 'energy', 'hfc', 'complex', 'phase', 'specdiff', 'kl', 'mkl', 'specflux', 'centroid', 'spread', 'skewness', 'kurtosis', 'slope', 'decrease', 'rolloff']
buf_size = 2048

class aubio_specdesc(TestCase):

    def test_members(self):
        if False:
            return 10
        o = specdesc()
        for method in methods:
            o = specdesc(method, buf_size)
            assert_equal([o.buf_size, o.method], [buf_size, method])
            spec = cvec(buf_size)
            spec.norm[0] = 1
            spec.norm[1] = 1.0 / 2.0
            o(spec)
            spec.norm = random.random_sample((len(spec.norm),)).astype(float_type)
            spec.phas = random.random_sample((len(spec.phas),)).astype(float_type)
            assert o(spec) != 0.0

    def test_phase(self):
        if False:
            while True:
                i = 10
        o = specdesc('phase', buf_size)
        spec = cvec(buf_size)
        assert_equal(o(spec), 0.0)
        spec.phas = random.random_sample((len(spec.phas),)).astype(float_type)
        spec.norm[:] = 1
        assert o(spec) != 0.0

    def test_specdiff(self):
        if False:
            return 10
        o = specdesc('phase', buf_size)
        spec = cvec(buf_size)
        assert_equal(o(spec), 0.0)
        spec.phas = random.random_sample((len(spec.phas),)).astype(float_type)
        spec.norm[:] = 1
        assert o(spec) != 0.0

    def test_hfc(self):
        if False:
            for i in range(10):
                print('nop')
        o = specdesc('hfc')
        c = cvec()
        assert_equal(0.0, o(c))
        a = arange(c.length, dtype=float_type)
        c.norm = a
        assert_equal(a, c.norm)
        assert_equal(sum(a * (a + 1)), o(c))

    def test_complex(self):
        if False:
            return 10
        o = specdesc('complex')
        c = cvec()
        assert_equal(0.0, o(c))
        a = arange(c.length, dtype=float_type)
        c.norm = a
        assert_equal(a, c.norm)
        assert_equal(sum(a), o(c))
        assert_equal(0, o(c))

    def test_kl(self):
        if False:
            while True:
                i = 10
        o = specdesc('kl')
        c = cvec()
        assert_equal(0.0, o(c))
        a = arange(c.length, dtype=float_type)
        c.norm = a
        assert_almost_equal(sum(a * log(1.0 + a / 0.1)) / o(c), 1.0, decimal=6)

    def test_mkl(self):
        if False:
            for i in range(10):
                print('nop')
        o = specdesc('mkl')
        c = cvec()
        assert_equal(0.0, o(c))
        a = arange(c.length, dtype=float_type)
        c.norm = a
        assert_almost_equal(sum(log(1.0 + a / 0.1)) / o(c), 1, decimal=6)

    def test_specflux(self):
        if False:
            i = 10
            return i + 15
        o = specdesc('specflux')
        c = cvec()
        assert_equal(0.0, o(c))
        a = arange(c.length, dtype=float_type)
        c.norm = a
        assert_equal(sum(a), o(c))
        assert_equal(0, o(c))
        c.norm = zeros(c.length, dtype=float_type)
        assert_equal(0, o(c))

    def test_centroid(self):
        if False:
            while True:
                i = 10
        o = specdesc('centroid')
        c = cvec()
        assert_equal(0.0, o(c))
        a = arange(c.length, dtype=float_type)
        c.norm = a
        centroid = sum(a * a) / sum(a)
        assert_almost_equal(centroid, o(c), decimal=2)
        c.norm = a * 0.5
        assert_almost_equal(centroid, o(c), decimal=2)

    def test_spread(self):
        if False:
            for i in range(10):
                print('nop')
        o = specdesc('spread')
        c = cvec(1024)
        ramp = arange(c.length, dtype=float_type)
        assert_equal(0.0, o(c))
        a = ramp
        c.norm = a
        centroid = sum(a * a) / sum(a)
        spread = sum(a * pow(ramp - centroid, 2.0)) / sum(a)
        assert_almost_equal(o(c), spread, decimal=1)

    def test_skewness(self):
        if False:
            while True:
                i = 10
        o = specdesc('skewness')
        c = cvec()
        assert_equal(0.0, o(c))
        a = arange(c.length, dtype=float_type)
        c.norm = a
        centroid = sum(a * a) / sum(a)
        spread = sum((a - centroid) ** 2 * a) / sum(a)
        skewness = sum((a - centroid) ** 3 * a) / sum(a) / spread ** 1.5
        assert_almost_equal(skewness, o(c), decimal=2)
        c.norm = a * 3
        assert_almost_equal(skewness, o(c), decimal=2)

    def test_kurtosis(self):
        if False:
            print('Hello World!')
        o = specdesc('kurtosis')
        c = cvec()
        assert_equal(0.0, o(c))
        a = arange(c.length, dtype=float_type)
        c.norm = a
        centroid = sum(a * a) / sum(a)
        spread = sum((a - centroid) ** 2 * a) / sum(a)
        kurtosis = sum((a - centroid) ** 4 * a) / sum(a) / spread ** 2
        assert_almost_equal(kurtosis, o(c), decimal=2)

    def test_slope(self):
        if False:
            i = 10
            return i + 15
        o = specdesc('slope')
        c = cvec()
        assert_equal(0.0, o(c))
        a = arange(c.length * 2, 0, -2, dtype=float_type)
        k = arange(c.length, dtype=float_type)
        c.norm = a
        num = len(a) * sum(k * a) - sum(k) * sum(a)
        den = len(a) * sum(k ** 2) - sum(k) ** 2
        slope = num / den / sum(a)
        assert_almost_equal(slope, o(c), decimal=5)
        a = arange(0, c.length * 2, +2, dtype=float_type)
        c.norm = a
        num = len(a) * sum(k * a) - sum(k) * sum(a)
        den = len(a) * sum(k ** 2) - sum(k) ** 2
        slope = num / den / sum(a)
        assert_almost_equal(slope, o(c), decimal=5)
        a = arange(0, c.length * 2, +2, dtype=float_type)
        c.norm = a * 2
        assert_almost_equal(slope, o(c), decimal=5)

    def test_decrease(self):
        if False:
            while True:
                i = 10
        o = specdesc('decrease')
        c = cvec()
        assert_equal(0.0, o(c))
        a = arange(c.length * 2, 0, -2, dtype=float_type)
        k = arange(c.length, dtype=float_type)
        c.norm = a
        decrease = sum((a[1:] - a[0]) / k[1:]) / sum(a[1:])
        assert_almost_equal(decrease, o(c), decimal=5)
        a = arange(0, c.length * 2, +2, dtype=float_type)
        c.norm = a
        decrease = sum((a[1:] - a[0]) / k[1:]) / sum(a[1:])
        assert_almost_equal(decrease, o(c), decimal=5)
        a = arange(0, c.length * 2, +2, dtype=float_type)
        c.norm = a * 2
        decrease = sum((a[1:] - a[0]) / k[1:]) / sum(a[1:])
        assert_almost_equal(decrease, o(c), decimal=5)

    def test_rolloff(self):
        if False:
            i = 10
            return i + 15
        o = specdesc('rolloff')
        c = cvec()
        assert_equal(0.0, o(c))
        a = arange(c.length * 2, 0, -2, dtype=float_type)
        c.norm = a
        cumsum = 0.95 * sum(a * a)
        i = 0
        rollsum = 0
        while rollsum < cumsum:
            rollsum += a[i] * a[i]
            i += 1
        rolloff = i
        assert_equal(rolloff, o(c))

class aubio_specdesc_wrong(TestCase):

    def test_negative(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ValueError):
            specdesc('default', -10)

    def test_unknown(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(RuntimeError):
            specdesc('unknown', 512)
if __name__ == '__main__':
    from unittest import main
    main()