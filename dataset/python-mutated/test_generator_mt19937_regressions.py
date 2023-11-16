from numpy.testing import assert_, assert_array_equal
import numpy as np
import pytest
from numpy.random import Generator, MT19937

class TestRegression:

    def setup_method(self):
        if False:
            while True:
                i = 10
        self.mt19937 = Generator(MT19937(121263137472525314065))

    def test_vonmises_range(self):
        if False:
            return 10
        for mu in np.linspace(-7.0, 7.0, 5):
            r = self.mt19937.vonmises(mu, 1, 50)
            assert_(np.all(r > -np.pi) and np.all(r <= np.pi))

    def test_hypergeometric_range(self):
        if False:
            print('Hello World!')
        assert_(np.all(self.mt19937.hypergeometric(3, 18, 11, size=10) < 4))
        assert_(np.all(self.mt19937.hypergeometric(18, 3, 11, size=10) > 0))
        args = (2 ** 20 - 2, 2 ** 20 - 2, 2 ** 20 - 2)
        assert_(self.mt19937.hypergeometric(*args) > 0)

    def test_logseries_convergence(self):
        if False:
            print('Hello World!')
        N = 1000
        rvsn = self.mt19937.logseries(0.8, size=N)
        freq = np.sum(rvsn == 1) / N
        msg = f'Frequency was {freq:f}, should be > 0.45'
        assert_(freq > 0.45, msg)
        freq = np.sum(rvsn == 2) / N
        msg = f'Frequency was {freq:f}, should be < 0.23'
        assert_(freq < 0.23, msg)

    def test_shuffle_mixed_dimension(self):
        if False:
            i = 10
            return i + 15
        for t in [[1, 2, 3, None], [(1, 1), (2, 2), (3, 3), None], [1, (2, 2), (3, 3), None], [(1, 1), 2, 3, None]]:
            mt19937 = Generator(MT19937(12345))
            shuffled = np.array(t, dtype=object)
            mt19937.shuffle(shuffled)
            expected = np.array([t[2], t[0], t[3], t[1]], dtype=object)
            assert_array_equal(np.array(shuffled, dtype=object), expected)

    def test_call_within_randomstate(self):
        if False:
            return 10
        res = np.array([1, 8, 0, 1, 5, 3, 3, 8, 1, 4])
        for i in range(3):
            mt19937 = Generator(MT19937(i))
            m = Generator(MT19937(4321))
            assert_array_equal(m.choice(10, size=10, p=np.ones(10) / 10.0), res)

    def test_multivariate_normal_size_types(self):
        if False:
            i = 10
            return i + 15
        self.mt19937.multivariate_normal([0], [[0]], size=1)
        self.mt19937.multivariate_normal([0], [[0]], size=np.int_(1))
        self.mt19937.multivariate_normal([0], [[0]], size=np.int64(1))

    def test_beta_small_parameters(self):
        if False:
            i = 10
            return i + 15
        x = self.mt19937.beta(0.0001, 0.0001, size=100)
        assert_(not np.any(np.isnan(x)), 'Nans in mt19937.beta')

    def test_beta_very_small_parameters(self):
        if False:
            print('Hello World!')
        self.mt19937.beta(1e-49, 1e-40)

    def test_beta_ridiculously_small_parameters(self):
        if False:
            i = 10
            return i + 15
        tiny = np.finfo(1.0).tiny
        x = self.mt19937.beta(tiny / 32, tiny / 40, size=50)
        assert not np.any(np.isnan(x))

    def test_choice_sum_of_probs_tolerance(self):
        if False:
            return 10
        a = [1, 2, 3]
        counts = [4, 4, 2]
        for dt in (np.float16, np.float32, np.float64):
            probs = np.array(counts, dtype=dt) / sum(counts)
            c = self.mt19937.choice(a, p=probs)
            assert_(c in a)
            with pytest.raises(ValueError):
                self.mt19937.choice(a, p=probs * 0.9)

    def test_shuffle_of_array_of_different_length_strings(self):
        if False:
            i = 10
            return i + 15
        a = np.array(['a', 'a' * 1000])
        for _ in range(100):
            self.mt19937.shuffle(a)
        import gc
        gc.collect()

    def test_shuffle_of_array_of_objects(self):
        if False:
            for i in range(10):
                print('nop')
        a = np.array([np.arange(1), np.arange(4)], dtype=object)
        for _ in range(1000):
            self.mt19937.shuffle(a)
        import gc
        gc.collect()

    def test_permutation_subclass(self):
        if False:
            while True:
                i = 10

        class N(np.ndarray):
            pass
        mt19937 = Generator(MT19937(1))
        orig = np.arange(3).view(N)
        perm = mt19937.permutation(orig)
        assert_array_equal(perm, np.array([2, 0, 1]))
        assert_array_equal(orig, np.arange(3).view(N))

        class M:
            a = np.arange(5)

            def __array__(self):
                if False:
                    while True:
                        i = 10
                return self.a
        mt19937 = Generator(MT19937(1))
        m = M()
        perm = mt19937.permutation(m)
        assert_array_equal(perm, np.array([4, 1, 3, 0, 2]))
        assert_array_equal(m.__array__(), np.arange(5))

    def test_gamma_0(self):
        if False:
            return 10
        assert self.mt19937.standard_gamma(0.0) == 0.0
        assert_array_equal(self.mt19937.standard_gamma([0.0]), 0.0)
        actual = self.mt19937.standard_gamma([0.0], dtype='float')
        expected = np.array([0.0], dtype=np.float32)
        assert_array_equal(actual, expected)

    def test_geometric_tiny_prob(self):
        if False:
            while True:
                i = 10
        assert_array_equal(self.mt19937.geometric(p=1e-30, size=3), np.iinfo(np.int64).max)