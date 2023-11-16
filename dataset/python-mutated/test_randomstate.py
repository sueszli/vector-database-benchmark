import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import assert_, assert_raises, assert_equal, assert_warns, assert_no_warnings, assert_array_equal, assert_array_almost_equal, suppress_warnings, IS_WASM
from numpy.random import MT19937, PCG64
from numpy import random
INT_FUNCS = {'binomial': (100.0, 0.6), 'geometric': (0.5,), 'hypergeometric': (20, 20, 10), 'logseries': (0.5,), 'multinomial': (20, np.ones(6) / 6.0), 'negative_binomial': (100, 0.5), 'poisson': (10.0,), 'zipf': (2,)}
if np.iinfo(np.long).max < 2 ** 32:
    INT_FUNC_HASHES = {'binomial': '2fbead005fc63942decb5326d36a1f32fe2c9d32c904ee61e46866b88447c263', 'logseries': '23ead5dcde35d4cfd4ef2c105e4c3d43304b45dc1b1444b7823b9ee4fa144ebb', 'geometric': '0d764db64f5c3bad48c8c33551c13b4d07a1e7b470f77629bef6c985cac76fcf', 'hypergeometric': '7b59bf2f1691626c5815cdcd9a49e1dd68697251d4521575219e4d2a1b8b2c67', 'multinomial': 'd754fa5b92943a38ec07630de92362dd2e02c43577fc147417dc5b9db94ccdd3', 'negative_binomial': '8eb216f7cb2a63cf55605422845caaff002fddc64a7dc8b2d45acd477a49e824', 'poisson': '70c891d76104013ebd6f6bcf30d403a9074b886ff62e4e6b8eb605bf1a4673b7', 'zipf': '01f074f97517cd5d21747148ac6ca4074dde7fcb7acbaec0a936606fecacd93f'}
else:
    INT_FUNC_HASHES = {'binomial': '8626dd9d052cb608e93d8868de0a7b347258b199493871a1dc56e2a26cacb112', 'geometric': '8edd53d272e49c4fc8fbbe6c7d08d563d62e482921f3131d0a0e068af30f0db9', 'hypergeometric': '83496cc4281c77b786c9b7ad88b74d42e01603a55c60577ebab81c3ba8d45657', 'logseries': '65878a38747c176bc00e930ebafebb69d4e1e16cd3a704e264ea8f5e24f548db', 'multinomial': '7a984ae6dca26fd25374479e118b22f55db0aedccd5a0f2584ceada33db98605', 'negative_binomial': 'd636d968e6a24ae92ab52fe11c46ac45b0897e98714426764e820a7d77602a61', 'poisson': '956552176f77e7c9cb20d0118fc9cf690be488d790ed4b4c4747b965e61b0bb4', 'zipf': 'f84ba7feffda41e606e20b28dfc0f1ea9964a74574513d4a4cbc98433a8bfa45'}

@pytest.fixture(scope='module', params=INT_FUNCS)
def int_func(request):
    if False:
        return 10
    return (request.param, INT_FUNCS[request.param], INT_FUNC_HASHES[request.param])

@pytest.fixture
def restore_singleton_bitgen():
    if False:
        print('Hello World!')
    'Ensures that the singleton bitgen is restored after a test'
    orig_bitgen = np.random.get_bit_generator()
    yield
    np.random.set_bit_generator(orig_bitgen)

def assert_mt19937_state_equal(a, b):
    if False:
        return 10
    assert_equal(a['bit_generator'], b['bit_generator'])
    assert_array_equal(a['state']['key'], b['state']['key'])
    assert_array_equal(a['state']['pos'], b['state']['pos'])
    assert_equal(a['has_gauss'], b['has_gauss'])
    assert_equal(a['gauss'], b['gauss'])

class TestSeed:

    def test_scalar(self):
        if False:
            i = 10
            return i + 15
        s = random.RandomState(0)
        assert_equal(s.randint(1000), 684)
        s = random.RandomState(4294967295)
        assert_equal(s.randint(1000), 419)

    def test_array(self):
        if False:
            return 10
        s = random.RandomState(range(10))
        assert_equal(s.randint(1000), 468)
        s = random.RandomState(np.arange(10))
        assert_equal(s.randint(1000), 468)
        s = random.RandomState([0])
        assert_equal(s.randint(1000), 973)
        s = random.RandomState([4294967295])
        assert_equal(s.randint(1000), 265)

    def test_invalid_scalar(self):
        if False:
            print('Hello World!')
        assert_raises(TypeError, random.RandomState, -0.5)
        assert_raises(ValueError, random.RandomState, -1)

    def test_invalid_array(self):
        if False:
            for i in range(10):
                print('nop')
        assert_raises(TypeError, random.RandomState, [-0.5])
        assert_raises(ValueError, random.RandomState, [-1])
        assert_raises(ValueError, random.RandomState, [4294967296])
        assert_raises(ValueError, random.RandomState, [1, 2, 4294967296])
        assert_raises(ValueError, random.RandomState, [1, -2, 4294967296])

    def test_invalid_array_shape(self):
        if False:
            return 10
        assert_raises(ValueError, random.RandomState, np.array([], dtype=np.int64))
        assert_raises(ValueError, random.RandomState, [[1, 2, 3]])
        assert_raises(ValueError, random.RandomState, [[1, 2, 3], [4, 5, 6]])

    def test_cannot_seed(self):
        if False:
            for i in range(10):
                print('nop')
        rs = random.RandomState(PCG64(0))
        with assert_raises(TypeError):
            rs.seed(1234)

    def test_invalid_initialization(self):
        if False:
            i = 10
            return i + 15
        assert_raises(ValueError, random.RandomState, MT19937)

class TestBinomial:

    def test_n_zero(self):
        if False:
            for i in range(10):
                print('nop')
        zeros = np.zeros(2, dtype='int')
        for p in [0, 0.5, 1]:
            assert_(random.binomial(0, p) == 0)
            assert_array_equal(random.binomial(zeros, p), zeros)

    def test_p_is_nan(self):
        if False:
            return 10
        assert_raises(ValueError, random.binomial, 1, np.nan)

class TestMultinomial:

    def test_basic(self):
        if False:
            print('Hello World!')
        random.multinomial(100, [0.2, 0.8])

    def test_zero_probability(self):
        if False:
            return 10
        random.multinomial(100, [0.2, 0.8, 0.0, 0.0, 0.0])

    def test_int_negative_interval(self):
        if False:
            return 10
        assert_(-5 <= random.randint(-5, -1) < -1)
        x = random.randint(-5, -1, 5)
        assert_(np.all(-5 <= x))
        assert_(np.all(x < -1))

    def test_size(self):
        if False:
            for i in range(10):
                print('nop')
        p = [0.5, 0.5]
        assert_equal(random.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.multinomial(1, p, [2, 2]).shape, (2, 2, 2))
        assert_equal(random.multinomial(1, p, (2, 2)).shape, (2, 2, 2))
        assert_equal(random.multinomial(1, p, np.array((2, 2))).shape, (2, 2, 2))
        assert_raises(TypeError, random.multinomial, 1, p, float(1))

    def test_invalid_prob(self):
        if False:
            return 10
        assert_raises(ValueError, random.multinomial, 100, [1.1, 0.2])
        assert_raises(ValueError, random.multinomial, 100, [-0.1, 0.9])

    def test_invalid_n(self):
        if False:
            i = 10
            return i + 15
        assert_raises(ValueError, random.multinomial, -1, [0.8, 0.2])

    def test_p_non_contiguous(self):
        if False:
            i = 10
            return i + 15
        p = np.arange(15.0)
        p /= np.sum(p[1::3])
        pvals = p[1::3]
        random.seed(1432985819)
        non_contig = random.multinomial(100, pvals=pvals)
        random.seed(1432985819)
        contig = random.multinomial(100, pvals=np.ascontiguousarray(pvals))
        assert_array_equal(non_contig, contig)

    def test_multinomial_pvals_float32(self):
        if False:
            while True:
                i = 10
        x = np.array([0.99, 0.99, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09], dtype=np.float32)
        pvals = x / x.sum()
        match = '[\\w\\s]*pvals array is cast to 64-bit floating'
        with pytest.raises(ValueError, match=match):
            random.multinomial(1, pvals)

    def test_multinomial_n_float(self):
        if False:
            return 10
        random.multinomial(100.5, [0.2, 0.8])

class TestSetState:

    def setup_method(self):
        if False:
            print('Hello World!')
        self.seed = 1234567890
        self.random_state = random.RandomState(self.seed)
        self.state = self.random_state.get_state()

    def test_basic(self):
        if False:
            return 10
        old = self.random_state.tomaxint(16)
        self.random_state.set_state(self.state)
        new = self.random_state.tomaxint(16)
        assert_(np.all(old == new))

    def test_gaussian_reset(self):
        if False:
            for i in range(10):
                print('nop')
        old = self.random_state.standard_normal(size=3)
        self.random_state.set_state(self.state)
        new = self.random_state.standard_normal(size=3)
        assert_(np.all(old == new))

    def test_gaussian_reset_in_media_res(self):
        if False:
            print('Hello World!')
        self.random_state.standard_normal()
        state = self.random_state.get_state()
        old = self.random_state.standard_normal(size=3)
        self.random_state.set_state(state)
        new = self.random_state.standard_normal(size=3)
        assert_(np.all(old == new))

    def test_backwards_compatibility(self):
        if False:
            for i in range(10):
                print('nop')
        old_state = self.state[:-2]
        x1 = self.random_state.standard_normal(size=16)
        self.random_state.set_state(old_state)
        x2 = self.random_state.standard_normal(size=16)
        self.random_state.set_state(self.state)
        x3 = self.random_state.standard_normal(size=16)
        assert_(np.all(x1 == x2))
        assert_(np.all(x1 == x3))

    def test_negative_binomial(self):
        if False:
            while True:
                i = 10
        self.random_state.negative_binomial(0.5, 0.5)

    def test_get_state_warning(self):
        if False:
            return 10
        rs = random.RandomState(PCG64())
        with suppress_warnings() as sup:
            w = sup.record(RuntimeWarning)
            state = rs.get_state()
            assert_(len(w) == 1)
            assert isinstance(state, dict)
            assert state['bit_generator'] == 'PCG64'

    def test_invalid_legacy_state_setting(self):
        if False:
            print('Hello World!')
        state = self.random_state.get_state()
        new_state = ('Unknown',) + state[1:]
        assert_raises(ValueError, self.random_state.set_state, new_state)
        assert_raises(TypeError, self.random_state.set_state, np.array(new_state, dtype=object))
        state = self.random_state.get_state(legacy=False)
        del state['bit_generator']
        assert_raises(ValueError, self.random_state.set_state, state)

    def test_pickle(self):
        if False:
            print('Hello World!')
        self.random_state.seed(0)
        self.random_state.random_sample(100)
        self.random_state.standard_normal()
        pickled = self.random_state.get_state(legacy=False)
        assert_equal(pickled['has_gauss'], 1)
        rs_unpick = pickle.loads(pickle.dumps(self.random_state))
        unpickled = rs_unpick.get_state(legacy=False)
        assert_mt19937_state_equal(pickled, unpickled)

    def test_state_setting(self):
        if False:
            return 10
        attr_state = self.random_state.__getstate__()
        self.random_state.standard_normal()
        self.random_state.__setstate__(attr_state)
        state = self.random_state.get_state(legacy=False)
        assert_mt19937_state_equal(attr_state, state)

    def test_repr(self):
        if False:
            for i in range(10):
                print('nop')
        assert repr(self.random_state).startswith('RandomState(MT19937)')

class TestRandint:
    rfunc = random.randint
    itype = [np.bool_, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]

    def test_unsupported_type(self):
        if False:
            while True:
                i = 10
        assert_raises(TypeError, self.rfunc, 1, dtype=float)

    def test_bounds_checking(self):
        if False:
            return 10
        for dt in self.itype:
            lbnd = 0 if dt is np.bool_ else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool_ else np.iinfo(dt).max + 1
            assert_raises(ValueError, self.rfunc, lbnd - 1, ubnd, dtype=dt)
            assert_raises(ValueError, self.rfunc, lbnd, ubnd + 1, dtype=dt)
            assert_raises(ValueError, self.rfunc, ubnd, lbnd, dtype=dt)
            assert_raises(ValueError, self.rfunc, 1, 0, dtype=dt)

    def test_rng_zero_and_extremes(self):
        if False:
            while True:
                i = 10
        for dt in self.itype:
            lbnd = 0 if dt is np.bool_ else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool_ else np.iinfo(dt).max + 1
            tgt = ubnd - 1
            assert_equal(self.rfunc(tgt, tgt + 1, size=1000, dtype=dt), tgt)
            tgt = lbnd
            assert_equal(self.rfunc(tgt, tgt + 1, size=1000, dtype=dt), tgt)
            tgt = (lbnd + ubnd) // 2
            assert_equal(self.rfunc(tgt, tgt + 1, size=1000, dtype=dt), tgt)

    def test_full_range(self):
        if False:
            i = 10
            return i + 15
        for dt in self.itype:
            lbnd = 0 if dt is np.bool_ else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool_ else np.iinfo(dt).max + 1
            try:
                self.rfunc(lbnd, ubnd, dtype=dt)
            except Exception as e:
                raise AssertionError('No error should have been raised, but one was with the following message:\n\n%s' % str(e))

    def test_in_bounds_fuzz(self):
        if False:
            for i in range(10):
                print('nop')
        random.seed()
        for dt in self.itype[1:]:
            for ubnd in [4, 8, 16]:
                vals = self.rfunc(2, ubnd, size=2 ** 16, dtype=dt)
                assert_(vals.max() < ubnd)
                assert_(vals.min() >= 2)
        vals = self.rfunc(0, 2, size=2 ** 16, dtype=np.bool_)
        assert_(vals.max() < 2)
        assert_(vals.min() >= 0)

    def test_repeatability(self):
        if False:
            print('Hello World!')
        tgt = {'bool': '509aea74d792fb931784c4b0135392c65aec64beee12b0cc167548a2c3d31e71', 'int16': '7b07f1a920e46f6d0fe02314155a2330bcfd7635e708da50e536c5ebb631a7d4', 'int32': 'e577bfed6c935de944424667e3da285012e741892dcb7051a8f1ce68ab05c92f', 'int64': '0fbead0b06759df2cfb55e43148822d4a1ff953c7eb19a5b08445a63bb64fa9e', 'int8': '001aac3a5acb935a9b186cbe14a1ca064b8bb2dd0b045d48abeacf74d0203404', 'uint16': '7b07f1a920e46f6d0fe02314155a2330bcfd7635e708da50e536c5ebb631a7d4', 'uint32': 'e577bfed6c935de944424667e3da285012e741892dcb7051a8f1ce68ab05c92f', 'uint64': '0fbead0b06759df2cfb55e43148822d4a1ff953c7eb19a5b08445a63bb64fa9e', 'uint8': '001aac3a5acb935a9b186cbe14a1ca064b8bb2dd0b045d48abeacf74d0203404'}
        for dt in self.itype[1:]:
            random.seed(1234)
            if sys.byteorder == 'little':
                val = self.rfunc(0, 6, size=1000, dtype=dt)
            else:
                val = self.rfunc(0, 6, size=1000, dtype=dt).byteswap()
            res = hashlib.sha256(val.view(np.int8)).hexdigest()
            assert_(tgt[np.dtype(dt).name] == res)
        random.seed(1234)
        val = self.rfunc(0, 2, size=1000, dtype=bool).view(np.int8)
        res = hashlib.sha256(val).hexdigest()
        assert_(tgt[np.dtype(bool).name] == res)

    @pytest.mark.skipif(np.iinfo('l').max < 2 ** 32, reason='Cannot test with 32-bit C long')
    def test_repeatability_32bit_boundary_broadcasting(self):
        if False:
            while True:
                i = 10
        desired = np.array([[[3992670689, 2438360420, 2557845020], [4107320065, 4142558326, 3216529513], [1605979228, 2807061240, 665605495]], [[3211410639, 4128781000, 457175120], [1712592594, 1282922662, 3081439808], [3997822960, 2008322436, 1563495165]], [[1398375547, 4269260146, 115316740], [3414372578, 3437564012, 2112038651], [3572980305, 2260248732, 3908238631]], [[2561372503, 223155946, 3127879445], [441282060, 3514786552, 2148440361], [1629275283, 3479737011, 3003195987]], [[412181688, 940383289, 3047321305], [2978368172, 764731833, 2282559898], [105711276, 720447391, 3596512484]]])
        for size in [None, (5, 3, 3)]:
            random.seed(12345)
            x = self.rfunc([[-1], [0], [1]], [2 ** 32 - 1, 2 ** 32, 2 ** 32 + 1], size=size)
            assert_array_equal(x, desired if size is not None else desired[0])

    def test_int64_uint64_corner_case(self):
        if False:
            print('Hello World!')
        dt = np.int64
        tgt = np.iinfo(np.int64).max
        lbnd = np.int64(np.iinfo(np.int64).max)
        ubnd = np.uint64(np.iinfo(np.int64).max + 1)
        actual = random.randint(lbnd, ubnd, dtype=dt)
        assert_equal(actual, tgt)

    def test_respect_dtype_singleton(self):
        if False:
            while True:
                i = 10
        for dt in self.itype:
            lbnd = 0 if dt is np.bool_ else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool_ else np.iinfo(dt).max + 1
            sample = self.rfunc(lbnd, ubnd, dtype=dt)
            assert_equal(sample.dtype, np.dtype(dt))
        for dt in (bool, int):
            op_dtype = 'long' if dt is int else 'bool'
            lbnd = 0 if dt is bool else np.iinfo(op_dtype).min
            ubnd = 2 if dt is bool else np.iinfo(op_dtype).max + 1
            sample = self.rfunc(lbnd, ubnd, dtype=dt)
            assert_(not hasattr(sample, 'dtype'))
            assert_equal(type(sample), dt)

class TestRandomDist:

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        self.seed = 1234567890

    def test_rand(self):
        if False:
            while True:
                i = 10
        random.seed(self.seed)
        actual = random.rand(3, 2)
        desired = np.array([[0.61879477158568, 0.5916236277597466], [0.8886835890444966, 0.8916548001156082], [0.4575674820298663, 0.7781880808593471]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_rand_singleton(self):
        if False:
            for i in range(10):
                print('nop')
        random.seed(self.seed)
        actual = random.rand()
        desired = 0.61879477158568
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_randn(self):
        if False:
            print('Hello World!')
        random.seed(self.seed)
        actual = random.randn(3, 2)
        desired = np.array([[1.3401634577186312, 1.7375912277193608], [1.498988344300628, -0.2286433324536169], [2.031033998682787, 2.1703249460565526]])
        assert_array_almost_equal(actual, desired, decimal=15)
        random.seed(self.seed)
        actual = random.randn()
        assert_array_almost_equal(actual, desired[0, 0], decimal=15)

    def test_randint(self):
        if False:
            print('Hello World!')
        random.seed(self.seed)
        actual = random.randint(-99, 99, size=(3, 2))
        desired = np.array([[31, 3], [-52, 41], [-48, -66]])
        assert_array_equal(actual, desired)

    def test_random_integers(self):
        if False:
            return 10
        random.seed(self.seed)
        with suppress_warnings() as sup:
            w = sup.record(DeprecationWarning)
            actual = random.random_integers(-99, 99, size=(3, 2))
            assert_(len(w) == 1)
        desired = np.array([[31, 3], [-52, 41], [-48, -66]])
        assert_array_equal(actual, desired)
        random.seed(self.seed)
        with suppress_warnings() as sup:
            w = sup.record(DeprecationWarning)
            actual = random.random_integers(198, size=(3, 2))
            assert_(len(w) == 1)
        assert_array_equal(actual, desired + 100)

    def test_tomaxint(self):
        if False:
            for i in range(10):
                print('nop')
        random.seed(self.seed)
        rs = random.RandomState(self.seed)
        actual = rs.tomaxint(size=(3, 2))
        if np.iinfo(np.long).max == 2147483647:
            desired = np.array([[1328851649, 731237375], [1270502067, 320041495], [1908433478, 499156889]], dtype=np.int64)
        else:
            desired = np.array([[5707374374421908479, 5456764827585442327], [8196659375100692377, 8224063923314595285], [4220315081820346526, 7177518203184491332]], dtype=np.int64)
        assert_equal(actual, desired)
        rs.seed(self.seed)
        actual = rs.tomaxint()
        assert_equal(actual, desired[0, 0])

    def test_random_integers_max_int(self):
        if False:
            while True:
                i = 10
        with suppress_warnings() as sup:
            w = sup.record(DeprecationWarning)
            actual = random.random_integers(np.iinfo('l').max, np.iinfo('l').max)
            assert_(len(w) == 1)
        desired = np.iinfo('l').max
        assert_equal(actual, desired)
        with suppress_warnings() as sup:
            w = sup.record(DeprecationWarning)
            typer = np.dtype('l').type
            actual = random.random_integers(typer(np.iinfo('l').max), typer(np.iinfo('l').max))
            assert_(len(w) == 1)
        assert_equal(actual, desired)

    def test_random_integers_deprecated(self):
        if False:
            for i in range(10):
                print('nop')
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            assert_raises(DeprecationWarning, random.random_integers, np.iinfo('l').max)
            assert_raises(DeprecationWarning, random.random_integers, np.iinfo('l').max, np.iinfo('l').max)

    def test_random_sample(self):
        if False:
            for i in range(10):
                print('nop')
        random.seed(self.seed)
        actual = random.random_sample((3, 2))
        desired = np.array([[0.61879477158568, 0.5916236277597466], [0.8886835890444966, 0.8916548001156082], [0.4575674820298663, 0.7781880808593471]])
        assert_array_almost_equal(actual, desired, decimal=15)
        random.seed(self.seed)
        actual = random.random_sample()
        assert_array_almost_equal(actual, desired[0, 0], decimal=15)

    def test_choice_uniform_replace(self):
        if False:
            for i in range(10):
                print('nop')
        random.seed(self.seed)
        actual = random.choice(4, 4)
        desired = np.array([2, 3, 2, 3])
        assert_array_equal(actual, desired)

    def test_choice_nonuniform_replace(self):
        if False:
            while True:
                i = 10
        random.seed(self.seed)
        actual = random.choice(4, 4, p=[0.4, 0.4, 0.1, 0.1])
        desired = np.array([1, 1, 2, 2])
        assert_array_equal(actual, desired)

    def test_choice_uniform_noreplace(self):
        if False:
            print('Hello World!')
        random.seed(self.seed)
        actual = random.choice(4, 3, replace=False)
        desired = np.array([0, 1, 3])
        assert_array_equal(actual, desired)

    def test_choice_nonuniform_noreplace(self):
        if False:
            return 10
        random.seed(self.seed)
        actual = random.choice(4, 3, replace=False, p=[0.1, 0.3, 0.5, 0.1])
        desired = np.array([2, 3, 1])
        assert_array_equal(actual, desired)

    def test_choice_noninteger(self):
        if False:
            for i in range(10):
                print('nop')
        random.seed(self.seed)
        actual = random.choice(['a', 'b', 'c', 'd'], 4)
        desired = np.array(['c', 'd', 'c', 'd'])
        assert_array_equal(actual, desired)

    def test_choice_exceptions(self):
        if False:
            i = 10
            return i + 15
        sample = random.choice
        assert_raises(ValueError, sample, -1, 3)
        assert_raises(ValueError, sample, 3.0, 3)
        assert_raises(ValueError, sample, [[1, 2], [3, 4]], 3)
        assert_raises(ValueError, sample, [], 3)
        assert_raises(ValueError, sample, [1, 2, 3, 4], 3, p=[[0.25, 0.25], [0.25, 0.25]])
        assert_raises(ValueError, sample, [1, 2], 3, p=[0.4, 0.4, 0.2])
        assert_raises(ValueError, sample, [1, 2], 3, p=[1.1, -0.1])
        assert_raises(ValueError, sample, [1, 2], 3, p=[0.4, 0.4])
        assert_raises(ValueError, sample, [1, 2, 3], 4, replace=False)
        assert_raises(ValueError, sample, [1, 2, 3], -2, replace=False)
        assert_raises(ValueError, sample, [1, 2, 3], (-1,), replace=False)
        assert_raises(ValueError, sample, [1, 2, 3], (-1, 1), replace=False)
        assert_raises(ValueError, sample, [1, 2, 3], 2, replace=False, p=[1, 0, 0])

    def test_choice_return_shape(self):
        if False:
            while True:
                i = 10
        p = [0.1, 0.9]
        assert_(np.isscalar(random.choice(2, replace=True)))
        assert_(np.isscalar(random.choice(2, replace=False)))
        assert_(np.isscalar(random.choice(2, replace=True, p=p)))
        assert_(np.isscalar(random.choice(2, replace=False, p=p)))
        assert_(np.isscalar(random.choice([1, 2], replace=True)))
        assert_(random.choice([None], replace=True) is None)
        a = np.array([1, 2])
        arr = np.empty(1, dtype=object)
        arr[0] = a
        assert_(random.choice(arr, replace=True) is a)
        s = tuple()
        assert_(not np.isscalar(random.choice(2, s, replace=True)))
        assert_(not np.isscalar(random.choice(2, s, replace=False)))
        assert_(not np.isscalar(random.choice(2, s, replace=True, p=p)))
        assert_(not np.isscalar(random.choice(2, s, replace=False, p=p)))
        assert_(not np.isscalar(random.choice([1, 2], s, replace=True)))
        assert_(random.choice([None], s, replace=True).ndim == 0)
        a = np.array([1, 2])
        arr = np.empty(1, dtype=object)
        arr[0] = a
        assert_(random.choice(arr, s, replace=True).item() is a)
        s = (2, 3)
        p = [0.1, 0.1, 0.1, 0.1, 0.4, 0.2]
        assert_equal(random.choice(6, s, replace=True).shape, s)
        assert_equal(random.choice(6, s, replace=False).shape, s)
        assert_equal(random.choice(6, s, replace=True, p=p).shape, s)
        assert_equal(random.choice(6, s, replace=False, p=p).shape, s)
        assert_equal(random.choice(np.arange(6), s, replace=True).shape, s)
        assert_equal(random.randint(0, 0, size=(3, 0, 4)).shape, (3, 0, 4))
        assert_equal(random.randint(0, -10, size=0).shape, (0,))
        assert_equal(random.randint(10, 10, size=0).shape, (0,))
        assert_equal(random.choice(0, size=0).shape, (0,))
        assert_equal(random.choice([], size=(0,)).shape, (0,))
        assert_equal(random.choice(['a', 'b'], size=(3, 0, 4)).shape, (3, 0, 4))
        assert_raises(ValueError, random.choice, [], 10)

    def test_choice_nan_probabilities(self):
        if False:
            return 10
        a = np.array([42, 1, 2])
        p = [None, None, None]
        assert_raises(ValueError, random.choice, a, p=p)

    def test_choice_p_non_contiguous(self):
        if False:
            while True:
                i = 10
        p = np.ones(10) / 5
        p[1::2] = 3.0
        random.seed(self.seed)
        non_contig = random.choice(5, 3, p=p[::2])
        random.seed(self.seed)
        contig = random.choice(5, 3, p=np.ascontiguousarray(p[::2]))
        assert_array_equal(non_contig, contig)

    def test_bytes(self):
        if False:
            for i in range(10):
                print('nop')
        random.seed(self.seed)
        actual = random.bytes(10)
        desired = b'\x82Ui\x9e\xff\x97+Wf\xa5'
        assert_equal(actual, desired)

    def test_shuffle(self):
        if False:
            for i in range(10):
                print('nop')
        for conv in [lambda x: np.array([]), lambda x: x, lambda x: np.asarray(x).astype(np.int8), lambda x: np.asarray(x).astype(np.float32), lambda x: np.asarray(x).astype(np.complex64), lambda x: np.asarray(x).astype(object), lambda x: [(i, i) for i in x], lambda x: np.asarray([[i, i] for i in x]), lambda x: np.vstack([x, x]).T, lambda x: np.asarray([(i, i) for i in x], [('a', int), ('b', int)]).view(np.recarray), lambda x: np.asarray([(i, i) for i in x], [('a', object, (1,)), ('b', np.int32, (1,))])]:
            random.seed(self.seed)
            alist = conv([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
            random.shuffle(alist)
            actual = alist
            desired = conv([0, 1, 9, 6, 2, 4, 5, 8, 7, 3])
            assert_array_equal(actual, desired)

    def test_shuffle_masked(self):
        if False:
            for i in range(10):
                print('nop')
        a = np.ma.masked_values(np.reshape(range(20), (5, 4)) % 3 - 1, -1)
        b = np.ma.masked_values(np.arange(20) % 3 - 1, -1)
        a_orig = a.copy()
        b_orig = b.copy()
        for i in range(50):
            random.shuffle(a)
            assert_equal(sorted(a.data[~a.mask]), sorted(a_orig.data[~a_orig.mask]))
            random.shuffle(b)
            assert_equal(sorted(b.data[~b.mask]), sorted(b_orig.data[~b_orig.mask]))

        def test_shuffle_invalid_objects(self):
            if False:
                i = 10
                return i + 15
            x = np.array(3)
            assert_raises(TypeError, random.shuffle, x)

    def test_permutation(self):
        if False:
            while True:
                i = 10
        random.seed(self.seed)
        alist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        actual = random.permutation(alist)
        desired = [0, 1, 9, 6, 2, 4, 5, 8, 7, 3]
        assert_array_equal(actual, desired)
        random.seed(self.seed)
        arr_2d = np.atleast_2d([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]).T
        actual = random.permutation(arr_2d)
        assert_array_equal(actual, np.atleast_2d(desired).T)
        random.seed(self.seed)
        bad_x_str = 'abcd'
        assert_raises(IndexError, random.permutation, bad_x_str)
        random.seed(self.seed)
        bad_x_float = 1.2
        assert_raises(IndexError, random.permutation, bad_x_float)
        integer_val = 10
        desired = [9, 0, 8, 5, 1, 3, 4, 7, 6, 2]
        random.seed(self.seed)
        actual = random.permutation(integer_val)
        assert_array_equal(actual, desired)

    def test_beta(self):
        if False:
            for i in range(10):
                print('nop')
        random.seed(self.seed)
        actual = random.beta(0.1, 0.9, size=(3, 2))
        desired = np.array([[0.014534185051374606, 0.0005312976156628681], [1.8536661905843232e-06, 0.004192145168001106], [0.0001584051551084981, 0.00012625289194939765]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_binomial(self):
        if False:
            print('Hello World!')
        random.seed(self.seed)
        actual = random.binomial(100.123, 0.456, size=(3, 2))
        desired = np.array([[37, 43], [42, 48], [46, 45]])
        assert_array_equal(actual, desired)
        random.seed(self.seed)
        actual = random.binomial(100.123, 0.456)
        desired = 37
        assert_array_equal(actual, desired)

    def test_chisquare(self):
        if False:
            i = 10
            return i + 15
        random.seed(self.seed)
        actual = random.chisquare(50, size=(3, 2))
        desired = np.array([[63.878581755010906, 68.6840774891137], [65.77116116901506, 47.096867624389745], [72.38284031996952, 74.18408615260374]])
        assert_array_almost_equal(actual, desired, decimal=13)

    def test_dirichlet(self):
        if False:
            return 10
        random.seed(self.seed)
        alpha = np.array([51.72840233779265, 39.74494232180944])
        actual = random.dirichlet(alpha, size=(3, 2))
        desired = np.array([[[0.5453944457361156, 0.4546055542638844], [0.6234581682203941, 0.376541831779606]], [[0.5520600008578578, 0.44793999914214233], [0.589640233051543, 0.4103597669484569]], [[0.5926690928064783, 0.4073309071935218], [0.5697443174397521, 0.430255682560248]]])
        assert_array_almost_equal(actual, desired, decimal=15)
        bad_alpha = np.array([0.54, -1e-16])
        assert_raises(ValueError, random.dirichlet, bad_alpha)
        random.seed(self.seed)
        alpha = np.array([51.72840233779265, 39.74494232180944])
        actual = random.dirichlet(alpha)
        assert_array_almost_equal(actual, desired[0, 0], decimal=15)

    def test_dirichlet_size(self):
        if False:
            i = 10
            return i + 15
        p = np.array([51.72840233779265, 39.74494232180944])
        assert_equal(random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.dirichlet(p, [2, 2]).shape, (2, 2, 2))
        assert_equal(random.dirichlet(p, (2, 2)).shape, (2, 2, 2))
        assert_equal(random.dirichlet(p, np.array((2, 2))).shape, (2, 2, 2))
        assert_raises(TypeError, random.dirichlet, p, float(1))

    def test_dirichlet_bad_alpha(self):
        if False:
            return 10
        alpha = np.array([0.54, -1e-16])
        assert_raises(ValueError, random.dirichlet, alpha)

    def test_dirichlet_alpha_non_contiguous(self):
        if False:
            while True:
                i = 10
        a = np.array([51.72840233779265, -1.0, 39.74494232180944])
        alpha = a[::2]
        random.seed(self.seed)
        non_contig = random.dirichlet(alpha, size=(3, 2))
        random.seed(self.seed)
        contig = random.dirichlet(np.ascontiguousarray(alpha), size=(3, 2))
        assert_array_almost_equal(non_contig, contig)

    def test_exponential(self):
        if False:
            return 10
        random.seed(self.seed)
        actual = random.exponential(1.1234, size=(3, 2))
        desired = np.array([[1.0834264977501162, 1.0060788992455731], [2.466288300852167, 2.496681068099239], [0.6871743346136344, 1.6917566699357598]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_exponential_0(self):
        if False:
            for i in range(10):
                print('nop')
        assert_equal(random.exponential(scale=0), 0)
        assert_raises(ValueError, random.exponential, scale=-0.0)

    def test_f(self):
        if False:
            print('Hello World!')
        random.seed(self.seed)
        actual = random.f(12, 77, size=(3, 2))
        desired = np.array([[1.2197539441857588, 1.7513575979155978], [1.4480311501714649, 1.2210895948039626], [1.0217697575774063, 1.3443182762330042]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_gamma(self):
        if False:
            i = 10
            return i + 15
        random.seed(self.seed)
        actual = random.gamma(5, 3, size=(3, 2))
        desired = np.array([[24.605091886492872, 28.549935632072106], [26.134761102040642, 12.56988482927716], [31.718632757899606, 33.30143302795922]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_gamma_0(self):
        if False:
            for i in range(10):
                print('nop')
        assert_equal(random.gamma(shape=0, scale=0), 0)
        assert_raises(ValueError, random.gamma, shape=-0.0, scale=-0.0)

    def test_geometric(self):
        if False:
            i = 10
            return i + 15
        random.seed(self.seed)
        actual = random.geometric(0.123456789, size=(3, 2))
        desired = np.array([[8, 7], [17, 17], [5, 12]])
        assert_array_equal(actual, desired)

    def test_geometric_exceptions(self):
        if False:
            while True:
                i = 10
        assert_raises(ValueError, random.geometric, 1.1)
        assert_raises(ValueError, random.geometric, [1.1] * 10)
        assert_raises(ValueError, random.geometric, -0.1)
        assert_raises(ValueError, random.geometric, [-0.1] * 10)
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning)
            assert_raises(ValueError, random.geometric, np.nan)
            assert_raises(ValueError, random.geometric, [np.nan] * 10)

    def test_gumbel(self):
        if False:
            while True:
                i = 10
        random.seed(self.seed)
        actual = random.gumbel(loc=0.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[0.19591898743416816, 0.34405539668096674], [-1.4492522252274278, -1.4737481629844686], [1.1065109047880342, -0.6953584862623617]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_gumbel_0(self):
        if False:
            i = 10
            return i + 15
        assert_equal(random.gumbel(scale=0), 0)
        assert_raises(ValueError, random.gumbel, scale=-0.0)

    def test_hypergeometric(self):
        if False:
            for i in range(10):
                print('nop')
        random.seed(self.seed)
        actual = random.hypergeometric(10.1, 5.5, 14, size=(3, 2))
        desired = np.array([[10, 10], [10, 10], [9, 9]])
        assert_array_equal(actual, desired)
        actual = random.hypergeometric(5, 0, 3, size=4)
        desired = np.array([3, 3, 3, 3])
        assert_array_equal(actual, desired)
        actual = random.hypergeometric(15, 0, 12, size=4)
        desired = np.array([12, 12, 12, 12])
        assert_array_equal(actual, desired)
        actual = random.hypergeometric(0, 5, 3, size=4)
        desired = np.array([0, 0, 0, 0])
        assert_array_equal(actual, desired)
        actual = random.hypergeometric(0, 15, 12, size=4)
        desired = np.array([0, 0, 0, 0])
        assert_array_equal(actual, desired)

    def test_laplace(self):
        if False:
            i = 10
            return i + 15
        random.seed(self.seed)
        actual = random.laplace(loc=0.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[0.6659972111276016, 0.5282945255222194], [3.1279195951440713, 3.18202813572992], [-0.05391065675859356, 1.7490133624283732]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_laplace_0(self):
        if False:
            for i in range(10):
                print('nop')
        assert_equal(random.laplace(scale=0), 0)
        assert_raises(ValueError, random.laplace, scale=-0.0)

    def test_logistic(self):
        if False:
            print('Hello World!')
        random.seed(self.seed)
        actual = random.logistic(loc=0.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[1.0923283530501144, 0.8648196662399954], [4.278185906949502, 4.338970063469297], [-0.21682183359214885, 2.6337336538606033]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_lognormal(self):
        if False:
            while True:
                i = 10
        random.seed(self.seed)
        actual = random.lognormal(mean=0.123456789, sigma=2.0, size=(3, 2))
        desired = np.array([[16.506986316888838, 36.54846706092655], [22.678865999812817, 0.7161756105899577], [65.72798501792724, 86.84341601437161]])
        assert_array_almost_equal(actual, desired, decimal=13)

    def test_lognormal_0(self):
        if False:
            for i in range(10):
                print('nop')
        assert_equal(random.lognormal(sigma=0), 1)
        assert_raises(ValueError, random.lognormal, sigma=-0.0)

    def test_logseries(self):
        if False:
            print('Hello World!')
        random.seed(self.seed)
        actual = random.logseries(p=0.923456789, size=(3, 2))
        desired = np.array([[2, 2], [6, 17], [3, 6]])
        assert_array_equal(actual, desired)

    def test_logseries_zero(self):
        if False:
            for i in range(10):
                print('nop')
        assert random.logseries(0) == 1

    @pytest.mark.parametrize('value', [np.nextafter(0.0, -1), 1.0, np.nan, 5.0])
    def test_logseries_exceptions(self, value):
        if False:
            i = 10
            return i + 15
        with np.errstate(invalid='ignore'):
            with pytest.raises(ValueError):
                random.logseries(value)
            with pytest.raises(ValueError):
                random.logseries(np.array([value] * 10))
            with pytest.raises(ValueError):
                random.logseries(np.array([value] * 10)[::2])

    def test_multinomial(self):
        if False:
            i = 10
            return i + 15
        random.seed(self.seed)
        actual = random.multinomial(20, [1 / 6.0] * 6, size=(3, 2))
        desired = np.array([[[4, 3, 5, 4, 2, 2], [5, 2, 8, 2, 2, 1]], [[3, 4, 3, 6, 0, 4], [2, 1, 4, 3, 6, 4]], [[4, 4, 2, 5, 2, 3], [4, 3, 4, 2, 3, 4]]])
        assert_array_equal(actual, desired)

    def test_multivariate_normal(self):
        if False:
            while True:
                i = 10
        random.seed(self.seed)
        mean = (0.123456789, 10)
        cov = [[1, 0], [0, 1]]
        size = (3, 2)
        actual = random.multivariate_normal(mean, cov, size)
        desired = np.array([[[1.463620246718631, 11.73759122771936], [1.622445133300628, 9.771356667546383]], [[2.154490787682787, 12.170324946056553], [1.719909438201865, 9.230548443648306]], [[0.689515026297799, 9.880729819607714], [-0.023054015651998, 9.20109662354288]]])
        assert_array_almost_equal(actual, desired, decimal=15)
        actual = random.multivariate_normal(mean, cov)
        desired = np.array([0.895289569463708, 9.17180864067987])
        assert_array_almost_equal(actual, desired, decimal=15)
        mean = [0, 0]
        cov = [[1, 2], [2, 1]]
        assert_warns(RuntimeWarning, random.multivariate_normal, mean, cov)
        assert_no_warnings(random.multivariate_normal, mean, cov, check_valid='ignore')
        assert_raises(ValueError, random.multivariate_normal, mean, cov, check_valid='raise')
        cov = np.array([[1, 0.1], [0.1, 1]], dtype=np.float32)
        with suppress_warnings() as sup:
            random.multivariate_normal(mean, cov)
            w = sup.record(RuntimeWarning)
            assert len(w) == 0
        mu = np.zeros(2)
        cov = np.eye(2)
        assert_raises(ValueError, random.multivariate_normal, mean, cov, check_valid='other')
        assert_raises(ValueError, random.multivariate_normal, np.zeros((2, 1, 1)), cov)
        assert_raises(ValueError, random.multivariate_normal, mu, np.empty((3, 2)))
        assert_raises(ValueError, random.multivariate_normal, mu, np.eye(3))

    def test_negative_binomial(self):
        if False:
            for i in range(10):
                print('nop')
        random.seed(self.seed)
        actual = random.negative_binomial(n=100, p=0.12345, size=(3, 2))
        desired = np.array([[848, 841], [892, 611], [779, 647]])
        assert_array_equal(actual, desired)

    def test_negative_binomial_exceptions(self):
        if False:
            for i in range(10):
                print('nop')
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning)
            assert_raises(ValueError, random.negative_binomial, 100, np.nan)
            assert_raises(ValueError, random.negative_binomial, 100, [np.nan] * 10)

    def test_noncentral_chisquare(self):
        if False:
            print('Hello World!')
        random.seed(self.seed)
        actual = random.noncentral_chisquare(df=5, nonc=5, size=(3, 2))
        desired = np.array([[23.919053544985175, 13.353246927338263], [31.224526613297364, 16.600473994661773], [5.034615982627246, 17.949730890235195]])
        assert_array_almost_equal(actual, desired, decimal=14)
        actual = random.noncentral_chisquare(df=0.5, nonc=0.2, size=(3, 2))
        desired = np.array([[1.4714537782851667, 0.1505289926801266], [0.00943803056963588, 1.0264725161566617], [0.332334982684171, 0.15451287602753125]])
        assert_array_almost_equal(actual, desired, decimal=14)
        random.seed(self.seed)
        actual = random.noncentral_chisquare(df=5, nonc=0, size=(3, 2))
        desired = np.array([[9.597154162763948, 11.72548445029608], [10.413711048138335, 3.694475922923986], [13.484222138963087, 14.377255424602957]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_noncentral_f(self):
        if False:
            return 10
        random.seed(self.seed)
        actual = random.noncentral_f(dfnum=5, dfden=2, nonc=1, size=(3, 2))
        desired = np.array([[1.4059809967492667, 0.3420797317928576], [3.5771506926577255, 7.926326625778298], [0.4374159946354416, 1.177420875242832]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_noncentral_f_nan(self):
        if False:
            i = 10
            return i + 15
        random.seed(self.seed)
        actual = random.noncentral_f(dfnum=5, dfden=2, nonc=np.nan)
        assert np.isnan(actual)

    def test_normal(self):
        if False:
            print('Hello World!')
        random.seed(self.seed)
        actual = random.normal(loc=0.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[2.8037837044372624, 3.5986392444387216], [3.121433477601256, -0.3338298759072338], [4.185524786365574, 4.464106681113105]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_normal_0(self):
        if False:
            for i in range(10):
                print('nop')
        assert_equal(random.normal(scale=0), 0)
        assert_raises(ValueError, random.normal, scale=-0.0)

    def test_pareto(self):
        if False:
            print('Hello World!')
        random.seed(self.seed)
        actual = random.pareto(a=0.123456789, size=(3, 2))
        desired = np.array([[2468.5246043903485, 1412.8688081051835], [52828779.70294852, 65772098.10473288], [140.84032335039151, 198390.2551352517]])
        np.testing.assert_array_almost_equal_nulp(actual, desired, nulp=30)

    def test_poisson(self):
        if False:
            print('Hello World!')
        random.seed(self.seed)
        actual = random.poisson(lam=0.123456789, size=(3, 2))
        desired = np.array([[0, 0], [1, 0], [0, 0]])
        assert_array_equal(actual, desired)

    def test_poisson_exceptions(self):
        if False:
            for i in range(10):
                print('nop')
        lambig = np.iinfo('l').max
        lamneg = -1
        assert_raises(ValueError, random.poisson, lamneg)
        assert_raises(ValueError, random.poisson, [lamneg] * 10)
        assert_raises(ValueError, random.poisson, lambig)
        assert_raises(ValueError, random.poisson, [lambig] * 10)
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning)
            assert_raises(ValueError, random.poisson, np.nan)
            assert_raises(ValueError, random.poisson, [np.nan] * 10)

    def test_power(self):
        if False:
            print('Hello World!')
        random.seed(self.seed)
        actual = random.power(a=0.123456789, size=(3, 2))
        desired = np.array([[0.02048932883240791, 0.01424192241128213], [0.384460737485353, 0.39499689943484395], [0.00177699707563439, 0.13115505880863756]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_rayleigh(self):
        if False:
            i = 10
            return i + 15
        random.seed(self.seed)
        actual = random.rayleigh(scale=10, size=(3, 2))
        desired = np.array([[13.88824964942484, 13.383318339044731], [20.95413364294492, 21.082850158007126], [11.060665370068543, 17.35468505778271]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_rayleigh_0(self):
        if False:
            for i in range(10):
                print('nop')
        assert_equal(random.rayleigh(scale=0), 0)
        assert_raises(ValueError, random.rayleigh, scale=-0.0)

    def test_standard_cauchy(self):
        if False:
            return 10
        random.seed(self.seed)
        actual = random.standard_cauchy(size=(3, 2))
        desired = np.array([[0.7712766019644534, -6.556011619559106], [0.9358202339115831, -2.0747929301375945], [-4.746016442970119, 0.18338989290760804]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_standard_exponential(self):
        if False:
            while True:
                i = 10
        random.seed(self.seed)
        actual = random.standard_exponential(size=(3, 2))
        desired = np.array([[0.964417391623746, 0.8955660488210551], [2.195378583631981, 2.2224328539249054], [0.6116915921431676, 1.505925467274132]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_standard_gamma(self):
        if False:
            for i in range(10):
                print('nop')
        random.seed(self.seed)
        actual = random.standard_gamma(shape=3, size=(3, 2))
        desired = np.array([[5.508415313184551, 6.629534703019031], [5.939884849437792, 2.31044849402134], [7.548386142313171, 8.012756093271868]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_standard_gamma_0(self):
        if False:
            print('Hello World!')
        assert_equal(random.standard_gamma(shape=0), 0)
        assert_raises(ValueError, random.standard_gamma, shape=-0.0)

    def test_standard_normal(self):
        if False:
            for i in range(10):
                print('nop')
        random.seed(self.seed)
        actual = random.standard_normal(size=(3, 2))
        desired = np.array([[1.3401634577186312, 1.7375912277193608], [1.498988344300628, -0.2286433324536169], [2.031033998682787, 2.1703249460565526]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_randn_singleton(self):
        if False:
            i = 10
            return i + 15
        random.seed(self.seed)
        actual = random.randn()
        desired = np.array(1.3401634577186312)
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_standard_t(self):
        if False:
            i = 10
            return i + 15
        random.seed(self.seed)
        actual = random.standard_t(df=10, size=(3, 2))
        desired = np.array([[0.9714061186265996, -0.08830486548450577], [1.3631114368950532, -0.5531746390986707], [-0.18473749069684214, 0.6118153734175532]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_triangular(self):
        if False:
            for i in range(10):
                print('nop')
        random.seed(self.seed)
        actual = random.triangular(left=5.12, mode=10.23, right=20.34, size=(3, 2))
        desired = np.array([[12.681171789492158, 12.412920614919315], [16.201313773351583, 16.256921387476005], [11.204006909118203, 14.497814483582992]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_uniform(self):
        if False:
            while True:
                i = 10
        random.seed(self.seed)
        actual = random.uniform(low=1.23, high=10.54, size=(3, 2))
        desired = np.array([[6.99097932346268, 6.73801597444324], [9.503644214004263, 9.53130618907631], [5.489953257698055, 8.474931032800521]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_uniform_range_bounds(self):
        if False:
            i = 10
            return i + 15
        fmin = np.finfo('float').min
        fmax = np.finfo('float').max
        func = random.uniform
        assert_raises(OverflowError, func, -np.inf, 0)
        assert_raises(OverflowError, func, 0, np.inf)
        assert_raises(OverflowError, func, fmin, fmax)
        assert_raises(OverflowError, func, [-np.inf], [0])
        assert_raises(OverflowError, func, [0], [np.inf])
        random.uniform(low=np.nextafter(fmin, 1), high=fmax / 1e+17)

    def test_scalar_exception_propagation(self):
        if False:
            print('Hello World!')

        class ThrowingFloat(np.ndarray):

            def __float__(self):
                if False:
                    print('Hello World!')
                raise TypeError
        throwing_float = np.array(1.0).view(ThrowingFloat)
        assert_raises(TypeError, random.uniform, throwing_float, throwing_float)

        class ThrowingInteger(np.ndarray):

            def __int__(self):
                if False:
                    return 10
                raise TypeError
        throwing_int = np.array(1).view(ThrowingInteger)
        assert_raises(TypeError, random.hypergeometric, throwing_int, 1, 1)

    def test_vonmises(self):
        if False:
            i = 10
            return i + 15
        random.seed(self.seed)
        actual = random.vonmises(mu=1.23, kappa=1.54, size=(3, 2))
        desired = np.array([[2.2856757267390204, 2.8916383844228504], [0.38198375564286025, 2.5763802311389075], [1.1915377158835305, 1.8350984968182535]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_vonmises_small(self):
        if False:
            for i in range(10):
                print('nop')
        random.seed(self.seed)
        r = random.vonmises(mu=0.0, kappa=1.1e-08, size=10 ** 6)
        assert_(np.isfinite(r).all())

    def test_vonmises_large(self):
        if False:
            return 10
        random.seed(self.seed)
        actual = random.vonmises(mu=0.0, kappa=10000000.0, size=3)
        desired = np.array([0.0004634253748521111, 0.0003558873596114509, -0.0002337119622577433])
        assert_array_almost_equal(actual, desired, decimal=8)

    def test_vonmises_nan(self):
        if False:
            i = 10
            return i + 15
        random.seed(self.seed)
        r = random.vonmises(mu=0.0, kappa=np.nan)
        assert_(np.isnan(r))

    def test_wald(self):
        if False:
            return 10
        random.seed(self.seed)
        actual = random.wald(mean=1.23, scale=1.54, size=(3, 2))
        desired = np.array([[3.8293526571589, 5.131252491842855], [0.35045403618358717, 1.5083239687200354], [0.24124319895843183, 0.22031101461955038]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_weibull(self):
        if False:
            for i in range(10):
                print('nop')
        random.seed(self.seed)
        actual = random.weibull(a=1.23, size=(3, 2))
        desired = np.array([[0.9709734264876673, 0.9142289644356552], [1.8951777003496293, 1.9141435796047956], [0.6705778375239099, 1.394940466350668]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_weibull_0(self):
        if False:
            i = 10
            return i + 15
        random.seed(self.seed)
        assert_equal(random.weibull(a=0, size=12), np.zeros(12))
        assert_raises(ValueError, random.weibull, a=-0.0)

    def test_zipf(self):
        if False:
            return 10
        random.seed(self.seed)
        actual = random.zipf(a=1.23, size=(3, 2))
        desired = np.array([[66, 29], [1, 1], [3, 13]])
        assert_array_equal(actual, desired)

class TestBroadcast:

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        self.seed = 123456789

    def set_seed(self):
        if False:
            i = 10
            return i + 15
        random.seed(self.seed)

    def test_uniform(self):
        if False:
            while True:
                i = 10
        low = [0]
        high = [1]
        uniform = random.uniform
        desired = np.array([0.532833024789759, 0.5341366008904166, 0.509553035526467])
        self.set_seed()
        actual = uniform(low * 3, high)
        assert_array_almost_equal(actual, desired, decimal=14)
        self.set_seed()
        actual = uniform(low, high * 3)
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_normal(self):
        if False:
            while True:
                i = 10
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        normal = random.normal
        desired = np.array([2.2129019979039612, 2.128397797652002, 1.8417114045748335])
        self.set_seed()
        actual = normal(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, normal, loc * 3, bad_scale)
        self.set_seed()
        actual = normal(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, normal, loc, bad_scale * 3)

    def test_beta(self):
        if False:
            while True:
                i = 10
        a = [1]
        b = [2]
        bad_a = [-1]
        bad_b = [-2]
        beta = random.beta
        desired = np.array([0.19843558305989056, 0.07523033640942364, 0.24976865978980844])
        self.set_seed()
        actual = beta(a * 3, b)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, beta, bad_a * 3, b)
        assert_raises(ValueError, beta, a * 3, bad_b)
        self.set_seed()
        actual = beta(a, b * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, beta, bad_a, b * 3)
        assert_raises(ValueError, beta, a, bad_b * 3)

    def test_exponential(self):
        if False:
            i = 10
            return i + 15
        scale = [1]
        bad_scale = [-1]
        exponential = random.exponential
        desired = np.array([0.7610685365884524, 0.7638628227869165, 0.712438131258918])
        self.set_seed()
        actual = exponential(scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, exponential, bad_scale * 3)

    def test_standard_gamma(self):
        if False:
            print('Hello World!')
        shape = [1]
        bad_shape = [-1]
        std_gamma = random.standard_gamma
        desired = np.array([0.7610685365884524, 0.7638628227869165, 0.712438131258918])
        self.set_seed()
        actual = std_gamma(shape * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, std_gamma, bad_shape * 3)

    def test_gamma(self):
        if False:
            while True:
                i = 10
        shape = [1]
        scale = [2]
        bad_shape = [-1]
        bad_scale = [-2]
        gamma = random.gamma
        desired = np.array([1.5221370731769048, 1.527725645573833, 1.424876262517836])
        self.set_seed()
        actual = gamma(shape * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gamma, bad_shape * 3, scale)
        assert_raises(ValueError, gamma, shape * 3, bad_scale)
        self.set_seed()
        actual = gamma(shape, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gamma, bad_shape, scale * 3)
        assert_raises(ValueError, gamma, shape, bad_scale * 3)

    def test_f(self):
        if False:
            while True:
                i = 10
        dfnum = [1]
        dfden = [2]
        bad_dfnum = [-1]
        bad_dfden = [-2]
        f = random.f
        desired = np.array([0.800389516382648, 0.8676871963536351, 2.72510951683868])
        self.set_seed()
        actual = f(dfnum * 3, dfden)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, f, bad_dfnum * 3, dfden)
        assert_raises(ValueError, f, dfnum * 3, bad_dfden)
        self.set_seed()
        actual = f(dfnum, dfden * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, f, bad_dfnum, dfden * 3)
        assert_raises(ValueError, f, dfnum, bad_dfden * 3)

    def test_noncentral_f(self):
        if False:
            for i in range(10):
                print('nop')
        dfnum = [2]
        dfden = [3]
        nonc = [4]
        bad_dfnum = [0]
        bad_dfden = [-1]
        bad_nonc = [-2]
        nonc_f = random.noncentral_f
        desired = np.array([9.139394326370521, 13.025456344595602, 8.801809835910055])
        self.set_seed()
        actual = nonc_f(dfnum * 3, dfden, nonc)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert np.all(np.isnan(nonc_f(dfnum, dfden, [np.nan] * 3)))
        assert_raises(ValueError, nonc_f, bad_dfnum * 3, dfden, nonc)
        assert_raises(ValueError, nonc_f, dfnum * 3, bad_dfden, nonc)
        assert_raises(ValueError, nonc_f, dfnum * 3, dfden, bad_nonc)
        self.set_seed()
        actual = nonc_f(dfnum, dfden * 3, nonc)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_f, bad_dfnum, dfden * 3, nonc)
        assert_raises(ValueError, nonc_f, dfnum, bad_dfden * 3, nonc)
        assert_raises(ValueError, nonc_f, dfnum, dfden * 3, bad_nonc)
        self.set_seed()
        actual = nonc_f(dfnum, dfden, nonc * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_f, bad_dfnum, dfden, nonc * 3)
        assert_raises(ValueError, nonc_f, dfnum, bad_dfden, nonc * 3)
        assert_raises(ValueError, nonc_f, dfnum, dfden, bad_nonc * 3)

    def test_noncentral_f_small_df(self):
        if False:
            print('Hello World!')
        self.set_seed()
        desired = np.array([6.869638627492048, 0.785880199263955])
        actual = random.noncentral_f(0.9, 0.9, 2, size=2)
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_chisquare(self):
        if False:
            i = 10
            return i + 15
        df = [1]
        bad_df = [-1]
        chisquare = random.chisquare
        desired = np.array([0.5702280113308829, 0.5194770210884078, 0.1320969254923558])
        self.set_seed()
        actual = chisquare(df * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, chisquare, bad_df * 3)

    def test_noncentral_chisquare(self):
        if False:
            while True:
                i = 10
        df = [1]
        nonc = [2]
        bad_df = [-1]
        bad_nonc = [-2]
        nonc_chi = random.noncentral_chisquare
        desired = np.array([9.001559946791376, 4.580413504971874, 6.087230243283456])
        self.set_seed()
        actual = nonc_chi(df * 3, nonc)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_chi, bad_df * 3, nonc)
        assert_raises(ValueError, nonc_chi, df * 3, bad_nonc)
        self.set_seed()
        actual = nonc_chi(df, nonc * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_chi, bad_df, nonc * 3)
        assert_raises(ValueError, nonc_chi, df, bad_nonc * 3)

    def test_standard_t(self):
        if False:
            print('Hello World!')
        df = [1]
        bad_df = [-1]
        t = random.standard_t
        desired = np.array([3.0702872575217643, 5.856072516736161, 1.0274791436474273])
        self.set_seed()
        actual = t(df * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, t, bad_df * 3)
        assert_raises(ValueError, random.standard_t, bad_df * 3)

    def test_vonmises(self):
        if False:
            print('Hello World!')
        mu = [2]
        kappa = [1]
        bad_kappa = [-1]
        vonmises = random.vonmises
        desired = np.array([2.9883443664201312, -2.7064099483995943, -1.8672476700665914])
        self.set_seed()
        actual = vonmises(mu * 3, kappa)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, vonmises, mu * 3, bad_kappa)
        self.set_seed()
        actual = vonmises(mu, kappa * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, vonmises, mu, bad_kappa * 3)

    def test_pareto(self):
        if False:
            i = 10
            return i + 15
        a = [1]
        bad_a = [-1]
        pareto = random.pareto
        desired = np.array([1.1405622680198362, 1.146551976204453, 1.0389564467453547])
        self.set_seed()
        actual = pareto(a * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, pareto, bad_a * 3)
        assert_raises(ValueError, random.pareto, bad_a * 3)

    def test_weibull(self):
        if False:
            while True:
                i = 10
        a = [1]
        bad_a = [-1]
        weibull = random.weibull
        desired = np.array([0.7610685365884524, 0.7638628227869165, 0.712438131258918])
        self.set_seed()
        actual = weibull(a * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, weibull, bad_a * 3)
        assert_raises(ValueError, random.weibull, bad_a * 3)

    def test_power(self):
        if False:
            while True:
                i = 10
        a = [1]
        bad_a = [-1]
        power = random.power
        desired = np.array([0.532833024789759, 0.5341366008904166, 0.509553035526467])
        self.set_seed()
        actual = power(a * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, power, bad_a * 3)
        assert_raises(ValueError, random.power, bad_a * 3)

    def test_laplace(self):
        if False:
            print('Hello World!')
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        laplace = random.laplace
        desired = np.array([0.06792135602850716, 0.07071564222697133, 0.019290950698972624])
        self.set_seed()
        actual = laplace(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, laplace, loc * 3, bad_scale)
        self.set_seed()
        actual = laplace(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, laplace, loc, bad_scale * 3)

    def test_gumbel(self):
        if False:
            for i in range(10):
                print('nop')
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        gumbel = random.gumbel
        desired = np.array([0.2730318639556768, 0.26936705726291116, 0.3390622039303794])
        self.set_seed()
        actual = gumbel(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gumbel, loc * 3, bad_scale)
        self.set_seed()
        actual = gumbel(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gumbel, loc, bad_scale * 3)

    def test_logistic(self):
        if False:
            print('Hello World!')
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        logistic = random.logistic
        desired = np.array([0.1315213583758617, 0.13675915696285773, 0.038216792802833396])
        self.set_seed()
        actual = logistic(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, logistic, loc * 3, bad_scale)
        self.set_seed()
        actual = logistic(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, logistic, loc, bad_scale * 3)
        assert_equal(random.logistic(1.0, 0.0), 1.0)

    def test_lognormal(self):
        if False:
            while True:
                i = 10
        mean = [0]
        sigma = [1]
        bad_sigma = [-1]
        lognormal = random.lognormal
        desired = np.array([9.142208604484843, 8.401395287012626, 6.307323411657867])
        self.set_seed()
        actual = lognormal(mean * 3, sigma)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, lognormal, mean * 3, bad_sigma)
        assert_raises(ValueError, random.lognormal, mean * 3, bad_sigma)
        self.set_seed()
        actual = lognormal(mean, sigma * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, lognormal, mean, bad_sigma * 3)
        assert_raises(ValueError, random.lognormal, mean, bad_sigma * 3)

    def test_rayleigh(self):
        if False:
            while True:
                i = 10
        scale = [1]
        bad_scale = [-1]
        rayleigh = random.rayleigh
        desired = np.array([1.2337491937897689, 1.2360119924878694, 1.193681809578179])
        self.set_seed()
        actual = rayleigh(scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, rayleigh, bad_scale * 3)

    def test_wald(self):
        if False:
            i = 10
            return i + 15
        mean = [0.5]
        scale = [1]
        bad_mean = [0]
        bad_scale = [-2]
        wald = random.wald
        desired = np.array([0.11873681120271318, 0.12450084820795027, 0.9096122728408238])
        self.set_seed()
        actual = wald(mean * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, wald, bad_mean * 3, scale)
        assert_raises(ValueError, wald, mean * 3, bad_scale)
        assert_raises(ValueError, random.wald, bad_mean * 3, scale)
        assert_raises(ValueError, random.wald, mean * 3, bad_scale)
        self.set_seed()
        actual = wald(mean, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, wald, bad_mean, scale * 3)
        assert_raises(ValueError, wald, mean, bad_scale * 3)
        assert_raises(ValueError, wald, 0.0, 1)
        assert_raises(ValueError, wald, 0.5, 0.0)

    def test_triangular(self):
        if False:
            i = 10
            return i + 15
        left = [1]
        right = [3]
        mode = [2]
        bad_left_one = [3]
        bad_mode_one = [4]
        (bad_left_two, bad_mode_two) = right * 2
        triangular = random.triangular
        desired = np.array([2.03339048710429, 2.0347400359389356, 2.0095991069536208])
        self.set_seed()
        actual = triangular(left * 3, mode, right)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, triangular, bad_left_one * 3, mode, right)
        assert_raises(ValueError, triangular, left * 3, bad_mode_one, right)
        assert_raises(ValueError, triangular, bad_left_two * 3, bad_mode_two, right)
        self.set_seed()
        actual = triangular(left, mode * 3, right)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, triangular, bad_left_one, mode * 3, right)
        assert_raises(ValueError, triangular, left, bad_mode_one * 3, right)
        assert_raises(ValueError, triangular, bad_left_two, bad_mode_two * 3, right)
        self.set_seed()
        actual = triangular(left, mode, right * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, triangular, bad_left_one, mode, right * 3)
        assert_raises(ValueError, triangular, left, bad_mode_one, right * 3)
        assert_raises(ValueError, triangular, bad_left_two, bad_mode_two, right * 3)
        assert_raises(ValueError, triangular, 10.0, 0.0, 20.0)
        assert_raises(ValueError, triangular, 10.0, 25.0, 20.0)
        assert_raises(ValueError, triangular, 10.0, 10.0, 10.0)

    def test_binomial(self):
        if False:
            return 10
        n = [1]
        p = [0.5]
        bad_n = [-1]
        bad_p_one = [-1]
        bad_p_two = [1.5]
        binom = random.binomial
        desired = np.array([1, 1, 1])
        self.set_seed()
        actual = binom(n * 3, p)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, binom, bad_n * 3, p)
        assert_raises(ValueError, binom, n * 3, bad_p_one)
        assert_raises(ValueError, binom, n * 3, bad_p_two)
        self.set_seed()
        actual = binom(n, p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, binom, bad_n, p * 3)
        assert_raises(ValueError, binom, n, bad_p_one * 3)
        assert_raises(ValueError, binom, n, bad_p_two * 3)

    def test_negative_binomial(self):
        if False:
            return 10
        n = [1]
        p = [0.5]
        bad_n = [-1]
        bad_p_one = [-1]
        bad_p_two = [1.5]
        neg_binom = random.negative_binomial
        desired = np.array([1, 0, 1])
        self.set_seed()
        actual = neg_binom(n * 3, p)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, neg_binom, bad_n * 3, p)
        assert_raises(ValueError, neg_binom, n * 3, bad_p_one)
        assert_raises(ValueError, neg_binom, n * 3, bad_p_two)
        self.set_seed()
        actual = neg_binom(n, p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, neg_binom, bad_n, p * 3)
        assert_raises(ValueError, neg_binom, n, bad_p_one * 3)
        assert_raises(ValueError, neg_binom, n, bad_p_two * 3)

    def test_poisson(self):
        if False:
            while True:
                i = 10
        max_lam = random.RandomState()._poisson_lam_max
        lam = [1]
        bad_lam_one = [-1]
        bad_lam_two = [max_lam * 2]
        poisson = random.poisson
        desired = np.array([1, 1, 0])
        self.set_seed()
        actual = poisson(lam * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, poisson, bad_lam_one * 3)
        assert_raises(ValueError, poisson, bad_lam_two * 3)

    def test_zipf(self):
        if False:
            return 10
        a = [2]
        bad_a = [0]
        zipf = random.zipf
        desired = np.array([2, 2, 1])
        self.set_seed()
        actual = zipf(a * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, zipf, bad_a * 3)
        with np.errstate(invalid='ignore'):
            assert_raises(ValueError, zipf, np.nan)
            assert_raises(ValueError, zipf, [0, 0, np.nan])

    def test_geometric(self):
        if False:
            for i in range(10):
                print('nop')
        p = [0.5]
        bad_p_one = [-1]
        bad_p_two = [1.5]
        geom = random.geometric
        desired = np.array([2, 2, 2])
        self.set_seed()
        actual = geom(p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, geom, bad_p_one * 3)
        assert_raises(ValueError, geom, bad_p_two * 3)

    def test_hypergeometric(self):
        if False:
            while True:
                i = 10
        ngood = [1]
        nbad = [2]
        nsample = [2]
        bad_ngood = [-1]
        bad_nbad = [-2]
        bad_nsample_one = [0]
        bad_nsample_two = [4]
        hypergeom = random.hypergeometric
        desired = np.array([1, 1, 1])
        self.set_seed()
        actual = hypergeom(ngood * 3, nbad, nsample)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, hypergeom, bad_ngood * 3, nbad, nsample)
        assert_raises(ValueError, hypergeom, ngood * 3, bad_nbad, nsample)
        assert_raises(ValueError, hypergeom, ngood * 3, nbad, bad_nsample_one)
        assert_raises(ValueError, hypergeom, ngood * 3, nbad, bad_nsample_two)
        self.set_seed()
        actual = hypergeom(ngood, nbad * 3, nsample)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, hypergeom, bad_ngood, nbad * 3, nsample)
        assert_raises(ValueError, hypergeom, ngood, bad_nbad * 3, nsample)
        assert_raises(ValueError, hypergeom, ngood, nbad * 3, bad_nsample_one)
        assert_raises(ValueError, hypergeom, ngood, nbad * 3, bad_nsample_two)
        self.set_seed()
        actual = hypergeom(ngood, nbad, nsample * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, hypergeom, bad_ngood, nbad, nsample * 3)
        assert_raises(ValueError, hypergeom, ngood, bad_nbad, nsample * 3)
        assert_raises(ValueError, hypergeom, ngood, nbad, bad_nsample_one * 3)
        assert_raises(ValueError, hypergeom, ngood, nbad, bad_nsample_two * 3)
        assert_raises(ValueError, hypergeom, -1, 10, 20)
        assert_raises(ValueError, hypergeom, 10, -1, 20)
        assert_raises(ValueError, hypergeom, 10, 10, 0)
        assert_raises(ValueError, hypergeom, 10, 10, 25)

    def test_logseries(self):
        if False:
            for i in range(10):
                print('nop')
        p = [0.5]
        bad_p_one = [2]
        bad_p_two = [-1]
        logseries = random.logseries
        desired = np.array([1, 1, 1])
        self.set_seed()
        actual = logseries(p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, logseries, bad_p_one * 3)
        assert_raises(ValueError, logseries, bad_p_two * 3)

@pytest.mark.skipif(IS_WASM, reason="can't start thread")
class TestThread:

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        self.seeds = range(4)

    def check_function(self, function, sz):
        if False:
            print('Hello World!')
        from threading import Thread
        out1 = np.empty((len(self.seeds),) + sz)
        out2 = np.empty((len(self.seeds),) + sz)
        t = [Thread(target=function, args=(random.RandomState(s), o)) for (s, o) in zip(self.seeds, out1)]
        [x.start() for x in t]
        [x.join() for x in t]
        for (s, o) in zip(self.seeds, out2):
            function(random.RandomState(s), o)
        if np.intp().dtype.itemsize == 4 and sys.platform == 'win32':
            assert_array_almost_equal(out1, out2)
        else:
            assert_array_equal(out1, out2)

    def test_normal(self):
        if False:
            while True:
                i = 10

        def gen_random(state, out):
            if False:
                return 10
            out[...] = state.normal(size=10000)
        self.check_function(gen_random, sz=(10000,))

    def test_exp(self):
        if False:
            for i in range(10):
                print('nop')

        def gen_random(state, out):
            if False:
                i = 10
                return i + 15
            out[...] = state.exponential(scale=np.ones((100, 1000)))
        self.check_function(gen_random, sz=(100, 1000))

    def test_multinomial(self):
        if False:
            return 10

        def gen_random(state, out):
            if False:
                while True:
                    i = 10
            out[...] = state.multinomial(10, [1 / 6.0] * 6, size=10000)
        self.check_function(gen_random, sz=(10000, 6))

class TestSingleEltArrayInput:

    def setup_method(self):
        if False:
            while True:
                i = 10
        self.argOne = np.array([2])
        self.argTwo = np.array([3])
        self.argThree = np.array([4])
        self.tgtShape = (1,)

    def test_one_arg_funcs(self):
        if False:
            for i in range(10):
                print('nop')
        funcs = (random.exponential, random.standard_gamma, random.chisquare, random.standard_t, random.pareto, random.weibull, random.power, random.rayleigh, random.poisson, random.zipf, random.geometric, random.logseries)
        probfuncs = (random.geometric, random.logseries)
        for func in funcs:
            if func in probfuncs:
                out = func(np.array([0.5]))
            else:
                out = func(self.argOne)
            assert_equal(out.shape, self.tgtShape)

    def test_two_arg_funcs(self):
        if False:
            print('Hello World!')
        funcs = (random.uniform, random.normal, random.beta, random.gamma, random.f, random.noncentral_chisquare, random.vonmises, random.laplace, random.gumbel, random.logistic, random.lognormal, random.wald, random.binomial, random.negative_binomial)
        probfuncs = (random.binomial, random.negative_binomial)
        for func in funcs:
            if func in probfuncs:
                argTwo = np.array([0.5])
            else:
                argTwo = self.argTwo
            out = func(self.argOne, argTwo)
            assert_equal(out.shape, self.tgtShape)
            out = func(self.argOne[0], argTwo)
            assert_equal(out.shape, self.tgtShape)
            out = func(self.argOne, argTwo[0])
            assert_equal(out.shape, self.tgtShape)

    def test_three_arg_funcs(self):
        if False:
            print('Hello World!')
        funcs = [random.noncentral_f, random.triangular, random.hypergeometric]
        for func in funcs:
            out = func(self.argOne, self.argTwo, self.argThree)
            assert_equal(out.shape, self.tgtShape)
            out = func(self.argOne[0], self.argTwo, self.argThree)
            assert_equal(out.shape, self.tgtShape)
            out = func(self.argOne, self.argTwo[0], self.argThree)
            assert_equal(out.shape, self.tgtShape)

def test_integer_dtype(int_func):
    if False:
        i = 10
        return i + 15
    random.seed(123456789)
    (fname, args, sha256) = int_func
    f = getattr(random, fname)
    actual = f(*args, size=2)
    assert_(actual.dtype == np.dtype('l'))

def test_integer_repeat(int_func):
    if False:
        for i in range(10):
            print('nop')
    random.seed(123456789)
    (fname, args, sha256) = int_func
    f = getattr(random, fname)
    val = f(*args, size=1000000)
    if sys.byteorder != 'little':
        val = val.byteswap()
    res = hashlib.sha256(val.view(np.int8)).hexdigest()
    assert_(res == sha256)

def test_broadcast_size_error():
    if False:
        print('Hello World!')
    with pytest.raises(ValueError):
        random.binomial(1, [0.3, 0.7], size=(2, 1))
    with pytest.raises(ValueError):
        random.binomial([1, 2], 0.3, size=(2, 1))
    with pytest.raises(ValueError):
        random.binomial([1, 2], [0.3, 0.7], size=(2, 1))

def test_randomstate_ctor_old_style_pickle():
    if False:
        return 10
    rs = np.random.RandomState(MT19937(0))
    rs.standard_normal(1)
    (ctor, args, state_a) = rs.__reduce__()
    assert args[:1] == ('MT19937',)
    b = ctor(*args[:1])
    b.set_state(state_a)
    state_b = b.get_state(legacy=False)
    assert_equal(state_a['bit_generator'], state_b['bit_generator'])
    assert_array_equal(state_a['state']['key'], state_b['state']['key'])
    assert_array_equal(state_a['state']['pos'], state_b['state']['pos'])
    assert_equal(state_a['has_gauss'], state_b['has_gauss'])
    assert_equal(state_a['gauss'], state_b['gauss'])

def test_hot_swap(restore_singleton_bitgen):
    if False:
        print('Hello World!')
    def_bg = np.random.default_rng(0)
    bg = def_bg.bit_generator
    np.random.set_bit_generator(bg)
    assert isinstance(np.random.mtrand._rand._bit_generator, type(bg))
    second_bg = np.random.get_bit_generator()
    assert bg is second_bg

def test_seed_alt_bit_gen(restore_singleton_bitgen):
    if False:
        while True:
            i = 10
    bg = PCG64(0)
    np.random.set_bit_generator(bg)
    state = np.random.get_state(legacy=False)
    np.random.seed(1)
    new_state = np.random.get_state(legacy=False)
    print(state)
    print(new_state)
    assert state['bit_generator'] == 'PCG64'
    assert state['state']['state'] != new_state['state']['state']
    assert state['state']['inc'] != new_state['state']['inc']

def test_state_error_alt_bit_gen(restore_singleton_bitgen):
    if False:
        i = 10
        return i + 15
    state = np.random.get_state()
    bg = PCG64(0)
    np.random.set_bit_generator(bg)
    with pytest.raises(ValueError, match='state must be for a PCG64'):
        np.random.set_state(state)

def test_swap_worked(restore_singleton_bitgen):
    if False:
        print('Hello World!')
    np.random.seed(98765)
    vals = np.random.randint(0, 2 ** 30, 10)
    bg = PCG64(0)
    state = bg.state
    np.random.set_bit_generator(bg)
    state_direct = np.random.get_state(legacy=False)
    for field in state:
        assert state[field] == state_direct[field]
    np.random.seed(98765)
    pcg_vals = np.random.randint(0, 2 ** 30, 10)
    assert not np.all(vals == pcg_vals)
    new_state = bg.state
    assert new_state['state']['state'] != state['state']['state']
    assert new_state['state']['inc'] == new_state['state']['inc']

def test_swapped_singleton_against_direct(restore_singleton_bitgen):
    if False:
        while True:
            i = 10
    np.random.set_bit_generator(PCG64(98765))
    singleton_vals = np.random.randint(0, 2 ** 30, 10)
    rg = np.random.RandomState(PCG64(98765))
    non_singleton_vals = rg.randint(0, 2 ** 30, 10)
    assert_equal(non_singleton_vals, singleton_vals)