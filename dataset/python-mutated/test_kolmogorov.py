import itertools
import sys
import pytest
import numpy as np
from numpy.testing import assert_
from scipy.special._testutils import FuncData
from scipy.special import kolmogorov, kolmogi, smirnov, smirnovi
from scipy.special._ufuncs import _kolmogc, _kolmogci, _kolmogp, _smirnovc, _smirnovci, _smirnovp
_rtol = 1e-10

class TestSmirnov:

    def test_nan(self):
        if False:
            for i in range(10):
                print('nop')
        assert_(np.isnan(smirnov(1, np.nan)))

    def test_basic(self):
        if False:
            while True:
                i = 10
        dataset = [(1, 0.1, 0.9), (1, 0.875, 0.125), (2, 0.875, 0.125 * 0.125), (3, 0.875, 0.125 * 0.125 * 0.125)]
        dataset = np.asarray(dataset)
        FuncData(smirnov, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])
        dataset[:, -1] = 1 - dataset[:, -1]
        FuncData(_smirnovc, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

    def test_x_equals_0(self):
        if False:
            return 10
        dataset = [(n, 0, 1) for n in itertools.chain(range(2, 20), range(1010, 1020))]
        dataset = np.asarray(dataset)
        FuncData(smirnov, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])
        dataset[:, -1] = 1 - dataset[:, -1]
        FuncData(_smirnovc, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

    def test_x_equals_1(self):
        if False:
            i = 10
            return i + 15
        dataset = [(n, 1, 0) for n in itertools.chain(range(2, 20), range(1010, 1020))]
        dataset = np.asarray(dataset)
        FuncData(smirnov, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])
        dataset[:, -1] = 1 - dataset[:, -1]
        FuncData(_smirnovc, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

    def test_x_equals_0point5(self):
        if False:
            i = 10
            return i + 15
        dataset = [(1, 0.5, 0.5), (2, 0.5, 0.25), (3, 0.5, 0.166666666667), (4, 0.5, 0.09375), (5, 0.5, 0.056), (6, 0.5, 0.0327932098765), (7, 0.5, 0.0191958707681), (8, 0.5, 0.0112953186035), (9, 0.5, 0.00661933257355), (10, 0.5, 0.003888705)]
        dataset = np.asarray(dataset)
        FuncData(smirnov, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])
        dataset[:, -1] = 1 - dataset[:, -1]
        FuncData(_smirnovc, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

    def test_n_equals_1(self):
        if False:
            print('Hello World!')
        x = np.linspace(0, 1, 101, endpoint=True)
        dataset = np.column_stack([[1] * len(x), x, 1 - x])
        FuncData(smirnov, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])
        dataset[:, -1] = 1 - dataset[:, -1]
        FuncData(_smirnovc, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

    def test_n_equals_2(self):
        if False:
            while True:
                i = 10
        x = np.linspace(0.5, 1, 101, endpoint=True)
        p = np.power(1 - x, 2)
        n = np.array([2] * len(x))
        dataset = np.column_stack([n, x, p])
        FuncData(smirnov, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])
        dataset[:, -1] = 1 - dataset[:, -1]
        FuncData(_smirnovc, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

    def test_n_equals_3(self):
        if False:
            i = 10
            return i + 15
        x = np.linspace(0.7, 1, 31, endpoint=True)
        p = np.power(1 - x, 3)
        n = np.array([3] * len(x))
        dataset = np.column_stack([n, x, p])
        FuncData(smirnov, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])
        dataset[:, -1] = 1 - dataset[:, -1]
        FuncData(_smirnovc, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

    def test_n_large(self):
        if False:
            i = 10
            return i + 15
        x = 0.4
        pvals = np.array([smirnov(n, x) for n in range(400, 1100, 20)])
        dfs = np.diff(pvals)
        assert_(np.all(dfs <= 0), msg='Not all diffs negative %s' % dfs)

class TestSmirnovi:

    def test_nan(self):
        if False:
            i = 10
            return i + 15
        assert_(np.isnan(smirnovi(1, np.nan)))

    def test_basic(self):
        if False:
            print('Hello World!')
        dataset = [(1, 0.4, 0.6), (1, 0.6, 0.4), (1, 0.99, 0.01), (1, 0.01, 0.99), (2, 0.125 * 0.125, 0.875), (3, 0.125 * 0.125 * 0.125, 0.875), (10, 1.0 / 16 ** 10, 1 - 1.0 / 16)]
        dataset = np.asarray(dataset)
        FuncData(smirnovi, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])
        dataset[:, 1] = 1 - dataset[:, 1]
        FuncData(_smirnovci, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

    def test_x_equals_0(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = [(n, 0, 1) for n in itertools.chain(range(2, 20), range(1010, 1020))]
        dataset = np.asarray(dataset)
        FuncData(smirnovi, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])
        dataset[:, 1] = 1 - dataset[:, 1]
        FuncData(_smirnovci, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

    def test_x_equals_1(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = [(n, 1, 0) for n in itertools.chain(range(2, 20), range(1010, 1020))]
        dataset = np.asarray(dataset)
        FuncData(smirnovi, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])
        dataset[:, 1] = 1 - dataset[:, 1]
        FuncData(_smirnovci, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

    def test_n_equals_1(self):
        if False:
            i = 10
            return i + 15
        pp = np.linspace(0, 1, 101, endpoint=True)
        dataset = np.column_stack([[1] * len(pp), pp, 1 - pp])
        FuncData(smirnovi, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])
        dataset[:, 1] = 1 - dataset[:, 1]
        FuncData(_smirnovci, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

    def test_n_equals_2(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.linspace(0.5, 1, 101, endpoint=True)
        p = np.power(1 - x, 2)
        n = np.array([2] * len(x))
        dataset = np.column_stack([n, p, x])
        FuncData(smirnovi, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])
        dataset[:, 1] = 1 - dataset[:, 1]
        FuncData(_smirnovci, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

    def test_n_equals_3(self):
        if False:
            i = 10
            return i + 15
        x = np.linspace(0.7, 1, 31, endpoint=True)
        p = np.power(1 - x, 3)
        n = np.array([3] * len(x))
        dataset = np.column_stack([n, p, x])
        FuncData(smirnovi, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])
        dataset[:, 1] = 1 - dataset[:, 1]
        FuncData(_smirnovci, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

    def test_round_trip(self):
        if False:
            print('Hello World!')

        def _sm_smi(n, p):
            if False:
                i = 10
                return i + 15
            return smirnov(n, smirnovi(n, p))

        def _smc_smci(n, p):
            if False:
                for i in range(10):
                    print('nop')
            return _smirnovc(n, _smirnovci(n, p))
        dataset = [(1, 0.4, 0.4), (1, 0.6, 0.6), (2, 0.875, 0.875), (3, 0.875, 0.875), (3, 0.125, 0.125), (10, 0.999, 0.999), (10, 0.0001, 0.0001)]
        dataset = np.asarray(dataset)
        FuncData(_sm_smi, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])
        FuncData(_smc_smci, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

    def test_x_equals_0point5(self):
        if False:
            return 10
        dataset = [(1, 0.5, 0.5), (2, 0.5, 0.366025403784), (2, 0.25, 0.5), (3, 0.5, 0.297156508177), (4, 0.5, 0.255520481121), (5, 0.5, 0.234559536069), (6, 0.5, 0.21715965898), (7, 0.5, 0.202722580034), (8, 0.5, 0.190621765256), (9, 0.5, 0.180363501362), (10, 0.5, 0.17157867006)]
        dataset = np.asarray(dataset)
        FuncData(smirnovi, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])
        dataset[:, 1] = 1 - dataset[:, 1]
        FuncData(_smirnovci, dataset, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

class TestSmirnovp:

    def test_nan(self):
        if False:
            i = 10
            return i + 15
        assert_(np.isnan(_smirnovp(1, np.nan)))

    def test_basic(self):
        if False:
            while True:
                i = 10
        n1_10 = np.arange(1, 10)
        dataset0 = np.column_stack([n1_10, np.full_like(n1_10, 0), np.full_like(n1_10, -1)])
        FuncData(_smirnovp, dataset0, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])
        n2_10 = np.arange(2, 10)
        dataset1 = np.column_stack([n2_10, np.full_like(n2_10, 1.0), np.full_like(n2_10, 0)])
        FuncData(_smirnovp, dataset1, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

    def test_oneminusoneovern(self):
        if False:
            i = 10
            return i + 15
        n = np.arange(1, 20)
        x = 1.0 / n
        xm1 = 1 - 1.0 / n
        pp1 = -n * x ** (n - 1)
        pp1 -= (1 - np.sign(n - 2) ** 2) * 0.5
        dataset1 = np.column_stack([n, xm1, pp1])
        FuncData(_smirnovp, dataset1, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

    def test_oneovertwon(self):
        if False:
            for i in range(10):
                print('nop')
        n = np.arange(1, 20)
        x = 1.0 / 2 / n
        pp = -(n * x + 1) * (1 + x) ** (n - 2)
        dataset0 = np.column_stack([n, x, pp])
        FuncData(_smirnovp, dataset0, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

    def test_oneovern(self):
        if False:
            return 10
        n = 2 ** np.arange(1, 10)
        x = 1.0 / n
        pp = -(n * x + 1) * (1 + x) ** (n - 2) + 0.5
        dataset0 = np.column_stack([n, x, pp])
        FuncData(_smirnovp, dataset0, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

    @pytest.mark.xfail(sys.maxsize <= 2 ** 32, reason='requires 64-bit platform')
    def test_oneovernclose(self):
        if False:
            return 10
        n = np.arange(3, 20)
        x = 1.0 / n - 2 * np.finfo(float).eps
        pp = -(n * x + 1) * (1 + x) ** (n - 2)
        dataset0 = np.column_stack([n, x, pp])
        FuncData(_smirnovp, dataset0, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])
        x = 1.0 / n + 2 * np.finfo(float).eps
        pp = -(n * x + 1) * (1 + x) ** (n - 2) + 1
        dataset1 = np.column_stack([n, x, pp])
        FuncData(_smirnovp, dataset1, (0, 1), 2, rtol=_rtol).check(dtypes=[int, float, float])

class TestKolmogorov:

    def test_nan(self):
        if False:
            while True:
                i = 10
        assert_(np.isnan(kolmogorov(np.nan)))

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        dataset = [(0, 1.0), (0.5, 0.9639452436648751), (0.8275735551899077, 0.5), (1, 0.26999967167735456), (2, 0.0006709252557796953)]
        dataset = np.asarray(dataset)
        FuncData(kolmogorov, dataset, (0,), 1, rtol=_rtol).check()

    def test_linspace(self):
        if False:
            print('Hello World!')
        x = np.linspace(0, 2.0, 21)
        dataset = [1.0, 1.0, 0.999999999999495, 0.9999906941986655, 0.9971923267772983, 0.9639452436648751, 0.8642827790506042, 0.711235195029689, 0.5441424115741981, 0.3927307079406543, 0.2699996716773546, 0.1777181926064012, 0.1122496666707249, 0.0680922218447664, 0.0396818795381144, 0.0222179626165251, 0.0119520432391966, 0.0061774306344441, 0.0030676213475797, 0.0014636048371873, 0.0006709252557797]
        dataset_c = [0.0, 6.609305242245699e-53, 5.050407338670114e-13, 9.305801334566668e-06, 0.0028076732227017, 0.0360547563351249, 0.1357172209493958, 0.288764804970311, 0.4558575884258019, 0.6072692920593457, 0.7300003283226455, 0.8222818073935988, 0.8877503333292751, 0.9319077781552336, 0.9603181204618857, 0.9777820373834749, 0.9880479567608034, 0.9938225693655559, 0.9969323786524203, 0.9985363951628127, 0.9993290747442203]
        dataset = np.column_stack([x, dataset])
        FuncData(kolmogorov, dataset, (0,), 1, rtol=_rtol).check()
        dataset_c = np.column_stack([x, dataset_c])
        FuncData(_kolmogc, dataset_c, (0,), 1, rtol=_rtol).check()

    def test_linspacei(self):
        if False:
            i = 10
            return i + 15
        p = np.linspace(0, 1.0, 21, endpoint=True)
        dataset = [np.inf, 1.3580986393225507, 1.2238478702170823, 1.1379465424937751, 1.072749174939648, 1.019184720253686, 0.9730633753323726, 0.9320695842357622, 0.8947644549851197, 0.8601710725555463, 0.8275735551899077, 0.7964065373291559, 0.7661855555617682, 0.736454288817191, 0.706732652306898, 0.6764476915028201, 0.6448126061663567, 0.6105590999244391, 0.5711732651063401, 0.5196103791686224, 0.0]
        dataset_c = [0.0, 0.5196103791686225, 0.5711732651063401, 0.6105590999244391, 0.6448126061663567, 0.6764476915028201, 0.706732652306898, 0.736454288817191, 0.7661855555617682, 0.7964065373291559, 0.8275735551899077, 0.8601710725555463, 0.8947644549851196, 0.9320695842357622, 0.9730633753323727, 1.019184720253686, 1.072749174939648, 1.1379465424937754, 1.2238478702170825, 1.358098639322551, np.inf]
        dataset = np.column_stack([p[1:], dataset[1:]])
        FuncData(kolmogi, dataset, (0,), 1, rtol=_rtol).check()
        dataset_c = np.column_stack([p[:-1], dataset_c[:-1]])
        FuncData(_kolmogci, dataset_c, (0,), 1, rtol=_rtol).check()

    def test_smallx(self):
        if False:
            for i in range(10):
                print('nop')
        epsilon = 0.1 ** np.arange(1, 14)
        x = np.array([0.571173265106, 0.441027698518, 0.374219690278, 0.331392659217, 0.300820537459, 0.277539353999, 0.259023494805, 0.243829561254, 0.231063086389, 0.220135543236, 0.210641372041, 0.202290283658, 0.19487060742])
        dataset = np.column_stack([x, 1 - epsilon])
        FuncData(kolmogorov, dataset, (0,), 1, rtol=_rtol).check()

    def test_round_trip(self):
        if False:
            print('Hello World!')

        def _ki_k(_x):
            if False:
                print('Hello World!')
            return kolmogi(kolmogorov(_x))

        def _kci_kc(_x):
            if False:
                while True:
                    i = 10
            return _kolmogci(_kolmogc(_x))
        x = np.linspace(0.0, 2.0, 21, endpoint=True)
        x02 = x[(x == 0) | (x > 0.21)]
        dataset02 = np.column_stack([x02, x02])
        FuncData(_ki_k, dataset02, (0,), 1, rtol=_rtol).check()
        dataset = np.column_stack([x, x])
        FuncData(_kci_kc, dataset, (0,), 1, rtol=_rtol).check()

class TestKolmogi:

    def test_nan(self):
        if False:
            print('Hello World!')
        assert_(np.isnan(kolmogi(np.nan)))

    def test_basic(self):
        if False:
            print('Hello World!')
        dataset = [(1.0, 0), (0.9639452436648751, 0.5), (0.9, 0.571173265106), (0.5, 0.8275735551899077), (0.26999967167735456, 1), (0.0006709252557796953, 2)]
        dataset = np.asarray(dataset)
        FuncData(kolmogi, dataset, (0,), 1, rtol=_rtol).check()

    def test_smallpcdf(self):
        if False:
            return 10
        epsilon = 0.5 ** np.arange(1, 55, 3)
        x = np.array([0.8275735551899077, 0.5345255069097583, 0.4320114038786941, 0.3736868442620478, 0.3345161714909591, 0.3057833329315859, 0.2835052890528936, 0.2655578150208676, 0.2506869966107999, 0.2380971058736669, 0.2272549289962079, 0.217787636160004, 0.2094254686862041, 0.2019676748836232, 0.1952612948137504, 0.1891874239646641, 0.1836520225050326, 0.1785795904846466])
        dataset = np.column_stack([1 - epsilon, x])
        FuncData(kolmogi, dataset, (0,), 1, rtol=_rtol).check()
        dataset = np.column_stack([epsilon, x])
        FuncData(_kolmogci, dataset, (0,), 1, rtol=_rtol).check()

    def test_smallpsf(self):
        if False:
            for i in range(10):
                print('nop')
        epsilon = 0.5 ** np.arange(1, 55, 3)
        x = np.array([0.8275735551899077, 1.3163786275161036, 1.6651092133663343, 1.9525136345289607, 2.2027324540033235, 2.427292943746085, 2.6327688477341593, 2.823330050922026, 3.0018183401530627, 3.170273508408889, 3.330218444630791, 3.482825815311332, 3.629021415015205, 3.769551326282596, 3.9050272690877326, 4.035958218708255, 4.162773055788489, 4.285837174326453])
        dataset = np.column_stack([epsilon, x])
        FuncData(kolmogi, dataset, (0,), 1, rtol=_rtol).check()
        dataset = np.column_stack([1 - epsilon, x])
        FuncData(_kolmogci, dataset, (0,), 1, rtol=_rtol).check()

    def test_round_trip(self):
        if False:
            while True:
                i = 10

        def _k_ki(_p):
            if False:
                for i in range(10):
                    print('nop')
            return kolmogorov(kolmogi(_p))
        p = np.linspace(0.1, 1.0, 10, endpoint=True)
        dataset = np.column_stack([p, p])
        FuncData(_k_ki, dataset, (0,), 1, rtol=_rtol).check()

class TestKolmogp:

    def test_nan(self):
        if False:
            return 10
        assert_(np.isnan(_kolmogp(np.nan)))

    def test_basic(self):
        if False:
            while True:
                i = 10
        dataset = [(0.0, -0.0), (0.2, -1.532420541338916e-10), (0.4, -0.1012254419260496), (0.6, -1.324123244249925), (0.8, -1.627024345636592), (1.0, -1.071948558356941), (1.2, -0.538512430720529), (1.4, -0.2222133182429472), (1.6, -0.07649302775520538), (1.8, -0.02208687346347873), (2.0, -0.005367402045629683)]
        dataset = np.asarray(dataset)
        FuncData(_kolmogp, dataset, (0,), 1, rtol=_rtol).check()