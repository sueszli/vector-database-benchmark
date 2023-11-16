import copy
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from scipy.stats._multicomp import _pvalue_dunnett, DunnettResult

class TestDunnett:
    samples_1 = [[24.0, 27.0, 33.0, 32.0, 28.0, 19.0, 37.0, 31.0, 36.0, 36.0, 34.0, 38.0, 32.0, 38.0, 32.0], [26.0, 24.0, 26.0, 25.0, 29.0, 29.5, 16.5, 36.0, 44.0], [25.0, 27.0, 19.0], [25.0, 20.0], [28.0]]
    control_1 = [18.0, 15.0, 18.0, 16.0, 17.0, 15.0, 14.0, 14.0, 14.0, 15.0, 15.0, 14.0, 15.0, 14.0, 22.0, 18.0, 21.0, 21.0, 10.0, 10.0, 11.0, 9.0, 25.0, 26.0, 17.5, 16.0, 15.5, 14.5, 22.0, 22.0, 24.0, 22.5, 29.0, 24.5, 20.0, 18.0, 18.5, 17.5, 26.5, 13.0, 16.5, 13.0, 13.0, 13.0, 28.0, 27.0, 34.0, 31.0, 29.0, 27.0, 24.0, 23.0, 38.0, 36.0, 25.0, 38.0, 26.0, 22.0, 36.0, 27.0, 27.0, 32.0, 28.0, 31.0]
    pvalue_1 = [4.727e-06, 0.022346, 0.97912, 0.99953, 0.86579]
    p_1_twosided = [0.0001, 0.02237, 0.97913, 0.99953, 0.86583]
    p_1_greater = [0.0001, 0.011217, 0.7685, 0.896991, 0.577211]
    p_1_less = [1, 1, 0.9966, 0.98398, 0.99953]
    statistic_1 = [5.27356, 2.9127, 0.60831, 0.27002, 0.96637]
    ci_1_twosided = [[5.3633917835622, 0.7296142201217, -8.3879817106607, -11.9090753452911, -11.7655021543469], [15.9709832164378, 13.8936496687672, 13.4556900439941, 14.6434503452911, 25.4998771543469]]
    ci_1_greater = [5.9036402398526, 1.4000632918725, -7.2754756323636, -10.5567456382391, -9.8675629499576]
    ci_1_less = [15.4306165948619, 13.2230539537359, 12.3429406339544, 13.2908248513211, 23.601522825166]
    pvalues_1 = dict(twosided=p_1_twosided, less=p_1_less, greater=p_1_greater)
    cis_1 = dict(twosided=ci_1_twosided, less=ci_1_less, greater=ci_1_greater)
    case_1 = dict(samples=samples_1, control=control_1, statistic=statistic_1, pvalues=pvalues_1, cis=cis_1)
    samples_2 = [[9.76, 8.8, 7.68, 9.36], [12.8, 9.68, 12.16, 9.2, 10.55]]
    control_2 = [7.4, 8.5, 7.2, 8.24, 9.84, 8.32]
    pvalue_2 = [0.6201, 0.0058]
    p_2_twosided = [0.620102, 0.0058254]
    p_2_greater = [0.3249776, 0.0029139]
    p_2_less = [0.91676, 0.99984]
    statistic_2 = [0.85703, 3.69375]
    ci_2_twosided = [[-1.2564116462124, 0.8396273539789], [2.5564116462124, 4.4163726460211]]
    ci_2_greater = [-0.9588591188156, 1.1187563667543]
    ci_2_less = [2.2588591188156, 4.1372436332457]
    pvalues_2 = dict(twosided=p_2_twosided, less=p_2_less, greater=p_2_greater)
    cis_2 = dict(twosided=ci_2_twosided, less=ci_2_less, greater=ci_2_greater)
    case_2 = dict(samples=samples_2, control=control_2, statistic=statistic_2, pvalues=pvalues_2, cis=cis_2)
    samples_3 = [[55, 64, 64], [55, 49, 52], [50, 44, 41]]
    control_3 = [55, 47, 48]
    pvalue_3 = [0.0364, 0.8966, 0.4091]
    p_3_twosided = [0.036407, 0.896539, 0.409295]
    p_3_greater = [0.018277, 0.521109, 0.981892]
    p_3_less = [0.99944, 0.90054, 0.20974]
    statistic_3 = [3.09073, 0.56195, -1.40488]
    ci_3_twosided = [[0.7529028025053, -8.2470971974947, -15.2470971974947], [21.2470971974947, 12.2470971974947, 5.2470971974947]]
    ci_3_greater = [2.4023682323149, -6.5976317676851, -13.5976317676851]
    ci_3_less = [19.5984402363662, 10.5984402363662, 3.5984402363662]
    pvalues_3 = dict(twosided=p_3_twosided, less=p_3_less, greater=p_3_greater)
    cis_3 = dict(twosided=ci_3_twosided, less=ci_3_less, greater=ci_3_greater)
    case_3 = dict(samples=samples_3, control=control_3, statistic=statistic_3, pvalues=pvalues_3, cis=cis_3)
    samples_4 = [[3.8, 2.7, 4.0, 2.4], [2.8, 3.4, 3.7, 2.2, 2.0]]
    control_4 = [2.9, 3.0, 2.5, 2.6, 3.2]
    pvalue_4 = [0.5832, 0.9982]
    p_4_twosided = [0.58317, 0.99819]
    p_4_greater = [0.30225, 0.69115]
    p_4_less = [0.91929, 0.65212]
    statistic_4 = [0.90875, -0.05007]
    ci_4_twosided = [[-0.6898153448579, -1.0333456251632], [1.4598153448579, 0.9933456251632]]
    ci_4_greater = [-0.5186459268412, -0.8719655502147]
    ci_4_less = [1.2886459268412, 0.8319655502147]
    pvalues_4 = dict(twosided=p_4_twosided, less=p_4_less, greater=p_4_greater)
    cis_4 = dict(twosided=ci_4_twosided, less=ci_4_less, greater=ci_4_greater)
    case_4 = dict(samples=samples_4, control=control_4, statistic=statistic_4, pvalues=pvalues_4, cis=cis_4)

    @pytest.mark.parametrize('rho, n_groups, df, statistic, pvalue, alternative', [(0.5, 1, 10, 1.81, 0.05, 'greater'), (0.5, 3, 10, 2.34, 0.05, 'greater'), (0.5, 2, 30, 1.99, 0.05, 'greater'), (0.5, 5, 30, 2.33, 0.05, 'greater'), (0.5, 4, 12, 3.32, 0.01, 'greater'), (0.5, 7, 12, 3.56, 0.01, 'greater'), (0.5, 2, 60, 2.64, 0.01, 'greater'), (0.5, 4, 60, 2.87, 0.01, 'greater'), (0.5, 4, 60, [2.87, 2.21], [0.01, 0.05], 'greater'), (0.5, 1, 10, 2.23, 0.05, 'two-sided'), (0.5, 3, 10, 2.81, 0.05, 'two-sided'), (0.5, 2, 30, 2.32, 0.05, 'two-sided'), (0.5, 3, 20, 2.57, 0.05, 'two-sided'), (0.5, 4, 12, 3.76, 0.01, 'two-sided'), (0.5, 7, 12, 4.08, 0.01, 'two-sided'), (0.5, 2, 60, 2.9, 0.01, 'two-sided'), (0.5, 4, 60, 3.14, 0.01, 'two-sided'), (0.5, 4, 60, [3.14, 2.55], [0.01, 0.05], 'two-sided')])
    def test_critical_values(self, rho, n_groups, df, statistic, pvalue, alternative):
        if False:
            print('Hello World!')
        rng = np.random.default_rng(165250594791731684851746311027739134893)
        rho = np.full((n_groups, n_groups), rho)
        np.fill_diagonal(rho, 1)
        statistic = np.array(statistic)
        res = _pvalue_dunnett(rho=rho, df=df, statistic=statistic, alternative=alternative, rng=rng)
        assert_allclose(res, pvalue, atol=0.005)

    @pytest.mark.parametrize('samples, control, pvalue, statistic', [(samples_1, control_1, pvalue_1, statistic_1), (samples_2, control_2, pvalue_2, statistic_2), (samples_3, control_3, pvalue_3, statistic_3), (samples_4, control_4, pvalue_4, statistic_4)])
    def test_basic(self, samples, control, pvalue, statistic):
        if False:
            while True:
                i = 10
        rng = np.random.default_rng(11681140010308601919115036826969764808)
        res = stats.dunnett(*samples, control=control, random_state=rng)
        assert isinstance(res, DunnettResult)
        assert_allclose(res.statistic, statistic, rtol=5e-05)
        assert_allclose(res.pvalue, pvalue, rtol=0.01, atol=0.0001)

    @pytest.mark.parametrize('alternative', ['two-sided', 'less', 'greater'])
    def test_ttest_ind(self, alternative):
        if False:
            for i in range(10):
                print('nop')
        rng = np.random.default_rng(114184017807316971636137493526995620351)
        for _ in range(10):
            sample = rng.integers(-100, 100, size=(10,))
            control = rng.integers(-100, 100, size=(10,))
            res = stats.dunnett(sample, control=control, alternative=alternative, random_state=rng)
            ref = stats.ttest_ind(sample, control, alternative=alternative, random_state=rng)
            assert_allclose(res.statistic, ref.statistic, rtol=0.001, atol=1e-05)
            assert_allclose(res.pvalue, ref.pvalue, rtol=0.001, atol=1e-05)

    @pytest.mark.parametrize('alternative, pvalue', [('less', [0, 1]), ('greater', [1, 0]), ('two-sided', [0, 0])])
    def test_alternatives(self, alternative, pvalue):
        if False:
            print('Hello World!')
        rng = np.random.default_rng(114184017807316971636137493526995620351)
        sample_less = rng.integers(0, 20, size=(10,))
        control = rng.integers(80, 100, size=(10,))
        sample_greater = rng.integers(160, 180, size=(10,))
        res = stats.dunnett(sample_less, sample_greater, control=control, alternative=alternative, random_state=rng)
        assert_allclose(res.pvalue, pvalue, atol=1e-07)
        ci = res.confidence_interval()
        if alternative == 'less':
            assert np.isneginf(ci.low).all()
            assert -100 < ci.high[0] < -60
            assert 60 < ci.high[1] < 100
        elif alternative == 'greater':
            assert -100 < ci.low[0] < -60
            assert 60 < ci.low[1] < 100
            assert np.isposinf(ci.high).all()
        elif alternative == 'two-sided':
            assert -100 < ci.low[0] < -60
            assert 60 < ci.low[1] < 100
            assert -100 < ci.high[0] < -60
            assert 60 < ci.high[1] < 100

    @pytest.mark.parametrize('case', [case_1, case_2, case_3, case_4])
    @pytest.mark.parametrize('alternative', ['less', 'greater', 'two-sided'])
    def test_against_R_multicomp_glht(self, case, alternative):
        if False:
            print('Hello World!')
        rng = np.random.default_rng(189117774084579816190295271136455278291)
        samples = case['samples']
        control = case['control']
        alternatives = {'less': 'less', 'greater': 'greater', 'two-sided': 'twosided'}
        p_ref = case['pvalues'][alternative.replace('-', '')]
        res = stats.dunnett(*samples, control=control, alternative=alternative, random_state=rng)
        assert_allclose(res.pvalue, p_ref, rtol=0.005, atol=0.0001)
        ci_ref = case['cis'][alternatives[alternative]]
        if alternative == 'greater':
            ci_ref = [ci_ref, np.inf]
        elif alternative == 'less':
            ci_ref = [-np.inf, ci_ref]
        assert res._ci is None
        assert res._ci_cl is None
        ci = res.confidence_interval(confidence_level=0.95)
        assert_allclose(ci.low, ci_ref[0], rtol=0.005, atol=1e-05)
        assert_allclose(ci.high, ci_ref[1], rtol=0.005, atol=1e-05)
        assert res._ci is ci
        assert res._ci_cl == 0.95
        ci_ = res.confidence_interval(confidence_level=0.95)
        assert ci_ is ci

    @pytest.mark.parametrize('alternative', ['two-sided', 'less', 'greater'])
    def test_str(self, alternative):
        if False:
            while True:
                i = 10
        rng = np.random.default_rng(189117774084579816190295271136455278291)
        res = stats.dunnett(*self.samples_3, control=self.control_3, alternative=alternative, random_state=rng)
        res_str = str(res)
        assert '(Sample 2 - Control)' in res_str
        assert '95.0%' in res_str
        if alternative == 'less':
            assert '-inf' in res_str
            assert '19.' in res_str
        elif alternative == 'greater':
            assert 'inf' in res_str
            assert '-13.' in res_str
        else:
            assert 'inf' not in res_str
            assert '21.' in res_str

    def test_warnings(self):
        if False:
            i = 10
            return i + 15
        rng = np.random.default_rng(189117774084579816190295271136455278291)
        res = stats.dunnett(*self.samples_3, control=self.control_3, random_state=rng)
        msg = 'Computation of the confidence interval did not converge'
        with pytest.warns(UserWarning, match=msg):
            res._allowance(tol=1e-05)

    def test_raises(self):
        if False:
            for i in range(10):
                print('nop')
        (samples, control) = (self.samples_3, self.control_3)
        with pytest.raises(ValueError, match='alternative must be'):
            stats.dunnett(*samples, control=control, alternative='bob')
        samples_ = copy.deepcopy(samples)
        samples_[0] = [samples_[0]]
        with pytest.raises(ValueError, match='must be 1D arrays'):
            stats.dunnett(*samples_, control=control)
        control_ = copy.deepcopy(control)
        control_ = [control_]
        with pytest.raises(ValueError, match='must be 1D arrays'):
            stats.dunnett(*samples, control=control_)
        samples_ = copy.deepcopy(samples)
        samples_[1] = []
        with pytest.raises(ValueError, match='at least 1 observation'):
            stats.dunnett(*samples_, control=control)
        control_ = []
        with pytest.raises(ValueError, match='at least 1 observation'):
            stats.dunnett(*samples, control=control_)
        res = stats.dunnett(*samples, control=control)
        with pytest.raises(ValueError, match='Confidence level must'):
            res.confidence_interval(confidence_level=3)

    @pytest.mark.filterwarnings('ignore:Computation of the confidence')
    @pytest.mark.parametrize('n_samples', [1, 2, 3])
    def test_shapes(self, n_samples):
        if False:
            print('Hello World!')
        rng = np.random.default_rng(689448934110805334)
        samples = rng.normal(size=(n_samples, 10))
        control = rng.normal(size=10)
        res = stats.dunnett(*samples, control=control, random_state=rng)
        assert res.statistic.shape == (n_samples,)
        assert res.pvalue.shape == (n_samples,)
        ci = res.confidence_interval()
        assert ci.low.shape == (n_samples,)
        assert ci.high.shape == (n_samples,)