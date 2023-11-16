from __future__ import division
from nose_parameterized import parameterized
from six.moves import range
import numpy as np
import pandas as pd
import talib
from numpy.random import RandomState
from zipline.lib.adjusted_array import AdjustedArray
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import BollingerBands, Aroon, FastStochasticOscillator, IchimokuKinkoHyo, LinearWeightedMovingAverage, RateOfChangePercentage, TrueRange, MovingAverageConvergenceDivergenceSignal, AnnualizedVolatility, RSI
from zipline.testing import check_allclose, parameter_space
from zipline.testing.fixtures import ZiplineTestCase
from zipline.testing.predicates import assert_equal
from .base import BaseUSEquityPipelineTestCase

class BollingerBandsTestCase(BaseUSEquityPipelineTestCase):

    def closes(self, mask_last_sid):
        if False:
            for i in range(10):
                print('nop')
        data = self.arange_data(dtype=np.float64)
        if mask_last_sid:
            data[:, -1] = np.nan
        return data

    def expected_bbands(self, window_length, k, closes):
        if False:
            print('Hello World!')
        'Compute the expected data (without adjustments) for the given\n        window, k, and closes array.\n\n        This uses talib.BBANDS to generate the expected data.\n        '
        lower_cols = []
        middle_cols = []
        upper_cols = []
        (ndates, nassets) = closes.shape
        for n in range(nassets):
            close_col = closes[:, n]
            if np.isnan(close_col).all():
                (upper, middle, lower) = [np.full(ndates, np.nan)] * 3
            else:
                (upper, middle, lower) = talib.BBANDS(close_col, window_length, k, k)
            upper_cols.append(upper)
            middle_cols.append(middle)
            lower_cols.append(lower)
        where = np.s_[window_length - 1:]
        uppers = np.column_stack(upper_cols)[where]
        middles = np.column_stack(middle_cols)[where]
        lowers = np.column_stack(lower_cols)[where]
        return (uppers, middles, lowers)

    @parameter_space(window_length={5, 10, 20}, k={1.5, 2, 2.5}, mask_last_sid={True, False}, __fail_fast=True)
    def test_bollinger_bands(self, window_length, k, mask_last_sid):
        if False:
            while True:
                i = 10
        closes = self.closes(mask_last_sid=mask_last_sid)
        mask = ~np.isnan(closes)
        bbands = BollingerBands(window_length=window_length, k=k)
        expected = self.expected_bbands(window_length, k, closes)
        self.check_terms(terms={'upper': bbands.upper, 'middle': bbands.middle, 'lower': bbands.lower}, expected={'upper': expected[0], 'middle': expected[1], 'lower': expected[2]}, initial_workspace={USEquityPricing.close: AdjustedArray(data=closes, adjustments={}, missing_value=np.nan)}, mask=self.build_mask(mask))

    def test_bollinger_bands_output_ordering(self):
        if False:
            return 10
        bbands = BollingerBands(window_length=5, k=2)
        (lower, middle, upper) = bbands
        self.assertIs(lower, bbands.lower)
        self.assertIs(middle, bbands.middle)
        self.assertIs(upper, bbands.upper)

class AroonTestCase(ZiplineTestCase):
    window_length = 10
    nassets = 5
    dtype = [('down', 'f8'), ('up', 'f8')]

    @parameterized.expand([(np.arange(window_length), np.arange(window_length) + 1, np.recarray(shape=(nassets,), dtype=dtype, buf=np.array([0, 100] * nassets, dtype='f8'))), (np.arange(window_length, 0, -1), np.arange(window_length, 0, -1) - 1, np.recarray(shape=(nassets,), dtype=dtype, buf=np.array([100, 0] * nassets, dtype='f8'))), (np.array([10, 10, 10, 1, 10, 10, 10, 10, 10, 10]), np.array([1, 1, 1, 1, 1, 10, 1, 1, 1, 1]), np.recarray(shape=(nassets,), dtype=dtype, buf=np.array([100 * 3 / 9, 100 * 5 / 9] * nassets, dtype='f8')))])
    def test_aroon_basic(self, lows, highs, expected_out):
        if False:
            while True:
                i = 10
        aroon = Aroon(window_length=self.window_length)
        today = pd.Timestamp('2014', tz='utc')
        assets = pd.Index(np.arange(self.nassets, dtype=np.int64))
        shape = (self.nassets,)
        out = np.recarray(shape=shape, dtype=self.dtype, buf=np.empty(shape=shape, dtype=self.dtype))
        aroon.compute(today, assets, out, lows, highs)
        assert_equal(out, expected_out)

class TestFastStochasticOscillator(ZiplineTestCase):
    """
    Test the Fast Stochastic Oscillator
    """

    def test_fso_expected_basic(self):
        if False:
            print('Hello World!')
        '\n        Simple test of expected output from fast stochastic oscillator\n        '
        fso = FastStochasticOscillator()
        today = pd.Timestamp('2015')
        assets = np.arange(3, dtype=np.float64)
        out = np.empty(shape=(3,), dtype=np.float64)
        highs = np.full((50, 3), 3, dtype=np.float64)
        lows = np.full((50, 3), 2, dtype=np.float64)
        closes = np.full((50, 3), 4, dtype=np.float64)
        fso.compute(today, assets, out, closes, lows, highs)
        assert_equal(out, np.full((3,), 200, dtype=np.float64))

    @parameter_space(seed=range(5))
    def test_fso_expected_with_talib(self, seed):
        if False:
            print('Hello World!')
        '\n        Test the output that is returned from the fast stochastic oscillator\n        is the same as that from the ta-lib STOCHF function.\n        '
        window_length = 14
        nassets = 6
        rng = np.random.RandomState(seed=seed)
        input_size = (window_length, nassets)
        closes = 9.0 + rng.random_sample(input_size) * 3.0
        highs = 13.0 + rng.random_sample(input_size) * 2.0
        lows = 6.0 + rng.random_sample(input_size) * 2.0
        expected_out_k = []
        for i in range(nassets):
            (fastk, fastd) = talib.STOCHF(high=highs[:, i], low=lows[:, i], close=closes[:, i], fastk_period=window_length, fastd_period=1)
            expected_out_k.append(fastk[-1])
        expected_out_k = np.array(expected_out_k)
        today = pd.Timestamp('2015')
        out = np.empty(shape=(nassets,), dtype=np.float)
        assets = np.arange(nassets, dtype=np.float)
        fso = FastStochasticOscillator()
        fso.compute(today, assets, out, closes, lows, highs)
        assert_equal(out, expected_out_k, array_decimal=6)

class IchimokuKinkoHyoTestCase(ZiplineTestCase):

    def test_ichimoku_kinko_hyo(self):
        if False:
            i = 10
            return i + 15
        window_length = 52
        today = pd.Timestamp('2014', tz='utc')
        nassets = 5
        assets = pd.Index(np.arange(nassets))
        days_col = np.arange(window_length)[:, np.newaxis]
        highs = np.arange(nassets) + 2 + days_col
        closes = np.arange(nassets) + 1 + days_col
        lows = np.arange(nassets) + days_col
        tenkan_sen_length = 9
        kijun_sen_length = 26
        chikou_span_length = 26
        ichimoku_kinko_hyo = IchimokuKinkoHyo(window_length=window_length, tenkan_sen_length=tenkan_sen_length, kijun_sen_length=kijun_sen_length, chikou_span_length=chikou_span_length)
        dtype = [('tenkan_sen', 'f8'), ('kijun_sen', 'f8'), ('senkou_span_a', 'f8'), ('senkou_span_b', 'f8'), ('chikou_span', 'f8')]
        out = np.recarray(shape=(nassets,), dtype=dtype, buf=np.empty(shape=(nassets,), dtype=dtype))
        ichimoku_kinko_hyo.compute(today, assets, out, highs, lows, closes, tenkan_sen_length, kijun_sen_length, chikou_span_length)
        expected_tenkan_sen = np.array([(53 + 43) / 2, (54 + 44) / 2, (55 + 45) / 2, (56 + 46) / 2, (57 + 47) / 2])
        expected_kijun_sen = np.array([(53 + 26) / 2, (54 + 27) / 2, (55 + 28) / 2, (56 + 29) / 2, (57 + 30) / 2])
        expected_senkou_span_a = (expected_tenkan_sen + expected_kijun_sen) / 2
        expected_senkou_span_b = np.array([(53 + 0) / 2, (54 + 1) / 2, (55 + 2) / 2, (56 + 3) / 2, (57 + 4) / 2])
        expected_chikou_span = np.array([27.0, 28.0, 29.0, 30.0, 31.0])
        assert_equal(out.tenkan_sen, expected_tenkan_sen, msg='tenkan_sen')
        assert_equal(out.kijun_sen, expected_kijun_sen, msg='kijun_sen')
        assert_equal(out.senkou_span_a, expected_senkou_span_a, msg='senkou_span_a')
        assert_equal(out.senkou_span_b, expected_senkou_span_b, msg='senkou_span_b')
        assert_equal(out.chikou_span, expected_chikou_span, msg='chikou_span')

    @parameter_space(arg={'tenkan_sen_length', 'kijun_sen_length', 'chikou_span_length'})
    def test_input_validation(self, arg):
        if False:
            i = 10
            return i + 15
        window_length = 52
        with self.assertRaises(ValueError) as e:
            IchimokuKinkoHyo(**{arg: window_length + 1})
        assert_equal(str(e.exception), '%s must be <= the window_length: 53 > 52' % arg)

class TestRateOfChangePercentage(ZiplineTestCase):

    @parameterized.expand([('constant', [2.0] * 10, 0.0), ('step', [2.0] + [1.0] * 9, -50.0), ('linear', [2.0 + x for x in range(10)], 450.0), ('quadratic', [2.0 + x ** 2 for x in range(10)], 4050.0)])
    def test_rate_of_change_percentage(self, test_name, data, expected):
        if False:
            print('Hello World!')
        window_length = len(data)
        rocp = RateOfChangePercentage(inputs=(USEquityPricing.close,), window_length=window_length)
        today = pd.Timestamp('2014')
        assets = np.arange(5, dtype=np.int64)
        data = np.array(data)[:, np.newaxis] * np.ones(len(assets))
        out = np.zeros(len(assets))
        rocp.compute(today, assets, out, data)
        assert_equal(out, np.full((len(assets),), expected))

class TestLinearWeightedMovingAverage(ZiplineTestCase):

    def test_wma1(self):
        if False:
            for i in range(10):
                print('nop')
        wma1 = LinearWeightedMovingAverage(inputs=(USEquityPricing.close,), window_length=10)
        today = pd.Timestamp('2015')
        assets = np.arange(5, dtype=np.int64)
        data = np.ones((10, 5))
        out = np.zeros(data.shape[1])
        wma1.compute(today, assets, out, data)
        assert_equal(out, np.ones(5))

    def test_wma2(self):
        if False:
            i = 10
            return i + 15
        wma2 = LinearWeightedMovingAverage(inputs=(USEquityPricing.close,), window_length=10)
        today = pd.Timestamp('2015')
        assets = np.arange(5, dtype=np.int64)
        data = np.arange(50, dtype=np.float64).reshape((10, 5))
        out = np.zeros(data.shape[1])
        wma2.compute(today, assets, out, data)
        assert_equal(out, np.array([30.0, 31.0, 32.0, 33.0, 34.0]))

class TestTrueRange(ZiplineTestCase):

    def test_tr_basic(self):
        if False:
            for i in range(10):
                print('nop')
        tr = TrueRange()
        today = pd.Timestamp('2014')
        assets = np.arange(3, dtype=np.int64)
        out = np.empty(3, dtype=np.float64)
        highs = np.full((2, 3), 3.0)
        lows = np.full((2, 3), 2.0)
        closes = np.full((2, 3), 1.0)
        tr.compute(today, assets, out, highs, lows, closes)
        assert_equal(out, np.full((3,), 2.0))

class MovingAverageConvergenceDivergenceTestCase(ZiplineTestCase):

    def expected_ewma(self, data_df, window):
        if False:
            for i in range(10):
                print('nop')
        return data_df.rolling(window).apply(lambda sub: pd.DataFrame(sub).ewm(span=window).mean().values[-1])

    @parameter_space(seed=range(5))
    def test_MACD_window_length_generation(self, seed):
        if False:
            while True:
                i = 10
        rng = RandomState(seed)
        signal_period = rng.randint(1, 90)
        fast_period = rng.randint(signal_period + 1, signal_period + 100)
        slow_period = rng.randint(fast_period + 1, fast_period + 100)
        ewma = MovingAverageConvergenceDivergenceSignal(fast_period=fast_period, slow_period=slow_period, signal_period=signal_period)
        assert_equal(ewma.window_length, slow_period + signal_period - 1)

    def test_bad_inputs(self):
        if False:
            while True:
                i = 10
        template = 'MACDSignal() expected a value greater than or equal to 1 for argument %r, but got 0 instead.'
        with self.assertRaises(ValueError) as e:
            MovingAverageConvergenceDivergenceSignal(fast_period=0)
        self.assertEqual(template % 'fast_period', str(e.exception))
        with self.assertRaises(ValueError) as e:
            MovingAverageConvergenceDivergenceSignal(slow_period=0)
        self.assertEqual(template % 'slow_period', str(e.exception))
        with self.assertRaises(ValueError) as e:
            MovingAverageConvergenceDivergenceSignal(signal_period=0)
        self.assertEqual(template % 'signal_period', str(e.exception))
        with self.assertRaises(ValueError) as e:
            MovingAverageConvergenceDivergenceSignal(fast_period=5, slow_period=4)
        expected = "'slow_period' must be greater than 'fast_period', but got\nslow_period=4, fast_period=5"
        self.assertEqual(expected, str(e.exception))

    @parameter_space(seed=range(2), fast_period=[3, 5], slow_period=[8, 10], signal_period=[3, 9], __fail_fast=True)
    def test_moving_average_convergence_divergence(self, seed, fast_period, slow_period, signal_period):
        if False:
            for i in range(10):
                print('nop')
        rng = RandomState(seed)
        nassets = 3
        macd = MovingAverageConvergenceDivergenceSignal(fast_period=fast_period, slow_period=slow_period, signal_period=signal_period)
        today = pd.Timestamp('2016', tz='utc')
        assets = pd.Index(np.arange(nassets))
        out = np.empty(shape=(nassets,), dtype=np.float64)
        close = rng.rand(macd.window_length, nassets)
        macd.compute(today, assets, out, close, fast_period, slow_period, signal_period)
        close_df = pd.DataFrame(close)
        fast_ewma = self.expected_ewma(close_df, fast_period)
        slow_ewma = self.expected_ewma(close_df, slow_period)
        signal_ewma = self.expected_ewma(fast_ewma - slow_ewma, signal_period)
        self.assertTrue(signal_ewma.iloc[:-1].isnull().all().all())
        expected_signal = signal_ewma.values[-1]
        np.testing.assert_almost_equal(out, expected_signal, decimal=8)

class RSITestCase(ZiplineTestCase):

    @parameterized.expand([(100, np.array([41.032913785966, 51.553585468393, 51.022005016446])), (101, np.array([43.506969935466, 46.145367530182, 50.57407044197])), (102, np.array([46.610102205934, 47.646892444315, 52.13182788538]))])
    def test_rsi(self, seed_value, expected):
        if False:
            for i in range(10):
                print('nop')
        rsi = RSI()
        today = np.datetime64(1, 'ns')
        assets = np.arange(3)
        out = np.empty((3,), dtype=float)
        np.random.seed(seed_value)
        test_data = np.abs(np.random.randn(15, 3))
        out = np.empty((3,), dtype=float)
        rsi.compute(today, assets, out, test_data)
        check_allclose(expected, out)

    def test_rsi_all_positive_returns(self):
        if False:
            i = 10
            return i + 15
        '\n        RSI indicator should be 100 in the case of 14 days of positive returns.\n        '
        rsi = RSI()
        today = np.datetime64(1, 'ns')
        assets = np.arange(1)
        out = np.empty((1,), dtype=float)
        closes = np.linspace(46, 60, num=15)
        closes.shape = (15, 1)
        rsi.compute(today, assets, out, closes)
        self.assertEqual(out[0], 100.0)

    def test_rsi_all_negative_returns(self):
        if False:
            print('Hello World!')
        '\n        RSI indicator should be 0 in the case of 14 days of negative returns.\n        '
        rsi = RSI()
        today = np.datetime64(1, 'ns')
        assets = np.arange(1)
        out = np.empty((1,), dtype=float)
        closes = np.linspace(46, 32, num=15)
        closes.shape = (15, 1)
        rsi.compute(today, assets, out, closes)
        self.assertEqual(out[0], 0.0)

    def test_rsi_same_returns(self):
        if False:
            i = 10
            return i + 15
        '\n        RSI indicator should be the same for two price series with the same\n        returns, even if the prices are different.\n        '
        rsi = RSI()
        today = np.datetime64(1, 'ns')
        assets = np.arange(2)
        out = np.empty((2,), dtype=float)
        example_case = np.array([46.125, 47.125, 46.4375, 46.9375, 44.9375, 44.25, 44.625, 45.75, 47.8125, 47.5625, 47.0, 44.5625, 46.3125, 47.6875, 46.6875])
        double = example_case * 2
        closes = np.vstack((example_case, double)).T
        rsi.compute(today, assets, out, closes)
        self.assertAlmostEqual(out[0], out[1])

class AnnualizedVolatilityTestCase(ZiplineTestCase):
    """
    Test Annualized Volatility
    """

    def test_simple_volatility(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Simple test for uniform returns should generate 0 volatility\n        '
        nassets = 3
        ann_vol = AnnualizedVolatility()
        today = pd.Timestamp('2016', tz='utc')
        assets = np.arange(nassets, dtype=np.float64)
        returns = np.full((ann_vol.window_length, nassets), 0.004, dtype=np.float64)
        out = np.empty(shape=(nassets,), dtype=np.float64)
        ann_vol.compute(today, assets, out, returns, 252)
        expected_vol = np.zeros(nassets)
        np.testing.assert_almost_equal(out, expected_vol, decimal=8)

    def test_volatility(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check volatility results against values calculated manually\n        '
        nassets = 3
        ann_vol = AnnualizedVolatility()
        today = pd.Timestamp('2016', tz='utc')
        assets = np.arange(nassets, dtype=np.float64)
        returns = np.random.normal(loc=0.001, scale=0.01, size=(ann_vol.window_length, nassets))
        out = np.empty(shape=(nassets,), dtype=np.float64)
        ann_vol.compute(today, assets, out, returns, 252)
        mean = np.mean(returns, axis=0)
        annualized_variance = ((returns - mean) ** 2).sum(axis=0) / returns.shape[0] * 252
        expected_vol = np.sqrt(annualized_variance)
        np.testing.assert_almost_equal(out, expected_vol, decimal=8)