import warnings
from mock import patch
import numpy as np
import pandas as pd
from pandas.core.common import PerformanceWarning
from zipline.finance.trading import SimulationParameters
from zipline.testing import MockDailyBarReader, create_daily_df_for_asset, create_minute_df_for_asset, str_to_seconds
from zipline.testing.fixtures import WithCreateBarData, WithMakeAlgo, ZiplineTestCase
from zipline.zipline_warnings import ZiplineDeprecationWarning
simple_algo = '\nfrom zipline.api import sid, order\ndef initialize(context):\n    pass\n\ndef handle_data(context, data):\n    assert sid(1) in data\n    assert sid(2) in data\n    assert len(data) == 3\n    for asset in data:\n        pass\n'
history_algo = '\nfrom zipline.api import sid, history\n\ndef initialize(context):\n    context.sid1 = sid(1)\n\ndef handle_data(context, data):\n    context.history_window = history(5, "1m", "volume")\n'
history_bts_algo = '\nfrom zipline.api import sid, history, record\n\ndef initialize(context):\n    context.sid3 = sid(3)\n    context.num_bts = 0\n\ndef before_trading_start(context, data):\n    context.num_bts += 1\n\n    # Get history at the second BTS (beginning of second day)\n    if context.num_bts == 2:\n        record(history=history(5, "1m", "volume"))\n\ndef handle_data(context, data):\n    pass\n'
simple_transforms_algo = '\nfrom zipline.api import sid\ndef initialize(context):\n    context.count = 0\n\ndef handle_data(context, data):\n    if context.count == 2:\n        context.mavg = data[sid(1)].mavg(5)\n        context.vwap = data[sid(1)].vwap(5)\n        context.stddev = data[sid(1)].stddev(5)\n        context.returns = data[sid(1)].returns()\n\n    context.count += 1\n'
manipulation_algo = '\ndef initialize(context):\n    context.asset1 = sid(1)\n    context.asset2 = sid(2)\n\ndef handle_data(context, data):\n    assert len(data) == 2\n    assert len(data.keys()) == 2\n    assert context.asset1 in data.keys()\n    assert context.asset2 in data.keys()\n'
sid_accessor_algo = '\nfrom zipline.api import sid\n\ndef initialize(context):\n    context.asset1 = sid(1)\n\ndef handle_data(context,data):\n    assert data[sid(1)].sid == context.asset1\n    assert data[sid(1)]["sid"] == context.asset1\n'
data_items_algo = '\nfrom zipline.api import sid\n\ndef initialize(context):\n    context.asset1 = sid(1)\n    context.asset2 = sid(2)\n\ndef handle_data(context, data):\n    iter_list = list(data.iteritems())\n    items_list = data.items()\n    assert iter_list == items_list\n'
reference_missing_position_by_int_algo = '\ndef initialize(context):\n    pass\n\ndef handle_data(context, data):\n    context.portfolio.positions[24]\n'
reference_missing_position_by_unexpected_type_algo = '\ndef initialize(context):\n    pass\n\ndef handle_data(context, data):\n    context.portfolio.positions["foobar"]\n'

class TestAPIShim(WithCreateBarData, WithMakeAlgo, ZiplineTestCase):
    START_DATE = pd.Timestamp('2016-01-05', tz='UTC')
    END_DATE = pd.Timestamp('2016-01-28', tz='UTC')
    SIM_PARAMS_DATA_FREQUENCY = 'minute'
    sids = ASSET_FINDER_EQUITY_SIDS = (1, 2, 3)

    @classmethod
    def make_equity_minute_bar_data(cls):
        if False:
            print('Hello World!')
        for sid in cls.sids:
            yield (sid, create_minute_df_for_asset(cls.trading_calendar, cls.SIM_PARAMS_START, cls.SIM_PARAMS_END))

    @classmethod
    def make_equity_daily_bar_data(cls, country_code, sids):
        if False:
            while True:
                i = 10
        for sid in sids:
            yield (sid, create_daily_df_for_asset(cls.trading_calendar, cls.SIM_PARAMS_START, cls.SIM_PARAMS_END))

    @classmethod
    def make_splits_data(cls):
        if False:
            print('Hello World!')
        return pd.DataFrame([{'effective_date': str_to_seconds('2016-01-06'), 'ratio': 0.5, 'sid': 3}])

    @classmethod
    def make_adjustment_writer_equity_daily_bar_reader(cls):
        if False:
            while True:
                i = 10
        return MockDailyBarReader(dates=cls.nyse_calendar.sessions_in_range(cls.START_DATE, cls.END_DATE))

    @classmethod
    def init_class_fixtures(cls):
        if False:
            for i in range(10):
                print('nop')
        super(TestAPIShim, cls).init_class_fixtures()
        cls.asset1 = cls.asset_finder.retrieve_asset(1)
        cls.asset2 = cls.asset_finder.retrieve_asset(2)
        cls.asset3 = cls.asset_finder.retrieve_asset(3)

    def create_algo(self, code, filename=None, sim_params=None):
        if False:
            i = 10
            return i + 15
        if sim_params is None:
            sim_params = self.sim_params
        return self.make_algo(script=code, sim_params=sim_params, algo_filename=filename)

    def test_old_new_data_api_paths(self):
        if False:
            return 10
        '\n        Test that the new and old data APIs hit the same code paths.\n\n        We want to ensure that the old data API(data[sid(N)].field and\n        similar)  and the new data API(data.current(sid(N), field) and\n        similar) hit the same code paths on the DataPortal.\n        '
        test_start_minute = self.trading_calendar.minutes_for_session(self.sim_params.sessions[0])[1]
        test_end_minute = self.trading_calendar.minutes_for_session(self.sim_params.sessions[0])[-1]
        bar_data = self.create_bardata(lambda : test_end_minute)
        ohlcvp_fields = ['open', 'high', 'lowclose', 'volume', 'price']
        spot_value_meth = 'zipline.data.data_portal.DataPortal.get_spot_value'

        def assert_get_spot_value_called(fun, field):
            if False:
                i = 10
                return i + 15
            '\n            Assert that get_spot_value was called during the execution of fun.\n\n            Takes in a function fun and a string field.\n            '
            with patch(spot_value_meth) as gsv:
                fun()
                gsv.assert_called_with(self.asset1, field, test_end_minute, 'minute')
        for field in ohlcvp_fields:
            assert_get_spot_value_called(lambda : getattr(bar_data[self.asset1], field), field)
            assert_get_spot_value_called(lambda : bar_data.current(self.asset1, field), field)
        history_meth = 'zipline.data.data_portal.DataPortal.get_history_window'

        def assert_get_history_window_called(fun, is_legacy):
            if False:
                return 10
            '\n            Assert that get_history_window was called during fun().\n\n            Takes in a function fun and a boolean is_legacy.\n            '
            with patch(history_meth) as ghw:
                fun()
                if is_legacy:
                    ghw.assert_called_with([self.asset1, self.asset2, self.asset3], test_end_minute, 5, '1m', 'volume', 'minute', True)
                else:
                    ghw.assert_called_with([self.asset1, self.asset2, self.asset3], test_end_minute, 5, '1m', 'volume', 'minute')
        test_sim_params = SimulationParameters(start_session=test_start_minute, end_session=test_end_minute, data_frequency='minute', trading_calendar=self.trading_calendar)
        history_algorithm = self.create_algo(history_algo, sim_params=test_sim_params)
        assert_get_history_window_called(lambda : history_algorithm.run(), is_legacy=True)
        assert_get_history_window_called(lambda : bar_data.history([self.asset1, self.asset2, self.asset3], 'volume', 5, '1m'), is_legacy=False)

    def test_sid_accessor(self):
        if False:
            while True:
                i = 10
        '\n        Test that we maintain backwards compat for sid access on a data object.\n\n        We want to support both data[sid(24)].sid, as well as\n        data[sid(24)]["sid"]. Since these are deprecated and will eventually\n        cease to be supported, we also want to assert that we\'re seeing a\n        deprecation warning.\n        '
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('ignore', PerformanceWarning)
            warnings.simplefilter('default', ZiplineDeprecationWarning)
            algo = self.create_algo(sid_accessor_algo)
            algo.run()
            self.assertEqual(2, len(w))
            for warning in w:
                self.assertEqual(ZiplineDeprecationWarning, warning.category)
                self.assertEqual('`data[sid(N)]` is deprecated. Use `data.current`.', str(warning.message))

    def test_data_items(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that we maintain backwards compat for data.[items | iteritems].\n\n        We also want to assert that we warn that iterating over the assets\n        in `data` is deprecated.\n        '
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('ignore', PerformanceWarning)
            warnings.simplefilter('default', ZiplineDeprecationWarning)
            algo = self.create_algo(data_items_algo)
            algo.run()
            self.assertEqual(4, len(w))
            for (idx, warning) in enumerate(w):
                self.assertEqual(ZiplineDeprecationWarning, warning.category)
                if idx % 2 == 0:
                    self.assertEqual('Iterating over the assets in `data` is deprecated.', str(warning.message))
                else:
                    self.assertEqual('`data[sid(N)]` is deprecated. Use `data.current`.', str(warning.message))

    def test_iterate_data(self):
        if False:
            print('Hello World!')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('ignore', PerformanceWarning)
            warnings.simplefilter('default', ZiplineDeprecationWarning)
            algo = self.create_algo(simple_algo)
            algo.run()
            self.assertEqual(4, len(w))
            line_nos = [warning.lineno for warning in w]
            self.assertEqual(4, len(set(line_nos)))
            for (idx, warning) in enumerate(w):
                self.assertEqual(ZiplineDeprecationWarning, warning.category)
                self.assertEqual('<string>', warning.filename)
                self.assertEqual(line_nos[idx], warning.lineno)
                if idx < 2:
                    self.assertEqual('Checking whether an asset is in data is deprecated.', str(warning.message))
                else:
                    self.assertEqual('Iterating over the assets in `data` is deprecated.', str(warning.message))

    def test_history(self):
        if False:
            return 10
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('ignore', PerformanceWarning)
            warnings.simplefilter('default', ZiplineDeprecationWarning)
            sim_params = self.sim_params.create_new(self.sim_params.sessions[1], self.sim_params.end_session)
            algo = self.create_algo(history_algo, sim_params=sim_params)
            algo.run()
            self.assertEqual(1, len(w))
            self.assertEqual(ZiplineDeprecationWarning, w[0].category)
            self.assertEqual('<string>', w[0].filename)
            self.assertEqual(8, w[0].lineno)
            self.assertEqual('The `history` method is deprecated.  Use `data.history` instead.', str(w[0].message))

    def test_old_new_history_bts_paths(self):
        if False:
            return 10
        '\n        Tests that calling history in before_trading_start gets us the correct\n        values, which involves 1) calling data_portal.get_history_window as of\n        the previous market minute, 2) getting adjustments between the previous\n        market minute and the current time, and 3) applying those adjustments\n        '
        algo = self.create_algo(history_bts_algo)
        algo.run()
        expected_vol_without_split = np.arange(386, 391) * 100
        expected_vol_with_split = np.arange(386, 391) * 200
        window = algo.recorded_vars['history']
        np.testing.assert_array_equal(window[self.asset1].values, expected_vol_without_split)
        np.testing.assert_array_equal(window[self.asset2].values, expected_vol_without_split)
        np.testing.assert_array_equal(window[self.asset3].values, expected_vol_with_split)

    def test_simple_transforms(self):
        if False:
            print('Hello World!')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('ignore', PerformanceWarning)
            warnings.simplefilter('default', ZiplineDeprecationWarning)
            sim_params = SimulationParameters(start_session=self.sim_params.sessions[8], end_session=self.sim_params.sessions[-1], data_frequency='minute', trading_calendar=self.trading_calendar)
            algo = self.create_algo(simple_transforms_algo, sim_params=sim_params)
            algo.run()
            self.assertEqual(8, len(w))
            transforms = ['mavg', 'vwap', 'stddev', 'returns']
            for (idx, line_no) in enumerate(range(8, 12)):
                warning1 = w[idx * 2]
                warning2 = w[idx * 2 + 1]
                self.assertEqual('<string>', warning1.filename)
                self.assertEqual('<string>', warning2.filename)
                self.assertEqual(line_no, warning1.lineno)
                self.assertEqual(line_no, warning2.lineno)
                self.assertEqual('`data[sid(N)]` is deprecated. Use `data.current`.', str(warning1.message))
                self.assertEqual('The `{0}` method is deprecated.'.format(transforms[idx]), str(warning2.message))
            self.assertEqual(2342, algo.mavg)
            self.assertAlmostEqual(2428.92599, algo.vwap, places=5)
            self.assertAlmostEqual(451.34355, algo.stddev, places=5)
            self.assertAlmostEqual(346, algo.returns)

    def test_manipulation(self):
        if False:
            i = 10
            return i + 15
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('ignore', PerformanceWarning)
            warnings.simplefilter('default', ZiplineDeprecationWarning)
            algo = self.create_algo(simple_algo)
            algo.run()
            self.assertEqual(4, len(w))
            for (idx, warning) in enumerate(w):
                self.assertEqual('<string>', warning.filename)
                self.assertEqual(7 + idx, warning.lineno)
                if idx < 2:
                    self.assertEqual('Checking whether an asset is in data is deprecated.', str(warning.message))
                else:
                    self.assertEqual('Iterating over the assets in `data` is deprecated.', str(warning.message))

    def test_reference_empty_position_by_int(self):
        if False:
            for i in range(10):
                print('nop')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('default', ZiplineDeprecationWarning)
            algo = self.create_algo(reference_missing_position_by_int_algo)
            algo.run()
            self.assertEqual(1, len(w))
            self.assertEqual(str(w[0].message), 'Referencing positions by integer is deprecated. Use an asset instead.')

    def test_reference_empty_position_by_unexpected_type(self):
        if False:
            i = 10
            return i + 15
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('default', ZiplineDeprecationWarning)
            algo = self.create_algo(reference_missing_position_by_unexpected_type_algo)
            algo.run()
            self.assertEqual(1, len(w))
            self.assertEqual(str(w[0].message), 'Position lookup expected a value of type Asset but got str instead.')