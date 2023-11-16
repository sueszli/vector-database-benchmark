from collections import OrderedDict
from numpy import array, append, nan, full
from numpy.testing import assert_almost_equal
import pandas as pd
from pandas import Timedelta
from six import iteritems
from zipline.assets import Equity, Future
from zipline.data.data_portal import HISTORY_FREQUENCIES, OHLCV_FIELDS
from zipline.data.minute_bars import FUTURES_MINUTES_PER_DAY, US_EQUITIES_MINUTES_PER_DAY
from zipline.testing import parameter_space
from zipline.testing.fixtures import ZiplineTestCase, WithTradingSessions, WithDataPortal, alias
from zipline.testing.predicates import assert_equal
from zipline.utils.numpy_utils import float64_dtype

class DataPortalTestBase(WithDataPortal, WithTradingSessions):
    ASSET_FINDER_EQUITY_SIDS = (1, 2, 3)
    DIVIDEND_ASSET_SID = 3
    START_DATE = pd.Timestamp('2016-08-01')
    END_DATE = pd.Timestamp('2016-08-08')
    TRADING_CALENDAR_STRS = ('NYSE', 'us_futures')
    EQUITY_DAILY_BAR_SOURCE_FROM_MINUTE = True
    OHLC_RATIOS_PER_SID = {10001: 100000}

    @classmethod
    def make_root_symbols_info(self):
        if False:
            for i in range(10):
                print('nop')
        return pd.DataFrame({'root_symbol': ['BAR', 'BUZ'], 'root_symbol_id': [1, 2], 'exchange': ['CMES', 'CMES']})

    @classmethod
    def make_futures_info(cls):
        if False:
            for i in range(10):
                print('nop')
        trading_sessions = cls.trading_sessions['us_futures']
        return pd.DataFrame({'sid': [10000, 10001], 'root_symbol': ['BAR', 'BUZ'], 'symbol': ['BARA', 'BUZZ'], 'start_date': [trading_sessions[1], trading_sessions[0]], 'end_date': [cls.END_DATE, cls.END_DATE], 'notice_date': [cls.END_DATE, cls.END_DATE], 'expiration_date': [cls.END_DATE, cls.END_DATE], 'tick_size': [0.01, 0.0001], 'multiplier': [500, 50000], 'exchange': ['CMES', 'CMES']})

    @classmethod
    def make_equity_minute_bar_data(cls):
        if False:
            i = 10
            return i + 15
        trading_calendar = cls.trading_calendars[Equity]
        dts = trading_calendar.minutes_for_session(cls.trading_days[0])
        dfs = []
        dfs.append(pd.DataFrame({'open': full(len(dts), nan), 'high': full(len(dts), nan), 'low': full(len(dts), nan), 'close': full(len(dts), nan), 'volume': full(len(dts), 0)}, index=dts))
        dts = trading_calendar.minutes_for_session(cls.trading_days[1])
        dfs.append(pd.DataFrame({'open': append(100.5, full(len(dts) - 1, nan)), 'high': append(100.9, full(len(dts) - 1, nan)), 'low': append(100.1, full(len(dts) - 1, nan)), 'close': append(100.3, full(len(dts) - 1, nan)), 'volume': append(1000, full(len(dts) - 1, nan))}, index=dts))
        dts = trading_calendar.minutes_for_session(cls.trading_days[2])
        dfs.append(pd.DataFrame({'open': [nan, 103.5, 102.5, 104.5, 101.5, nan], 'high': [nan, 103.9, 102.9, 104.9, 101.9, nan], 'low': [nan, 103.1, 102.1, 104.1, 101.1, nan], 'close': [nan, 103.3, 102.3, 104.3, 101.3, nan], 'volume': [0, 1003, 1002, 1004, 1001, 0]}, index=dts[:6]))
        dts = trading_calendar.minutes_for_session(cls.trading_days[3])
        dfs.append(pd.DataFrame({'open': full(len(dts), nan), 'high': full(len(dts), nan), 'low': full(len(dts), nan), 'close': full(len(dts), nan), 'volume': full(len(dts), 0)}, index=dts))
        asset1_df = pd.concat(dfs)
        yield (1, asset1_df)
        asset2_df = pd.DataFrame({'open': 1.0055, 'high': 1.0059, 'low': 1.0051, 'close': 1.0055, 'volume': 100}, index=asset1_df.index)
        yield (2, asset2_df)
        yield (cls.DIVIDEND_ASSET_SID, asset2_df.copy())

    @classmethod
    def make_future_minute_bar_data(cls):
        if False:
            i = 10
            return i + 15
        trading_calendar = cls.trading_calendars[Future]
        trading_sessions = cls.trading_sessions['us_futures']
        dts = trading_calendar.minutes_for_session(trading_sessions[1])
        dfs = []
        dfs.append(pd.DataFrame({'open': full(len(dts), nan), 'high': full(len(dts), nan), 'low': full(len(dts), nan), 'close': full(len(dts), nan), 'volume': full(len(dts), 0)}, index=dts))
        dts = trading_calendar.minutes_for_session(trading_sessions[2])
        dfs.append(pd.DataFrame({'open': append(200.5, full(len(dts) - 1, nan)), 'high': append(200.9, full(len(dts) - 1, nan)), 'low': append(200.1, full(len(dts) - 1, nan)), 'close': append(200.3, full(len(dts) - 1, nan)), 'volume': append(2000, full(len(dts) - 1, nan))}, index=dts))
        dts = trading_calendar.minutes_for_session(trading_sessions[3])
        dfs.append(pd.DataFrame({'open': [nan, 203.5, 202.5, 204.5, 201.5, nan], 'high': [nan, 203.9, 202.9, 204.9, 201.9, nan], 'low': [nan, 203.1, 202.1, 204.1, 201.1, nan], 'close': [nan, 203.3, 202.3, 204.3, 201.3, nan], 'volume': [0, 2003, 2002, 2004, 2001, 0]}, index=dts[:6]))
        dts = trading_calendar.minutes_for_session(trading_sessions[4])
        dfs.append(pd.DataFrame({'open': full(len(dts), nan), 'high': full(len(dts), nan), 'low': full(len(dts), nan), 'close': full(len(dts), nan), 'volume': full(len(dts), 0)}, index=dts))
        asset10000_df = pd.concat(dfs)
        yield (10000, asset10000_df)
        missing_dts = trading_calendar.minutes_for_session(trading_sessions[0])
        asset10001_df = pd.DataFrame({'open': 1.00549, 'high': 1.00591, 'low': 1.00507, 'close': 1.0055, 'volume': 100}, index=missing_dts.append(asset10000_df.index))
        yield (10001, asset10001_df)

    @classmethod
    def make_dividends_data(cls):
        if False:
            i = 10
            return i + 15
        return pd.DataFrame([{'ex_date': cls.trading_days[2].to_datetime64(), 'record_date': cls.trading_days[2].to_datetime64(), 'declared_date': cls.trading_days[2].to_datetime64(), 'pay_date': cls.trading_days[2].to_datetime64(), 'amount': 0.5, 'sid': cls.DIVIDEND_ASSET_SID}], columns=['ex_date', 'record_date', 'declared_date', 'pay_date', 'amount', 'sid'])

    def test_get_last_traded_equity_minute(self):
        if False:
            print('Hello World!')
        trading_calendar = self.trading_calendars[Equity]
        dts = trading_calendar.minutes_for_session(self.trading_days[0])
        asset = self.asset_finder.retrieve_asset(1)
        self.assertTrue(pd.isnull(self.data_portal.get_last_traded_dt(asset, dts[0], 'minute')))
        dts = trading_calendar.minutes_for_session(self.trading_days[2])
        self.assertEqual(dts[1], self.data_portal.get_last_traded_dt(asset, dts[1], 'minute'))
        self.assertEqual(dts[4], self.data_portal.get_last_traded_dt(asset, dts[5], 'minute'))

    def test_get_last_traded_future_minute(self):
        if False:
            return 10
        asset = self.asset_finder.retrieve_asset(10000)
        trading_calendar = self.trading_calendars[Future]
        dts = trading_calendar.minutes_for_session(self.trading_days[0])
        self.assertTrue(pd.isnull(self.data_portal.get_last_traded_dt(asset, dts[0], 'minute')))
        dts = trading_calendar.minutes_for_session(self.trading_days[3])
        self.assertEqual(dts[1], self.data_portal.get_last_traded_dt(asset, dts[1], 'minute'))
        self.assertEqual(dts[4], self.data_portal.get_last_traded_dt(asset, dts[5], 'minute'))

    def test_get_last_traded_dt_equity_daily(self):
        if False:
            return 10
        asset = self.asset_finder.retrieve_asset(1)
        self.assertTrue(pd.isnull(self.data_portal.get_last_traded_dt(asset, self.trading_days[0], 'daily')))
        self.assertEqual(self.trading_days[1], self.data_portal.get_last_traded_dt(asset, self.trading_days[1], 'daily'))
        self.assertEqual(self.trading_days[2], self.data_portal.get_last_traded_dt(asset, self.trading_days[3], 'daily'))

    def test_get_spot_value_equity_minute(self):
        if False:
            print('Hello World!')
        trading_calendar = self.trading_calendars[Equity]
        asset = self.asset_finder.retrieve_asset(1)
        dts = trading_calendar.minutes_for_session(self.trading_days[2])
        dt = dts[1]
        expected = OrderedDict({'open': 103.5, 'high': 103.9, 'low': 103.1, 'close': 103.3, 'volume': 1003, 'price': 103.3})
        result = [self.data_portal.get_spot_value(asset, field, dt, 'minute') for field in expected.keys()]
        assert_almost_equal(array(list(expected.values())), result)
        dt = dts[100]
        expected = OrderedDict({'open': nan, 'high': nan, 'low': nan, 'close': nan, 'volume': 0, 'price': 101.3})
        result = [self.data_portal.get_spot_value(asset, field, dt, 'minute') for field in expected.keys()]
        assert_almost_equal(array(list(expected.values())), result)

    def test_get_spot_value_future_minute(self):
        if False:
            for i in range(10):
                print('nop')
        trading_calendar = self.trading_calendars[Future]
        asset = self.asset_finder.retrieve_asset(10000)
        dts = trading_calendar.minutes_for_session(self.trading_days[3])
        dt = dts[1]
        expected = OrderedDict({'open': 203.5, 'high': 203.9, 'low': 203.1, 'close': 203.3, 'volume': 2003, 'price': 203.3})
        result = [self.data_portal.get_spot_value(asset, field, dt, 'minute') for field in expected.keys()]
        assert_almost_equal(array(list(expected.values())), result)
        dt = dts[100]
        expected = OrderedDict({'open': nan, 'high': nan, 'low': nan, 'close': nan, 'volume': 0, 'price': 201.3})
        result = [self.data_portal.get_spot_value(asset, field, dt, 'minute') for field in expected.keys()]
        assert_almost_equal(array(list(expected.values())), result)

    def test_get_spot_value_multiple_assets(self):
        if False:
            while True:
                i = 10
        equity = self.asset_finder.retrieve_asset(1)
        future = self.asset_finder.retrieve_asset(10000)
        trading_calendar = self.trading_calendars[Future]
        dts = trading_calendar.minutes_for_session(self.trading_days[3])
        expected = pd.DataFrame({equity: [nan, nan, nan, nan, 0, 101.3], future: [203.5, 203.9, 203.1, 203.3, 2003, 203.3]}, index=['open', 'high', 'low', 'close', 'volume', 'price'])
        result = [self.data_portal.get_spot_value(assets=[equity, future], field=field, dt=dts[1], data_frequency='minute') for field in expected.index]
        assert_almost_equal(expected.values.tolist(), result)

    @parameter_space(data_frequency=['daily', 'minute'], field=['close', 'price'])
    def test_get_adjustments(self, data_frequency, field):
        if False:
            for i in range(10):
                print('nop')
        asset = self.asset_finder.retrieve_asset(self.DIVIDEND_ASSET_SID)
        calendar = self.trading_calendars[Equity]
        day = calendar.day
        dividend_date = self.trading_days[2]
        prev_day_price = 1.006
        dividend_amount = 0.5
        ratio = 1.0 - dividend_amount / prev_day_price
        cases = OrderedDict([((dividend_date - day, dividend_date - day), 1.0), ((dividend_date - day, dividend_date), ratio), ((dividend_date - day, dividend_date + day), ratio), ((dividend_date, dividend_date), 1.0), ((dividend_date, dividend_date + day), 1.0), ((dividend_date + day, dividend_date + day), 1.0)])
        for ((dt, perspective_dt), expected) in iteritems(cases):
            if data_frequency == 'minute':
                dt = calendar.session_open(dt)
                perspective_dt = calendar.session_open(perspective_dt)
            val = self.data_portal.get_adjustments(asset, field, dt, perspective_dt)[0]
            assert_almost_equal(val, expected, err_msg='at dt={} perspective={}'.format(dt, perspective_dt))

    def test_bar_count_for_simple_transforms(self):
        if False:
            while True:
                i = 10
        july_9_dt = self.trading_calendar.open_and_close_for_session(pd.Timestamp('2015-07-09', tz='UTC'))[0] + Timedelta('30 minutes')
        self.assertEqual(3 * 390 + 31, self.data_portal._get_minute_count_for_transform(july_9_dt, 4))
        nov_30_dt = self.trading_calendar.open_and_close_for_session(pd.Timestamp('2015-11-30', tz='UTC'))[0] + Timedelta('30 minutes')
        self.assertEqual(390 + 390 + 210 + 31, self.data_portal._get_minute_count_for_transform(nov_30_dt, 4))

    def test_get_last_traded_dt_minute(self):
        if False:
            while True:
                i = 10
        minutes = self.nyse_calendar.minutes_for_session(self.trading_days[2])
        equity = self.asset_finder.retrieve_asset(1)
        result = self.data_portal.get_last_traded_dt(equity, minutes[3], 'minute')
        self.assertEqual(minutes[3], result, 'Asset 1 had a trade on third minute, so should return that as the last trade on that dt.')
        result = self.data_portal.get_last_traded_dt(equity, minutes[5], 'minute')
        self.assertEqual(minutes[4], result, 'Asset 1 had a trade on fourth minute, so should return that as the last trade on the fifth.')
        future = self.asset_finder.retrieve_asset(10000)
        calendar = self.trading_calendars[Future]
        minutes = calendar.minutes_for_session(self.trading_days[3])
        result = self.data_portal.get_last_traded_dt(future, minutes[3], 'minute')
        self.assertEqual(minutes[3], result, 'Asset 10000 had a trade on the third minute, so return that as the last trade on that dt.')
        result = self.data_portal.get_last_traded_dt(future, minutes[5], 'minute')
        self.assertEqual(minutes[4], result, 'Asset 10000 had a trade on fourth minute, so should return that as the last trade on the fifth.')

    def test_get_empty_splits(self):
        if False:
            return 10
        splits = self.data_portal.get_splits([], self.trading_days[2])
        self.assertEqual([], splits)

    @parameter_space(frequency=HISTORY_FREQUENCIES, field=OHLCV_FIELDS)
    def test_price_rounding(self, frequency, field):
        if False:
            i = 10
            return i + 15
        equity = self.asset_finder.retrieve_asset(2)
        future = self.asset_finder.retrieve_asset(10001)
        cf = self.data_portal.asset_finder.create_continuous_future('BUZ', 0, 'calendar', None)
        minutes = self.nyse_calendar.minutes_for_session(self.trading_days[0])
        if frequency == '1m':
            minute = minutes[0]
            expected_equity_volume = 100
            expected_future_volume = 100
            data_frequency = 'minute'
        else:
            minute = minutes[0].normalize()
            expected_equity_volume = 100 * US_EQUITIES_MINUTES_PER_DAY
            expected_future_volume = 100 * FUTURES_MINUTES_PER_DAY
            data_frequency = 'daily'
        expected_equity_values = {'open': 1.006, 'high': 1.006, 'low': 1.005, 'close': 1.006, 'volume': expected_equity_volume}
        expected_future_values = {'open': 1.0055, 'high': 1.0059, 'low': 1.0051, 'close': 1.0055, 'volume': expected_future_volume}
        result = self.data_portal.get_history_window(assets=[equity, future, cf], end_dt=minute, bar_count=1, frequency=frequency, field=field, data_frequency=data_frequency)
        expected_result = pd.DataFrame({equity: expected_equity_values[field], future: expected_future_values[field], cf: expected_future_values[field]}, index=[minute], dtype=float64_dtype)
        assert_equal(result, expected_result)

class TestDataPortal(DataPortalTestBase, ZiplineTestCase):
    DATA_PORTAL_LAST_AVAILABLE_SESSION = None
    DATA_PORTAL_LAST_AVAILABLE_MINUTE = None

class TestDataPortalExplicitLastAvailable(DataPortalTestBase, ZiplineTestCase):
    DATA_PORTAL_LAST_AVAILABLE_SESSION = alias('START_DATE')
    DATA_PORTAL_LAST_AVAILABLE_MINUTE = alias('END_DATE')