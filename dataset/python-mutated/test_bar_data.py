from datetime import timedelta, time
from itertools import chain
from nose_parameterized import parameterized
import numpy as np
from numpy import nan
from numpy.testing import assert_almost_equal
import pandas as pd
from toolz import concat
from trading_calendars import get_calendar
from trading_calendars.utils.pandas_utils import days_at_time
from zipline._protocol import handle_non_market_minutes
from zipline.finance.asset_restrictions import Restriction, HistoricalRestrictions, RESTRICTION_STATES
from zipline.testing import MockDailyBarReader, create_daily_df_for_asset, create_minute_df_for_asset, str_to_seconds
from zipline.testing.fixtures import WithCreateBarData, WithDataPortal, ZiplineTestCase
OHLC = ['open', 'high', 'low', 'close']
OHLCP = OHLC + ['price']
ALL_FIELDS = OHLCP + ['volume', 'last_traded']
field_info = {'open': 1, 'high': 2, 'low': -1, 'close': 0}

def str_to_ts(dt_str):
    if False:
        print('Hello World!')
    return pd.Timestamp(dt_str, tz='UTC')

class WithBarDataChecks(object):

    def assert_same(self, val1, val2):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.assertEqual(val1, val2)
        except AssertionError:
            if val1 is pd.NaT:
                self.assertTrue(val2 is pd.NaT)
            elif np.isnan(val1):
                self.assertTrue(np.isnan(val2))
            else:
                raise

    def check_internal_consistency(self, bar_data):
        if False:
            print('Hello World!')
        df = bar_data.current([self.ASSET1, self.ASSET2], ALL_FIELDS)
        asset1_multi_field = bar_data.current(self.ASSET1, ALL_FIELDS)
        asset2_multi_field = bar_data.current(self.ASSET2, ALL_FIELDS)
        for field in ALL_FIELDS:
            asset1_value = bar_data.current(self.ASSET1, field)
            asset2_value = bar_data.current(self.ASSET2, field)
            multi_asset_series = bar_data.current([self.ASSET1, self.ASSET2], field)
            self.assert_same(multi_asset_series.loc[self.ASSET1], asset1_value)
            self.assert_same(multi_asset_series.loc[self.ASSET2], asset2_value)
            self.assert_same(df.loc[self.ASSET1][field], asset1_value)
            self.assert_same(df.loc[self.ASSET2][field], asset2_value)
            self.assert_same(asset1_multi_field[field], asset1_value)
            self.assert_same(asset2_multi_field[field], asset2_value)
        for field in ['data_portal', 'simulation_dt_func', 'data_frequency', '_views', '_universe_func', '_last_calculated_universe', '_universe_last_updatedat']:
            with self.assertRaises(AttributeError):
                getattr(bar_data, field)

class TestMinuteBarData(WithCreateBarData, WithBarDataChecks, WithDataPortal, ZiplineTestCase):
    START_DATE = pd.Timestamp('2016-01-05', tz='UTC')
    END_DATE = ASSET_FINDER_EQUITY_END_DATE = pd.Timestamp('2016-01-07', tz='UTC')
    ASSET_FINDER_EQUITY_SIDS = (1, 2, 3, 4, 5)
    SPLIT_ASSET_SID = 3
    ILLIQUID_SPLIT_ASSET_SID = 4
    HILARIOUSLY_ILLIQUID_ASSET_SID = 5

    @classmethod
    def make_equity_minute_bar_data(cls):
        if False:
            return 10
        for sid in (1, cls.SPLIT_ASSET_SID):
            yield (sid, create_minute_df_for_asset(cls.trading_calendar, cls.equity_minute_bar_days[0], cls.equity_minute_bar_days[-1]))
        for sid in (2, cls.ILLIQUID_SPLIT_ASSET_SID):
            yield (sid, create_minute_df_for_asset(cls.trading_calendar, cls.equity_minute_bar_days[0], cls.equity_minute_bar_days[-1], 10))
        yield (cls.HILARIOUSLY_ILLIQUID_ASSET_SID, create_minute_df_for_asset(cls.trading_calendar, cls.equity_minute_bar_days[0], cls.equity_minute_bar_days[-1], 50))

    @classmethod
    def make_futures_info(cls):
        if False:
            i = 10
            return i + 15
        return pd.DataFrame.from_dict({6: {'symbol': 'CLG06', 'root_symbol': 'CL', 'start_date': pd.Timestamp('2005-12-01', tz='UTC'), 'notice_date': pd.Timestamp('2005-12-20', tz='UTC'), 'expiration_date': pd.Timestamp('2006-01-20', tz='UTC'), 'exchange': 'ICEUS'}, 7: {'symbol': 'CLK06', 'root_symbol': 'CL', 'start_date': pd.Timestamp('2005-12-01', tz='UTC'), 'notice_date': pd.Timestamp('2006-03-20', tz='UTC'), 'expiration_date': pd.Timestamp('2006-04-20', tz='UTC'), 'exchange': 'ICEUS'}}, orient='index')

    @classmethod
    def make_splits_data(cls):
        if False:
            i = 10
            return i + 15
        return pd.DataFrame([{'effective_date': str_to_seconds('2016-01-06'), 'ratio': 0.5, 'sid': cls.SPLIT_ASSET_SID}, {'effective_date': str_to_seconds('2016-01-06'), 'ratio': 0.5, 'sid': cls.ILLIQUID_SPLIT_ASSET_SID}])

    @classmethod
    def init_class_fixtures(cls):
        if False:
            while True:
                i = 10
        super(TestMinuteBarData, cls).init_class_fixtures()
        cls.ASSET1 = cls.asset_finder.retrieve_asset(1)
        cls.ASSET2 = cls.asset_finder.retrieve_asset(2)
        cls.SPLIT_ASSET = cls.asset_finder.retrieve_asset(cls.SPLIT_ASSET_SID)
        cls.ILLIQUID_SPLIT_ASSET = cls.asset_finder.retrieve_asset(cls.ILLIQUID_SPLIT_ASSET_SID)
        cls.HILARIOUSLY_ILLIQUID_ASSET = cls.asset_finder.retrieve_asset(cls.HILARIOUSLY_ILLIQUID_ASSET_SID)
        cls.ASSETS = [cls.ASSET1, cls.ASSET2]

    def test_current_session(self):
        if False:
            print('Hello World!')
        regular_minutes = self.trading_calendar.minutes_for_sessions_in_range(self.equity_minute_bar_days[0], self.equity_minute_bar_days[-1])
        bts_minutes = days_at_time(self.equity_minute_bar_days, time(8, 45), 'US/Eastern')
        three_oh_six_am_minutes = days_at_time(self.equity_minute_bar_days, time(3, 6), 'US/Eastern')
        all_minutes = [regular_minutes, bts_minutes, three_oh_six_am_minutes]
        for minute in list(concat(all_minutes)):
            bar_data = self.create_bardata(lambda : minute)
            self.assertEqual(self.trading_calendar.minute_to_session_label(minute), bar_data.current_session)

    def test_current_session_minutes(self):
        if False:
            while True:
                i = 10
        first_day_minutes = self.trading_calendar.minutes_for_session(self.equity_minute_bar_days[0])
        for minute in first_day_minutes:
            bar_data = self.create_bardata(lambda : minute)
            np.testing.assert_array_equal(first_day_minutes, bar_data.current_session_minutes)

    def test_minute_before_assets_trading(self):
        if False:
            print('Hello World!')
        minutes = self.trading_calendar.minutes_for_session(self.trading_calendar.previous_session_label(self.equity_minute_bar_days[0]))
        for (idx, minute) in enumerate(minutes):
            bar_data = self.create_bardata(lambda : minute)
            self.check_internal_consistency(bar_data)
            self.assertFalse(bar_data.can_trade(self.ASSET1))
            self.assertFalse(bar_data.can_trade(self.ASSET2))
            self.assertFalse(bar_data.is_stale(self.ASSET1))
            self.assertFalse(bar_data.is_stale(self.ASSET2))
            for field in ALL_FIELDS:
                for asset in self.ASSETS:
                    asset_value = bar_data.current(asset, field)
                    if field in OHLCP:
                        self.assertTrue(np.isnan(asset_value))
                    elif field == 'volume':
                        self.assertEqual(0, asset_value)
                    elif field == 'last_traded':
                        self.assertTrue(asset_value is pd.NaT)

    def test_regular_minute(self):
        if False:
            for i in range(10):
                print('nop')
        minutes = self.trading_calendar.minutes_for_session(self.equity_minute_bar_days[0])
        for (idx, minute) in enumerate(minutes):
            bar_data = self.create_bardata(lambda : minute)
            self.check_internal_consistency(bar_data)
            asset2_has_data = (idx + 1) % 10 == 0
            self.assertTrue(bar_data.can_trade(self.ASSET1))
            self.assertFalse(bar_data.is_stale(self.ASSET1))
            if idx < 9:
                self.assertFalse(bar_data.can_trade(self.ASSET2))
                self.assertFalse(bar_data.is_stale(self.ASSET2))
            else:
                self.assertTrue(bar_data.can_trade(self.ASSET2))
                if asset2_has_data:
                    self.assertFalse(bar_data.is_stale(self.ASSET2))
                else:
                    self.assertTrue(bar_data.is_stale(self.ASSET2))
            for field in ALL_FIELDS:
                asset1_value = bar_data.current(self.ASSET1, field)
                asset2_value = bar_data.current(self.ASSET2, field)
                if idx == 0 and field == 'low':
                    self.assertTrue(np.isnan(asset1_value))
                elif field in OHLC:
                    self.assertEqual(idx + 1 + field_info[field], asset1_value)
                    if asset2_has_data:
                        self.assertEqual(idx + 1 + field_info[field], asset2_value)
                    else:
                        self.assertTrue(np.isnan(asset2_value))
                elif field == 'volume':
                    self.assertEqual((idx + 1) * 100, asset1_value)
                    if asset2_has_data:
                        self.assertEqual((idx + 1) * 100, asset2_value)
                    else:
                        self.assertEqual(0, asset2_value)
                elif field == 'price':
                    self.assertEqual(idx + 1, asset1_value)
                    if asset2_has_data:
                        self.assertEqual(idx + 1, asset2_value)
                    elif idx < 9:
                        self.assertTrue(np.isnan(asset2_value))
                    else:
                        self.assertEqual(idx // 10 * 10, asset2_value)
                elif field == 'last_traded':
                    self.assertEqual(minute, asset1_value)
                    if idx < 9:
                        self.assertTrue(asset2_value is pd.NaT)
                    elif asset2_has_data:
                        self.assertEqual(minute, asset2_value)
                    else:
                        last_traded_minute = minutes[idx // 10 * 10]
                        self.assertEqual(last_traded_minute - timedelta(minutes=1), asset2_value)

    def test_minute_of_last_day(self):
        if False:
            i = 10
            return i + 15
        minutes = self.trading_calendar.minutes_for_session(self.equity_daily_bar_days[-1])
        for (idx, minute) in enumerate(minutes):
            bar_data = self.create_bardata(lambda : minute)
            self.assertTrue(bar_data.can_trade(self.ASSET1))
            self.assertTrue(bar_data.can_trade(self.ASSET2))

    def test_minute_after_assets_stopped(self):
        if False:
            i = 10
            return i + 15
        minutes = self.trading_calendar.minutes_for_session(self.trading_calendar.next_session_label(self.equity_minute_bar_days[-1]))
        last_trading_minute = self.trading_calendar.minutes_for_session(self.equity_minute_bar_days[-1])[-1]
        for (idx, minute) in enumerate(minutes):
            bar_data = self.create_bardata(lambda : minute)
            self.assertFalse(bar_data.can_trade(self.ASSET1))
            self.assertFalse(bar_data.can_trade(self.ASSET2))
            self.assertFalse(bar_data.is_stale(self.ASSET1))
            self.assertFalse(bar_data.is_stale(self.ASSET2))
            self.check_internal_consistency(bar_data)
            for field in ALL_FIELDS:
                for asset in self.ASSETS:
                    asset_value = bar_data.current(asset, field)
                    if field in OHLCP:
                        self.assertTrue(np.isnan(asset_value))
                    elif field == 'volume':
                        self.assertEqual(0, asset_value)
                    elif field == 'last_traded':
                        self.assertEqual(last_trading_minute, asset_value)

    def test_get_value_is_unadjusted(self):
        if False:
            print('Hello World!')
        splits = self.adjustment_reader.get_adjustments_for_sid('splits', self.SPLIT_ASSET.sid)
        self.assertEqual(1, len(splits))
        split = splits[0]
        self.assertEqual(split[0], pd.Timestamp('2016-01-06', tz='UTC'))
        minutes = self.trading_calendar.minutes_for_sessions_in_range(self.equity_minute_bar_days[0], self.equity_minute_bar_days[1])
        for (idx, minute) in enumerate(minutes):
            bar_data = self.create_bardata(lambda : minute)
            self.assertEqual(idx + 1, bar_data.current(self.SPLIT_ASSET, 'price'))

    def test_get_value_is_adjusted_if_needed(self):
        if False:
            for i in range(10):
                print('nop')
        day0_minutes = self.trading_calendar.minutes_for_session(self.equity_minute_bar_days[0])
        day1_minutes = self.trading_calendar.minutes_for_session(self.equity_minute_bar_days[1])
        for (idx, minute) in enumerate(day0_minutes[-10:-1]):
            bar_data = self.create_bardata(lambda : minute)
            self.assertEqual(380, bar_data.current(self.ILLIQUID_SPLIT_ASSET, 'price'))
        bar_data = self.create_bardata(lambda : day0_minutes[-1])
        self.assertEqual(390, bar_data.current(self.ILLIQUID_SPLIT_ASSET, 'price'))
        for (idx, minute) in enumerate(day1_minutes[0:9]):
            bar_data = self.create_bardata(lambda : minute)
            self.assertEqual(195, bar_data.current(self.ILLIQUID_SPLIT_ASSET, 'price'))

    def test_get_value_at_midnight(self):
        if False:
            return 10
        day = self.equity_minute_bar_days[1]
        eight_fortyfive_am_eastern = pd.Timestamp('{0}-{1}-{2} 8:45'.format(day.year, day.month, day.day), tz='US/Eastern')
        bar_data = self.create_bardata(lambda : day)
        bar_data2 = self.create_bardata(lambda : eight_fortyfive_am_eastern)
        with handle_non_market_minutes(bar_data), handle_non_market_minutes(bar_data2):
            for bd in [bar_data, bar_data2]:
                for field in ['close', 'price']:
                    self.assertEqual(390, bd.current(self.ASSET1, field))
                self.assertEqual(350, bd.current(self.HILARIOUSLY_ILLIQUID_ASSET, 'price'))
                self.assertTrue(np.isnan(bd.current(self.HILARIOUSLY_ILLIQUID_ASSET, 'high')))
                self.assertEqual(0, bd.current(self.HILARIOUSLY_ILLIQUID_ASSET, 'volume'))

    def test_get_value_during_non_market_hours(self):
        if False:
            i = 10
            return i + 15
        bar_data = self.create_bardata(simulation_dt_func=lambda : pd.Timestamp('2016-01-06 4:15', tz='US/Eastern'))
        self.assertTrue(np.isnan(bar_data.current(self.ASSET1, 'open')))
        self.assertTrue(np.isnan(bar_data.current(self.ASSET1, 'high')))
        self.assertTrue(np.isnan(bar_data.current(self.ASSET1, 'low')))
        self.assertTrue(np.isnan(bar_data.current(self.ASSET1, 'close')))
        self.assertEqual(0, bar_data.current(self.ASSET1, 'volume'))
        self.assertEqual(390, bar_data.current(self.ASSET1, 'price'))

    def test_can_trade_equity_same_cal_outside_lifetime(self):
        if False:
            for i in range(10):
                print('nop')
        session_before_asset1_start = self.trading_calendar.previous_session_label(self.ASSET1.start_date)
        minutes_for_session = self.trading_calendar.minutes_for_session(session_before_asset1_start)
        minutes_to_check = chain([minutes_for_session[0] - pd.Timedelta(minutes=1)], minutes_for_session)
        for minute in minutes_to_check:
            bar_data = self.create_bardata(simulation_dt_func=lambda : minute)
            self.assertFalse(bar_data.can_trade(self.ASSET1))
        session_after_asset1_end = self.trading_calendar.next_session_label(self.ASSET1.end_date)
        bts_after_asset1_end = session_after_asset1_end.replace(hour=8, minute=45).tz_convert(None).tz_localize('US/Eastern')
        minutes_to_check = chain(self.trading_calendar.minutes_for_session(session_after_asset1_end), [bts_after_asset1_end])
        for minute in minutes_to_check:
            bar_data = self.create_bardata(simulation_dt_func=lambda : minute)
            self.assertFalse(bar_data.can_trade(self.ASSET1))

    def test_can_trade_equity_same_cal_exchange_closed(self):
        if False:
            while True:
                i = 10
        minutes = self.trading_calendar.minutes_for_sessions_in_range(self.ASSET1.start_date, self.ASSET1.end_date)
        for minute in minutes:
            bar_data = self.create_bardata(simulation_dt_func=lambda : minute)
            self.assertTrue(bar_data.can_trade(self.ASSET1))

    def test_can_trade_equity_same_cal_no_last_price(self):
        if False:
            for i in range(10):
                print('nop')
        minutes_in_session = self.trading_calendar.minutes_for_session(self.ASSET1.start_date)
        for minute in minutes_in_session[0:49]:
            bar_data = self.create_bardata(simulation_dt_func=lambda : minute)
            self.assertFalse(bar_data.can_trade(self.HILARIOUSLY_ILLIQUID_ASSET))
        for minute in minutes_in_session[50:]:
            bar_data = self.create_bardata(simulation_dt_func=lambda : minute)
            self.assertTrue(bar_data.can_trade(self.HILARIOUSLY_ILLIQUID_ASSET))

    def test_is_stale_during_non_market_hours(self):
        if False:
            while True:
                i = 10
        bar_data = self.create_bardata(lambda : self.equity_minute_bar_days[1])
        with handle_non_market_minutes(bar_data):
            self.assertTrue(bar_data.is_stale(self.HILARIOUSLY_ILLIQUID_ASSET))

    def test_overnight_adjustments(self):
        if False:
            for i in range(10):
                print('nop')
        splits = self.adjustment_reader.get_adjustments_for_sid('splits', self.SPLIT_ASSET.sid)
        self.assertEqual(1, len(splits))
        split = splits[0]
        self.assertEqual(split[0], pd.Timestamp('2016-01-06', tz='UTC'))
        day = self.equity_daily_bar_days[1]
        eight_fortyfive_am_eastern = pd.Timestamp('{0}-{1}-{2} 8:45'.format(day.year, day.month, day.day), tz='US/Eastern')
        bar_data = self.create_bardata(lambda : eight_fortyfive_am_eastern)
        expected = {'open': 391 / 2.0, 'high': 392 / 2.0, 'low': 389 / 2.0, 'close': 390 / 2.0, 'volume': 39000 * 2.0, 'price': 390 / 2.0}
        with handle_non_market_minutes(bar_data):
            for field in OHLCP + ['volume']:
                value = bar_data.current(self.SPLIT_ASSET, field)
                self.assertEqual(value, expected[field])

    def test_can_trade_restricted(self):
        if False:
            while True:
                i = 10
        '\n        Test that can_trade will return False for a sid if it is restricted\n        on that dt\n        '
        minutes_to_check = [(str_to_ts('2016-01-05 14:31'), False), (str_to_ts('2016-01-06 14:31'), False), (str_to_ts('2016-01-07 14:31'), True), (str_to_ts('2016-01-07 15:00'), False), (str_to_ts('2016-01-07 15:30'), True)]
        rlm = HistoricalRestrictions([Restriction(1, str_to_ts('2016-01-05'), RESTRICTION_STATES.FROZEN), Restriction(1, str_to_ts('2016-01-07'), RESTRICTION_STATES.ALLOWED), Restriction(1, str_to_ts('2016-01-07 15:00'), RESTRICTION_STATES.FROZEN), Restriction(1, str_to_ts('2016-01-07 15:30'), RESTRICTION_STATES.ALLOWED)])
        for info in minutes_to_check:
            bar_data = self.create_bardata(simulation_dt_func=lambda : info[0], restrictions=rlm)
            self.assertEqual(bar_data.can_trade(self.ASSET1), info[1])

class TestMinuteBarDataFuturesCalendar(WithCreateBarData, WithBarDataChecks, ZiplineTestCase):
    START_DATE = pd.Timestamp('2016-01-05', tz='UTC')
    END_DATE = ASSET_FINDER_EQUITY_END_DATE = pd.Timestamp('2016-01-07', tz='UTC')
    ASSET_FINDER_EQUITY_SIDS = [1]

    @classmethod
    def make_equity_minute_bar_data(cls):
        if False:
            while True:
                i = 10
        yield (1, create_minute_df_for_asset(cls.trading_calendar, cls.equity_minute_bar_days[0], cls.equity_minute_bar_days[-1]))

    @classmethod
    def make_futures_info(cls):
        if False:
            while True:
                i = 10
        return pd.DataFrame.from_dict({6: {'symbol': 'CLH16', 'root_symbol': 'CL', 'start_date': pd.Timestamp('2016-01-04', tz='UTC'), 'notice_date': pd.Timestamp('2016-01-19', tz='UTC'), 'expiration_date': pd.Timestamp('2016-02-19', tz='UTC'), 'exchange': 'ICEUS'}, 7: {'symbol': 'FVH16', 'root_symbol': 'FV', 'start_date': pd.Timestamp('2016-01-04', tz='UTC'), 'notice_date': pd.Timestamp('2016-01-22', tz='UTC'), 'expiration_date': pd.Timestamp('2016-02-22', tz='UTC'), 'auto_close_date': pd.Timestamp('2016-01-20', tz='UTC'), 'exchange': 'CMES'}}, orient='index')

    @classmethod
    def init_class_fixtures(cls):
        if False:
            print('Hello World!')
        super(TestMinuteBarDataFuturesCalendar, cls).init_class_fixtures()
        cls.trading_calendar = get_calendar('CMES')

    def test_can_trade_multiple_exchange_closed(self):
        if False:
            i = 10
            return i + 15
        nyse_asset = self.asset_finder.retrieve_asset(1)
        ice_asset = self.asset_finder.retrieve_asset(6)
        minutes_to_check = [(pd.Timestamp('2016-01-05 20:00', tz='US/Eastern'), False, False), (pd.Timestamp('2016-01-05 20:01', tz='US/Eastern'), False, True), (pd.Timestamp('2016-01-05 20:02', tz='US/Eastern'), False, True), (pd.Timestamp('2016-01-06 00:00', tz='US/Eastern'), False, True), (pd.Timestamp('2016-01-06 9:30', tz='US/Eastern'), False, True), (pd.Timestamp('2016-01-06 9:31', tz='US/Eastern'), True, True), (pd.Timestamp('2016-01-06 9:32', tz='US/Eastern'), True, True), (pd.Timestamp('2016-01-06 15:59', tz='US/Eastern'), True, True), (pd.Timestamp('2016-01-06 16:00', tz='US/Eastern'), True, True), (pd.Timestamp('2016-01-06 16:01', tz='US/Eastern'), False, True), (pd.Timestamp('2016-01-06 17:59', tz='US/Eastern'), False, True), (pd.Timestamp('2016-01-06 18:00', tz='US/Eastern'), False, True), (pd.Timestamp('2016-01-06 18:01', tz='US/Eastern'), False, False)]
        for info in minutes_to_check:
            bar_data = self.create_bardata(simulation_dt_func=lambda : info[0])
            series = bar_data.can_trade([nyse_asset, ice_asset])
            self.assertEqual(info[1], series.loc[nyse_asset])
            self.assertEqual(info[2], series.loc[ice_asset])

    def test_can_trade_delisted(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that can_trade returns False for an asset after its auto close\n        date.\n        '
        auto_closing_asset = self.asset_finder.retrieve_asset(7)
        minutes_to_check = [(pd.Timestamp('2016-01-20 00:00:00', tz='UTC'), True), (pd.Timestamp('2016-01-20 23:00:00', tz='UTC'), True), (pd.Timestamp('2016-01-20 23:01:00', tz='UTC'), False), (pd.Timestamp('2016-01-20 23:59:00', tz='UTC'), False), (pd.Timestamp('2016-01-21 00:00:00', tz='UTC'), False), (pd.Timestamp('2016-01-21 00:01:00', tz='UTC'), False), (pd.Timestamp('2016-01-22 00:00:00', tz='UTC'), False)]
        for info in minutes_to_check:
            bar_data = self.create_bardata(simulation_dt_func=lambda : info[0])
            self.assertEqual(bar_data.can_trade(auto_closing_asset), info[1])

class TestDailyBarData(WithCreateBarData, WithBarDataChecks, WithDataPortal, ZiplineTestCase):
    START_DATE = pd.Timestamp('2016-01-05', tz='UTC')
    END_DATE = ASSET_FINDER_EQUITY_END_DATE = pd.Timestamp('2016-01-11', tz='UTC')
    CREATE_BARDATA_DATA_FREQUENCY = 'daily'
    ASSET_FINDER_EQUITY_SIDS = set(range(1, 9))
    SPLIT_ASSET_SID = 3
    ILLIQUID_SPLIT_ASSET_SID = 4
    MERGER_ASSET_SID = 5
    ILLIQUID_MERGER_ASSET_SID = 6
    DIVIDEND_ASSET_SID = 7
    ILLIQUID_DIVIDEND_ASSET_SID = 8

    @classmethod
    def make_equity_info(cls):
        if False:
            print('Hello World!')
        frame = super(TestDailyBarData, cls).make_equity_info()
        frame.loc[[1, 2], 'end_date'] = pd.Timestamp('2016-01-08', tz='UTC')
        return frame

    @classmethod
    def make_splits_data(cls):
        if False:
            while True:
                i = 10
        return pd.DataFrame.from_records([{'effective_date': str_to_seconds('2016-01-06'), 'ratio': 0.5, 'sid': cls.SPLIT_ASSET_SID}, {'effective_date': str_to_seconds('2016-01-07'), 'ratio': 0.5, 'sid': cls.ILLIQUID_SPLIT_ASSET_SID}])

    @classmethod
    def make_mergers_data(cls):
        if False:
            print('Hello World!')
        return pd.DataFrame.from_records([{'effective_date': str_to_seconds('2016-01-06'), 'ratio': 0.5, 'sid': cls.MERGER_ASSET_SID}, {'effective_date': str_to_seconds('2016-01-07'), 'ratio': 0.6, 'sid': cls.ILLIQUID_MERGER_ASSET_SID}])

    @classmethod
    def make_dividends_data(cls):
        if False:
            while True:
                i = 10
        return pd.DataFrame.from_records([{'ex_date': pd.Timestamp('2016-01-06', tz='UTC').to_datetime64(), 'record_date': pd.Timestamp('2016-01-06', tz='UTC').to_datetime64(), 'declared_date': pd.Timestamp('2016-01-06', tz='UTC').to_datetime64(), 'pay_date': pd.Timestamp('2016-01-06', tz='UTC').to_datetime64(), 'amount': 2.0, 'sid': cls.DIVIDEND_ASSET_SID}, {'ex_date': pd.Timestamp('2016-01-07', tz='UTC').to_datetime64(), 'record_date': pd.Timestamp('2016-01-07', tz='UTC').to_datetime64(), 'declared_date': pd.Timestamp('2016-01-07', tz='UTC').to_datetime64(), 'pay_date': pd.Timestamp('2016-01-07', tz='UTC').to_datetime64(), 'amount': 4.0, 'sid': cls.ILLIQUID_DIVIDEND_ASSET_SID}], columns=['ex_date', 'record_date', 'declared_date', 'pay_date', 'amount', 'sid'])

    @classmethod
    def make_adjustment_writer_equity_daily_bar_reader(cls):
        if False:
            i = 10
            return i + 15
        return MockDailyBarReader(dates=cls.trading_calendar.sessions_in_range(cls.START_DATE, cls.END_DATE))

    @classmethod
    def make_equity_daily_bar_data(cls, country_code, sids):
        if False:
            while True:
                i = 10
        for sid in sids:
            asset = cls.asset_finder.retrieve_asset(sid)
            yield (sid, create_daily_df_for_asset(cls.trading_calendar, asset.start_date, asset.end_date, interval=2 - sid % 2))

    @classmethod
    def init_class_fixtures(cls):
        if False:
            print('Hello World!')
        super(TestDailyBarData, cls).init_class_fixtures()
        cls.ASSET1 = cls.asset_finder.retrieve_asset(1)
        cls.ASSET2 = cls.asset_finder.retrieve_asset(2)
        cls.SPLIT_ASSET = cls.asset_finder.retrieve_asset(cls.SPLIT_ASSET_SID)
        cls.ILLIQUID_SPLIT_ASSET = cls.asset_finder.retrieve_asset(cls.ILLIQUID_SPLIT_ASSET_SID)
        cls.MERGER_ASSET = cls.asset_finder.retrieve_asset(cls.MERGER_ASSET_SID)
        cls.ILLIQUID_MERGER_ASSET = cls.asset_finder.retrieve_asset(cls.ILLIQUID_MERGER_ASSET_SID)
        cls.DIVIDEND_ASSET = cls.asset_finder.retrieve_asset(cls.DIVIDEND_ASSET_SID)
        cls.ILLIQUID_DIVIDEND_ASSET = cls.asset_finder.retrieve_asset(cls.ILLIQUID_DIVIDEND_ASSET_SID)
        cls.ASSETS = [cls.ASSET1, cls.ASSET2]

    def get_last_minute_of_session(self, session_label):
        if False:
            return 10
        return self.trading_calendar.open_and_close_for_session(session_label)[1]

    def test_current_session(self):
        if False:
            print('Hello World!')
        for session in self.trading_calendar.sessions_in_range(self.equity_daily_bar_days[0], self.equity_daily_bar_days[-1]):
            bar_data = self.create_bardata(simulation_dt_func=lambda : self.get_last_minute_of_session(session))
            self.assertEqual(session, bar_data.current_session)

    def test_day_before_assets_trading(self):
        if False:
            i = 10
            return i + 15
        minute = self.get_last_minute_of_session(self.trading_calendar.previous_session_label(self.equity_daily_bar_days[0]))
        bar_data = self.create_bardata(simulation_dt_func=lambda : minute)
        self.check_internal_consistency(bar_data)
        self.assertFalse(bar_data.can_trade(self.ASSET1))
        self.assertFalse(bar_data.can_trade(self.ASSET2))
        self.assertFalse(bar_data.is_stale(self.ASSET1))
        self.assertFalse(bar_data.is_stale(self.ASSET2))
        for field in ALL_FIELDS:
            for asset in self.ASSETS:
                asset_value = bar_data.current(asset, field)
                if field in OHLCP:
                    self.assertTrue(np.isnan(asset_value))
                elif field == 'volume':
                    self.assertEqual(0, asset_value)
                elif field == 'last_traded':
                    self.assertTrue(asset_value is pd.NaT)

    def test_semi_active_day(self):
        if False:
            for i in range(10):
                print('nop')
        bar_data = self.create_bardata(simulation_dt_func=lambda : self.get_last_minute_of_session(self.equity_daily_bar_days[0]))
        self.check_internal_consistency(bar_data)
        self.assertTrue(bar_data.can_trade(self.ASSET1))
        self.assertFalse(bar_data.can_trade(self.ASSET2))
        self.assertFalse(bar_data.is_stale(self.ASSET1))
        self.assertFalse(bar_data.is_stale(self.ASSET2))
        self.assertEqual(3, bar_data.current(self.ASSET1, 'open'))
        self.assertEqual(4, bar_data.current(self.ASSET1, 'high'))
        self.assertEqual(1, bar_data.current(self.ASSET1, 'low'))
        self.assertEqual(2, bar_data.current(self.ASSET1, 'close'))
        self.assertEqual(200, bar_data.current(self.ASSET1, 'volume'))
        self.assertEqual(2, bar_data.current(self.ASSET1, 'price'))
        self.assertEqual(self.equity_daily_bar_days[0], bar_data.current(self.ASSET1, 'last_traded'))
        for field in OHLCP:
            self.assertTrue(np.isnan(bar_data.current(self.ASSET2, field)), field)
        self.assertEqual(0, bar_data.current(self.ASSET2, 'volume'))
        self.assertTrue(bar_data.current(self.ASSET2, 'last_traded') is pd.NaT)

    def test_fully_active_day(self):
        if False:
            print('Hello World!')
        bar_data = self.create_bardata(simulation_dt_func=lambda : self.get_last_minute_of_session(self.equity_daily_bar_days[1]))
        self.check_internal_consistency(bar_data)
        for asset in self.ASSETS:
            self.assertTrue(bar_data.can_trade(asset))
            self.assertFalse(bar_data.is_stale(asset))
            self.assertEqual(4, bar_data.current(asset, 'open'))
            self.assertEqual(5, bar_data.current(asset, 'high'))
            self.assertEqual(2, bar_data.current(asset, 'low'))
            self.assertEqual(3, bar_data.current(asset, 'close'))
            self.assertEqual(300, bar_data.current(asset, 'volume'))
            self.assertEqual(3, bar_data.current(asset, 'price'))
            self.assertEqual(self.equity_daily_bar_days[1], bar_data.current(asset, 'last_traded'))

    def test_last_active_day(self):
        if False:
            for i in range(10):
                print('nop')
        bar_data = self.create_bardata(simulation_dt_func=lambda : self.get_last_minute_of_session(self.equity_daily_bar_days[-1]))
        self.check_internal_consistency(bar_data)
        for asset in self.ASSETS:
            if asset in (1, 2):
                self.assertFalse(bar_data.can_trade(asset))
            else:
                self.assertTrue(bar_data.can_trade(asset))
            self.assertFalse(bar_data.is_stale(asset))
            if asset in (1, 2):
                assert_almost_equal(nan, bar_data.current(asset, 'open'))
                assert_almost_equal(nan, bar_data.current(asset, 'high'))
                assert_almost_equal(nan, bar_data.current(asset, 'low'))
                assert_almost_equal(nan, bar_data.current(asset, 'close'))
                assert_almost_equal(0, bar_data.current(asset, 'volume'))
                assert_almost_equal(nan, bar_data.current(asset, 'price'))
            else:
                self.assertEqual(6, bar_data.current(asset, 'open'))
                self.assertEqual(7, bar_data.current(asset, 'high'))
                self.assertEqual(4, bar_data.current(asset, 'low'))
                self.assertEqual(5, bar_data.current(asset, 'close'))
                self.assertEqual(500, bar_data.current(asset, 'volume'))
                self.assertEqual(5, bar_data.current(asset, 'price'))

    def test_after_assets_dead(self):
        if False:
            print('Hello World!')
        session = self.END_DATE
        bar_data = self.create_bardata(simulation_dt_func=lambda : session)
        self.check_internal_consistency(bar_data)
        for asset in self.ASSETS:
            self.assertFalse(bar_data.can_trade(asset))
            self.assertFalse(bar_data.is_stale(asset))
            for field in OHLCP:
                self.assertTrue(np.isnan(bar_data.current(asset, field)))
            self.assertEqual(0, bar_data.current(asset, 'volume'))
            last_traded_dt = bar_data.current(asset, 'last_traded')
            if asset in (self.ASSET1, self.ASSET2):
                self.assertEqual(self.equity_daily_bar_days[3], last_traded_dt)

    @parameterized.expand([('split', 2, 3, 3, 1.5), ('merger', 2, 3, 3, 1.8), ('dividend', 2, 3, 3, 2.88)])
    def test_get_value_adjustments(self, adjustment_type, liquid_day_0_price, liquid_day_1_price, illiquid_day_0_price, illiquid_day_1_price_adjusted):
        if False:
            i = 10
            return i + 15
        'Test the behaviour of spot prices during adjustments.'
        table_name = adjustment_type + 's'
        liquid_asset = getattr(self, adjustment_type.upper() + '_ASSET')
        illiquid_asset = getattr(self, 'ILLIQUID_' + adjustment_type.upper() + '_ASSET')
        adjustments = self.adjustment_reader.get_adjustments_for_sid(table_name, liquid_asset.sid)
        self.assertEqual(1, len(adjustments))
        adjustment = adjustments[0]
        self.assertEqual(adjustment[0], pd.Timestamp('2016-01-06', tz='UTC'))
        bar_data = self.create_bardata(simulation_dt_func=lambda : self.equity_daily_bar_days[0])
        self.assertEqual(liquid_day_0_price, bar_data.current(liquid_asset, 'price'))
        bar_data = self.create_bardata(simulation_dt_func=lambda : self.equity_daily_bar_days[1])
        self.assertEqual(liquid_day_1_price, bar_data.current(liquid_asset, 'price'))
        bar_data = self.create_bardata(simulation_dt_func=lambda : self.equity_daily_bar_days[1])
        self.assertEqual(illiquid_day_0_price, bar_data.current(illiquid_asset, 'price'))
        bar_data = self.create_bardata(simulation_dt_func=lambda : self.equity_daily_bar_days[2])
        self.assertAlmostEqual(illiquid_day_1_price_adjusted, bar_data.current(illiquid_asset, 'price'))

    def test_can_trade_restricted(self):
        if False:
            print('Hello World!')
        '\n        Test that can_trade will return False for a sid if it is restricted\n        on that dt\n        '
        minutes_to_check = [(pd.Timestamp('2016-01-05', tz='UTC'), False), (pd.Timestamp('2016-01-06', tz='UTC'), False), (pd.Timestamp('2016-01-07', tz='UTC'), True)]
        rlm = HistoricalRestrictions([Restriction(1, str_to_ts('2016-01-05'), RESTRICTION_STATES.FROZEN), Restriction(1, str_to_ts('2016-01-07'), RESTRICTION_STATES.ALLOWED)])
        for info in minutes_to_check:
            bar_data = self.create_bardata(simulation_dt_func=lambda : info[0], restrictions=rlm)
            self.assertEqual(bar_data.can_trade(self.ASSET1), info[1])