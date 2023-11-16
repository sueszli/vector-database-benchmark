from collections import OrderedDict
from textwrap import dedent
from nose_parameterized import parameterized
import numpy as np
from numpy import nan
import pandas as pd
from six import iteritems
from zipline._protocol import handle_non_market_minutes, BarData
from zipline.assets import Asset, Equity
from zipline.errors import HistoryInInitialize, HistoryWindowStartsBeforeData
from zipline.finance.asset_restrictions import NoRestrictions
from zipline.testing import create_minute_df_for_asset, str_to_seconds, MockDailyBarReader
import zipline.testing.fixtures as zf
OHLC = ['open', 'high', 'low', 'close']
OHLCP = OHLC + ['price']
ALL_FIELDS = OHLCP + ['volume']

class WithHistory(zf.WithCreateBarData, zf.WithDataPortal):
    TRADING_START_DT = TRADING_ENV_MIN_DATE = START_DATE = pd.Timestamp('2014-01-03', tz='UTC')
    TRADING_END_DT = END_DATE = pd.Timestamp('2016-01-29', tz='UTC')
    SPLIT_ASSET_SID = 4
    DIVIDEND_ASSET_SID = 5
    MERGER_ASSET_SID = 6
    HALF_DAY_TEST_ASSET_SID = 7
    SHORT_ASSET_SID = 8

    @classmethod
    def init_class_fixtures(cls):
        if False:
            print('Hello World!')
        super(WithHistory, cls).init_class_fixtures()
        cls.trading_days = cls.trading_calendar.sessions_in_range(cls.TRADING_START_DT, cls.TRADING_END_DT)
        cls.ASSET1 = cls.asset_finder.retrieve_asset(1)
        cls.ASSET2 = cls.asset_finder.retrieve_asset(2)
        cls.ASSET3 = cls.asset_finder.retrieve_asset(3)
        cls.SPLIT_ASSET = cls.asset_finder.retrieve_asset(cls.SPLIT_ASSET_SID)
        cls.DIVIDEND_ASSET = cls.asset_finder.retrieve_asset(cls.DIVIDEND_ASSET_SID)
        cls.MERGER_ASSET = cls.asset_finder.retrieve_asset(cls.MERGER_ASSET_SID)
        cls.HALF_DAY_TEST_ASSET = cls.asset_finder.retrieve_asset(cls.HALF_DAY_TEST_ASSET_SID)
        cls.SHORT_ASSET = cls.asset_finder.retrieve_asset(cls.SHORT_ASSET_SID)

    @classmethod
    def make_equity_info(cls):
        if False:
            print('Hello World!')
        jan_5_2015 = pd.Timestamp('2015-01-05', tz='UTC')
        day_after_12312015 = pd.Timestamp('2016-01-04', tz='UTC')
        return pd.DataFrame.from_dict({1: {'start_date': pd.Timestamp('2014-01-03', tz='UTC'), 'end_date': cls.TRADING_END_DT, 'symbol': 'ASSET1', 'exchange': 'TEST'}, 2: {'start_date': jan_5_2015, 'end_date': day_after_12312015, 'symbol': 'ASSET2', 'exchange': 'TEST'}, 3: {'start_date': jan_5_2015, 'end_date': day_after_12312015, 'symbol': 'ASSET3', 'exchange': 'TEST'}, cls.SPLIT_ASSET_SID: {'start_date': jan_5_2015, 'end_date': day_after_12312015, 'symbol': 'SPLIT_ASSET', 'exchange': 'TEST'}, cls.DIVIDEND_ASSET_SID: {'start_date': jan_5_2015, 'end_date': day_after_12312015, 'symbol': 'DIVIDEND_ASSET', 'exchange': 'TEST'}, cls.MERGER_ASSET_SID: {'start_date': jan_5_2015, 'end_date': day_after_12312015, 'symbol': 'MERGER_ASSET', 'exchange': 'TEST'}, cls.HALF_DAY_TEST_ASSET_SID: {'start_date': pd.Timestamp('2014-07-02', tz='UTC'), 'end_date': day_after_12312015, 'symbol': 'HALF_DAY_TEST_ASSET', 'exchange': 'TEST'}, cls.SHORT_ASSET_SID: {'start_date': pd.Timestamp('2015-01-05', tz='UTC'), 'end_date': pd.Timestamp('2015-01-06', tz='UTC'), 'symbol': 'SHORT_ASSET', 'exchange': 'TEST'}}, orient='index')

    @classmethod
    def make_splits_data(cls):
        if False:
            while True:
                i = 10
        return pd.DataFrame([{'effective_date': str_to_seconds('2015-01-06'), 'ratio': 0.25, 'sid': cls.SPLIT_ASSET_SID}, {'effective_date': str_to_seconds('2015-01-07'), 'ratio': 0.5, 'sid': cls.SPLIT_ASSET_SID}])

    @classmethod
    def make_mergers_data(cls):
        if False:
            for i in range(10):
                print('nop')
        return pd.DataFrame([{'effective_date': str_to_seconds('2015-01-06'), 'ratio': 0.25, 'sid': cls.MERGER_ASSET_SID}, {'effective_date': str_to_seconds('2015-01-07'), 'ratio': 0.5, 'sid': cls.MERGER_ASSET_SID}])

    @classmethod
    def make_dividends_data(cls):
        if False:
            return 10
        return pd.DataFrame([{'ex_date': pd.Timestamp('2015-01-06', tz='UTC').to_datetime64(), 'record_date': pd.Timestamp('2015-01-06', tz='UTC').to_datetime64(), 'declared_date': pd.Timestamp('2015-01-06', tz='UTC').to_datetime64(), 'pay_date': pd.Timestamp('2015-01-06', tz='UTC').to_datetime64(), 'amount': 2.0, 'sid': cls.DIVIDEND_ASSET_SID}, {'ex_date': pd.Timestamp('2015-01-07', tz='UTC').to_datetime64(), 'record_date': pd.Timestamp('2015-01-07', tz='UTC').to_datetime64(), 'declared_date': pd.Timestamp('2015-01-07', tz='UTC').to_datetime64(), 'pay_date': pd.Timestamp('2015-01-07', tz='UTC').to_datetime64(), 'amount': 4.0, 'sid': cls.DIVIDEND_ASSET_SID}], columns=['ex_date', 'record_date', 'declared_date', 'pay_date', 'amount', 'sid'])

    @classmethod
    def make_adjustment_writer_equity_daily_bar_reader(cls):
        if False:
            for i in range(10):
                print('nop')
        return MockDailyBarReader(dates=cls.trading_calendar.sessions_in_range(cls.TRADING_START_DT, cls.TRADING_END_DT))

    def verify_regular_dt(self, idx, dt, mode, fields=None, assets=None):
        if False:
            i = 10
            return i + 15
        if mode == 'daily':
            freq = '1d'
        else:
            freq = '1m'
        cal = self.trading_calendar
        equity_cal = self.trading_calendars[Equity]

        def reindex_to_primary_calendar(a, field):
            if False:
                while True:
                    i = 10
            '\n            Reindex an array of prices from a window on the NYSE\n            calendar by the window on the primary calendar with the same\n            dt and window size.\n            '
            if mode == 'daily':
                dts = cal.sessions_window(dt, -9)
                equity_sess = equity_cal.minute_to_session_label(dt)
                equity_dts = equity_cal.sessions_window(equity_sess, -9)
            elif mode == 'minute':
                dts = cal.minutes_window(dt, -10)
                equity_dts = equity_cal.minutes_window(dt, -10)
            output = pd.Series(index=equity_dts, data=a).reindex(dts)
            if field == 'volume':
                return output.fillna(0)
            elif field == 'price':
                return output.fillna(method='ffill')
            else:
                return output
        fields = fields if fields is not None else ALL_FIELDS
        assets = assets if assets is not None else [self.ASSET2, self.ASSET3]
        bar_data = self.create_bardata(simulation_dt_func=lambda : dt)
        check_internal_consistency(bar_data, assets, fields, 10, freq)
        for field in fields:
            for asset in assets:
                asset_series = bar_data.history(asset, field, 10, freq)
                base = MINUTE_FIELD_INFO[field] + 2
                if idx < 9:
                    missing_count = 9 - idx
                    present_count = 9 - missing_count
                    if field in OHLCP:
                        if asset == self.ASSET2:
                            np.testing.assert_array_equal(np.full(missing_count, np.nan), asset_series[0:missing_count])
                            np.testing.assert_array_equal(np.array(range(base, base + present_count + 1)), asset_series[9 - present_count:])
                        if asset == self.ASSET3:
                            np.testing.assert_array_equal(np.full(10, np.nan), asset_series)
                    elif field == 'volume':
                        if asset == self.ASSET2:
                            np.testing.assert_array_equal(np.zeros(missing_count), asset_series[0:missing_count])
                            np.testing.assert_array_equal(np.array(range(base, base + present_count + 1)) * 100, asset_series[9 - present_count:])
                        if asset == self.ASSET3:
                            np.testing.assert_array_equal(np.zeros(10), asset_series)
                else:
                    position_from_end = (idx + 1) % 10 + 1
                    value_for_asset3 = (idx + 1) // 10 * 10 + MINUTE_FIELD_INFO[field] + 1
                    if field in OHLC:
                        asset3_answer_key = np.full(10, np.nan)
                        asset3_answer_key[-position_from_end] = value_for_asset3
                        asset3_answer_key = reindex_to_primary_calendar(asset3_answer_key, field)
                        if asset == self.ASSET2:
                            np.testing.assert_array_equal(reindex_to_primary_calendar(np.array(range(base + idx - 9, base + idx + 1)), field), asset_series)
                        if asset == self.ASSET3:
                            np.testing.assert_array_equal(asset3_answer_key, asset_series)
                    elif field == 'volume':
                        asset3_answer_key = np.zeros(10)
                        asset3_answer_key[-position_from_end] = value_for_asset3 * 100
                        asset3_answer_key = reindex_to_primary_calendar(asset3_answer_key, field)
                        if asset == self.ASSET2:
                            np.testing.assert_array_equal(reindex_to_primary_calendar(np.array(range(base + idx - 9, base + idx + 1)) * 100, field), asset_series)
                        if asset == self.ASSET3:
                            np.testing.assert_array_equal(asset3_answer_key, asset_series)
                    elif field == 'price':
                        if asset == self.ASSET2:
                            np.testing.assert_array_equal(reindex_to_primary_calendar(range(idx - 7, idx + 3), field=field), asset_series)
                        if asset == self.ASSET3:
                            second_begin = dt - equity_cal.day * (position_from_end - 1)
                            first_end = second_begin - cal.day
                            first_part = asset_series[:first_end]
                            second_part = asset_series[second_begin:]
                            decile_count = (idx + 1) // 10
                            if len(second_part) >= 10:
                                np.testing.assert_array_equal(np.full(len(first_part), np.nan), first_part)
                            elif decile_count == 1:
                                np.testing.assert_array_equal(np.full(len(first_part), np.nan), first_part)
                                np.testing.assert_array_equal(np.array([11] * len(second_part)), second_part)
                            else:
                                np.testing.assert_array_equal(np.array([decile_count * 10 - 9] * len(first_part)), first_part)
                                np.testing.assert_array_equal(np.array([decile_count * 10 + 1] * len(second_part)), second_part)

def check_internal_consistency(bar_data, assets, fields, bar_count, freq):
    if False:
        return 10
    if isinstance(assets, Asset):
        asset_list = [assets]
    else:
        asset_list = assets
    if isinstance(fields, str):
        field_list = [fields]
    else:
        field_list = fields
    multi_field_dict = {asset: bar_data.history(asset, field_list, bar_count, freq) for asset in asset_list}
    multi_asset_dict = {field: bar_data.history(asset_list, field, bar_count, freq) for field in fields}
    panel = bar_data.history(asset_list, field_list, bar_count, freq)
    for field in field_list:
        for asset in asset_list:
            series = bar_data.history(asset, field, bar_count, freq)
            np.testing.assert_array_equal(series, multi_asset_dict[field][asset])
            np.testing.assert_array_equal(series, multi_field_dict[asset][field])
            np.testing.assert_array_equal(series, panel[field][asset])
MINUTE_FIELD_INFO = {'open': 1, 'high': 2, 'low': -1, 'close': 0, 'price': 0, 'volume': 0}

class MinuteEquityHistoryTestCase(WithHistory, zf.WithMakeAlgo, zf.ZiplineTestCase):
    EQUITY_DAILY_BAR_SOURCE_FROM_MINUTE = True
    DATA_PORTAL_FIRST_TRADING_DAY = zf.alias('TRADING_START_DT')

    @classmethod
    def make_equity_minute_bar_data(cls):
        if False:
            print('Hello World!')
        equities_cal = cls.trading_calendars[Equity]
        data = {}
        sids = {2, 5, cls.SHORT_ASSET_SID, cls.HALF_DAY_TEST_ASSET_SID}
        for sid in sids:
            asset = cls.asset_finder.retrieve_asset(sid)
            data[sid] = create_minute_df_for_asset(equities_cal, asset.start_date, asset.end_date, start_val=2)
        data[1] = create_minute_df_for_asset(equities_cal, pd.Timestamp('2014-01-03', tz='utc'), pd.Timestamp('2016-01-29', tz='utc'), start_val=2)
        asset2 = cls.asset_finder.retrieve_asset(2)
        data[asset2.sid] = create_minute_df_for_asset(equities_cal, asset2.start_date, equities_cal.previous_session_label(asset2.end_date), start_val=2, minute_blacklist=[pd.Timestamp('2015-01-08 14:31', tz='UTC'), pd.Timestamp('2015-01-08 21:00', tz='UTC')])
        data[cls.MERGER_ASSET_SID] = data[cls.SPLIT_ASSET_SID] = pd.concat((create_minute_df_for_asset(equities_cal, pd.Timestamp('2015-01-05', tz='UTC'), pd.Timestamp('2015-01-05', tz='UTC'), start_val=8000), create_minute_df_for_asset(equities_cal, pd.Timestamp('2015-01-06', tz='UTC'), pd.Timestamp('2015-01-06', tz='UTC'), start_val=2000), create_minute_df_for_asset(equities_cal, pd.Timestamp('2015-01-07', tz='UTC'), pd.Timestamp('2015-01-07', tz='UTC'), start_val=1000), create_minute_df_for_asset(equities_cal, pd.Timestamp('2015-01-08', tz='UTC'), pd.Timestamp('2015-01-08', tz='UTC'), start_val=1000)))
        asset3 = cls.asset_finder.retrieve_asset(3)
        data[3] = create_minute_df_for_asset(equities_cal, asset3.start_date, asset3.end_date, start_val=2, interval=10)
        return iteritems(data)

    def test_history_in_initialize(self):
        if False:
            return 10
        algo_text = dedent("            from zipline.api import history\n\n            def initialize(context):\n                history([1], 10, '1d', 'price')\n\n            def handle_data(context, data):\n                pass\n            ")
        algo = self.make_algo(script=algo_text)
        with self.assertRaises(HistoryInInitialize):
            algo.run()

    def test_negative_bar_count(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Negative bar counts leak future information.\n        '
        with self.assertRaisesRegex(ValueError, 'bar_count must be >= 1, but got -1'):
            self.data_portal.get_history_window([self.ASSET1], pd.Timestamp('2015-01-07 14:35', tz='UTC'), -1, '1d', 'close', 'minute')

    def test_daily_splits_and_mergers(self):
        if False:
            print('Hello World!')
        jan5 = pd.Timestamp('2015-01-05', tz='UTC')
        for asset in [self.SPLIT_ASSET, self.MERGER_ASSET]:
            window1 = self.data_portal.get_history_window([asset], self.trading_calendar.open_and_close_for_session(jan5)[1], 2, '1d', 'close', 'minute')[asset]
            np.testing.assert_array_equal(np.array([np.nan, 8389]), window1)
            window2 = self.data_portal.get_history_window([asset], pd.Timestamp('2015-01-06 14:35', tz='UTC'), 2, '1d', 'close', 'minute')[asset]
            np.testing.assert_array_equal([2097.25, 2004], window2)
            window3 = self.data_portal.get_history_window([asset], pd.Timestamp('2015-01-07 14:35', tz='UTC'), 3, '1d', 'close', 'minute')[asset]
            np.testing.assert_array_equal([1048.625, 1194.5, 1004.0], window3)
            window4 = self.data_portal.get_history_window([asset], pd.Timestamp('2015-01-08 14:40', tz='UTC'), 2, '1d', 'close', 'minute')[asset]
            np.testing.assert_array_equal([1389, 1009], window4)

    def test_daily_dividends(self):
        if False:
            return 10
        jan5 = pd.Timestamp('2015-01-05', tz='UTC')
        asset = self.DIVIDEND_ASSET
        window1 = self.data_portal.get_history_window([asset], self.trading_calendar.session_close(jan5), 2, '1d', 'close', 'minute')[asset]
        np.testing.assert_array_equal(np.array([nan, 391]), window1)
        window2 = self.data_portal.get_history_window([asset], pd.Timestamp('2015-01-06 14:35', tz='UTC'), 2, '1d', 'close', 'minute')[asset]
        np.testing.assert_array_equal([383.18, 396], window2)
        window3 = self.data_portal.get_history_window([asset], pd.Timestamp('2015-01-07 14:35', tz='UTC'), 3, '1d', 'close', 'minute')[asset]
        np.testing.assert_array_equal([367.853, 749.76, 786], window3)
        window4 = self.data_portal.get_history_window([asset], pd.Timestamp('2015-01-08 14:40', tz='UTC'), 2, '1d', 'close', 'minute')[asset]
        np.testing.assert_array_equal([1171, 1181], window4)

    def test_minute_before_assets_trading(self):
        if False:
            return 10
        minutes = self.trading_calendar.minutes_for_session(self.trading_calendar.previous_session_label(pd.Timestamp('2015-01-05', tz='UTC')))[0:60]
        for (idx, minute) in enumerate(minutes):
            bar_data = self.create_bardata(lambda : minute)
            check_internal_consistency(bar_data, [self.ASSET2, self.ASSET3], ALL_FIELDS, 10, '1m')
            for field in ALL_FIELDS:
                asset2_series = bar_data.history(self.ASSET2, field, 10, '1m')
                asset3_series = bar_data.history(self.ASSET3, field, 10, '1m')
                if field == 'volume':
                    np.testing.assert_array_equal(np.zeros(10), asset2_series)
                    np.testing.assert_array_equal(np.zeros(10), asset3_series)
                else:
                    np.testing.assert_array_equal(np.full(10, np.nan), asset2_series)
                    np.testing.assert_array_equal(np.full(10, np.nan), asset3_series)

    @parameterized.expand([('open_sid_2', 'open', 2), ('high_sid_2', 'high', 2), ('low_sid_2', 'low', 2), ('close_sid_2', 'close', 2), ('volume_sid_2', 'volume', 2), ('open_sid_3', 'open', 3), ('high_sid_3', 'high', 3), ('low_sid_3', 'low', 3), ('close_sid_3', 'close', 3), ('volume_sid_3', 'volume', 3)])
    def test_minute_regular(self, name, field, sid):
        if False:
            print('Hello World!')
        asset = self.asset_finder.retrieve_asset(sid)
        minutes = self.trading_calendars[Equity].minutes_for_session(pd.Timestamp('2015-01-05', tz='UTC'))[0:60]
        for (idx, minute) in enumerate(minutes):
            self.verify_regular_dt(idx, minute, 'minute', assets=[asset], fields=[field])

    def test_minute_sunday_midnight(self):
        if False:
            for i in range(10):
                print('nop')
        sunday_midnight = pd.Timestamp('2015-01-09', tz='UTC')
        trading_minutes = self.trading_calendar.all_minutes
        last_minute = trading_minutes[trading_minutes <= sunday_midnight][-1]
        sunday_midnight_bar_data = self.create_bardata(lambda : sunday_midnight)
        last_minute_bar_data = self.create_bardata(lambda : last_minute)
        with handle_non_market_minutes(sunday_midnight_bar_data):
            for field in ALL_FIELDS:
                np.testing.assert_array_equal(sunday_midnight_bar_data.history(self.ASSET2, field, 30, '1m'), last_minute_bar_data.history(self.ASSET2, field, 30, '1m'))

    def test_minute_after_asset_stopped(self):
        if False:
            i = 10
            return i + 15
        minutes = self.trading_calendars[Equity].minutes_for_session(pd.Timestamp('2015-01-07', tz='UTC'))[0:60]
        for (idx, minute) in enumerate(minutes):
            bar_data = self.create_bardata(lambda : minute)
            check_internal_consistency(bar_data, self.SHORT_ASSET, ALL_FIELDS, 30, '1m')
        data_portal = self.make_data_portal()
        window_start = pd.Timestamp('2015-01-06 20:47', tz='UTC')
        window_end = pd.Timestamp('2015-01-07 14:46', tz='UTC')
        bar_data = BarData(data_portal=data_portal, simulation_dt_func=lambda : minutes[15], data_frequency='minute', restrictions=NoRestrictions(), trading_calendar=self.trading_calendar)
        bar_count = len(self.trading_calendar.minutes_in_range(window_start, window_end))
        window = bar_data.history(self.SHORT_ASSET, ALL_FIELDS, bar_count, '1m')
        for field in ALL_FIELDS:
            if field == 'volume':
                np.testing.assert_array_equal(range(76800, 78101, 100), window['volume'][0:14])
                np.testing.assert_array_equal(np.zeros(16), window['volume'][-16:])
            else:
                np.testing.assert_array_equal(np.array(range(768, 782)) + MINUTE_FIELD_INFO[field], window[field][0:14])
                np.testing.assert_array_equal(np.full(16, np.nan), window[field][-16:])
        window = bar_data.history(self.SHORT_ASSET, ALL_FIELDS, 5, '1m')
        for field in ALL_FIELDS:
            if field == 'volume':
                np.testing.assert_array_equal(np.zeros(5), window['volume'])
            else:
                np.testing.assert_array_equal(np.full(5, np.nan), window[field])

    def test_minute_splits_and_mergers(self):
        if False:
            while True:
                i = 10
        jan5 = pd.Timestamp('2015-01-05', tz='UTC')
        for asset in [self.SPLIT_ASSET, self.MERGER_ASSET]:
            equity_cal = self.trading_calendars[Equity]
            window1 = self.data_portal.get_history_window([asset], equity_cal.open_and_close_for_session(jan5)[1], 10, '1m', 'close', 'minute')[asset]
            np.testing.assert_array_equal(np.array(range(8380, 8390)), window1)
            window2_start = pd.Timestamp('2015-01-05 20:56', tz='UTC')
            window2_end = pd.Timestamp('2015-01-06 14:35', tz='UTC')
            window2_count = len(self.trading_calendar.minutes_in_range(window2_start, window2_end))
            window2 = self.data_portal.get_history_window([asset], pd.Timestamp('2015-01-06 14:35', tz='UTC'), window2_count, '1m', 'close', 'minute')[asset]
            np.testing.assert_array_equal([2096.25, 2096.5, 2096.75, 2097, 2097.25], window2[:5])
            np.testing.assert_array_equal([2000, 2001, 2002, 2003, 2004], window2[-5:])
            window3_start = pd.Timestamp('2015-01-05 20:56', tz='UTC')
            window3_end = pd.Timestamp('2015-01-07 14:35', tz='UTC')
            window3_minutes = self.trading_calendar.minutes_in_range(window3_start, window3_end)
            window3_count = len(window3_minutes)
            window3 = self.data_portal.get_history_window([asset], pd.Timestamp('2015-01-07 14:35', tz='UTC'), window3_count, '1m', 'close', 'minute')[asset]
            np.testing.assert_array_equal([1048.125, 1048.25, 1048.375, 1048.5, 1048.625], window3[0:5])
            middle_day_open_i = window3_minutes.searchsorted(pd.Timestamp('2015-01-06 14:31', tz='UTC'))
            middle_day_close_i = window3_minutes.searchsorted(pd.Timestamp('2015-01-06 21:00', tz='UTC'))
            np.testing.assert_array_equal(np.array(range(2000, 2390), dtype='float64') / 2, window3[middle_day_open_i:middle_day_close_i + 1])
            np.testing.assert_array_equal(range(1000, 1005), window3[-5:])
            window4 = self.data_portal.get_history_window([asset], pd.Timestamp('2015-01-07 14:40', tz='UTC'), 5, '1m', 'close', 'minute')[asset]
            np.testing.assert_array_equal(range(1005, 1010), window4)

    def test_minute_dividends(self):
        if False:
            print('Hello World!')
        window1 = self.data_portal.get_history_window([self.DIVIDEND_ASSET], pd.Timestamp('2015-01-05 21:00', tz='UTC'), 10, '1m', 'close', 'minute')[self.DIVIDEND_ASSET]
        np.testing.assert_array_equal(np.array(range(382, 392)), window1)
        window2_start = pd.Timestamp('2015-01-05 20:56', tz='UTC')
        window2_end = pd.Timestamp('2015-01-06 14:35', tz='UTC')
        window2_count = len(self.trading_calendar.minutes_in_range(window2_start, window2_end))
        window2 = self.data_portal.get_history_window([self.DIVIDEND_ASSET], window2_end, window2_count, '1m', 'close', 'minute')[self.DIVIDEND_ASSET]
        np.testing.assert_array_almost_equal(np.array(range(387, 392), dtype='float64') * 0.98, window2[0:5])
        np.testing.assert_array_equal(range(392, 397), window2[-5:])
        window3_start = pd.Timestamp('2015-01-05 20:56', tz='UTC')
        window3_end = pd.Timestamp('2015-01-07 14:35', tz='UTC')
        window3_minutes = self.trading_calendar.minutes_in_range(window3_start, window3_end)
        window3_count = len(window3_minutes)
        window3 = self.data_portal.get_history_window([self.DIVIDEND_ASSET], window3_end, window3_count, '1m', 'close', 'minute')[self.DIVIDEND_ASSET]
        np.testing.assert_array_almost_equal(np.around(np.array(range(387, 392), dtype='float64') * 0.9408, 3), window3[0:5])
        middle_day_open_i = window3_minutes.searchsorted(pd.Timestamp('2015-01-06 14:31', tz='UTC'))
        middle_day_close_i = window3_minutes.searchsorted(pd.Timestamp('2015-01-06 21:00', tz='UTC'))
        np.testing.assert_array_almost_equal(np.array(range(392, 782), dtype='float64') * 0.96, window3[middle_day_open_i:middle_day_close_i + 1])
        np.testing.assert_array_equal(np.array(range(782, 787)), window3[-5:])

    def test_passing_iterable_to_history_regular_hours(self):
        if False:
            return 10
        current_dt = pd.Timestamp('2015-01-06 9:45', tz='US/Eastern')
        bar_data = self.create_bardata(lambda : current_dt)
        bar_data.history(pd.Index([self.ASSET1, self.ASSET2]), 'high', 5, '1m')

    def test_passing_iterable_to_history_bts(self):
        if False:
            i = 10
            return i + 15
        current_dt = pd.Timestamp('2015-01-07 8:45', tz='US/Eastern')
        bar_data = self.create_bardata(lambda : current_dt)
        with handle_non_market_minutes(bar_data):
            bar_data.history(pd.Index([self.ASSET1, self.ASSET2]), 'high', 5, '1m')

    def test_overnight_adjustments(self):
        if False:
            return 10
        current_dt = pd.Timestamp('2015-01-06 8:45', tz='US/Eastern')
        bar_data = self.create_bardata(lambda : current_dt)
        adj_expected = {'open': np.arange(8381, 8391) / 4.0, 'high': np.arange(8382, 8392) / 4.0, 'low': np.arange(8379, 8389) / 4.0, 'close': np.arange(8380, 8390) / 4.0, 'volume': np.arange(8380, 8390) * 100 * 4.0, 'price': np.arange(8380, 8390) / 4.0}
        expected = {'open': np.arange(383, 393) / 2.0, 'high': np.arange(384, 394) / 2.0, 'low': np.arange(381, 391) / 2.0, 'close': np.arange(382, 392) / 2.0, 'volume': np.arange(382, 392) * 100 * 2.0, 'price': np.arange(382, 392) / 2.0}
        window_start = pd.Timestamp('2015-01-05 20:51', tz='UTC')
        window_end = pd.Timestamp('2015-01-06 13:44', tz='UTC')
        window_length = len(self.trading_calendar.minutes_in_range(window_start, window_end))
        with handle_non_market_minutes(bar_data):
            for field in ALL_FIELDS:
                values = bar_data.history(self.SPLIT_ASSET, field, window_length, '1m')
                np.testing.assert_array_equal(values.values[:10], adj_expected[field], err_msg=field)
            values = bar_data.history(self.SPLIT_ASSET, ['open', 'volume'], window_length, '1m')
            np.testing.assert_array_equal(values.open.values[:10], adj_expected['open'])
            np.testing.assert_array_equal(values.volume.values[:10], adj_expected['volume'])
            values = bar_data.history([self.SPLIT_ASSET, self.ASSET2], 'open', window_length, '1m')
            np.testing.assert_array_equal(values[self.SPLIT_ASSET].values[:10], adj_expected['open'])
            np.testing.assert_array_equal(values[self.ASSET2].values[:10], expected['open'] * 2)
            values = bar_data.history([self.SPLIT_ASSET, self.ASSET2], ['open', 'volume'], window_length, '1m')
            np.testing.assert_array_equal(values.open[self.SPLIT_ASSET].values[:10], adj_expected['open'])
            np.testing.assert_array_equal(values.volume[self.SPLIT_ASSET].values[:10], adj_expected['volume'])
            np.testing.assert_array_equal(values.open[self.ASSET2].values[:10], expected['open'] * 2)
            np.testing.assert_array_equal(values.volume[self.ASSET2].values[:10], expected['volume'] / 2)

    def test_minute_early_close(self):
        if False:
            return 10
        cal = self.trading_calendar
        window_start = pd.Timestamp('2014-07-03 16:46:00', tz='UTC')
        window_end = pd.Timestamp('2014-07-07 13:35:00', tz='UTC')
        bar_count = len(cal.minutes_in_range(window_start, window_end))
        window = self.data_portal.get_history_window([self.HALF_DAY_TEST_ASSET], window_end, bar_count, '1m', 'close', 'minute')[self.HALF_DAY_TEST_ASSET]
        expected = range(587, 607)
        np.testing.assert_array_equal(window[:15], expected[:15])
        np.testing.assert_array_equal(window[15:-5], np.full(len(window) - 20, np.nan))
        np.testing.assert_array_equal(window[-5:], expected[-5:])
        self.assertEqual(window.index[14], pd.Timestamp('2014-07-03 17:00', tz='UTC'))
        self.assertEqual(window.index[-5], pd.Timestamp('2014-07-07 13:31', tz='UTC'))

    def test_minute_different_lifetimes(self):
        if False:
            i = 10
            return i + 15
        cal = self.trading_calendar
        equity_cal = self.trading_calendars[Equity]
        day = self.trading_calendar.next_session_label(self.TRADING_START_DT)
        window_start = pd.Timestamp('2014-01-03 19:22', tz='UTC')
        window_end = pd.Timestamp('2014-01-06 14:31', tz='UTC')
        bar_count = len(cal.minutes_in_range(window_start, window_end))
        equity_cal = self.trading_calendars[Equity]
        (first_equity_open, _) = equity_cal.open_and_close_for_session(day)
        asset1_minutes = equity_cal.minutes_for_sessions_in_range(self.ASSET1.start_date, self.ASSET1.end_date)
        asset1_idx = asset1_minutes.searchsorted(first_equity_open)
        window = self.data_portal.get_history_window([self.ASSET1, self.ASSET2], first_equity_open, bar_count, '1m', 'close', 'minute')
        expected = range(asset1_idx - 97, asset1_idx + 3)
        np.testing.assert_array_equal(window[self.ASSET1][:99], expected[:99])
        np.testing.assert_array_equal(window[self.ASSET1][99:-1], np.full(len(window) - 100, np.nan))
        np.testing.assert_array_equal(window[self.ASSET1][-1:], expected[-1:])
        np.testing.assert_array_equal(window[self.ASSET2], np.full(len(window), np.nan))

    def test_history_window_before_first_trading_day(self):
        if False:
            print('Hello World!')
        first_day_minutes = self.trading_calendar.minutes_for_session(self.TRADING_START_DT)
        exp_msg = 'History window extends before 2014-01-03. To use this history window, start the backtest on or after 2014-01-06.'
        for field in OHLCP:
            with self.assertRaisesRegex(HistoryWindowStartsBeforeData, exp_msg):
                self.data_portal.get_history_window([self.ASSET1], first_day_minutes[5], 15, '1m', field, 'minute')[self.ASSET1]

    def test_daily_history_blended(self):
        if False:
            i = 10
            return i + 15
        day = pd.Timestamp('2015-01-07', tz='UTC')
        minutes = self.trading_calendar.minutes_for_session(day)
        equity_cal = self.trading_calendars[Equity]
        equity_minutes = equity_cal.minutes_for_session(day)
        (equity_open, equity_close) = (equity_minutes[0], equity_minutes[-1])
        for minute in minutes:
            idx = equity_minutes.searchsorted(min(minute, equity_close))
            for field in ALL_FIELDS:
                window = self.data_portal.get_history_window([self.ASSET2], minute, 3, '1d', field, 'minute')[self.ASSET2]
                self.assertEqual(len(window), 3)
                if field == 'open':
                    self.assertEqual(window[0], 3)
                    self.assertEqual(window[1], 393)
                elif field == 'high':
                    self.assertEqual(window[0], 393)
                    self.assertEqual(window[1], 783)
                elif field == 'low':
                    self.assertEqual(window[0], 1)
                    self.assertEqual(window[1], 391)
                elif field == 'close':
                    self.assertEqual(window[0], 391)
                    self.assertEqual(window[1], 781)
                elif field == 'volume':
                    self.assertEqual(window[0], 7663500)
                    self.assertEqual(window[1], 22873500)
                last_val = -1
                if minute < equity_open:
                    if field == 'volume':
                        last_val = 0
                    elif field == 'price':
                        last_val = window[1]
                    else:
                        last_val = nan
                elif field == 'open':
                    last_val = 783
                elif field == 'high':
                    last_val = 784 + idx
                elif field == 'low':
                    last_val = 781
                elif field == 'close' or field == 'price':
                    last_val = 782 + idx
                elif field == 'volume':
                    last_val = sum(np.array(range(782, 782 + idx + 1)) * 100)
                np.testing.assert_equal(window[-1], last_val)

    @parameterized.expand(ALL_FIELDS)
    def test_daily_history_blended_gaps(self, field):
        if False:
            for i in range(10):
                print('nop')
        day = pd.Timestamp('2015-01-08', tz='UTC')
        minutes = self.trading_calendar.minutes_for_session(day)
        equity_cal = self.trading_calendars[Equity]
        equity_minutes = equity_cal.minutes_for_session(day)
        (equity_open, equity_close) = (equity_minutes[0], equity_minutes[-1])
        for minute in minutes:
            idx = equity_minutes.searchsorted(min(minute, equity_close))
            window = self.data_portal.get_history_window([self.ASSET2], minute, 3, '1d', field, 'minute')[self.ASSET2]
            self.assertEqual(len(window), 3)
            if field == 'open':
                self.assertEqual(window[0], 393)
                self.assertEqual(window[1], 783)
            elif field == 'high':
                self.assertEqual(window[0], 783)
                self.assertEqual(window[1], 1173)
            elif field == 'low':
                self.assertEqual(window[0], 391)
                self.assertEqual(window[1], 781)
            elif field == 'close':
                self.assertEqual(window[0], 781)
                self.assertEqual(window[1], 1171)
            elif field == 'price':
                self.assertEqual(window[0], 781)
                self.assertEqual(window[1], 1171)
            elif field == 'volume':
                self.assertEqual(window[0], 22873500)
                self.assertEqual(window[1], 38083500)
            last_val = -1
            if minute < equity_open:
                if field == 'volume':
                    last_val = 0
                elif field == 'price':
                    last_val = window[1]
                else:
                    last_val = nan
            elif field == 'open':
                if idx == 0:
                    last_val = np.nan
                else:
                    last_val = 1174.0
            elif field == 'high':
                if idx == 0:
                    last_val = np.nan
                elif idx == 389:
                    last_val = 1562.0
                else:
                    last_val = 1174.0 + idx
            elif field == 'low':
                if idx == 0:
                    last_val = np.nan
                else:
                    last_val = 1172.0
            elif field == 'close':
                if idx == 0:
                    last_val = np.nan
                elif idx == 389:
                    last_val = 1172.0 + 388
                else:
                    last_val = 1172.0 + idx
            elif field == 'price':
                if idx == 0:
                    last_val = 1171.0
                elif idx == 389:
                    last_val = 1172.0 + 388
                else:
                    last_val = 1172.0 + idx
            elif field == 'volume':
                if idx == 0:
                    last_val = 0
                elif idx == 389:
                    last_val = sum(np.array(range(1173, 1172 + 388 + 1)) * 100)
                else:
                    last_val = sum(np.array(range(1173, 1172 + idx + 1)) * 100)
            np.testing.assert_almost_equal(window[-1], last_val, err_msg='field={0} minute={1}'.format(field, minute))

    @parameterized.expand([('bar_count%s' % x, x) for x in [1, 2, 3]])
    def test_daily_history_minute_gaps_price_ffill(self, test_name, bar_count):
        if False:
            i = 10
            return i + 15
        for (day_idx, day) in enumerate([pd.Timestamp('2015-01-05', tz='UTC'), pd.Timestamp('2015-01-06', tz='UTC'), pd.Timestamp('2015-01-12', tz='UTC')]):
            session_minutes = self.trading_calendar.minutes_for_session(day)
            equity_cal = self.trading_calendars[Equity]
            equity_minutes = equity_cal.minutes_for_session(day)
            if day_idx == 0:
                minutes_to_test = OrderedDict([(session_minutes[0], np.nan), (equity_minutes[0], np.nan), (equity_minutes[1], np.nan), (equity_minutes[8], np.nan), (equity_minutes[9], 11.0), (equity_minutes[10], 11.0), (equity_minutes[-2], 381.0), (equity_minutes[-1], 391.0), (session_minutes[-1], 391.0)])
            elif day_idx == 1:
                minutes_to_test = OrderedDict([(session_minutes[0], 391.0), (equity_minutes[0], 391.0), (equity_minutes[8], 391.0), (equity_minutes[9], 401.0), (equity_minutes[-1], 781.0), (session_minutes[-1], 781.0)])
            else:
                minutes_to_test = OrderedDict([(session_minutes[0], 1951.0), (equity_minutes[0], 1951.0), (equity_minutes[8], 1951.0), (equity_minutes[9], 1961.0)])
            for (minute, expected) in minutes_to_test.items():
                window = self.data_portal.get_history_window([self.ASSET3], minute, bar_count, '1d', 'price', 'minute')[self.ASSET3]
                self.assertEqual(len(window), bar_count, 'Unexpected window length at {}. Expected {}, but was {}.'.format(minute, bar_count, len(window)))
                np.testing.assert_allclose(window[-1], expected, err_msg='at minute {}'.format(minute))

class NoPrefetchMinuteEquityHistoryTestCase(MinuteEquityHistoryTestCase):
    DATA_PORTAL_MINUTE_HISTORY_PREFETCH = 0
    DATA_PORTAL_DAILY_HISTORY_PREFETCH = 0

class DailyEquityHistoryTestCase(WithHistory, zf.ZiplineTestCase):
    CREATE_BARDATA_DATA_FREQUENCY = 'daily'

    @classmethod
    def make_equity_daily_bar_data(cls, country_code, sids):
        if False:
            return 10
        yield (1, cls.create_df_for_asset(cls.START_DATE, pd.Timestamp('2016-01-30', tz='UTC')))
        yield (3, cls.create_df_for_asset(pd.Timestamp('2015-01-05', tz='UTC'), pd.Timestamp('2015-12-31', tz='UTC'), interval=10, force_zeroes=True))
        yield (cls.SHORT_ASSET_SID, cls.create_df_for_asset(pd.Timestamp('2015-01-05', tz='UTC'), pd.Timestamp('2015-01-06', tz='UTC')))
        for sid in {2, 4, 5, 6}:
            asset = cls.asset_finder.retrieve_asset(sid)
            yield (sid, cls.create_df_for_asset(asset.start_date, asset.end_date))

    @classmethod
    def create_df_for_asset(cls, start_day, end_day, interval=1, force_zeroes=False):
        if False:
            for i in range(10):
                print('nop')
        sessions = cls.trading_calendars[Equity].sessions_in_range(start_day, end_day)
        sessions_count = len(sessions)
        sessions_arr = np.array(range(2, sessions_count + 2))
        df = pd.DataFrame({'open': sessions_arr + 1, 'high': sessions_arr + 2, 'low': sessions_arr - 1, 'close': sessions_arr, 'volume': 100 * sessions_arr}, index=sessions)
        if interval > 1:
            counter = 0
            while counter < sessions_count:
                df[counter:counter + interval - 1] = 0
                counter += interval
        return df

    def test_daily_before_assets_trading(self):
        if False:
            i = 10
            return i + 15
        days = self.trading_calendar.sessions_in_range(pd.Timestamp('2014-12-15', tz='UTC'), pd.Timestamp('2014-12-18', tz='UTC'))
        for (idx, day) in enumerate(days):
            bar_data = self.create_bardata(simulation_dt_func=lambda : day)
            check_internal_consistency(bar_data, [self.ASSET2, self.ASSET3], ALL_FIELDS, 10, '1d')
            for field in ALL_FIELDS:
                asset2_series = bar_data.history(self.ASSET2, field, 10, '1d')
                asset3_series = bar_data.history(self.ASSET3, field, 10, '1d')
                if field == 'volume':
                    np.testing.assert_array_equal(np.zeros(10), asset2_series)
                    np.testing.assert_array_equal(np.zeros(10), asset3_series)
                else:
                    np.testing.assert_array_equal(np.full(10, np.nan), asset2_series)
                    np.testing.assert_array_equal(np.full(10, np.nan), asset3_series)

    def test_daily_regular(self):
        if False:
            i = 10
            return i + 15
        jan5 = pd.Timestamp('2015-01-05')
        days = self.trading_calendars[Equity].sessions_window(jan5, 30)
        for (idx, day) in enumerate(days):
            self.verify_regular_dt(idx, day, 'daily')

    def test_daily_some_assets_stopped(self):
        if False:
            print('Hello World!')
        bar_data = self.create_bardata(simulation_dt_func=lambda : pd.Timestamp('2016-01-06', tz='UTC'))
        for field in OHLCP:
            window = bar_data.history([self.ASSET1, self.ASSET2], field, 15, '1d')
            np.testing.assert_array_equal(np.full(2, np.nan), window[self.ASSET2][-2:])
            self.assertFalse(np.isnan(window[self.ASSET2][-3]))
        volume_window = bar_data.history([self.ASSET1, self.ASSET2], 'volume', 15, '1d')
        np.testing.assert_array_equal(np.zeros(2), volume_window[self.ASSET2][-2:])
        self.assertNotEqual(0, volume_window[self.ASSET2][-3])

    def test_daily_after_asset_stopped(self):
        if False:
            i = 10
            return i + 15
        days = self.trading_calendar.sessions_in_range(pd.Timestamp('2015-01-07', tz='UTC'), pd.Timestamp('2015-01-08', tz='UTC'))
        for (idx, day) in enumerate(days):
            bar_data = self.create_bardata(simulation_dt_func=lambda : day)
            check_internal_consistency(bar_data, self.SHORT_ASSET, ALL_FIELDS, 2, '1d')
            for field in ALL_FIELDS:
                asset_series = bar_data.history(self.SHORT_ASSET, field, 2, '1d')
                if idx == 0:
                    if field in OHLCP:
                        self.assertEqual(3 + MINUTE_FIELD_INFO[field], asset_series.iloc[0])
                        self.assertTrue(np.isnan(asset_series.iloc[1]))
                    elif field == 'volume':
                        self.assertEqual(300, asset_series.iloc[0])
                        self.assertEqual(0, asset_series.iloc[1])
                elif field in OHLCP:
                    self.assertTrue(np.isnan(asset_series.iloc[0]))
                    self.assertTrue(np.isnan(asset_series.iloc[1]))
                elif field == 'volume':
                    self.assertEqual(0, asset_series.iloc[0])
                    self.assertEqual(0, asset_series.iloc[1])

    def test_daily_splits_and_mergers(self):
        if False:
            for i in range(10):
                print('nop')
        for asset in [self.SPLIT_ASSET, self.MERGER_ASSET]:
            window1 = self.data_portal.get_history_window([asset], pd.Timestamp('2015-01-05', tz='UTC'), 1, '1d', 'close', 'daily')[asset]
            np.testing.assert_array_equal(window1, [2])
            window1_volume = self.data_portal.get_history_window([asset], pd.Timestamp('2015-01-05', tz='UTC'), 1, '1d', 'volume', 'daily')[asset]
            np.testing.assert_array_equal(window1_volume, [200])
            window2 = self.data_portal.get_history_window([asset], pd.Timestamp('2015-01-06', tz='UTC'), 2, '1d', 'close', 'daily')[asset]
            np.testing.assert_array_equal([0.5, 3], window2)
            window2_volume = self.data_portal.get_history_window([asset], pd.Timestamp('2015-01-06', tz='UTC'), 2, '1d', 'volume', 'daily')[asset]
            if asset == self.SPLIT_ASSET:
                np.testing.assert_array_equal(window2_volume, [800, 300])
            elif asset == self.MERGER_ASSET:
                np.testing.assert_array_equal(window2_volume, [200, 300])
            window3 = self.data_portal.get_history_window([asset], pd.Timestamp('2015-01-07', tz='UTC'), 3, '1d', 'close', 'daily')[asset]
            np.testing.assert_array_equal([0.25, 1.5, 4], window3)
            window3_volume = self.data_portal.get_history_window([asset], pd.Timestamp('2015-01-07', tz='UTC'), 3, '1d', 'volume', 'daily')[asset]
            if asset == self.SPLIT_ASSET:
                np.testing.assert_array_equal(window3_volume, [1600, 600, 400])
            elif asset == self.MERGER_ASSET:
                np.testing.assert_array_equal(window3_volume, [200, 300, 400])

    def test_daily_dividends(self):
        if False:
            for i in range(10):
                print('nop')
        window1 = self.data_portal.get_history_window([self.DIVIDEND_ASSET], pd.Timestamp('2015-01-05', tz='UTC'), 1, '1d', 'close', 'daily')[self.DIVIDEND_ASSET]
        np.testing.assert_array_equal(window1, [2])
        window2 = self.data_portal.get_history_window([self.DIVIDEND_ASSET], pd.Timestamp('2015-01-06', tz='UTC'), 2, '1d', 'close', 'daily')[self.DIVIDEND_ASSET]
        np.testing.assert_array_equal([1.96, 3], window2)
        window3 = self.data_portal.get_history_window([self.DIVIDEND_ASSET], pd.Timestamp('2015-01-07', tz='UTC'), 3, '1d', 'close', 'daily')[self.DIVIDEND_ASSET]
        np.testing.assert_array_equal([1.882, 2.88, 4], window3)

    def test_daily_blended_some_assets_stopped(self):
        if False:
            for i in range(10):
                print('nop')
        bar_data = self.create_bardata(simulation_dt_func=lambda : pd.Timestamp('2016-01-06 16:00', tz='UTC'))
        for field in OHLCP:
            window = bar_data.history([self.ASSET1, self.ASSET2], field, 15, '1d')
            np.testing.assert_array_equal(np.full(2, np.nan), window[self.ASSET2][-2:])
            self.assertFalse(np.isnan(window[self.ASSET2][-3]))
        volume_window = bar_data.history([self.ASSET1, self.ASSET2], 'volume', 15, '1d')
        np.testing.assert_array_equal(np.zeros(2), volume_window[self.ASSET2][-2:])
        self.assertNotEqual(0, volume_window[self.ASSET2][-3])

    def test_history_window_before_first_trading_day(self):
        if False:
            for i in range(10):
                print('nop')
        second_day = self.trading_calendar.next_session_label(self.TRADING_START_DT)
        exp_msg = 'History window extends before 2014-01-03. To use this history window, start the backtest on or after 2014-01-09.'
        with self.assertRaisesRegex(HistoryWindowStartsBeforeData, exp_msg):
            self.data_portal.get_history_window([self.ASSET1], second_day, 4, '1d', 'price', 'daily')[self.ASSET1]
        with self.assertRaisesRegex(HistoryWindowStartsBeforeData, exp_msg):
            self.data_portal.get_history_window([self.ASSET1], second_day, 4, '1d', 'volume', 'daily')[self.ASSET1]
        first_minute = self.trading_calendar.schedule.market_open[self.TRADING_START_DT]
        with self.assertRaisesRegex(HistoryWindowStartsBeforeData, exp_msg):
            self.data_portal.get_history_window([self.ASSET2], first_minute, 4, '1d', 'close', 'daily')[self.ASSET2]

    def test_history_window_different_order(self):
        if False:
            return 10
        '\n        Prevent regression on a bug where the passing the same assets, but\n        in a different order would return a history window with the values,\n        but not the keys, in order of the first history call.\n        '
        day = self.ASSET2.end_date
        window_1 = self.data_portal.get_history_window([self.ASSET1, self.ASSET2], day, 4, '1d', 'close', 'daily')
        window_2 = self.data_portal.get_history_window([self.ASSET2, self.ASSET1], day, 4, '1d', 'close', 'daily')
        np.testing.assert_almost_equal(window_1[self.ASSET1].values, window_2[self.ASSET1].values)
        np.testing.assert_almost_equal(window_1[self.ASSET2].values, window_2[self.ASSET2].values)

    def test_history_window_out_of_order_dates(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Use a history window with non-monotonically increasing dates.\n        A scenario which does not occur during simulations, but useful\n        for using a history loader in a notebook.\n        '
        window_1 = self.data_portal.get_history_window([self.ASSET1], pd.Timestamp('2014-02-07', tz='UTC'), 4, '1d', 'close', 'daily')
        window_2 = self.data_portal.get_history_window([self.ASSET1], pd.Timestamp('2014-02-05', tz='UTC'), 4, '1d', 'close', 'daily')
        window_3 = self.data_portal.get_history_window([self.ASSET1], pd.Timestamp('2014-02-07', tz='UTC'), 4, '1d', 'close', 'daily')
        window_4 = self.data_portal.get_history_window([self.ASSET1], pd.Timestamp('2014-01-22', tz='UTC'), 4, '1d', 'close', 'daily')
        np.testing.assert_almost_equal(window_1.values, window_3.values)
        offsets = np.arange(4)

        def assert_window_prices(window, prices):
            if False:
                i = 10
                return i + 15
            np.testing.assert_almost_equal(window.loc[:, self.ASSET1], prices)
        assert_window_prices(window_1, 23 + offsets)
        assert_window_prices(window_2, 21 + offsets)
        assert_window_prices(window_3, 23 + offsets)
        if not self.trading_calendar.is_session('2014-01-20'):
            assert_window_prices(window_4, 11 + offsets)
        else:
            assert_window_prices(window_4, [12, nan, 13, 14])

class NoPrefetchDailyEquityHistoryTestCase(DailyEquityHistoryTestCase):
    DATA_PORTAL_MINUTE_HISTORY_PREFETCH = 0
    DATA_PORTAL_DAILY_HISTORY_PREFETCH = 0

class MinuteEquityHistoryFuturesCalendarTestCase(MinuteEquityHistoryTestCase):
    TRADING_CALENDAR_STRS = ('NYSE', 'us_futures')
    TRADING_CALENDAR_PRIMARY_CAL = 'us_futures'

class DailyEquityHistoryFuturesCalendarTestCase(DailyEquityHistoryTestCase):
    TRADING_CALENDAR_STRS = ('NYSE', 'us_futures')
    TRADING_CALENDAR_PRIMARY_CAL = 'us_futures'