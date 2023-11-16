"""
Tests for USEquityPricingLoader and related classes.
"""
from nose_parameterized import parameterized
from numpy import arange, datetime64, float64, ones, uint32
from numpy.testing import assert_allclose, assert_array_equal
from pandas import concat, DataFrame, Int64Index, Timestamp
from pandas.util.testing import assert_frame_equal
from toolz.curried.operator import getitem
from zipline.lib.adjustment import Float64Multiply
from zipline.pipeline.domain import US_EQUITIES
from zipline.pipeline.loaders.synthetic import NullAdjustmentReader, make_bar_data, expected_bar_values_2d
from zipline.pipeline.loaders.equity_pricing_loader import USEquityPricingLoader
from zipline.errors import WindowLengthTooLong
from zipline.pipeline.data import USEquityPricing
from zipline.testing import seconds_to_timestamp, str_to_seconds, MockDailyBarReader
from zipline.testing.fixtures import WithAdjustmentReader, ZiplineTestCase
TEST_CALENDAR_START = Timestamp('2015-06-01', tz='UTC')
TEST_CALENDAR_STOP = Timestamp('2015-06-30', tz='UTC')
TEST_QUERY_START = Timestamp('2015-06-10', tz='UTC')
TEST_QUERY_STOP = Timestamp('2015-06-19', tz='UTC')
EQUITY_INFO = DataFrame([{'start_date': '2015-06-01', 'end_date': '2015-06-05'}, {'start_date': '2015-06-22', 'end_date': '2015-06-30'}, {'start_date': '2015-06-02', 'end_date': '2015-06-30'}, {'start_date': '2015-06-01', 'end_date': '2015-06-15'}, {'start_date': '2015-06-12', 'end_date': '2015-06-18'}, {'start_date': '2015-06-15', 'end_date': '2015-06-25'}], index=arange(1, 7), columns=['start_date', 'end_date']).astype(datetime64)
EQUITY_INFO['symbol'] = [chr(ord('A') + n) for n in range(len(EQUITY_INFO))]
EQUITY_INFO['exchange'] = 'TEST'
TEST_QUERY_SIDS = EQUITY_INFO.index
SPLITS = DataFrame([{'effective_date': str_to_seconds('2015-06-03'), 'ratio': 1.103, 'sid': 1}, {'effective_date': str_to_seconds('2015-06-10'), 'ratio': 3.11, 'sid': 3}, {'effective_date': str_to_seconds('2015-06-12'), 'ratio': 3.112, 'sid': 3}, {'effective_date': str_to_seconds('2015-06-21'), 'ratio': 6.121, 'sid': 6}, {'effective_date': str_to_seconds('2015-06-11'), 'ratio': 3.111, 'sid': 3}, {'effective_date': str_to_seconds('2015-06-19'), 'ratio': 3.119, 'sid': 3}], columns=['effective_date', 'ratio', 'sid'])
MERGERS = DataFrame([{'effective_date': str_to_seconds('2015-06-03'), 'ratio': 1.203, 'sid': 1}, {'effective_date': str_to_seconds('2015-06-10'), 'ratio': 3.21, 'sid': 3}, {'effective_date': str_to_seconds('2015-06-12'), 'ratio': 3.212, 'sid': 3}, {'effective_date': str_to_seconds('2015-06-25'), 'ratio': 6.225, 'sid': 6}, {'effective_date': str_to_seconds('2015-06-12'), 'ratio': 4.212, 'sid': 4}, {'effective_date': str_to_seconds('2015-06-19'), 'ratio': 3.219, 'sid': 3}], columns=['effective_date', 'ratio', 'sid'])
DIVIDENDS = DataFrame([{'declared_date': Timestamp('2015-05-01', tz='UTC').to_datetime64(), 'ex_date': Timestamp('2015-06-01', tz='UTC').to_datetime64(), 'record_date': Timestamp('2015-06-03', tz='UTC').to_datetime64(), 'pay_date': Timestamp('2015-06-05', tz='UTC').to_datetime64(), 'amount': 90.0, 'sid': 1}, {'declared_date': Timestamp('2015-06-01', tz='UTC').to_datetime64(), 'ex_date': Timestamp('2015-06-10', tz='UTC').to_datetime64(), 'record_date': Timestamp('2015-06-15', tz='UTC').to_datetime64(), 'pay_date': Timestamp('2015-06-17', tz='UTC').to_datetime64(), 'amount': 80.0, 'sid': 3}, {'declared_date': Timestamp('2015-06-01', tz='UTC').to_datetime64(), 'ex_date': Timestamp('2015-06-12', tz='UTC').to_datetime64(), 'record_date': Timestamp('2015-06-15', tz='UTC').to_datetime64(), 'pay_date': Timestamp('2015-06-17', tz='UTC').to_datetime64(), 'amount': 70.0, 'sid': 3}, {'declared_date': Timestamp('2015-06-01', tz='UTC').to_datetime64(), 'ex_date': Timestamp('2015-06-25', tz='UTC').to_datetime64(), 'record_date': Timestamp('2015-06-28', tz='UTC').to_datetime64(), 'pay_date': Timestamp('2015-06-30', tz='UTC').to_datetime64(), 'amount': 60.0, 'sid': 6}, {'declared_date': Timestamp('2015-06-01', tz='UTC').to_datetime64(), 'ex_date': Timestamp('2015-06-15', tz='UTC').to_datetime64(), 'record_date': Timestamp('2015-06-18', tz='UTC').to_datetime64(), 'pay_date': Timestamp('2015-06-20', tz='UTC').to_datetime64(), 'amount': 50.0, 'sid': 3}, {'declared_date': Timestamp('2015-06-01', tz='UTC').to_datetime64(), 'ex_date': Timestamp('2015-06-19', tz='UTC').to_datetime64(), 'record_date': Timestamp('2015-06-22', tz='UTC').to_datetime64(), 'pay_date': Timestamp('2015-06-30', tz='UTC').to_datetime64(), 'amount': 40.0, 'sid': 3}], columns=['declared_date', 'ex_date', 'record_date', 'pay_date', 'amount', 'sid'])
DIVIDENDS_EXPECTED = DataFrame([{'effective_date': str_to_seconds('2015-06-01'), 'ratio': 0.1, 'sid': 1}, {'effective_date': str_to_seconds('2015-06-10'), 'ratio': 0.2, 'sid': 3}, {'effective_date': str_to_seconds('2015-06-12'), 'ratio': 0.3, 'sid': 3}, {'effective_date': str_to_seconds('2015-06-25'), 'ratio': 0.4, 'sid': 6}, {'effective_date': str_to_seconds('2015-06-15'), 'ratio': 0.5, 'sid': 3}, {'effective_date': str_to_seconds('2015-06-19'), 'ratio': 0.6, 'sid': 3}], columns=['effective_date', 'ratio', 'sid'])

class USEquityPricingLoaderTestCase(WithAdjustmentReader, ZiplineTestCase):
    START_DATE = TEST_CALENDAR_START
    END_DATE = TEST_CALENDAR_STOP
    asset_ids = (1, 2, 3)

    @classmethod
    def make_equity_info(cls):
        if False:
            print('Hello World!')
        return EQUITY_INFO

    @classmethod
    def make_splits_data(cls):
        if False:
            i = 10
            return i + 15
        return SPLITS

    @classmethod
    def make_mergers_data(cls):
        if False:
            while True:
                i = 10
        return MERGERS

    @classmethod
    def make_dividends_data(cls):
        if False:
            return 10
        return DIVIDENDS

    @classmethod
    def make_adjustment_writer_equity_daily_bar_reader(cls):
        if False:
            print('Hello World!')
        return MockDailyBarReader(dates=cls.calendar_days_between(cls.START_DATE, cls.END_DATE))

    @classmethod
    def make_equity_daily_bar_data(cls, country_code, sids):
        if False:
            i = 10
            return i + 15
        return make_bar_data(EQUITY_INFO, cls.equity_daily_bar_days)

    @classmethod
    def init_class_fixtures(cls):
        if False:
            print('Hello World!')
        super(USEquityPricingLoaderTestCase, cls).init_class_fixtures()
        cls.sids = TEST_QUERY_SIDS
        cls.asset_info = EQUITY_INFO

    def test_input_sanity(self):
        if False:
            print('Hello World!')
        for table in (SPLITS, MERGERS):
            for (eff_date_secs, _, sid) in table.itertuples(index=False):
                eff_date = Timestamp(eff_date_secs, unit='s')
                (asset_start, asset_end) = EQUITY_INFO.ix[sid, ['start_date', 'end_date']]
                self.assertGreaterEqual(eff_date, asset_start)
                self.assertLessEqual(eff_date, asset_end)

    @classmethod
    def calendar_days_between(cls, start_date, end_date, shift=0):
        if False:
            for i in range(10):
                print('nop')
        slice_ = cls.equity_daily_bar_days.slice_indexer(start_date, end_date)
        start = slice_.start + shift
        stop = slice_.stop + shift
        if start < 0:
            raise KeyError(start_date, shift)
        return cls.equity_daily_bar_days[start:stop]

    def expected_adjustments(self, start_date, end_date, tables, adjustment_type):
        if False:
            return 10
        price_adjustments = {}
        volume_adjustments = {}
        should_include_price_adjustments = adjustment_type == 'all' or adjustment_type == 'price'
        should_include_volume_adjustments = adjustment_type == 'all' or adjustment_type == 'volume'
        query_days = self.calendar_days_between(start_date, end_date)
        start_loc = query_days.get_loc(start_date)
        for table in tables:
            for (eff_date_secs, ratio, sid) in table.itertuples(index=False):
                eff_date = Timestamp(eff_date_secs, unit='s', tz='UTC')
                if not start_date <= eff_date <= end_date:
                    continue
                eff_date_loc = query_days.get_loc(eff_date)
                delta = eff_date_loc - start_loc
                if should_include_price_adjustments:
                    price_adjustments.setdefault(delta, []).append(Float64Multiply(first_row=0, last_row=delta, first_col=sid - 1, last_col=sid - 1, value=ratio))
                if table is SPLITS and should_include_volume_adjustments:
                    volume_adjustments.setdefault(delta, []).append(Float64Multiply(first_row=0, last_row=delta, first_col=sid - 1, last_col=sid - 1, value=1.0 / ratio))
        output = {}
        if should_include_price_adjustments:
            output['price_adjustments'] = price_adjustments
        if should_include_volume_adjustments:
            output['volume_adjustments'] = volume_adjustments
        return output

    @parameterized([([SPLITS, MERGERS, DIVIDENDS_EXPECTED], 'all'), ([SPLITS, MERGERS, DIVIDENDS_EXPECTED], 'price'), ([SPLITS, MERGERS, DIVIDENDS_EXPECTED], 'volume'), ([SPLITS, MERGERS, None], 'all'), ([SPLITS, MERGERS, None], 'price')])
    def test_load_adjustments(self, tables, adjustment_type):
        if False:
            while True:
                i = 10
        query_days = self.calendar_days_between(TEST_QUERY_START, TEST_QUERY_STOP)
        adjustments = self.adjustment_reader.load_adjustments(query_days, self.sids, should_include_splits=tables[0] is not None, should_include_mergers=tables[1] is not None, should_include_dividends=tables[2] is not None, adjustment_type=adjustment_type)
        expected_adjustments = self.expected_adjustments(TEST_QUERY_START, TEST_QUERY_STOP, [table for table in tables if table is not None], adjustment_type)
        if adjustment_type == 'all' or adjustment_type == 'price':
            expected_price_adjustments = expected_adjustments['price']
            for key in expected_price_adjustments:
                price_adjustment = adjustments['price'][key]
                for (j, adj) in enumerate(price_adjustment):
                    expected = expected_price_adjustments[key][j]
                    self.assertEqual(adj.first_row, expected.first_row)
                    self.assertEqual(adj.last_row, expected.last_row)
                    self.assertEqual(adj.first_col, expected.first_col)
                    self.assertEqual(adj.last_col, expected.last_col)
                    assert_allclose(adj.value, expected.value)
        if adjustment_type == 'all' or adjustment_type == 'volume':
            expected_volume_adjustments = expected_adjustments['volume']
            for key in expected_volume_adjustments:
                volume_adjustment = adjustments['volume'][key]
                for (j, adj) in enumerate(volume_adjustment):
                    expected = expected_volume_adjustments[key][j]
                    self.assertEqual(adj.first_row, expected.first_row)
                    self.assertEqual(adj.last_row, expected.last_row)
                    self.assertEqual(adj.first_col, expected.first_col)
                    self.assertEqual(adj.last_col, expected.last_col)
                    assert_allclose(adj.value, expected.value)

    @parameterized([(True,), (False,)])
    def test_load_adjustments_to_df(self, convert_dts):
        if False:
            while True:
                i = 10
        reader = self.adjustment_reader
        adjustment_dfs = reader.unpack_db_to_component_dfs(convert_dates=convert_dts)
        name_and_raw = (('splits', SPLITS), ('mergers', MERGERS), ('dividends', DIVIDENDS_EXPECTED))

        def create_expected_table(df, name):
            if False:
                i = 10
                return i + 15
            expected_df = df.copy()
            if convert_dts:
                for colname in reader._datetime_int_cols[name]:
                    expected_df[colname] = expected_df[colname].astype('datetime64[s]')
            return expected_df

        def create_expected_div_table(df, name):
            if False:
                return 10
            expected_df = df.copy()
            if not convert_dts:
                for colname in reader._datetime_int_cols[name]:
                    expected_df[colname] = expected_df[colname].astype('datetime64[s]').astype(int)
            return expected_df
        for (action_name, raw_tbl) in name_and_raw:
            exp = create_expected_table(raw_tbl, action_name)
            assert_frame_equal(adjustment_dfs[action_name], exp)
        div_name = 'dividend_payouts'
        assert_frame_equal(adjustment_dfs[div_name], create_expected_div_table(DIVIDENDS, div_name))

    def test_read_no_adjustments(self):
        if False:
            return 10
        adjustment_reader = NullAdjustmentReader()
        columns = [USEquityPricing.close, USEquityPricing.volume]
        query_days = self.calendar_days_between(TEST_QUERY_START, TEST_QUERY_STOP)
        shifted_query_days = self.calendar_days_between(TEST_QUERY_START, TEST_QUERY_STOP, shift=-1)
        adjustments = adjustment_reader.load_pricing_adjustments([c.name for c in columns], query_days, self.sids)
        self.assertEqual(adjustments, [{}, {}])
        pricing_loader = USEquityPricingLoader.without_fx(self.bcolz_equity_daily_bar_reader, adjustment_reader)
        results = pricing_loader.load_adjusted_array(domain=US_EQUITIES, columns=columns, dates=query_days, sids=self.sids, mask=ones((len(query_days), len(self.sids)), dtype=bool))
        (closes, volumes) = map(getitem(results), columns)
        expected_baseline_closes = expected_bar_values_2d(shifted_query_days, self.sids, self.asset_info, 'close')
        expected_baseline_volumes = expected_bar_values_2d(shifted_query_days, self.sids, self.asset_info, 'volume')
        for windowlen in range(1, len(query_days) + 1):
            for (offset, window) in enumerate(closes.traverse(windowlen)):
                assert_array_equal(expected_baseline_closes[offset:offset + windowlen], window)
            for (offset, window) in enumerate(volumes.traverse(windowlen)):
                assert_array_equal(expected_baseline_volumes[offset:offset + windowlen], window)
        with self.assertRaises(WindowLengthTooLong):
            closes.traverse(windowlen + 1)
        with self.assertRaises(WindowLengthTooLong):
            volumes.traverse(windowlen + 1)

    def apply_adjustments(self, dates, assets, baseline_values, adjustments):
        if False:
            for i in range(10):
                print('nop')
        (min_date, max_date) = dates[[0, -1]]
        orig_dtype = baseline_values.dtype
        values = baseline_values.astype(float64).copy()
        for (eff_date_secs, ratio, sid) in adjustments.itertuples(index=False):
            eff_date = seconds_to_timestamp(eff_date_secs)
            if eff_date not in dates:
                continue
            eff_date_loc = dates.get_loc(eff_date)
            asset_col = assets.get_loc(sid)
            values[:eff_date_loc + 1, asset_col] *= ratio
        return values.astype(orig_dtype)

    def test_read_with_adjustments(self):
        if False:
            while True:
                i = 10
        columns = [USEquityPricing.high, USEquityPricing.volume]
        query_days = self.calendar_days_between(TEST_QUERY_START, TEST_QUERY_STOP)
        shifted_query_days = self.calendar_days_between(TEST_QUERY_START, TEST_QUERY_STOP, shift=-1)
        pricing_loader = USEquityPricingLoader.without_fx(self.bcolz_equity_daily_bar_reader, self.adjustment_reader)
        results = pricing_loader.load_adjusted_array(domain=US_EQUITIES, columns=columns, dates=query_days, sids=Int64Index(arange(1, 7)), mask=ones((len(query_days), 6), dtype=bool))
        (highs, volumes) = map(getitem(results), columns)
        expected_baseline_highs = expected_bar_values_2d(shifted_query_days, self.sids, self.asset_info, 'high')
        expected_baseline_volumes = expected_bar_values_2d(shifted_query_days, self.sids, self.asset_info, 'volume')
        for windowlen in range(1, len(query_days) + 1):
            for (offset, window) in enumerate(highs.traverse(windowlen)):
                baseline = expected_baseline_highs[offset:offset + windowlen]
                baseline_dates = query_days[offset:offset + windowlen]
                expected_adjusted_highs = self.apply_adjustments(baseline_dates, self.sids, baseline, concat([SPLITS, MERGERS, DIVIDENDS_EXPECTED], ignore_index=True))
                assert_allclose(expected_adjusted_highs, window)
            for (offset, window) in enumerate(volumes.traverse(windowlen)):
                baseline = expected_baseline_volumes[offset:offset + windowlen]
                baseline_dates = query_days[offset:offset + windowlen]
                adjustments = SPLITS.copy()
                adjustments.ratio = 1 / adjustments.ratio
                expected_adjusted_volumes = self.apply_adjustments(baseline_dates, self.sids, baseline, adjustments)
                assert_array_equal(expected_adjusted_volumes, window.astype(uint32))
        with self.assertRaises(WindowLengthTooLong):
            highs.traverse(windowlen + 1)
        with self.assertRaises(WindowLengthTooLong):
            volumes.traverse(windowlen + 1)