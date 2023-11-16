"""Tests for pipelines on international markets.
"""
from itertools import cycle, islice
from nose_parameterized import parameterized
import numpy as np
import pandas as pd
from trading_calendars import get_calendar
from zipline.assets.synthetic import make_rotating_equity_info
from zipline.data.in_memory_daily_bars import InMemoryDailyBarReader
from zipline.pipeline.domain import CA_EQUITIES, GB_EQUITIES, US_EQUITIES
from zipline.pipeline import Pipeline
from zipline.pipeline.data import EquityPricing, USEquityPricing
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.loaders.equity_pricing_loader import EquityPricingLoader
from zipline.pipeline.loaders.synthetic import NullAdjustmentReader
from zipline.testing.predicates import assert_equal
from zipline.testing.core import parameter_space, random_tick_prices
import zipline.testing.fixtures as zf

def T(s):
    if False:
        while True:
            i = 10
    return pd.Timestamp(s, tz='UTC')

class WithInternationalDailyBarData(zf.WithAssetFinder):
    """
    Fixture for generating international daily bars.

    Eventually this should be moved into zipline.testing.fixtures and should
    replace most of the existing machinery
    """
    DAILY_BAR_START_DATE = zf.alias('START_DATE')
    DAILY_BAR_END_DATE = zf.alias('END_DATE')
    DAILY_BAR_LOOKBACK_DAYS = 0
    INTERNATIONAL_PRICING_STARTING_PRICES = {'XNYS': 100, 'XTSE': 50, 'XLON': 25}
    INTERNATIONAL_PRICING_CURRENCIES = {'XNYS': ['USD'], 'XTSE': ['CAD'], 'XLON': ['GBP', 'EUR', 'USD']}
    assert INTERNATIONAL_PRICING_STARTING_PRICES.keys() == INTERNATIONAL_PRICING_CURRENCIES.keys()
    FX_RATES_CURRENCIES = ['USD', 'CAD', 'GBP', 'EUR']

    @classmethod
    def make_daily_bar_data(cls, assets, calendar, sessions):
        if False:
            i = 10
            return i + 15
        start = cls.INTERNATIONAL_PRICING_STARTING_PRICES[calendar.name]
        closes = random_tick_prices(start, len(sessions))
        opens = closes - 0.05
        highs = closes + 0.1
        lows = closes - 0.1
        volumes = np.arange(10000, 10000 + len(closes))
        base_frame = pd.DataFrame({'close': closes, 'open': opens, 'high': highs, 'low': lows, 'volume': volumes}, index=sessions)
        for asset in assets:
            sid = asset.sid
            yield (sid, base_frame + sid)

    @classmethod
    def make_currency_codes(cls, calendar, assets):
        if False:
            return 10
        currencies = cls.INTERNATIONAL_PRICING_CURRENCIES[calendar.name]
        return pd.Series(index=assets, data=list(islice(cycle(currencies), len(assets))))

    @classmethod
    def init_class_fixtures(cls):
        if False:
            print('Hello World!')
        super(WithInternationalDailyBarData, cls).init_class_fixtures()
        cls.daily_bar_sessions = {}
        cls.daily_bar_data = {}
        cls.daily_bar_readers = {}
        cls.daily_bar_currency_codes = {}
        for (calendar, assets) in cls.assets_by_calendar.items():
            name = calendar.name
            start_delta = cls.DAILY_BAR_LOOKBACK_DAYS * calendar.day
            start_session = cls.DAILY_BAR_START_DATE - start_delta
            sessions = calendar.sessions_in_range(start_session, cls.DAILY_BAR_END_DATE)
            cls.daily_bar_sessions[name] = sessions
            cls.daily_bar_data[name] = dict(cls.make_daily_bar_data(assets=assets, calendar=calendar, sessions=sessions))
            panel = pd.Panel.from_dict(cls.daily_bar_data[name]).transpose(2, 1, 0)
            cls.daily_bar_currency_codes[name] = cls.make_currency_codes(calendar, assets)
            cls.daily_bar_readers[name] = InMemoryDailyBarReader.from_panel(panel, calendar, currency_codes=cls.daily_bar_currency_codes[name])

class WithInternationalPricingPipelineEngine(zf.WithFXRates, WithInternationalDailyBarData):

    @classmethod
    def init_class_fixtures(cls):
        if False:
            i = 10
            return i + 15
        super(WithInternationalPricingPipelineEngine, cls).init_class_fixtures()
        adjustments = NullAdjustmentReader()
        cls.loaders = {GB_EQUITIES: EquityPricingLoader(cls.daily_bar_readers['XLON'], adjustments, cls.in_memory_fx_rate_reader), US_EQUITIES: EquityPricingLoader(cls.daily_bar_readers['XNYS'], adjustments, cls.in_memory_fx_rate_reader), CA_EQUITIES: EquityPricingLoader(cls.daily_bar_readers['XTSE'], adjustments, cls.in_memory_fx_rate_reader)}
        cls.engine = SimplePipelineEngine(get_loader=cls.get_loader, asset_finder=cls.asset_finder)

    @classmethod
    def get_loader(cls, column):
        if False:
            i = 10
            return i + 15
        return cls.loaders[column.domain]

    def run_pipeline(self, pipeline, start_date, end_date):
        if False:
            i = 10
            return i + 15
        return self.engine.run_pipeline(pipeline, start_date, end_date)

class InternationalEquityTestCase(WithInternationalPricingPipelineEngine, zf.ZiplineTestCase):
    START_DATE = T('2014-01-02')
    END_DATE = T('2014-02-06')
    EXCHANGE_INFO = pd.DataFrame.from_records([{'exchange': 'XNYS', 'country_code': 'US'}, {'exchange': 'XTSE', 'country_code': 'CA'}, {'exchange': 'XLON', 'country_code': 'GB'}])

    @classmethod
    def make_equity_info(cls):
        if False:
            print('Hello World!')
        out = pd.concat([make_rotating_equity_info(num_assets=20, first_start=cls.START_DATE, frequency=get_calendar(exchange).day, periods_between_starts=1, asset_lifetime=5, exchange=exchange) for exchange in cls.EXCHANGE_INFO.exchange], ignore_index=True)
        assert_equal(out.end_date.max(), cls.END_DATE)
        return out

    @classmethod
    def make_exchanges_info(cls, equities, futures, root_symbols):
        if False:
            while True:
                i = 10
        return cls.EXCHANGE_INFO

    @parameter_space(domain=[CA_EQUITIES, US_EQUITIES, GB_EQUITIES])
    def test_generic_pipeline_with_explicit_domain(self, domain):
        if False:
            while True:
                i = 10
        calendar = domain.calendar
        pipe = Pipeline({'open': EquityPricing.open.latest, 'high': EquityPricing.high.latest, 'low': EquityPricing.low.latest, 'close': EquityPricing.close.latest, 'volume': EquityPricing.volume.latest}, domain=domain)
        sessions = self.daily_bar_sessions[calendar.name]
        (start, end) = sessions[[-17, -10]]
        result = self.run_pipeline(pipe, start, end)
        all_assets = self.assets_by_calendar[calendar]
        expected_assets = [a for a in all_assets if alive_in_range(a, start, end, include_asset_start_date=False)]
        expected_dates = sessions[-17:-9]
        for col in pipe.columns:
            result_data = result[col].unstack()
            assert_equal(pd.Index(expected_assets), result_data.columns)
            assert_equal(expected_dates, result_data.index)
            for asset in expected_assets:
                for date in expected_dates:
                    value = result_data.at[date, asset]
                    self.check_expected_latest_value(calendar, col, date, asset, value)

    @parameterized.expand([('US', US_EQUITIES, 'XNYS'), ('CA', CA_EQUITIES, 'XTSE'), ('GB', GB_EQUITIES, 'XLON')])
    def test_currency_convert_prices(self, name, domain, calendar_name):
        if False:
            i = 10
            return i + 15
        pipe = Pipeline({'close': EquityPricing.close.latest, 'close_USD': EquityPricing.close.fx('USD').latest, 'close_CAD': EquityPricing.close.fx('CAD').latest, 'close_EUR': EquityPricing.close.fx('EUR').latest, 'close_GBP': EquityPricing.close.fx('GBP').latest}, domain=domain)
        sessions = self.daily_bar_sessions[calendar_name]
        execution_sessions = sessions[-17:-9]
        (start, end) = execution_sessions[[0, -1]]
        result = self.run_pipeline(pipe, start, end)
        closes_2d = result['close'].unstack(fill_value=np.nan)
        all_currency_codes = self.daily_bar_currency_codes[calendar_name]
        currency_codes = all_currency_codes.loc[[a.sid for a in closes_2d.columns]]
        fx_reader = self.in_memory_fx_rate_reader
        for target in self.FX_RATES_CURRENCIES:
            result_2d = result['close_' + target].unstack(fill_value=np.nan)
            expected_rates = fx_reader.get_rates(rate='mid', quote=target, bases=np.array(currency_codes, dtype=object), dts=sessions[-18:-10])
            expected_result_2d = closes_2d * expected_rates
            assert_equal(result_2d, expected_result_2d)

    @parameterized.expand([('US', US_EQUITIES, 'XNYS'), ('CA', CA_EQUITIES, 'XTSE'), ('GB', GB_EQUITIES, 'XLON')])
    def test_only_currency_converted_data(self, name, domain, calendar_name):
        if False:
            return 10
        pipe = Pipeline({'close_USD': EquityPricing.close.fx('USD').latest, 'close_EUR': EquityPricing.close.fx('EUR').latest}, domain=domain)
        (start, end) = self.daily_bar_sessions[calendar_name][-2:]
        result = self.run_pipeline(pipe, start, end)
        calendar = get_calendar(calendar_name)
        daily_bars = self.daily_bar_data[calendar_name]
        currency_codes = self.daily_bar_currency_codes[calendar_name]
        for ((dt, asset), row) in result.iterrows():
            price_date = dt - calendar.day
            expected_close = daily_bars[asset].loc[price_date, 'close']
            expected_base = currency_codes.loc[asset]
            expected_rate_USD = self.in_memory_fx_rate_reader.get_rate_scalar(rate='mid', quote='USD', base=expected_base, dt=price_date.asm8)
            expected_price = expected_close * expected_rate_USD
            assert_equal(row.close_USD, expected_price)
            expected_rate_EUR = self.in_memory_fx_rate_reader.get_rate_scalar(rate='mid', quote='EUR', base=expected_base, dt=price_date.asm8)
            expected_price = expected_close * expected_rate_EUR
            assert_equal(row.close_EUR, expected_price)

    def test_explicit_specialization_matches_implicit(self):
        if False:
            print('Hello World!')
        pipeline_specialized = Pipeline({'open': EquityPricing.open.latest, 'high': EquityPricing.high.latest, 'low': EquityPricing.low.latest, 'close': EquityPricing.close.latest, 'volume': EquityPricing.volume.latest}, domain=US_EQUITIES)
        dataset_specialized = Pipeline({'open': USEquityPricing.open.latest, 'high': USEquityPricing.high.latest, 'low': USEquityPricing.low.latest, 'close': USEquityPricing.close.latest, 'volume': USEquityPricing.volume.latest})
        sessions = self.daily_bar_sessions['XNYS']
        self.assert_identical_results(pipeline_specialized, dataset_specialized, sessions[1], sessions[-1])

    def test_cannot_convert_volume_data(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(TypeError) as exc:
            EquityPricing.volume.fx('EUR')
        assert_equal(str(exc.exception), 'The .fx() method cannot be called on EquityPricing.volume because it does not produce currency-denominated data.')

    def check_expected_latest_value(self, calendar, col, date, asset, value):
        if False:
            return 10
        'Check the expected result of column.latest from a pipeline.\n        '
        if np.isnan(value):
            self.assertTrue(date <= asset.start_date or date > asset.end_date)
        else:
            self.assertTrue(asset.start_date < date <= asset.end_date)
            bars = self.daily_bar_data[calendar.name]
            expected_value = bars[asset.sid].loc[date - calendar.day, col]
            assert_equal(value, expected_value)

    def assert_identical_results(self, left, right, start_date, end_date):
        if False:
            i = 10
            return i + 15
        'Assert that two pipelines produce the same results.\n        '
        left_result = self.run_pipeline(left, start_date, end_date)
        right_result = self.run_pipeline(right, start_date, end_date)
        assert_equal(left_result, right_result)

def alive_in_range(asset, start, end, include_asset_start_date=False):
    if False:
        i = 10
        return i + 15
    '\n    Check if an asset was alive in the range from start to end.\n\n    Parameters\n    ----------\n    asset : Asset\n        The asset to check\n    start : pd.Timestamp\n        Start of the interval.\n    end : pd.Timestamp\n        End of the interval.\n    include_asset_start_date : bool\n        Whether to include the start date of the asset when checking liveness.\n\n    Returns\n    -------\n    was_alive : bool\n        Whether or not ``asset`` was alive for any days in the range from\n        ``start`` to ``end``.\n    '
    if include_asset_start_date:
        asset_start = asset.start_date
    else:
        asset_start = asset.start_date + pd.Timedelta('1 day')
    return intervals_overlap((asset_start, asset.end_date), (start, end))

def intervals_overlap(a, b):
    if False:
        return 10
    '\n    Check whether a pair of datetime intervals overlap.\n\n    Parameters\n    ----------\n    a : (pd.Timestamp, pd.Timestamp)\n    b : (pd.Timestamp, pd.Timestamp)\n\n    Returns\n    -------\n    have_overlap : bool\n        Bool indicating whether there there is a non-empty intersection between\n        the intervals.\n    '
    a_strictly_before = a[1] < b[0]
    b_strictly_before = b[1] < a[0]
    return not (a_strictly_before or b_strictly_before)