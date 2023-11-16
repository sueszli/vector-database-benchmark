import os
import sqlite3
from unittest import TestCase
import warnings
from logbook import NullHandler, Logger
import numpy as np
import pandas as pd
from pandas.core.common import PerformanceWarning
from six import with_metaclass, iteritems, itervalues, PY2
import responses
from toolz import flip, groupby, merge
from trading_calendars import get_calendar, register_calendar_alias
import h5py
import zipline
from zipline.algorithm import TradingAlgorithm
from zipline.assets import Equity, Future
from zipline.assets.continuous_futures import CHAIN_PREDICATES
from zipline.data.benchmarks import get_benchmark_returns_from_file
from zipline.data.fx import DEFAULT_FX_RATE
from zipline.finance.asset_restrictions import NoRestrictions
from zipline.utils.memoize import classlazyval
from zipline.pipeline import SimplePipelineEngine
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.data.testing import TestingDataSet
from zipline.pipeline.domain import GENERIC, US_EQUITIES
from zipline.pipeline.loaders import USEquityPricingLoader
from zipline.pipeline.loaders.testing import make_seeded_random_loader
from zipline.protocol import BarData
from zipline.utils.compat import ExitStack
from zipline.utils.paths import ensure_directory, ensure_directory_containing
from .core import create_daily_bar_data, create_minute_bar_data, make_simple_equity_info, tmp_asset_finder, tmp_dir, write_hdf5_daily_bars
from .debug import debug_mro_failure
from ..data.adjustments import SQLiteAdjustmentReader, SQLiteAdjustmentWriter
from ..data.bcolz_daily_bars import BcolzDailyBarReader, BcolzDailyBarWriter
from ..data.data_portal import DataPortal, DEFAULT_MINUTE_HISTORY_PREFETCH, DEFAULT_DAILY_HISTORY_PREFETCH
from ..data.fx import InMemoryFXRateReader, HDF5FXRateReader, HDF5FXRateWriter
from ..data.hdf5_daily_bars import HDF5DailyBarReader, HDF5DailyBarWriter, MultiCountryDailyBarReader
from ..data.minute_bars import BcolzMinuteBarReader, BcolzMinuteBarWriter, US_EQUITIES_MINUTES_PER_DAY, FUTURES_MINUTES_PER_DAY
from ..data.resample import minute_frame_to_session_frame, MinuteResampleSessionBarReader
from ..finance.trading import SimulationParameters
from ..utils.classproperty import classproperty
from ..utils.final import FinalMeta, final
from ..utils.memoize import remember_last
zipline_dir = os.path.dirname(zipline.__file__)

class DebugMROMeta(FinalMeta):
    """Metaclass that helps debug MRO resolution errors.
    """

    def __new__(mcls, name, bases, clsdict):
        if False:
            return 10
        try:
            return super(DebugMROMeta, mcls).__new__(mcls, name, bases, clsdict)
        except TypeError as e:
            if '(MRO)' in str(e):
                msg = debug_mro_failure(name, bases)
                raise TypeError(msg)
            else:
                raise

class ZiplineTestCase(with_metaclass(DebugMROMeta, TestCase)):
    """
    Shared extensions to core unittest.TestCase.

    Overrides the default unittest setUp/tearDown functions with versions that
    use ExitStack to correctly clean up resources, even in the face of
    exceptions that occur during setUp/setUpClass.

    Subclasses **should not override setUp or setUpClass**!

    Instead, they should implement `init_instance_fixtures` for per-test-method
    resources, and `init_class_fixtures` for per-class resources.

    Resources that need to be cleaned up should be registered using
    either `enter_{class,instance}_context` or `add_{class,instance}_callback}.
    """
    _in_setup = False

    @final
    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        cls._static_class_attributes = set(vars(cls))
        cls._class_teardown_stack = ExitStack()
        try:
            cls._base_init_fixtures_was_called = False
            cls.init_class_fixtures()
            assert cls._base_init_fixtures_was_called, 'ZiplineTestCase.init_class_fixtures() was not called.\nThis probably means that you overrode init_class_fixtures without calling super().'
        except BaseException:
            cls.tearDownClass()
            raise

    @classmethod
    def init_class_fixtures(cls):
        if False:
            i = 10
            return i + 15
        '\n        Override and implement this classmethod to register resources that\n        should be created and/or torn down on a per-class basis.\n\n        Subclass implementations of this should always invoke this with super()\n        to ensure that fixture mixins work properly.\n        '
        if cls._in_setup:
            raise ValueError('Called init_class_fixtures from init_instance_fixtures. Did you write super(..., self).init_class_fixtures() instead of super(..., self).init_instance_fixtures()?')
        cls._base_init_fixtures_was_called = True

    @final
    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        stack = cls._class_teardown_stack
        for name in set(vars(cls)) - cls._static_class_attributes:
            delattr(cls, name)
        stack.close()

    @final
    @classmethod
    def enter_class_context(cls, context_manager):
        if False:
            print('Hello World!')
        '\n        Enter a context manager to be exited during the tearDownClass\n        '
        if cls._in_setup:
            raise ValueError('Attempted to enter a class context in init_instance_fixtures.\nDid you mean to call enter_instance_context?')
        return cls._class_teardown_stack.enter_context(context_manager)

    @final
    @classmethod
    def add_class_callback(cls, callback, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Register a callback to be executed during tearDownClass.\n\n        Parameters\n        ----------\n        callback : callable\n            The callback to invoke at the end of the test suite.\n        '
        if cls._in_setup:
            raise ValueError('Attempted to add a class callback in init_instance_fixtures.\nDid you mean to call add_instance_callback?')
        return cls._class_teardown_stack.callback(callback, *args, **kwargs)

    @final
    def setUp(self):
        if False:
            print('Hello World!')
        type(self)._in_setup = True
        self._pre_setup_attrs = set(vars(self))
        self._instance_teardown_stack = ExitStack()
        try:
            self._init_instance_fixtures_was_called = False
            self.init_instance_fixtures()
            assert self._init_instance_fixtures_was_called, 'ZiplineTestCase.init_instance_fixtures() was not called.\nThis probably means that you overrode init_instance_fixtures without calling super().'
        except BaseException:
            self.tearDown()
            raise
        finally:
            type(self)._in_setup = False

    def init_instance_fixtures(self):
        if False:
            print('Hello World!')
        self._init_instance_fixtures_was_called = True

    @final
    def tearDown(self):
        if False:
            while True:
                i = 10
        stack = self._instance_teardown_stack
        for attr in set(vars(self)) - self._pre_setup_attrs:
            delattr(self, attr)
        stack.close()

    @final
    def enter_instance_context(self, context_manager):
        if False:
            return 10
        '\n        Enter a context manager that should be exited during tearDown.\n        '
        return self._instance_teardown_stack.enter_context(context_manager)

    @final
    def add_instance_callback(self, callback):
        if False:
            for i in range(10):
                print('nop')
        '\n        Register a callback to be executed during tearDown.\n\n        Parameters\n        ----------\n        callback : callable\n            The callback to invoke at the end of each test.\n        '
        return self._instance_teardown_stack.callback(callback)
    if PY2:

        def assertRaisesRegex(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return self.assertRaisesRegexp(*args, **kwargs)

def alias(attr_name):
    if False:
        return 10
    "Make a fixture attribute an alias of another fixture's attribute by\n    default.\n\n    Parameters\n    ----------\n    attr_name : str\n        The name of the attribute to alias.\n\n    Returns\n    -------\n    p : classproperty\n        A class property that does the property aliasing.\n\n    Examples\n    --------\n    >>> class C(object):\n    ...     attr = 1\n    ...\n    >>> class D(C):\n    ...     attr_alias = alias('attr')\n    ...\n    >>> D.attr\n    1\n    >>> D.attr_alias\n    1\n    >>> class E(D):\n    ...     attr_alias = 2\n    ...\n    >>> E.attr\n    1\n    >>> E.attr_alias\n    2\n    "
    return classproperty(flip(getattr, attr_name))

class WithDefaultDateBounds(with_metaclass(DebugMROMeta, object)):
    """
    ZiplineTestCase mixin which makes it possible to synchronize date bounds
    across fixtures.

    This fixture should always be the last fixture in bases of any fixture or
    test case that uses it.

    Attributes
    ----------
    START_DATE : datetime
    END_DATE : datetime
        The date bounds to be used for fixtures that want to have consistent
        dates.
    """
    START_DATE = pd.Timestamp('2006-01-03', tz='utc')
    END_DATE = pd.Timestamp('2006-12-29', tz='utc')

class WithLogger(object):
    """
    ZiplineTestCase mixin providing cls.log_handler as an instance-level
    fixture.

    After init_instance_fixtures has been called `self.log_handler` will be a
    new ``logbook.NullHandler``.

    Methods
    -------
    make_log_handler() -> logbook.LogHandler
        A class method which constructs the new log handler object. By default
        this will construct a ``NullHandler``.
    """
    make_log_handler = NullHandler

    @classmethod
    def init_class_fixtures(cls):
        if False:
            return 10
        super(WithLogger, cls).init_class_fixtures()
        cls.log = Logger()
        cls.log_handler = cls.enter_class_context(cls.make_log_handler().applicationbound())

class WithAssetFinder(WithDefaultDateBounds):
    """
    ZiplineTestCase mixin providing cls.asset_finder as a class-level fixture.

    After init_class_fixtures has been called, `cls.asset_finder` is populated
    with an AssetFinder.

    Attributes
    ----------
    ASSET_FINDER_EQUITY_SIDS : iterable[int]
        The default sids to construct equity data for.
    ASSET_FINDER_EQUITY_SYMBOLS : iterable[str]
        The default symbols to use for the equities.
    ASSET_FINDER_EQUITY_START_DATE : datetime
        The default start date to create equity data for. This defaults to
        ``START_DATE``.
    ASSET_FINDER_EQUITY_END_DATE : datetime
        The default end date to create equity data for. This defaults to
        ``END_DATE``.
    ASSET_FINDER_EQUITY_NAMES: iterable[str]
        The default names to use for the equities.
    ASSET_FINDER_EQUITY_EXCHANGE : str
        The default exchange to assign each equity.
    ASSET_FINDER_COUNTRY_CODE : str
        The default country code to assign each exchange.

    Methods
    -------
    make_equity_info() -> pd.DataFrame
        A class method which constructs the dataframe of equity info to write
        to the class's asset db. By default this is empty.
    make_futures_info() -> pd.DataFrame
        A class method which constructs the dataframe of futures contract info
        to write to the class's asset db. By default this is empty.
    make_exchanges_info() -> pd.DataFrame
        A class method which constructs the dataframe of exchange information
        to write to the class's assets db. By default this is empty.
    make_root_symbols_info() -> pd.DataFrame
        A class method which constructs the dataframe of root symbols
        information to write to the class's assets db. By default this is
        empty.
    make_asset_finder_db_url() -> string
        A class method which returns the URL at which to create the SQLAlchemy
        engine. By default provides a URL for an in-memory database.
    make_asset_finder() -> pd.DataFrame
        A class method which constructs the actual asset finder object to use
        for the class. If this method is overridden then the ``make_*_info``
        methods may not be respected.

    See Also
    --------
    zipline.testing.make_simple_equity_info
    zipline.testing.make_jagged_equity_info
    zipline.testing.make_rotating_equity_info
    zipline.testing.make_future_info
    zipline.testing.make_commodity_future_info
    """
    ASSET_FINDER_EQUITY_SIDS = (ord('A'), ord('B'), ord('C'))
    ASSET_FINDER_EQUITY_SYMBOLS = None
    ASSET_FINDER_EQUITY_NAMES = None
    ASSET_FINDER_EQUITY_EXCHANGE = 'TEST'
    ASSET_FINDER_EQUITY_START_DATE = alias('START_DATE')
    ASSET_FINDER_EQUITY_END_DATE = alias('END_DATE')
    ASSET_FINDER_FUTURE_CHAIN_PREDICATES = CHAIN_PREDICATES
    ASSET_FINDER_COUNTRY_CODE = '??'

    @classmethod
    def _make_info(cls, *args):
        if False:
            for i in range(10):
                print('nop')
        return None
    make_futures_info = _make_info
    make_exchanges_info = _make_info
    make_root_symbols_info = _make_info
    make_equity_supplementary_mappings = _make_info
    del _make_info

    @classmethod
    def make_equity_info(cls):
        if False:
            for i in range(10):
                print('nop')
        return make_simple_equity_info(cls.ASSET_FINDER_EQUITY_SIDS, cls.ASSET_FINDER_EQUITY_START_DATE, cls.ASSET_FINDER_EQUITY_END_DATE, cls.ASSET_FINDER_EQUITY_SYMBOLS, cls.ASSET_FINDER_EQUITY_NAMES, cls.ASSET_FINDER_EQUITY_EXCHANGE)

    @classmethod
    def make_asset_finder_db_url(cls):
        if False:
            return 10
        return 'sqlite:///:memory:'

    @classmethod
    def make_asset_finder(cls):
        if False:
            print('Hello World!')
        'Returns a new AssetFinder\n\n        Returns\n        -------\n        asset_finder : zipline.assets.AssetFinder\n        '
        equities = cls.make_equity_info()
        futures = cls.make_futures_info()
        root_symbols = cls.make_root_symbols_info()
        exchanges = cls.make_exchanges_info(equities, futures, root_symbols)
        if exchanges is None:
            exchange_names = [df['exchange'] for df in (equities, futures, root_symbols) if df is not None]
            if exchange_names:
                exchanges = pd.DataFrame({'exchange': pd.concat(exchange_names).unique(), 'country_code': cls.ASSET_FINDER_COUNTRY_CODE})
        return cls.enter_class_context(tmp_asset_finder(url=cls.make_asset_finder_db_url(), equities=equities, futures=futures, exchanges=exchanges, root_symbols=root_symbols, equity_supplementary_mappings=cls.make_equity_supplementary_mappings(), future_chain_predicates=cls.ASSET_FINDER_FUTURE_CHAIN_PREDICATES))

    @classmethod
    def init_class_fixtures(cls):
        if False:
            i = 10
            return i + 15
        super(WithAssetFinder, cls).init_class_fixtures()
        cls.asset_finder = cls.make_asset_finder()

    @classlazyval
    def all_assets(cls):
        if False:
            while True:
                i = 10
        'A list of Assets for all sids in cls.asset_finder.\n        '
        return cls.asset_finder.retrieve_all(cls.asset_finder.sids)

    @classlazyval
    def exchange_names(cls):
        if False:
            while True:
                i = 10
        'A list of canonical exchange names for all exchanges in this suite.\n        '
        infos = itervalues(cls.asset_finder.exchange_info)
        return sorted((i.canonical_name for i in infos))

    @classlazyval
    def assets_by_calendar(cls):
        if False:
            return 10
        'A dict from calendar -> list of assets with that calendar.\n        '
        return groupby(lambda a: get_calendar(a.exchange), cls.all_assets)

    @classlazyval
    def all_calendars(cls):
        if False:
            i = 10
            return i + 15
        'A list of all calendars for assets in this test suite.\n        '
        return list(cls.assets_by_calendar)

class WithTradingCalendars(object):
    """
    ZiplineTestCase mixin providing cls.trading_calendar,
    cls.all_trading_calendars, cls.trading_calendar_for_asset_type as a
    class-level fixture.

    After ``init_class_fixtures`` has been called:
    - `cls.trading_calendar` is populated with a default of the nyse trading
    calendar for compatibility with existing tests
    - `cls.all_trading_calendars` is populated with the trading calendars
    keyed by name,
    - `cls.trading_calendar_for_asset_type` is populated with the trading
    calendars keyed by the asset type which uses the respective calendar.

    Attributes
    ----------
    TRADING_CALENDAR_STRS : iterable
        iterable of identifiers of the calendars to use.
    TRADING_CALENDAR_FOR_ASSET_TYPE : dict
        A dictionary which maps asset type names to the calendar associated
        with that asset type.
    """
    TRADING_CALENDAR_STRS = ('NYSE',)
    TRADING_CALENDAR_FOR_ASSET_TYPE = {Equity: 'NYSE', Future: 'us_futures'}
    TRADING_CALENDAR_PRIMARY_CAL = 'NYSE'

    @classmethod
    def init_class_fixtures(cls):
        if False:
            i = 10
            return i + 15
        super(WithTradingCalendars, cls).init_class_fixtures()
        cls.trading_calendars = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PerformanceWarning)
            for cal_str in set(cls.TRADING_CALENDAR_STRS) | {cls.TRADING_CALENDAR_PRIMARY_CAL}:
                calendar = get_calendar(cal_str)
                setattr(cls, '{0}_calendar'.format(cal_str.lower()), calendar)
                cls.trading_calendars[cal_str] = calendar
            type_to_cal = iteritems(cls.TRADING_CALENDAR_FOR_ASSET_TYPE)
            for (asset_type, cal_str) in type_to_cal:
                calendar = get_calendar(cal_str)
                cls.trading_calendars[asset_type] = calendar
        cls.trading_calendar = cls.trading_calendars[cls.TRADING_CALENDAR_PRIMARY_CAL]
STATIC_BENCHMARK_PATH = os.path.join(zipline_dir, 'resources', 'market_data', 'SPY_benchmark.csv')

@remember_last
def read_checked_in_benchmark_data():
    if False:
        print('Hello World!')
    return get_benchmark_returns_from_file(STATIC_BENCHMARK_PATH)

class WithBenchmarkReturns(WithDefaultDateBounds, WithTradingCalendars):
    """
    ZiplineTestCase mixin providing cls.benchmark_returns as a class-level
    attribute.
    """
    _default_treasury_curves = None

    @classproperty
    def BENCHMARK_RETURNS(cls):
        if False:
            i = 10
            return i + 15
        benchmark_returns = read_checked_in_benchmark_data()
        static_start_date = benchmark_returns.index[0].date()
        static_end_date = benchmark_returns.index[-1].date()
        warning_message = 'The WithBenchmarkReturns fixture uses static data between {static_start} and {static_end}. To use a start and end date of {given_start} and {given_end} you will have to update the file in {benchmark_path} to include the missing dates.'.format(static_start=static_start_date, static_end=static_end_date, given_start=cls.START_DATE.date(), given_end=cls.END_DATE.date(), benchmark_path=STATIC_BENCHMARK_PATH)
        if cls.START_DATE.date() < static_start_date or cls.END_DATE.date() > static_end_date:
            raise AssertionError(warning_message)
        return benchmark_returns

class WithSimParams(WithDefaultDateBounds):
    """
    ZiplineTestCase mixin providing cls.sim_params as a class level fixture.

    Attributes
    ----------
    SIM_PARAMS_CAPITAL_BASE : float
    SIM_PARAMS_DATA_FREQUENCY : {'daily', 'minute'}
    SIM_PARAMS_EMISSION_RATE : {'daily', 'minute'}
        Forwarded to ``SimulationParameters``.

    SIM_PARAMS_START : datetime
    SIM_PARAMS_END : datetime
        Forwarded to ``SimulationParameters``. If not
        explicitly overridden these will be ``START_DATE`` and ``END_DATE``

    Methods
    -------
    make_simparams(**overrides)
        Construct a ``SimulationParameters`` using the defaults defined by
        fixture configuration attributes. Any parameters to
        ``SimulationParameters`` can be overridden by passing them by keyword.

    See Also
    --------
    zipline.finance.trading.SimulationParameters
    """
    SIM_PARAMS_CAPITAL_BASE = 100000.0
    SIM_PARAMS_DATA_FREQUENCY = 'daily'
    SIM_PARAMS_EMISSION_RATE = 'daily'
    SIM_PARAMS_START = alias('START_DATE')
    SIM_PARAMS_END = alias('END_DATE')

    @classmethod
    def make_simparams(cls, **overrides):
        if False:
            for i in range(10):
                print('nop')
        kwargs = dict(start_session=cls.SIM_PARAMS_START, end_session=cls.SIM_PARAMS_END, capital_base=cls.SIM_PARAMS_CAPITAL_BASE, data_frequency=cls.SIM_PARAMS_DATA_FREQUENCY, emission_rate=cls.SIM_PARAMS_EMISSION_RATE, trading_calendar=cls.trading_calendar)
        kwargs.update(overrides)
        return SimulationParameters(**kwargs)

    @classmethod
    def init_class_fixtures(cls):
        if False:
            return 10
        super(WithSimParams, cls).init_class_fixtures()
        cls.sim_params = cls.make_simparams()

class WithTradingSessions(WithDefaultDateBounds, WithTradingCalendars):
    """
    ZiplineTestCase mixin providing cls.trading_days, cls.all_trading_sessions
    as a class-level fixture.

    After init_class_fixtures has been called, `cls.all_trading_sessions`
    is populated with a dictionary of calendar name to the DatetimeIndex
    containing the calendar trading days ranging from:

    (DATA_MAX_DAY - (cls.TRADING_DAY_COUNT) -> DATA_MAX_DAY)

    `cls.trading_days`, for compatibility with existing tests which make the
    assumption that trading days are equity only, defaults to the nyse trading
    sessions.

    Attributes
    ----------
    DATA_MAX_DAY : datetime
        The most recent trading day in the calendar.
    TRADING_DAY_COUNT : int
        The number of days to put in the calendar. The default value of
        ``TRADING_DAY_COUNT`` is 126 (half a trading-year). Inheritors can
        override TRADING_DAY_COUNT to request more or less data.
    """
    DATA_MIN_DAY = alias('START_DATE')
    DATA_MAX_DAY = alias('END_DATE')
    trading_days = alias('nyse_sessions')

    @classmethod
    def init_class_fixtures(cls):
        if False:
            while True:
                i = 10
        super(WithTradingSessions, cls).init_class_fixtures()
        cls.trading_sessions = {}
        for cal_str in cls.TRADING_CALENDAR_STRS:
            trading_calendar = cls.trading_calendars[cal_str]
            sessions = trading_calendar.sessions_in_range(cls.DATA_MIN_DAY, cls.DATA_MAX_DAY)
            setattr(cls, '{0}_sessions'.format(cal_str.lower()), sessions)
            cls.trading_sessions[cal_str] = sessions

class WithTmpDir(object):
    """
    ZiplineTestCase mixing providing cls.tmpdir as a class-level fixture.

    After init_class_fixtures has been called, `cls.tmpdir` is populated with
    a `testfixtures.TempDirectory` object whose path is `cls.TMP_DIR_PATH`.

    Attributes
    ----------
    TMP_DIR_PATH : str
        The path to the new directory to create. By default this is None
        which will create a unique directory in /tmp.
    """
    TMP_DIR_PATH = None

    @classmethod
    def init_class_fixtures(cls):
        if False:
            while True:
                i = 10
        super(WithTmpDir, cls).init_class_fixtures()
        cls.tmpdir = cls.enter_class_context(tmp_dir(path=cls.TMP_DIR_PATH))

class WithInstanceTmpDir(object):
    """
    ZiplineTestCase mixing providing self.tmpdir as an instance-level fixture.

    After init_instance_fixtures has been called, `self.tmpdir` is populated
    with a `testfixtures.TempDirectory` object whose path is
    `cls.TMP_DIR_PATH`.

    Attributes
    ----------
    INSTANCE_TMP_DIR_PATH : str
        The path to the new directory to create. By default this is None
        which will create a unique directory in /tmp.
    """
    INSTANCE_TMP_DIR_PATH = None

    def init_instance_fixtures(self):
        if False:
            for i in range(10):
                print('nop')
        super(WithInstanceTmpDir, self).init_instance_fixtures()
        self.instance_tmpdir = self.enter_instance_context(tmp_dir(path=self.INSTANCE_TMP_DIR_PATH))

class WithEquityDailyBarData(WithAssetFinder, WithTradingCalendars):
    """
    ZiplineTestCase mixin providing cls.make_equity_daily_bar_data.

    Attributes
    ----------
    EQUITY_DAILY_BAR_START_DATE : Timestamp
        The date at to which to start creating data. This defaults to
        ``START_DATE``.
    EQUITY_DAILY_BAR_END_DATE = Timestamp
        The end date up to which to create data. This defaults to ``END_DATE``.
    EQUITY_DAILY_BAR_SOURCE_FROM_MINUTE : bool
        If this flag is set, `make_equity_daily_bar_data` will read data from
        the minute bars defined by `WithEquityMinuteBarData`.
        The current default is `False`, but could be `True` in the future.
    EQUITY_DAILY_BAR_COUNTRY_CODES : tuple
        The countres to create data for. By default this is populated
        with all of the countries present in the asset finder.

    Methods
    -------
    make_equity_daily_bar_data(country_code, sids)
    make_equity_daily_bar_currency_codes(country_code, sids)

    See Also
    --------
    WithEquityMinuteBarData
    zipline.testing.create_daily_bar_data
    """
    EQUITY_DAILY_BAR_START_DATE = alias('START_DATE')
    EQUITY_DAILY_BAR_END_DATE = alias('END_DATE')
    EQUITY_DAILY_BAR_SOURCE_FROM_MINUTE = None

    @classproperty
    def EQUITY_DAILY_BAR_LOOKBACK_DAYS(cls):
        if False:
            return 10
        if cls.EQUITY_DAILY_BAR_SOURCE_FROM_MINUTE:
            return cls.EQUITY_MINUTE_BAR_LOOKBACK_DAYS
        else:
            return 0

    @classproperty
    def EQUITY_DAILY_BAR_COUNTRY_CODES(cls):
        if False:
            return 10
        return cls.asset_finder.country_codes

    @classmethod
    def _make_equity_daily_bar_from_minute(cls):
        if False:
            while True:
                i = 10
        assert issubclass(cls, WithEquityMinuteBarData), "Can't source daily data from minute without minute data!"
        assets = cls.asset_finder.retrieve_all(cls.asset_finder.equities_sids)
        minute_data = dict(cls.make_equity_minute_bar_data())
        for asset in assets:
            yield (asset.sid, minute_frame_to_session_frame(minute_data[asset.sid], cls.trading_calendars[Equity]))

    @classmethod
    def make_equity_daily_bar_data(cls, country_code, sids):
        if False:
            i = 10
            return i + 15
        "\n        Create daily pricing data.\n\n        Parameters\n        ----------\n        country_code : str\n            An ISO 3166 alpha-2 country code. Data should be created for\n            this country.\n        sids : tuple[int]\n            The sids to include in the data.\n\n        Yields\n        ------\n        (int, pd.DataFrame)\n            A sid, dataframe pair to be passed to a daily bar writer.\n            The dataframe should be indexed by date, with columns of\n            ('open', 'high', 'low', 'close', 'volume', 'day', & 'id').\n        "
        if cls.EQUITY_DAILY_BAR_SOURCE_FROM_MINUTE:
            return cls._make_equity_daily_bar_from_minute()
        else:
            return create_daily_bar_data(cls.equity_daily_bar_days, sids)

    @classmethod
    def make_equity_daily_bar_currency_codes(cls, country_code, sids):
        if False:
            i = 10
            return i + 15
        "Create listing currencies.\n\n        Default is to list all assets in USD.\n\n        Parameters\n        ----------\n        country_code : str\n            An ISO 3166 alpha-2 country code. Data should be created for\n            this country.\n        sids : tuple[int]\n            The sids to include in the data.\n\n        Returns\n        -------\n        currency_codes : pd.Series[int, str]\n            Map from sids to currency for that sid's prices.\n        "
        return pd.Series(index=list(sids), data='USD')

    @classmethod
    def init_class_fixtures(cls):
        if False:
            i = 10
            return i + 15
        super(WithEquityDailyBarData, cls).init_class_fixtures()
        trading_calendar = cls.trading_calendars[Equity]
        if trading_calendar.is_session(cls.EQUITY_DAILY_BAR_START_DATE):
            first_session = cls.EQUITY_DAILY_BAR_START_DATE
        else:
            first_session = trading_calendar.minute_to_session_label(pd.Timestamp(cls.EQUITY_DAILY_BAR_START_DATE))
        if cls.EQUITY_DAILY_BAR_LOOKBACK_DAYS > 0:
            first_session = trading_calendar.sessions_window(first_session, -1 * cls.EQUITY_DAILY_BAR_LOOKBACK_DAYS)[0]
        days = trading_calendar.sessions_in_range(first_session, cls.EQUITY_DAILY_BAR_END_DATE)
        cls.equity_daily_bar_days = days

class WithFutureDailyBarData(WithAssetFinder, WithTradingCalendars):
    """
    ZiplineTestCase mixin providing cls.make_future_daily_bar_data.

    Attributes
    ----------
    FUTURE_DAILY_BAR_START_DATE : Timestamp
        The date at to which to start creating data. This defaults to
        ``START_DATE``.
    FUTURE_DAILY_BAR_END_DATE = Timestamp
        The end date up to which to create data. This defaults to ``END_DATE``.
    FUTURE_DAILY_BAR_SOURCE_FROM_MINUTE : bool
        If this flag is set, `make_future_daily_bar_data` will read data from
        the minute bars defined by `WithFutureMinuteBarData`.
        The current default is `False`, but could be `True` in the future.

    Methods
    -------
    make_future_daily_bar_data() -> iterable[(int, pd.DataFrame)]
        A class method that returns an iterator of (sid, dataframe) pairs
        which will be written to the bcolz files that the class's
        ``BcolzDailyBarReader`` will read from. By default this creates
        some simple synthetic data with
        :func:`~zipline.testing.create_daily_bar_data`

    See Also
    --------
    WithFutureMinuteBarData
    zipline.testing.create_daily_bar_data
    """
    FUTURE_DAILY_BAR_USE_FULL_CALENDAR = False
    FUTURE_DAILY_BAR_START_DATE = alias('START_DATE')
    FUTURE_DAILY_BAR_END_DATE = alias('END_DATE')
    FUTURE_DAILY_BAR_SOURCE_FROM_MINUTE = None

    @classproperty
    def FUTURE_DAILY_BAR_LOOKBACK_DAYS(cls):
        if False:
            while True:
                i = 10
        if cls.FUTURE_DAILY_BAR_SOURCE_FROM_MINUTE:
            return cls.FUTURE_MINUTE_BAR_LOOKBACK_DAYS
        else:
            return 0

    @classmethod
    def _make_future_daily_bar_from_minute(cls):
        if False:
            return 10
        assert issubclass(cls, WithFutureMinuteBarData), "Can't source daily data from minute without minute data!"
        assets = cls.asset_finder.retrieve_all(cls.asset_finder.futures_sids)
        minute_data = dict(cls.make_future_minute_bar_data())
        for asset in assets:
            yield (asset.sid, minute_frame_to_session_frame(minute_data[asset.sid], cls.trading_calendars[Future]))

    @classmethod
    def make_future_daily_bar_data(cls):
        if False:
            while True:
                i = 10
        if cls.FUTURE_DAILY_BAR_SOURCE_FROM_MINUTE:
            return cls._make_future_daily_bar_from_minute()
        else:
            return create_daily_bar_data(cls.future_daily_bar_days, cls.asset_finder.futures_sids)

    @classmethod
    def init_class_fixtures(cls):
        if False:
            return 10
        super(WithFutureDailyBarData, cls).init_class_fixtures()
        trading_calendar = cls.trading_calendars[Future]
        if cls.FUTURE_DAILY_BAR_USE_FULL_CALENDAR:
            days = trading_calendar.all_sessions
        else:
            if trading_calendar.is_session(cls.FUTURE_DAILY_BAR_START_DATE):
                first_session = cls.FUTURE_DAILY_BAR_START_DATE
            else:
                first_session = trading_calendar.minute_to_session_label(pd.Timestamp(cls.FUTURE_DAILY_BAR_START_DATE))
            if cls.FUTURE_DAILY_BAR_LOOKBACK_DAYS > 0:
                first_session = trading_calendar.sessions_window(first_session, -1 * cls.FUTURE_DAILY_BAR_LOOKBACK_DAYS)[0]
            days = trading_calendar.sessions_in_range(first_session, cls.FUTURE_DAILY_BAR_END_DATE)
        cls.future_daily_bar_days = days

class WithBcolzEquityDailyBarReader(WithEquityDailyBarData, WithTmpDir):
    """
    ZiplineTestCase mixin providing cls.bcolz_daily_bar_path,
    cls.bcolz_daily_bar_ctable, and cls.bcolz_equity_daily_bar_reader
    class level fixtures.

    After init_class_fixtures has been called:
    - `cls.bcolz_daily_bar_path` is populated with
      `cls.tmpdir.getpath(cls.BCOLZ_DAILY_BAR_PATH)`.
    - `cls.bcolz_daily_bar_ctable` is populated with data returned from
      `cls.make_equity_daily_bar_data`. By default this calls
      :func:`zipline.pipeline.loaders.synthetic.make_equity_daily_bar_data`.
    - `cls.bcolz_equity_daily_bar_reader` is a daily bar reader
       pointing to the directory that was just written to.

    Attributes
    ----------
    BCOLZ_DAILY_BAR_PATH : str
        The path inside the tmpdir where this will be written.
    EQUITY_DAILY_BAR_LOOKBACK_DAYS : int
        The number of days of data to add before the first day. This is used
        when a test needs to use history, in which case this should be set to
        the largest history window that will be
        requested.
    EQUITY_DAILY_BAR_USE_FULL_CALENDAR : bool
        If this flag is set the ``equity_daily_bar_days`` will be the full
        set of trading days from the trading environment. This flag overrides
        ``EQUITY_DAILY_BAR_LOOKBACK_DAYS``.
    BCOLZ_DAILY_BAR_READ_ALL_THRESHOLD : int
        If this flag is set, use the value as the `read_all_threshold`
        parameter to BcolzDailyBarReader, otherwise use the default
        value.
    EQUITY_DAILY_BAR_SOURCE_FROM_MINUTE : bool
        If this flag is set, `make_equity_daily_bar_data` will read data from
        the minute bar reader defined by a `WithBcolzEquityMinuteBarReader`.

    Methods
    -------
    make_bcolz_daily_bar_rootdir_path() -> string
        A class method that returns the path for the rootdir of the daily
        bars ctable. By default this is a subdirectory BCOLZ_DAILY_BAR_PATH in
        the shared temp directory.

    See Also
    --------
    WithBcolzEquityMinuteBarReader
    WithDataPortal
    zipline.testing.create_daily_bar_data
    """
    BCOLZ_DAILY_BAR_PATH = 'daily_equity_pricing.bcolz'
    BCOLZ_DAILY_BAR_READ_ALL_THRESHOLD = None
    BCOLZ_DAILY_BAR_COUNTRY_CODE = None
    EQUITY_DAILY_BAR_SOURCE_FROM_MINUTE = False
    _write_method_name = 'write'
    INVALID_DATA_BEHAVIOR = 'warn'

    @classproperty
    def BCOLZ_DAILY_BAR_COUNTRY_CODE(cls):
        if False:
            for i in range(10):
                print('nop')
        return cls.EQUITY_DAILY_BAR_COUNTRY_CODES[0]

    @classmethod
    def make_bcolz_daily_bar_rootdir_path(cls):
        if False:
            return 10
        return cls.tmpdir.makedir(cls.BCOLZ_DAILY_BAR_PATH)

    @classmethod
    def init_class_fixtures(cls):
        if False:
            i = 10
            return i + 15
        super(WithBcolzEquityDailyBarReader, cls).init_class_fixtures()
        cls.bcolz_daily_bar_path = p = cls.make_bcolz_daily_bar_rootdir_path()
        days = cls.equity_daily_bar_days
        sids = cls.asset_finder.equities_sids_for_country_code(cls.BCOLZ_DAILY_BAR_COUNTRY_CODE)
        trading_calendar = cls.trading_calendars[Equity]
        cls.bcolz_daily_bar_ctable = t = getattr(BcolzDailyBarWriter(p, trading_calendar, days[0], days[-1]), cls._write_method_name)(cls.make_equity_daily_bar_data(country_code=cls.BCOLZ_DAILY_BAR_COUNTRY_CODE, sids=sids), invalid_data_behavior=cls.INVALID_DATA_BEHAVIOR)
        if cls.BCOLZ_DAILY_BAR_READ_ALL_THRESHOLD is not None:
            cls.bcolz_equity_daily_bar_reader = BcolzDailyBarReader(t, cls.BCOLZ_DAILY_BAR_READ_ALL_THRESHOLD)
        else:
            cls.bcolz_equity_daily_bar_reader = BcolzDailyBarReader(t)

class WithBcolzFutureDailyBarReader(WithFutureDailyBarData, WithTmpDir):
    """
    ZiplineTestCase mixin providing cls.bcolz_daily_bar_path,
    cls.bcolz_daily_bar_ctable, and cls.bcolz_future_daily_bar_reader
    class level fixtures.

    After init_class_fixtures has been called:
    - `cls.bcolz_daily_bar_path` is populated with
      `cls.tmpdir.getpath(cls.BCOLZ_DAILY_BAR_PATH)`.
    - `cls.bcolz_daily_bar_ctable` is populated with data returned from
      `cls.make_future_daily_bar_data`. By default this calls
      :func:`zipline.pipeline.loaders.synthetic.make_future_daily_bar_data`.
    - `cls.bcolz_future_daily_bar_reader` is a daily bar reader
       pointing to the directory that was just written to.

    Attributes
    ----------
    BCOLZ_DAILY_BAR_PATH : str
        The path inside the tmpdir where this will be written.
    FUTURE_DAILY_BAR_LOOKBACK_DAYS : int
        The number of days of data to add before the first day. This is used
        when a test needs to use history, in which case this should be set to
        the largest history window that will be
        requested.
    FUTURE_DAILY_BAR_USE_FULL_CALENDAR : bool
        If this flag is set the ``future_daily_bar_days`` will be the full
        set of trading days from the trading environment. This flag overrides
        ``FUTURE_DAILY_BAR_LOOKBACK_DAYS``.
    BCOLZ_FUTURE_DAILY_BAR_READ_ALL_THRESHOLD : int
        If this flag is set, use the value as the `read_all_threshold`
        parameter to BcolzDailyBarReader, otherwise use the default
        value.
    FUTURE_DAILY_BAR_SOURCE_FROM_MINUTE : bool
        If this flag is set, `make_future_daily_bar_data` will read data from
        the minute bar reader defined by a `WithBcolzFutureMinuteBarReader`.

    Methods
    -------
    make_bcolz_daily_bar_rootdir_path() -> string
        A class method that returns the path for the rootdir of the daily
        bars ctable. By default this is a subdirectory BCOLZ_DAILY_BAR_PATH in
        the shared temp directory.

    See Also
    --------
    WithBcolzFutureMinuteBarReader
    WithDataPortal
    zipline.testing.create_daily_bar_data
    """
    BCOLZ_FUTURE_DAILY_BAR_PATH = 'daily_future_pricing.bcolz'
    BCOLZ_FUTURE_DAILY_BAR_READ_ALL_THRESHOLD = None
    FUTURE_DAILY_BAR_SOURCE_FROM_MINUTE = False
    BCOLZ_FUTURE_DAILY_BAR_INVALID_DATA_BEHAVIOR = 'warn'
    BCOLZ_FUTURE_DAILY_BAR_WRITE_METHOD_NAME = 'write'

    @classmethod
    def make_bcolz_future_daily_bar_rootdir_path(cls):
        if False:
            print('Hello World!')
        return cls.tmpdir.makedir(cls.BCOLZ_FUTURE_DAILY_BAR_PATH)

    @classmethod
    def init_class_fixtures(cls):
        if False:
            print('Hello World!')
        super(WithBcolzFutureDailyBarReader, cls).init_class_fixtures()
        p = cls.make_bcolz_future_daily_bar_rootdir_path()
        cls.future_bcolz_daily_bar_path = p
        days = cls.future_daily_bar_days
        trading_calendar = cls.trading_calendars[Future]
        cls.future_bcolz_daily_bar_ctable = t = getattr(BcolzDailyBarWriter(p, trading_calendar, days[0], days[-1]), cls.BCOLZ_FUTURE_DAILY_BAR_WRITE_METHOD_NAME)(cls.make_future_daily_bar_data(), invalid_data_behavior=cls.BCOLZ_FUTURE_DAILY_BAR_INVALID_DATA_BEHAVIOR)
        if cls.BCOLZ_FUTURE_DAILY_BAR_READ_ALL_THRESHOLD is not None:
            cls.bcolz_future_daily_bar_reader = BcolzDailyBarReader(t, cls.BCOLZ_FUTURE_DAILY_BAR_READ_ALL_THRESHOLD)
        else:
            cls.bcolz_future_daily_bar_reader = BcolzDailyBarReader(t)

class WithBcolzEquityDailyBarReaderFromCSVs(WithBcolzEquityDailyBarReader):
    """
    ZiplineTestCase mixin that provides
    cls.bcolz_equity_daily_bar_reader from a mapping of sids to CSV
    file paths.
    """
    _write_method_name = 'write_csvs'

def _trading_days_for_minute_bars(calendar, start_date, end_date, lookback_days):
    if False:
        i = 10
        return i + 15
    first_session = calendar.minute_to_session_label(start_date)
    if lookback_days > 0:
        first_session = calendar.sessions_window(first_session, -1 * lookback_days)[0]
    return calendar.sessions_in_range(first_session, end_date)

class WithWriteHDF5DailyBars(WithEquityDailyBarData, WithTmpDir):
    """
    Fixture class defining the capability of writing HDF5 daily bars to disk.

    Uses cls.make_equity_daily_bar_data (inherited from WithEquityDailyBarData)
    to determine the data to write.

    Methods
    -------
    write_hdf5_daily_bars(cls, path, country_codes)
        Creates an HDF5 file on disk and populates it with pricing data.

    Attributes
    ----------
    HDF5_DAILY_BAR_CHUNK_SIZE
    """
    HDF5_DAILY_BAR_CHUNK_SIZE = 30

    @classmethod
    def write_hdf5_daily_bars(cls, path, country_codes):
        if False:
            i = 10
            return i + 15
        '\n        Write HDF5 pricing data using an HDF5DailyBarWriter.\n\n        Parameters\n        ----------\n        path : str\n            Location (relative to cls.tmpdir) at which to write data.\n        country_codes : list[str]\n            List of country codes to write.\n\n        Returns\n        -------\n        written : h5py.File\n             A read-only h5py.File pointing at the written data. The returned\n             file is registered to be closed automatically during class\n             teardown.\n        '
        ensure_directory_containing(path)
        writer = HDF5DailyBarWriter(path, cls.HDF5_DAILY_BAR_CHUNK_SIZE)
        write_hdf5_daily_bars(writer, cls.asset_finder, country_codes, cls.make_equity_daily_bar_data, cls.make_equity_daily_bar_currency_codes)
        return cls.enter_class_context(writer.h5_file(mode='r'))

class WithHDF5EquityMultiCountryDailyBarReader(WithWriteHDF5DailyBars):
    """
    Fixture providing cls.hdf5_daily_bar_path and
    cls.hdf5_equity_daily_bar_reader class level fixtures.

    After init_class_fixtures has been called:
    - `cls.hdf5_daily_bar_path` is populated with
      `cls.tmpdir.getpath(cls.HDF5_DAILY_BAR_PATH)`.
    - The file at `cls.hdf5_daily_bar_path` is populated with data returned
      from `cls.make_equity_daily_bar_data`. By default this calls
      :func:`zipline.pipeline.loaders.synthetic.make_equity_daily_bar_data`.

    - `cls.hdf5_equity_daily_bar_reader` is a daily bar reader pointing
      to the file that was just written to.

    Attributes
    ----------
    HDF5_DAILY_BAR_PATH : str
        The path inside the tmpdir where this will be written.
    HDF5_DAILY_BAR_COUNTRY_CODE : str
        The ISO 3166 alpha-2 country code for the country to write/read.

    Methods
    -------
    make_hdf5_daily_bar_path() -> string
        A class method that returns the path for the rootdir of the daily
        bars ctable. By default this is a subdirectory HDF5_DAILY_BAR_PATH in
        the shared temp directory.

    See Also
    --------
    WithDataPortal
    zipline.testing.create_daily_bar_data
    """
    HDF5_DAILY_BAR_PATH = 'daily_equity_pricing.h5'
    HDF5_DAILY_BAR_COUNTRY_CODES = alias('EQUITY_DAILY_BAR_COUNTRY_CODES')

    @classmethod
    def make_hdf5_daily_bar_path(cls):
        if False:
            for i in range(10):
                print('nop')
        return cls.tmpdir.getpath(cls.HDF5_DAILY_BAR_PATH)

    @classmethod
    def init_class_fixtures(cls):
        if False:
            while True:
                i = 10
        super(WithHDF5EquityMultiCountryDailyBarReader, cls).init_class_fixtures()
        cls.hdf5_daily_bar_path = path = cls.make_hdf5_daily_bar_path()
        f = cls.write_hdf5_daily_bars(path, cls.HDF5_DAILY_BAR_COUNTRY_CODES)
        cls.single_country_hdf5_equity_daily_bar_readers = {country_code: HDF5DailyBarReader.from_file(f, country_code) for country_code in f}
        cls.hdf5_equity_daily_bar_reader = MultiCountryDailyBarReader(cls.single_country_hdf5_equity_daily_bar_readers)

class WithEquityMinuteBarData(WithAssetFinder, WithTradingCalendars):
    """
    ZiplineTestCase mixin providing cls.equity_minute_bar_days.

    After init_class_fixtures has been called:
    - `cls.equity_minute_bar_days` has the range over which data has been
       generated.

    Attributes
    ----------
    EQUITY_MINUTE_BAR_LOOKBACK_DAYS : int
        The number of days of data to add before the first day.
        This is used when a test needs to use history, in which case this
        should be set to the largest history window that will be requested.
    EQUITY_MINUTE_BAR_START_DATE : Timestamp
        The date at to which to start creating data. This defaults to
        ``START_DATE``.
    EQUITY_MINUTE_BAR_END_DATE = Timestamp
        The end date up to which to create data. This defaults to ``END_DATE``.

    Methods
    -------
    make_equity_minute_bar_data() -> iterable[(int, pd.DataFrame)]
        Classmethod producing an iterator of (sid, minute_data) pairs.
        The default implementation invokes
        zipline.testing.core.create_minute_bar_data.

    See Also
    --------
    WithEquityDailyBarData
    zipline.testing.create_minute_bar_data
    """
    EQUITY_MINUTE_BAR_LOOKBACK_DAYS = 0
    EQUITY_MINUTE_BAR_START_DATE = alias('START_DATE')
    EQUITY_MINUTE_BAR_END_DATE = alias('END_DATE')

    @classmethod
    def make_equity_minute_bar_data(cls):
        if False:
            while True:
                i = 10
        trading_calendar = cls.trading_calendars[Equity]
        return create_minute_bar_data(trading_calendar.minutes_for_sessions_in_range(cls.equity_minute_bar_days[0], cls.equity_minute_bar_days[-1]), cls.asset_finder.equities_sids)

    @classmethod
    def init_class_fixtures(cls):
        if False:
            return 10
        super(WithEquityMinuteBarData, cls).init_class_fixtures()
        trading_calendar = cls.trading_calendars[Equity]
        cls.equity_minute_bar_days = _trading_days_for_minute_bars(trading_calendar, pd.Timestamp(cls.EQUITY_MINUTE_BAR_START_DATE), pd.Timestamp(cls.EQUITY_MINUTE_BAR_END_DATE), cls.EQUITY_MINUTE_BAR_LOOKBACK_DAYS)

class WithFutureMinuteBarData(WithAssetFinder, WithTradingCalendars):
    """
    ZiplineTestCase mixin providing cls.future_minute_bar_days.

    After init_class_fixtures has been called:
    - `cls.future_minute_bar_days` has the range over which data has been
       generated.

    Attributes
    ----------
    FUTURE_MINUTE_BAR_LOOKBACK_DAYS : int
        The number of days of data to add before the first day.
        This is used when a test needs to use history, in which case this
        should be set to the largest history window that will be requested.
    FUTURE_MINUTE_BAR_START_DATE : Timestamp
        The date at to which to start creating data. This defaults to
        ``START_DATE``.
    FUTURE_MINUTE_BAR_END_DATE = Timestamp
        The end date up to which to create data. This defaults to ``END_DATE``.

    Methods
    -------
    make_future_minute_bar_data() -> iterable[(int, pd.DataFrame)]
        A class method that returns a dict mapping sid to dataframe
        which will be written to into the the format of the inherited
        class which writes the minute bar data for use by a reader.
        By default this creates some simple sythetic data with
        :func:`~zipline.testing.create_minute_bar_data`

    See Also
    --------
    zipline.testing.create_minute_bar_data
    """
    FUTURE_MINUTE_BAR_LOOKBACK_DAYS = 0
    FUTURE_MINUTE_BAR_START_DATE = alias('START_DATE')
    FUTURE_MINUTE_BAR_END_DATE = alias('END_DATE')

    @classmethod
    def make_future_minute_bar_data(cls):
        if False:
            while True:
                i = 10
        trading_calendar = get_calendar('us_futures')
        return create_minute_bar_data(trading_calendar.minutes_for_sessions_in_range(cls.future_minute_bar_days[0], cls.future_minute_bar_days[-1]), cls.asset_finder.futures_sids)

    @classmethod
    def init_class_fixtures(cls):
        if False:
            print('Hello World!')
        super(WithFutureMinuteBarData, cls).init_class_fixtures()
        trading_calendar = get_calendar('us_futures')
        cls.future_minute_bar_days = _trading_days_for_minute_bars(trading_calendar, pd.Timestamp(cls.FUTURE_MINUTE_BAR_START_DATE), pd.Timestamp(cls.FUTURE_MINUTE_BAR_END_DATE), cls.FUTURE_MINUTE_BAR_LOOKBACK_DAYS)

class WithBcolzEquityMinuteBarReader(WithEquityMinuteBarData, WithTmpDir):
    """
    ZiplineTestCase mixin providing cls.bcolz_minute_bar_path,
    cls.bcolz_minute_bar_ctable, and cls.bcolz_equity_minute_bar_reader
    class level fixtures.

    After init_class_fixtures has been called:
    - `cls.bcolz_minute_bar_path` is populated with
      `cls.tmpdir.getpath(cls.BCOLZ_MINUTE_BAR_PATH)`.
    - `cls.bcolz_minute_bar_ctable` is populated with data returned from
      `cls.make_equity_minute_bar_data`. By default this calls
      :func:`zipline.pipeline.loaders.synthetic.make_equity_minute_bar_data`.
    - `cls.bcolz_equity_minute_bar_reader` is a minute bar reader
       pointing to the directory that was just written to.

    Attributes
    ----------
    BCOLZ_MINUTE_BAR_PATH : str
        The path inside the tmpdir where this will be written.

    Methods
    -------
    make_bcolz_minute_bar_rootdir_path() -> string
        A class method that returns the path for the directory that contains
        the minute bar ctables. By default this is a subdirectory
        BCOLZ_MINUTE_BAR_PATH in the shared temp directory.

    See Also
    --------
    WithBcolzEquityDailyBarReader
    WithDataPortal
    zipline.testing.create_minute_bar_data
    """
    BCOLZ_EQUITY_MINUTE_BAR_PATH = 'minute_equity_pricing'

    @classmethod
    def make_bcolz_equity_minute_bar_rootdir_path(cls):
        if False:
            for i in range(10):
                print('nop')
        return cls.tmpdir.makedir(cls.BCOLZ_EQUITY_MINUTE_BAR_PATH)

    @classmethod
    def init_class_fixtures(cls):
        if False:
            while True:
                i = 10
        super(WithBcolzEquityMinuteBarReader, cls).init_class_fixtures()
        cls.bcolz_equity_minute_bar_path = p = cls.make_bcolz_equity_minute_bar_rootdir_path()
        days = cls.equity_minute_bar_days
        writer = BcolzMinuteBarWriter(p, cls.trading_calendars[Equity], days[0], days[-1], US_EQUITIES_MINUTES_PER_DAY)
        writer.write(cls.make_equity_minute_bar_data())
        cls.bcolz_equity_minute_bar_reader = BcolzMinuteBarReader(p)

class WithBcolzFutureMinuteBarReader(WithFutureMinuteBarData, WithTmpDir):
    """
    ZiplineTestCase mixin providing cls.bcolz_minute_bar_path,
    cls.bcolz_minute_bar_ctable, and cls.bcolz_equity_minute_bar_reader
    class level fixtures.

    After init_class_fixtures has been called:
    - `cls.bcolz_minute_bar_path` is populated with
      `cls.tmpdir.getpath(cls.BCOLZ_MINUTE_BAR_PATH)`.
    - `cls.bcolz_minute_bar_ctable` is populated with data returned from
      `cls.make_equity_minute_bar_data`. By default this calls
      :func:`zipline.pipeline.loaders.synthetic.make_equity_minute_bar_data`.
    - `cls.bcolz_equity_minute_bar_reader` is a minute bar reader
       pointing to the directory that was just written to.

    Attributes
    ----------
    BCOLZ_FUTURE_MINUTE_BAR_PATH : str
        The path inside the tmpdir where this will be written.

    Methods
    -------
    make_bcolz_minute_bar_rootdir_path() -> string
        A class method that returns the path for the directory that contains
        the minute bar ctables. By default this is a subdirectory
        BCOLZ_MINUTE_BAR_PATH in the shared temp directory.

    See Also
    --------
    WithBcolzEquityDailyBarReader
    WithDataPortal
    zipline.testing.create_minute_bar_data
    """
    BCOLZ_FUTURE_MINUTE_BAR_PATH = 'minute_future_pricing'
    OHLC_RATIOS_PER_SID = None

    @classmethod
    def make_bcolz_future_minute_bar_rootdir_path(cls):
        if False:
            return 10
        return cls.tmpdir.makedir(cls.BCOLZ_FUTURE_MINUTE_BAR_PATH)

    @classmethod
    def init_class_fixtures(cls):
        if False:
            while True:
                i = 10
        super(WithBcolzFutureMinuteBarReader, cls).init_class_fixtures()
        trading_calendar = get_calendar('us_futures')
        cls.bcolz_future_minute_bar_path = p = cls.make_bcolz_future_minute_bar_rootdir_path()
        days = cls.future_minute_bar_days
        writer = BcolzMinuteBarWriter(p, trading_calendar, days[0], days[-1], FUTURES_MINUTES_PER_DAY, ohlc_ratios_per_sid=cls.OHLC_RATIOS_PER_SID)
        writer.write(cls.make_future_minute_bar_data())
        cls.bcolz_future_minute_bar_reader = BcolzMinuteBarReader(p)

class WithConstantEquityMinuteBarData(WithEquityMinuteBarData):
    EQUITY_MINUTE_CONSTANT_LOW = 3.0
    EQUITY_MINUTE_CONSTANT_OPEN = 4.0
    EQUITY_MINUTE_CONSTANT_CLOSE = 5.0
    EQUITY_MINUTE_CONSTANT_HIGH = 6.0
    EQUITY_MINUTE_CONSTANT_VOLUME = 100.0

    @classmethod
    def make_equity_minute_bar_data(cls):
        if False:
            i = 10
            return i + 15
        trading_calendar = cls.trading_calendars[Equity]
        sids = cls.asset_finder.equities_sids
        minutes = trading_calendar.minutes_for_sessions_in_range(cls.equity_minute_bar_days[0], cls.equity_minute_bar_days[-1])
        frame = pd.DataFrame({'open': cls.EQUITY_MINUTE_CONSTANT_OPEN, 'high': cls.EQUITY_MINUTE_CONSTANT_HIGH, 'low': cls.EQUITY_MINUTE_CONSTANT_LOW, 'close': cls.EQUITY_MINUTE_CONSTANT_CLOSE, 'volume': cls.EQUITY_MINUTE_CONSTANT_VOLUME}, index=minutes)
        return ((sid, frame) for sid in sids)

class WithConstantFutureMinuteBarData(WithFutureMinuteBarData):
    FUTURE_MINUTE_CONSTANT_LOW = 3.0
    FUTURE_MINUTE_CONSTANT_OPEN = 4.0
    FUTURE_MINUTE_CONSTANT_CLOSE = 5.0
    FUTURE_MINUTE_CONSTANT_HIGH = 6.0
    FUTURE_MINUTE_CONSTANT_VOLUME = 100.0

    @classmethod
    def make_future_minute_bar_data(cls):
        if False:
            print('Hello World!')
        trading_calendar = cls.trading_calendars[Future]
        sids = cls.asset_finder.futures_sids
        minutes = trading_calendar.minutes_for_sessions_in_range(cls.future_minute_bar_days[0], cls.future_minute_bar_days[-1])
        frame = pd.DataFrame({'open': cls.FUTURE_MINUTE_CONSTANT_OPEN, 'high': cls.FUTURE_MINUTE_CONSTANT_HIGH, 'low': cls.FUTURE_MINUTE_CONSTANT_LOW, 'close': cls.FUTURE_MINUTE_CONSTANT_CLOSE, 'volume': cls.FUTURE_MINUTE_CONSTANT_VOLUME}, index=minutes)
        return ((sid, frame) for sid in sids)

class WithAdjustmentReader(WithBcolzEquityDailyBarReader):
    """
    ZiplineTestCase mixin providing cls.adjustment_reader as a class level
    fixture.

    After init_class_fixtures has been called, `cls.adjustment_reader` will be
    populated with a new SQLiteAdjustmentReader object. The data that will be
    written can be passed by overriding `make_{field}_data` where field may
    be `splits`, `mergers` `dividends`, or `stock_dividends`.
    The daily bar reader used for this adjustment reader may be customized
    by overriding `make_adjustment_writer_equity_daily_bar_reader`.
    This is useful to providing a `MockDailyBarReader`.

    Methods
    -------
    make_splits_data() -> pd.DataFrame
        A class method that returns a dataframe of splits data to write to the
        class's adjustment db. By default this is empty.
    make_mergers_data() -> pd.DataFrame
        A class method that returns a dataframe of mergers data to write to the
        class's adjustment db. By default this is empty.
    make_dividends_data() -> pd.DataFrame
        A class method that returns a dataframe of dividends data to write to
        the class's adjustment db. By default this is empty.
    make_stock_dividends_data() -> pd.DataFrame
        A class method that returns a dataframe of stock dividends data to
        write to the class's adjustment db. By default this is empty.
    make_adjustment_db_conn_str() -> string
        A class method that returns the sqlite3 connection string for the
        database in to which the adjustments will be written. By default this
        is an in-memory database.
    make_adjustment_writer_equity_daily_bar_reader() -> pd.DataFrame
        A class method that returns the daily bar reader to use for the class's
        adjustment writer. By default this is the class's actual
        ``bcolz_equity_daily_bar_reader`` as inherited from
        ``WithBcolzEquityDailyBarReader``. This should probably not be
          overridden; however, some tests used a ``MockDailyBarReader``
         for this.
    make_adjustment_writer(conn: sqlite3.Connection) -> AdjustmentWriter
        A class method that constructs the adjustment which will be used
        to write the data into the connection to be used by the class's
        adjustment reader.

    See Also
    --------
    zipline.testing.MockDailyBarReader
    """

    @classmethod
    def _make_data(cls):
        if False:
            return 10
        return None
    make_splits_data = _make_data
    make_mergers_data = _make_data
    make_dividends_data = _make_data
    make_stock_dividends_data = _make_data
    del _make_data

    @classmethod
    def make_adjustment_writer(cls, conn):
        if False:
            for i in range(10):
                print('nop')
        return SQLiteAdjustmentWriter(conn, cls.make_adjustment_writer_equity_daily_bar_reader())

    @classmethod
    def make_adjustment_writer_equity_daily_bar_reader(cls):
        if False:
            return 10
        return cls.bcolz_equity_daily_bar_reader

    @classmethod
    def make_adjustment_db_conn_str(cls):
        if False:
            while True:
                i = 10
        return ':memory:'

    @classmethod
    def init_class_fixtures(cls):
        if False:
            while True:
                i = 10
        super(WithAdjustmentReader, cls).init_class_fixtures()
        conn = sqlite3.connect(cls.make_adjustment_db_conn_str())
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            cls.make_adjustment_writer(conn).write(splits=cls.make_splits_data(), mergers=cls.make_mergers_data(), dividends=cls.make_dividends_data(), stock_dividends=cls.make_stock_dividends_data())
        cls.adjustment_reader = SQLiteAdjustmentReader(conn)

class WithUSEquityPricingPipelineEngine(WithAdjustmentReader, WithTradingSessions):
    """
    Mixin providing the following as a class-level fixtures.
        - cls.data_root_dir
        - cls.findata_dir
        - cls.pipeline_engine
        - cls.adjustments_db_path

    """

    @classmethod
    def init_class_fixtures(cls):
        if False:
            i = 10
            return i + 15
        cls.data_root_dir = cls.enter_class_context(tmp_dir())
        cls.findata_dir = cls.data_root_dir.makedir('findata')
        super(WithUSEquityPricingPipelineEngine, cls).init_class_fixtures()
        loader = USEquityPricingLoader.without_fx(cls.bcolz_equity_daily_bar_reader, SQLiteAdjustmentReader(cls.adjustments_db_path))

        def get_loader(column):
            if False:
                for i in range(10):
                    print('nop')
            if column in USEquityPricing.columns:
                return loader
            else:
                raise AssertionError('No loader registered for %s' % column)
        cls.pipeline_engine = SimplePipelineEngine(get_loader=get_loader, asset_finder=cls.asset_finder, default_domain=US_EQUITIES)

    @classmethod
    def make_adjustment_db_conn_str(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.adjustments_db_path = os.path.join(cls.findata_dir, 'adjustments', cls.END_DATE.strftime('%Y-%m-%d-adjustments.db'))
        ensure_directory(os.path.dirname(cls.adjustments_db_path))
        return cls.adjustments_db_path

class WithSeededRandomPipelineEngine(WithTradingSessions, WithAssetFinder):
    """
    ZiplineTestCase mixin providing class-level fixtures for running pipelines
    against deterministically-generated random data.

    Attributes
    ----------
    SEEDED_RANDOM_PIPELINE_SEED : int
        Fixture input. Random seed used to initialize the random state loader.
    seeded_random_loader : SeededRandomLoader
        Fixture output. Loader capable of providing columns for
        zipline.pipeline.data.testing.TestingDataSet.
    seeded_random_engine : SimplePipelineEngine
        Fixture output.  A pipeline engine that will use seeded_random_loader
        as its only data provider.

    Methods
    -------
    run_pipeline(start_date, end_date)
        Run a pipeline with self.seeded_random_engine.

    See Also
    --------
    zipline.pipeline.loaders.synthetic.SeededRandomLoader
    zipline.pipeline.loaders.testing.make_seeded_random_loader
    zipline.pipeline.engine.SimplePipelineEngine
    """
    SEEDED_RANDOM_PIPELINE_SEED = 42
    SEEDED_RANDOM_PIPELINE_DEFAULT_DOMAIN = GENERIC

    @classmethod
    def init_class_fixtures(cls):
        if False:
            i = 10
            return i + 15
        super(WithSeededRandomPipelineEngine, cls).init_class_fixtures()
        cls._sids = cls.asset_finder.sids
        cls.seeded_random_loader = loader = make_seeded_random_loader(cls.SEEDED_RANDOM_PIPELINE_SEED, cls.trading_days, cls._sids, columns=cls.make_seeded_random_loader_columns())
        cls.seeded_random_engine = SimplePipelineEngine(get_loader=lambda column: loader, asset_finder=cls.asset_finder, default_domain=cls.SEEDED_RANDOM_PIPELINE_DEFAULT_DOMAIN, default_hooks=cls.make_seeded_random_pipeline_engine_hooks(), populate_initial_workspace=cls.make_seeded_random_populate_initial_workspace())

    @classmethod
    def make_seeded_random_pipeline_engine_hooks(cls):
        if False:
            return 10
        return []

    @classmethod
    def make_seeded_random_populate_initial_workspace(cls):
        if False:
            i = 10
            return i + 15
        return None

    @classmethod
    def make_seeded_random_loader_columns(cls):
        if False:
            i = 10
            return i + 15
        return TestingDataSet.columns

    def raw_expected_values(self, column, start_date, end_date):
        if False:
            return 10
        '\n        Get an array containing the raw values we expect to be produced for the\n        given dates between start_date and end_date, inclusive.\n        '
        all_values = self.seeded_random_loader.values(column.dtype, self.trading_days, self._sids)
        row_slice = self.trading_days.slice_indexer(start_date, end_date)
        return all_values[row_slice]

    def run_pipeline(self, pipeline, start_date, end_date, hooks=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Run a pipeline with self.seeded_random_engine.\n        '
        return self.seeded_random_engine.run_pipeline(pipeline, start_date, end_date, hooks=hooks)

    def run_chunked_pipeline(self, pipeline, start_date, end_date, chunksize, hooks=None):
        if False:
            print('Hello World!')
        '\n        Run a chunked pipeline with self.seeded_random_engine.\n        '
        return self.seeded_random_engine.run_chunked_pipeline(pipeline, start_date, end_date, chunksize=chunksize, hooks=hooks)

class WithDataPortal(WithAdjustmentReader, WithBcolzEquityMinuteBarReader, WithBcolzFutureMinuteBarReader):
    """
    ZiplineTestCase mixin providing self.data_portal as an instance level
    fixture.

    After init_instance_fixtures has been called, `self.data_portal` will be
    populated with a new data portal created by passing in the class's
    trading env, `cls.bcolz_equity_minute_bar_reader`,
    `cls.bcolz_equity_daily_bar_reader`, and `cls.adjustment_reader`.

    Attributes
    ----------
    DATA_PORTAL_USE_DAILY_DATA : bool
        Should the daily bar reader be used? Defaults to True.
    DATA_PORTAL_USE_MINUTE_DATA : bool
        Should the minute bar reader be used? Defaults to True.
    DATA_PORTAL_USE_ADJUSTMENTS : bool
        Should the adjustment reader be used? Defaults to True.

    Methods
    -------
    make_data_portal() -> DataPortal
        Method which returns the data portal to be used for each test case.
        If this is overridden, the ``DATA_PORTAL_USE_*`` attributes may not
        be respected.
    """
    DATA_PORTAL_USE_DAILY_DATA = True
    DATA_PORTAL_USE_MINUTE_DATA = True
    DATA_PORTAL_USE_ADJUSTMENTS = True
    DATA_PORTAL_FIRST_TRADING_DAY = None
    DATA_PORTAL_LAST_AVAILABLE_SESSION = None
    DATA_PORTAL_LAST_AVAILABLE_MINUTE = None
    DATA_PORTAL_MINUTE_HISTORY_PREFETCH = DEFAULT_MINUTE_HISTORY_PREFETCH
    DATA_PORTAL_DAILY_HISTORY_PREFETCH = DEFAULT_DAILY_HISTORY_PREFETCH

    def make_data_portal(self):
        if False:
            return 10
        if self.DATA_PORTAL_FIRST_TRADING_DAY is None:
            if self.DATA_PORTAL_USE_MINUTE_DATA:
                self.DATA_PORTAL_FIRST_TRADING_DAY = self.bcolz_equity_minute_bar_reader.first_trading_day
            elif self.DATA_PORTAL_USE_DAILY_DATA:
                self.DATA_PORTAL_FIRST_TRADING_DAY = self.bcolz_equity_daily_bar_reader.first_trading_day
        return DataPortal(self.asset_finder, self.trading_calendar, first_trading_day=self.DATA_PORTAL_FIRST_TRADING_DAY, equity_daily_reader=self.bcolz_equity_daily_bar_reader if self.DATA_PORTAL_USE_DAILY_DATA else None, equity_minute_reader=self.bcolz_equity_minute_bar_reader if self.DATA_PORTAL_USE_MINUTE_DATA else None, adjustment_reader=self.adjustment_reader if self.DATA_PORTAL_USE_ADJUSTMENTS else None, future_minute_reader=self.bcolz_future_minute_bar_reader if self.DATA_PORTAL_USE_MINUTE_DATA else None, future_daily_reader=MinuteResampleSessionBarReader(self.bcolz_future_minute_bar_reader.trading_calendar, self.bcolz_future_minute_bar_reader) if self.DATA_PORTAL_USE_MINUTE_DATA else None, last_available_session=self.DATA_PORTAL_LAST_AVAILABLE_SESSION, last_available_minute=self.DATA_PORTAL_LAST_AVAILABLE_MINUTE, minute_history_prefetch_length=self.DATA_PORTAL_MINUTE_HISTORY_PREFETCH, daily_history_prefetch_length=self.DATA_PORTAL_DAILY_HISTORY_PREFETCH)

    def init_instance_fixtures(self):
        if False:
            while True:
                i = 10
        super(WithDataPortal, self).init_instance_fixtures()
        self.data_portal = self.make_data_portal()

class WithResponses(object):
    """
    ZiplineTestCase mixin that provides self.responses as an instance
    fixture.

    After init_instance_fixtures has been called, `self.responses` will be
    a new `responses.RequestsMock` object. Users may add new endpoints to this
    with the `self.responses.add` method.
    """

    def init_instance_fixtures(self):
        if False:
            i = 10
            return i + 15
        super(WithResponses, self).init_instance_fixtures()
        self.responses = self.enter_instance_context(responses.RequestsMock())

class WithCreateBarData(WithDataPortal):
    CREATE_BARDATA_DATA_FREQUENCY = 'minute'

    def create_bardata(self, simulation_dt_func, restrictions=None):
        if False:
            print('Hello World!')
        return BarData(self.data_portal, simulation_dt_func, self.CREATE_BARDATA_DATA_FREQUENCY, self.trading_calendar, restrictions or NoRestrictions())

class WithMakeAlgo(WithBenchmarkReturns, WithSimParams, WithLogger, WithDataPortal):
    """
    ZiplineTestCase mixin that provides a ``make_algo`` method.
    """
    START_DATE = pd.Timestamp('2014-12-29', tz='UTC')
    END_DATE = pd.Timestamp('2015-1-05', tz='UTC')
    SIM_PARAMS_DATA_FREQUENCY = 'minute'
    DEFAULT_ALGORITHM_CLASS = TradingAlgorithm

    @classproperty
    def BENCHMARK_SID(cls):
        if False:
            print('Hello World!')
        'The sid to use as a benchmark.\n\n        Can be overridden to use an alternative benchmark.\n        '
        return cls.asset_finder.sids[0]

    def merge_with_inherited_algo_kwargs(self, overriding_type, suite_overrides, method_overrides):
        if False:
            while True:
                i = 10
        "\n        Helper for subclasses overriding ``make_algo_kwargs``.\n\n        A common pattern for tests using `WithMakeAlgoKwargs` is that a\n        particular test suite has a set of default keywords it wants to use\n        everywhere, but also accepts test-specific overrides.\n\n        Test suites that fit that pattern can call this method and pass the\n        test class, suite-specific overrides, and method-specific overrides,\n        and this method takes care of fetching parent class overrides and\n        merging them with the suite- and instance-specific overrides.\n\n        Parameters\n        ----------\n        overriding_type : type\n            The type from which we're being called. This is forwarded to\n            super().make_algo_kwargs()\n        suite_overrides : dict\n            Keywords which should take precedence over kwargs returned by\n            super(overriding_type, self).make_algo_kwargs().  These are\n            generally keyword arguments that are constant within a test suite.\n        method_overrides : dict\n            Keywords which should take precedence over `suite_overrides` and\n            superclass kwargs.  These are generally keyword arguments that are\n            overridden on a per-test basis.\n        "
        return super(overriding_type, self).make_algo_kwargs(**merge(suite_overrides, method_overrides))

    def make_algo_kwargs(self, **overrides):
        if False:
            i = 10
            return i + 15
        if self.BENCHMARK_SID is None:
            overrides.setdefault('benchmark_returns', self.BENCHMARK_RETURNS)
        return merge({'sim_params': self.sim_params, 'data_portal': self.data_portal, 'benchmark_sid': self.BENCHMARK_SID}, overrides)

    def make_algo(self, algo_class=None, **overrides):
        if False:
            return 10
        if algo_class is None:
            algo_class = self.DEFAULT_ALGORITHM_CLASS
        return algo_class(**self.make_algo_kwargs(**overrides))

    def run_algorithm(self, **overrides):
        if False:
            while True:
                i = 10
        '\n        Create and run an TradingAlgorithm in memory.\n        '
        return self.make_algo(**overrides).run()

class WithWerror(object):

    @classmethod
    def init_class_fixtures(cls):
        if False:
            return 10
        cls.enter_class_context(warnings.catch_warnings())
        warnings.simplefilter('error')
        super(WithWerror, cls).init_class_fixtures()
register_calendar_alias('TEST', 'NYSE')

class WithSeededRandomState(object):
    RANDOM_SEED = np.array(list('lmao'), dtype='S1').view('i4').item()

    def init_instance_fixtures(self):
        if False:
            print('Hello World!')
        super(WithSeededRandomState, self).init_instance_fixtures()
        self.rand = np.random.RandomState(self.RANDOM_SEED)

class WithFXRates(object):
    """Fixture providing a factory for in-memory exchange rate data.
    """
    FX_RATES_START_DATE = alias('START_DATE')
    FX_RATES_END_DATE = alias('END_DATE')
    FX_RATES_CALENDAR = '24/5'
    FX_RATES_CURRENCIES = ['USD', 'CAD', 'GBP', 'EUR']
    FX_RATES_RATE_NAMES = ['mid']
    HDF5_FX_CHUNK_SIZE = 75

    @classproperty
    def FX_RATES_DEFAULT_RATE(cls):
        if False:
            while True:
                i = 10
        return cls.FX_RATES_RATE_NAMES[0]

    @classmethod
    def init_class_fixtures(cls):
        if False:
            return 10
        super(WithFXRates, cls).init_class_fixtures()
        cal = get_calendar(cls.FX_RATES_CALENDAR)
        cls.fx_rates_sessions = cal.sessions_in_range(cls.FX_RATES_START_DATE, cls.FX_RATES_END_DATE)
        cls.fx_rates = cls.make_fx_rates(cls.FX_RATES_RATE_NAMES, cls.FX_RATES_CURRENCIES, cls.fx_rates_sessions)
        cls.in_memory_fx_rate_reader = InMemoryFXRateReader(cls.fx_rates, cls.FX_RATES_DEFAULT_RATE)

    @classmethod
    def make_fx_rates_from_reference(cls, reference):
        if False:
            for i in range(10):
                print('nop')
        '\n        Helper method for implementing make_fx_rates.\n\n        Takes a (dates x currencies) DataFrame of "reference" values, which are\n        assumed to be the "true" value of each currency in some unknown\n        external currency. Computes fx rates from A -> B as by dividing the\n        reference value for A by the reference value for B.\n\n        Parameters\n        ----------\n        reference : pd.DataFrame\n            DataFrame of "true" values for currencies.\n\n        Returns\n        -------\n        rates : dict[str, pd.DataFrame]\n            Map from quote currency to FX rates for that currency.\n        '
        out = {}
        for quote in reference.columns:
            out[quote] = reference.divide(reference[quote], axis=0)
        return out

    @classmethod
    def make_fx_rates(cls, rate_names, currencies, sessions):
        if False:
            i = 10
            return i + 15
        rng = np.random.RandomState(42)
        out = {}
        for rate_name in rate_names:
            cols = {}
            for currency in currencies:
                (start, end) = sorted(rng.uniform(0.5, 1.5, (2,)))
                cols[currency] = np.linspace(start, end, len(sessions))
            reference = pd.DataFrame(cols, index=sessions, columns=currencies)
            out[rate_name] = cls.make_fx_rates_from_reference(reference)
        return out

    @classmethod
    def write_h5_fx_rates(cls, path):
        if False:
            for i in range(10):
                print('nop')
        'Write cls.fx_rates to disk with an HDF5FXRateWriter.\n\n        Returns an HDF5FXRateReader that reader from written data.\n        '
        sessions = cls.fx_rates_sessions
        with h5py.File(path, 'w') as h5_file:
            writer = HDF5FXRateWriter(h5_file, cls.HDF5_FX_CHUNK_SIZE)
            fx_data = ((rate, quote, quote_frame.values) for (rate, rate_dict) in cls.fx_rates.items() for (quote, quote_frame) in rate_dict.items())
            writer.write(dts=sessions.values, currencies=np.array(cls.FX_RATES_CURRENCIES, dtype=object), data=fx_data)
        h5_file = cls.enter_class_context(h5py.File(path, 'r'))
        return HDF5FXRateReader(h5_file, default_rate=cls.FX_RATES_DEFAULT_RATE)

    @classmethod
    def get_expected_fx_rate_scalar(cls, rate, quote, base, dt):
        if False:
            return 10
        'Get the expected FX rate for the given scalar coordinates.\n        '
        if base is None:
            return np.nan
        if rate == DEFAULT_FX_RATE:
            rate = cls.FX_RATES_DEFAULT_RATE
        col = cls.fx_rates[rate][quote][base]
        if dt < col.index[0]:
            return np.nan
        ix = fast_get_loc_ffilled(col.index.values, dt.asm8)
        return col.values[ix]

    @classmethod
    def get_expected_fx_rates(cls, rate, quote, bases, dts):
        if False:
            while True:
                i = 10
        'Get an array of expected FX rates for the given indices.\n        '
        out = np.empty((len(dts), len(bases)), dtype='float64')
        for (i, dt) in enumerate(dts):
            for (j, base) in enumerate(bases):
                out[i, j] = cls.get_expected_fx_rate_scalar(rate, quote, base, dt)
        return out

    @classmethod
    def get_expected_fx_rates_columnar(cls, rate, quote, bases, dts):
        if False:
            print('Hello World!')
        assert len(bases) == len(dts)
        rates = [cls.get_expected_fx_rate_scalar(rate, quote, base, dt) for (base, dt) in zip(bases, dts)]
        return np.array(rates, dtype='float64')

def fast_get_loc_ffilled(dts, dt):
    if False:
        print('Hello World!')
    "\n    Equivalent to dts.get_loc(dt, method='ffill'), but with reasonable\n    microperformance.\n    "
    ix = dts.searchsorted(dt, side='right') - 1
    if ix < 0:
        raise KeyError(dt)
    return ix