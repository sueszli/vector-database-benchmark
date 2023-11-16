from collections import Iterable, namedtuple
from copy import copy
import warnings
from datetime import tzinfo, time
import logbook
import pytz
import pandas as pd
import numpy as np
from itertools import chain, repeat
from six import exec_, iteritems, itervalues, string_types
from trading_calendars.utils.pandas_utils import days_at_time
from trading_calendars import get_calendar
from zipline._protocol import handle_non_market_minutes
from zipline.errors import AttachPipelineAfterInitialize, CannotOrderDelistedAsset, DuplicatePipelineName, HistoryInInitialize, IncompatibleCommissionModel, IncompatibleSlippageModel, NoSuchPipeline, OrderDuringInitialize, OrderInBeforeTradingStart, PipelineOutputDuringInitialize, RegisterAccountControlPostInit, RegisterTradingControlPostInit, ScheduleFunctionInvalidCalendar, SetBenchmarkOutsideInitialize, SetCancelPolicyPostInit, SetCommissionPostInit, SetSlippagePostInit, UnsupportedCancelPolicy, UnsupportedDatetimeFormat, UnsupportedOrderParameters, ZeroCapitalError
from zipline.finance.blotter import SimulationBlotter
from zipline.finance.controls import LongOnly, MaxOrderCount, MaxOrderSize, MaxPositionSize, MaxLeverage, MinLeverage, RestrictedListOrder
from zipline.finance.execution import LimitOrder, MarketOrder, StopLimitOrder, StopOrder
from zipline.finance.asset_restrictions import Restrictions
from zipline.finance.cancel_policy import NeverCancel, CancelPolicy
from zipline.finance.asset_restrictions import NoRestrictions, StaticRestrictions, SecurityListRestrictions
from zipline.assets import Asset, Equity, Future
from zipline.gens.tradesimulation import AlgorithmSimulator
from zipline.finance.metrics import MetricsTracker, load as load_metrics_set
from zipline.pipeline import Pipeline
import zipline.pipeline.domain as domain
from zipline.pipeline.engine import ExplodingPipelineEngine, SimplePipelineEngine
from zipline.utils.api_support import api_method, require_initialized, require_not_initialized, ZiplineAPI, disallowed_in_before_trading_start
from zipline.utils.compat import ExitStack
from zipline.utils.input_validation import coerce_string, ensure_upper_case, error_keywords, expect_dtypes, expect_types, optional, optionally
from zipline.utils.numpy_utils import int64_dtype
from zipline.utils.pandas_utils import normalize_date
from zipline.utils.cache import ExpiringCache
from zipline.utils.pandas_utils import clear_dataframe_indexer_caches
import zipline.utils.events
from zipline.utils.events import EventManager, make_eventrule, date_rules, time_rules, calendars, AfterOpen, BeforeClose
from zipline.utils.math_utils import tolerant_equals, round_if_near_integer
from zipline.utils.preprocess import preprocess
from zipline.utils.security_list import SecurityList
import zipline.protocol
from zipline.sources.requests_csv import PandasRequestsCSV
from zipline.gens.sim_engine import MinuteSimulationClock
from zipline.sources.benchmark_source import BenchmarkSource
from zipline.zipline_warnings import ZiplineDeprecationWarning
log = logbook.Logger('ZiplineLog')
AttachedPipeline = namedtuple('AttachedPipeline', 'pipe chunks eager')

class NoBenchmark(ValueError):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(NoBenchmark, self).__init__('Must specify either benchmark_sid or benchmark_returns.')

class TradingAlgorithm(object):
    """A class that represents a trading strategy and parameters to execute
    the strategy.

    Parameters
    ----------
    *args, **kwargs
        Forwarded to ``initialize`` unless listed below.
    initialize : callable[context -> None], optional
        Function that is called at the start of the simulation to
        setup the initial context.
    handle_data : callable[(context, data) -> None], optional
        Function called on every bar. This is where most logic should be
        implemented.
    before_trading_start : callable[(context, data) -> None], optional
        Function that is called before any bars have been processed each
        day.
    analyze : callable[(context, DataFrame) -> None], optional
        Function that is called at the end of the backtest. This is passed
        the context and the performance results for the backtest.
    script : str, optional
        Algoscript that contains the definitions for the four algorithm
        lifecycle functions and any supporting code.
    namespace : dict, optional
        The namespace to execute the algoscript in. By default this is an
        empty namespace that will include only python built ins.
    algo_filename : str, optional
        The filename for the algoscript. This will be used in exception
        tracebacks. default: '<string>'.
    data_frequency : {'daily', 'minute'}, optional
        The duration of the bars.
    equities_metadata : dict or DataFrame or file-like object, optional
        If dict is provided, it must have the following structure:
        * keys are the identifiers
        * values are dicts containing the metadata, with the metadata
          field name as the key
        If pandas.DataFrame is provided, it must have the
        following structure:
        * column names must be the metadata fields
        * index must be the different asset identifiers
        * array contents should be the metadata value
        If an object with a ``read`` method is provided, ``read`` must
        return rows containing at least one of 'sid' or 'symbol' along
        with the other metadata fields.
    futures_metadata : dict or DataFrame or file-like object, optional
        The same layout as ``equities_metadata`` except that it is used
        for futures information.
    identifiers : list, optional
        Any asset identifiers that are not provided in the
        equities_metadata, but will be traded by this TradingAlgorithm.
    get_pipeline_loader : callable[BoundColumn -> PipelineLoader], optional
        The function that maps pipeline columns to their loaders.
    create_event_context : callable[BarData -> context manager], optional
        A function used to create a context mananger that wraps the
        execution of all events that are scheduled for a bar.
        This function will be passed the data for the bar and should
        return the actual context manager that will be entered.
    history_container_class : type, optional
        The type of history container to use. default: HistoryContainer
    platform : str, optional
        The platform the simulation is running on. This can be queried for
        in the simulation with ``get_environment``. This allows algorithms
        to conditionally execute code based on platform it is running on.
        default: 'zipline'
    adjustment_reader : AdjustmentReader
        The interface to the adjustments.
    """

    def __init__(self, sim_params, data_portal=None, asset_finder=None, namespace=None, script=None, algo_filename=None, initialize=None, handle_data=None, before_trading_start=None, analyze=None, trading_calendar=None, metrics_set=None, blotter=None, blotter_class=None, cancel_policy=None, benchmark_sid=None, benchmark_returns=None, platform='zipline', capital_changes=None, get_pipeline_loader=None, create_event_context=None, **initialize_kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.trading_controls = []
        self.account_controls = []
        self._recorded_vars = {}
        self.namespace = namespace or {}
        self._platform = platform
        self.logger = None
        self.data_portal = data_portal
        if self.data_portal is None:
            if asset_finder is None:
                raise ValueError('Must pass either data_portal or asset_finder to TradingAlgorithm()')
            self.asset_finder = asset_finder
        else:
            if asset_finder is not None and asset_finder is not data_portal.asset_finder:
                raise ValueError('Inconsistent asset_finders in TradingAlgorithm()')
            self.asset_finder = data_portal.asset_finder
        self.benchmark_returns = benchmark_returns
        self.sim_params = sim_params
        if trading_calendar is None:
            self.trading_calendar = sim_params.trading_calendar
        elif trading_calendar.name == sim_params.trading_calendar.name:
            self.trading_calendar = sim_params.trading_calendar
        else:
            raise ValueError('Conflicting calendars: trading_calendar={}, but sim_params.trading_calendar={}'.format(trading_calendar.name, self.sim_params.trading_calendar.name))
        self.metrics_tracker = None
        self._last_sync_time = pd.NaT
        self._metrics_set = metrics_set
        if self._metrics_set is None:
            self._metrics_set = load_metrics_set('default')
        self.init_engine(get_pipeline_loader)
        self._pipelines = {}
        self._pipeline_cache = ExpiringCache(cleanup=clear_dataframe_indexer_caches)
        if blotter is not None:
            self.blotter = blotter
        else:
            cancel_policy = cancel_policy or NeverCancel()
            blotter_class = blotter_class or SimulationBlotter
            self.blotter = blotter_class(cancel_policy=cancel_policy)
        self._symbol_lookup_date = None
        self.algoscript = script
        self._initialize = None
        self._before_trading_start = None
        self._analyze = None
        self._in_before_trading_start = False
        self.event_manager = EventManager(create_event_context)
        self._handle_data = None

        def noop(*args, **kwargs):
            if False:
                print('Hello World!')
            pass
        if self.algoscript is not None:
            unexpected_api_methods = set()
            if initialize is not None:
                unexpected_api_methods.add('initialize')
            if handle_data is not None:
                unexpected_api_methods.add('handle_data')
            if before_trading_start is not None:
                unexpected_api_methods.add('before_trading_start')
            if analyze is not None:
                unexpected_api_methods.add('analyze')
            if unexpected_api_methods:
                raise ValueError('TradingAlgorithm received a script and the following API methods as functions:\n{funcs}'.format(funcs=unexpected_api_methods))
            if algo_filename is None:
                algo_filename = '<string>'
            code = compile(self.algoscript, algo_filename, 'exec')
            exec_(code, self.namespace)
            self._initialize = self.namespace.get('initialize', noop)
            self._handle_data = self.namespace.get('handle_data', noop)
            self._before_trading_start = self.namespace.get('before_trading_start')
            self._analyze = self.namespace.get('analyze')
        else:
            self._initialize = initialize or (lambda self: None)
            self._handle_data = handle_data
            self._before_trading_start = before_trading_start
            self._analyze = analyze
        self.event_manager.add_event(zipline.utils.events.Event(zipline.utils.events.Always(), self.handle_data.__func__), prepend=True)
        if self.sim_params.capital_base <= 0:
            raise ZeroCapitalError()
        self.initialized = False
        self.initialize_kwargs = initialize_kwargs or {}
        self.benchmark_sid = benchmark_sid
        self.capital_changes = capital_changes or {}
        self.capital_change_deltas = {}
        self.restrictions = NoRestrictions()
        self._backwards_compat_universe = None

    def init_engine(self, get_loader):
        if False:
            i = 10
            return i + 15
        '\n        Construct and store a PipelineEngine from loader.\n\n        If get_loader is None, constructs an ExplodingPipelineEngine\n        '
        if get_loader is not None:
            self.engine = SimplePipelineEngine(get_loader, self.asset_finder, self.default_pipeline_domain(self.trading_calendar))
        else:
            self.engine = ExplodingPipelineEngine()

    def initialize(self, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Call self._initialize with `self` made available to Zipline API\n        functions.\n        '
        with ZiplineAPI(self):
            self._initialize(self, *args, **kwargs)

    def before_trading_start(self, data):
        if False:
            return 10
        self.compute_eager_pipelines()
        if self._before_trading_start is None:
            return
        self._in_before_trading_start = True
        with handle_non_market_minutes(data) if self.data_frequency == 'minute' else ExitStack():
            self._before_trading_start(self, data)
        self._in_before_trading_start = False

    def handle_data(self, data):
        if False:
            return 10
        if self._handle_data:
            self._handle_data(self, data)

    def analyze(self, perf):
        if False:
            return 10
        if self._analyze is None:
            return
        with ZiplineAPI(self):
            self._analyze(self, perf)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        N.B. this does not yet represent a string that can be used\n        to instantiate an exact copy of an algorithm.\n\n        However, it is getting close, and provides some value as something\n        that can be inspected interactively.\n        '
        return '\n{class_name}(\n    capital_base={capital_base}\n    sim_params={sim_params},\n    initialized={initialized},\n    slippage_models={slippage_models},\n    commission_models={commission_models},\n    blotter={blotter},\n    recorded_vars={recorded_vars})\n'.strip().format(class_name=self.__class__.__name__, capital_base=self.sim_params.capital_base, sim_params=repr(self.sim_params), initialized=self.initialized, slippage_models=repr(self.blotter.slippage_models), commission_models=repr(self.blotter.commission_models), blotter=repr(self.blotter), recorded_vars=repr(self.recorded_vars))

    def _create_clock(self):
        if False:
            while True:
                i = 10
        '\n        If the clock property is not set, then create one based on frequency.\n        '
        trading_o_and_c = self.trading_calendar.schedule.loc[self.sim_params.sessions]
        market_closes = trading_o_and_c['market_close']
        minutely_emission = False
        if self.sim_params.data_frequency == 'minute':
            market_opens = trading_o_and_c['market_open']
            minutely_emission = self.sim_params.emission_rate == 'minute'
            execution_opens = self.trading_calendar.execution_time_from_open(market_opens)
            execution_closes = self.trading_calendar.execution_time_from_close(market_closes)
        else:
            execution_closes = self.trading_calendar.execution_time_from_close(market_closes)
            execution_opens = execution_closes
        before_trading_start_minutes = days_at_time(self.sim_params.sessions, time(8, 45), 'US/Eastern')
        return MinuteSimulationClock(self.sim_params.sessions, execution_opens, execution_closes, before_trading_start_minutes, minute_emission=minutely_emission)

    def _create_benchmark_source(self):
        if False:
            while True:
                i = 10
        if self.benchmark_sid is not None:
            benchmark_asset = self.asset_finder.retrieve_asset(self.benchmark_sid)
            benchmark_returns = None
        else:
            if self.benchmark_returns is None:
                raise NoBenchmark()
            benchmark_asset = None
            benchmark_returns = self.benchmark_returns
        return BenchmarkSource(benchmark_asset=benchmark_asset, benchmark_returns=benchmark_returns, trading_calendar=self.trading_calendar, sessions=self.sim_params.sessions, data_portal=self.data_portal, emission_rate=self.sim_params.emission_rate)

    def _create_metrics_tracker(self):
        if False:
            for i in range(10):
                print('nop')
        return MetricsTracker(trading_calendar=self.trading_calendar, first_session=self.sim_params.start_session, last_session=self.sim_params.end_session, capital_base=self.sim_params.capital_base, emission_rate=self.sim_params.emission_rate, data_frequency=self.sim_params.data_frequency, asset_finder=self.asset_finder, metrics=self._metrics_set)

    def _create_generator(self, sim_params):
        if False:
            while True:
                i = 10
        if sim_params is not None:
            self.sim_params = sim_params
        self.metrics_tracker = metrics_tracker = self._create_metrics_tracker()
        self.on_dt_changed(self.sim_params.start_session)
        if not self.initialized:
            self.initialize(**self.initialize_kwargs)
            self.initialized = True
        benchmark_source = self._create_benchmark_source()
        self.trading_client = AlgorithmSimulator(self, sim_params, self.data_portal, self._create_clock(), benchmark_source, self.restrictions, universe_func=self._calculate_universe)
        metrics_tracker.handle_start_of_simulation(benchmark_source)
        return self.trading_client.transform()

    def _calculate_universe(self):
        if False:
            for i in range(10):
                print('nop')
        if self._backwards_compat_universe is None:
            self._backwards_compat_universe = self.asset_finder.retrieve_all(self.asset_finder.sids)
        return self._backwards_compat_universe

    def compute_eager_pipelines(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute any pipelines attached with eager=True.\n        '
        for (name, pipe) in self._pipelines.items():
            if pipe.eager:
                self.pipeline_output(name)

    def get_generator(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Override this method to add new logic to the construction\n        of the generator. Overrides can use the _create_generator\n        method to get a standard construction generator.\n        '
        return self._create_generator(self.sim_params)

    def run(self, data_portal=None):
        if False:
            while True:
                i = 10
        'Run the algorithm.\n        '
        if data_portal is not None:
            self.data_portal = data_portal
            self.asset_finder = data_portal.asset_finder
        elif self.data_portal is None:
            raise RuntimeError('No data portal in TradingAlgorithm.run().\nEither pass a DataPortal to TradingAlgorithm() or to run().')
        else:
            assert self.asset_finder is not None, 'Have data portal without asset_finder.'
        try:
            perfs = []
            for perf in self.get_generator():
                perfs.append(perf)
            daily_stats = self._create_daily_stats(perfs)
            self.analyze(daily_stats)
        finally:
            self.data_portal = None
            self.metrics_tracker = None
        return daily_stats

    def _create_daily_stats(self, perfs):
        if False:
            i = 10
            return i + 15
        daily_perfs = []
        for perf in perfs:
            if 'daily_perf' in perf:
                perf['daily_perf'].update(perf['daily_perf'].pop('recorded_vars'))
                perf['daily_perf'].update(perf['cumulative_risk_metrics'])
                daily_perfs.append(perf['daily_perf'])
            else:
                self.risk_report = perf
        daily_dts = pd.DatetimeIndex([p['period_close'] for p in daily_perfs], tz='UTC')
        daily_stats = pd.DataFrame(daily_perfs, index=daily_dts)
        return daily_stats

    def calculate_capital_changes(self, dt, emission_rate, is_interday, portfolio_value_adjustment=0.0):
        if False:
            print('Hello World!')
        '\n        If there is a capital change for a given dt, this means the the change\n        occurs before `handle_data` on the given dt. In the case of the\n        change being a target value, the change will be computed on the\n        portfolio value according to prices at the given dt\n\n        `portfolio_value_adjustment`, if specified, will be removed from the\n        portfolio_value of the cumulative performance when calculating deltas\n        from target capital changes.\n        '
        try:
            capital_change = self.capital_changes[dt]
        except KeyError:
            return
        self._sync_last_sale_prices()
        if capital_change['type'] == 'target':
            target = capital_change['value']
            capital_change_amount = target - (self.portfolio.portfolio_value - portfolio_value_adjustment)
            log.info('Processing capital change to target %s at %s. Capital change delta is %s' % (target, dt, capital_change_amount))
        elif capital_change['type'] == 'delta':
            target = None
            capital_change_amount = capital_change['value']
            log.info('Processing capital change of delta %s at %s' % (capital_change_amount, dt))
        else:
            log.error("Capital change %s does not indicate a valid type ('target' or 'delta')" % capital_change)
            return
        self.capital_change_deltas.update({dt: capital_change_amount})
        self.metrics_tracker.capital_change(capital_change_amount)
        yield {'capital_change': {'date': dt, 'type': 'cash', 'target': target, 'delta': capital_change_amount}}

    @api_method
    def get_environment(self, field='platform'):
        if False:
            return 10
        "Query the execution environment.\n\n        Parameters\n        ----------\n        field : {'platform', 'arena', 'data_frequency',\n                 'start', 'end', 'capital_base', 'platform', '*'}\n            The field to query. The options have the following meanings:\n              arena : str\n                  The arena from the simulation parameters. This will normally\n                  be ``'backtest'`` but some systems may use this distinguish\n                  live trading from backtesting.\n              data_frequency : {'daily', 'minute'}\n                  data_frequency tells the algorithm if it is running with\n                  daily data or minute data.\n              start : datetime\n                  The start date for the simulation.\n              end : datetime\n                  The end date for the simulation.\n              capital_base : float\n                  The starting capital for the simulation.\n              platform : str\n                  The platform that the code is running on. By default this\n                  will be the string 'zipline'. This can allow algorithms to\n                  know if they are running on the Quantopian platform instead.\n              * : dict[str -> any]\n                  Returns all of the fields in a dictionary.\n\n        Returns\n        -------\n        val : any\n            The value for the field queried. See above for more information.\n\n        Raises\n        ------\n        ValueError\n            Raised when ``field`` is not a valid option.\n        "
        env = {'arena': self.sim_params.arena, 'data_frequency': self.sim_params.data_frequency, 'start': self.sim_params.first_open, 'end': self.sim_params.last_close, 'capital_base': self.sim_params.capital_base, 'platform': self._platform}
        if field == '*':
            return env
        else:
            try:
                return env[field]
            except KeyError:
                raise ValueError('%r is not a valid field for get_environment' % field)

    @api_method
    def fetch_csv(self, url, pre_func=None, post_func=None, date_column='date', date_format=None, timezone=pytz.utc.zone, symbol=None, mask=True, symbol_column=None, special_params_checker=None, country_code=None, **kwargs):
        if False:
            return 10
        "Fetch a csv from a remote url and register the data so that it is\n        queryable from the ``data`` object.\n\n        Parameters\n        ----------\n        url : str\n            The url of the csv file to load.\n        pre_func : callable[pd.DataFrame -> pd.DataFrame], optional\n            A callback to allow preprocessing the raw data returned from\n            fetch_csv before dates are paresed or symbols are mapped.\n        post_func : callable[pd.DataFrame -> pd.DataFrame], optional\n            A callback to allow postprocessing of the data after dates and\n            symbols have been mapped.\n        date_column : str, optional\n            The name of the column in the preprocessed dataframe containing\n            datetime information to map the data.\n        date_format : str, optional\n            The format of the dates in the ``date_column``. If not provided\n            ``fetch_csv`` will attempt to infer the format. For information\n            about the format of this string, see :func:`pandas.read_csv`.\n        timezone : tzinfo or str, optional\n            The timezone for the datetime in the ``date_column``.\n        symbol : str, optional\n            If the data is about a new asset or index then this string will\n            be the name used to identify the values in ``data``. For example,\n            one may use ``fetch_csv`` to load data for VIX, then this field\n            could be the string ``'VIX'``.\n        mask : bool, optional\n            Drop any rows which cannot be symbol mapped.\n        symbol_column : str\n            If the data is attaching some new attribute to each asset then this\n            argument is the name of the column in the preprocessed dataframe\n            containing the symbols. This will be used along with the date\n            information to map the sids in the asset finder.\n        country_code : str, optional\n            Country code to use to disambiguate symbol lookups.\n        **kwargs\n            Forwarded to :func:`pandas.read_csv`.\n\n        Returns\n        -------\n        csv_data_source : zipline.sources.requests_csv.PandasRequestsCSV\n            A requests source that will pull data from the url specified.\n        "
        if country_code is None:
            country_code = self.default_fetch_csv_country_code(self.trading_calendar)
        csv_data_source = PandasRequestsCSV(url, pre_func, post_func, self.asset_finder, self.trading_calendar.day, self.sim_params.start_session, self.sim_params.end_session, date_column, date_format, timezone, symbol, mask, symbol_column, data_frequency=self.data_frequency, country_code=country_code, special_params_checker=special_params_checker, **kwargs)
        self.data_portal.handle_extra_source(csv_data_source.df, self.sim_params)
        return csv_data_source

    def add_event(self, rule, callback):
        if False:
            while True:
                i = 10
        "Adds an event to the algorithm's EventManager.\n\n        Parameters\n        ----------\n        rule : EventRule\n            The rule for when the callback should be triggered.\n        callback : callable[(context, data) -> None]\n            The function to execute when the rule is triggered.\n        "
        self.event_manager.add_event(zipline.utils.events.Event(rule, callback))

    @api_method
    def schedule_function(self, func, date_rule=None, time_rule=None, half_days=True, calendar=None):
        if False:
            i = 10
            return i + 15
        '\n        Schedule a function to be called repeatedly in the future.\n\n        Parameters\n        ----------\n        func : callable\n            The function to execute when the rule is triggered. ``func`` should\n            have the same signature as ``handle_data``.\n        date_rule : zipline.utils.events.EventRule, optional\n            Rule for the dates on which to execute ``func``. If not\n            passed, the function will run every trading day.\n        time_rule : zipline.utils.events.EventRule, optional\n            Rule for the time at which to execute ``func``. If not passed, the\n            function will execute at the end of the first market minute of the\n            day.\n        half_days : bool, optional\n            Should this rule fire on half days? Default is True.\n        calendar : Sentinel, optional\n            Calendar used to compute rules that depend on the trading calendar.\n\n        See Also\n        --------\n        :class:`zipline.api.date_rules`\n        :class:`zipline.api.time_rules`\n        '
        if isinstance(date_rule, (AfterOpen, BeforeClose)) and (not time_rule):
            warnings.warn('Got a time rule for the second positional argument date_rule. You should use keyword argument time_rule= when calling schedule_function without specifying a date_rule', stacklevel=3)
        date_rule = date_rule or date_rules.every_day()
        time_rule = time_rule or time_rules.every_minute() if self.sim_params.data_frequency == 'minute' else time_rules.every_minute()
        if calendar is None:
            cal = self.trading_calendar
        elif calendar is calendars.US_EQUITIES:
            cal = get_calendar('XNYS')
        elif calendar is calendars.US_FUTURES:
            cal = get_calendar('us_futures')
        else:
            raise ScheduleFunctionInvalidCalendar(given_calendar=calendar, allowed_calendars='[calendars.US_EQUITIES, calendars.US_FUTURES]')
        self.add_event(make_eventrule(date_rule, time_rule, cal, half_days), func)

    @api_method
    def record(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Track and record values each day.\n\n        Parameters\n        ----------\n        **kwargs\n            The names and values to record.\n\n        Notes\n        -----\n        These values will appear in the performance packets and the performance\n        dataframe passed to ``analyze`` and returned from\n        :func:`~zipline.run_algorithm`.\n        '
        args = [iter(args)] * 2
        positionals = zip(*args)
        for (name, value) in chain(positionals, iteritems(kwargs)):
            self._recorded_vars[name] = value

    @api_method
    def set_benchmark(self, benchmark):
        if False:
            i = 10
            return i + 15
        'Set the benchmark asset.\n\n        Parameters\n        ----------\n        benchmark : zipline.assets.Asset\n            The asset to set as the new benchmark.\n\n        Notes\n        -----\n        Any dividends payed out for that new benchmark asset will be\n        automatically reinvested.\n        '
        if self.initialized:
            raise SetBenchmarkOutsideInitialize()
        self.benchmark_sid = benchmark

    @api_method
    @preprocess(root_symbol_str=ensure_upper_case)
    def continuous_future(self, root_symbol_str, offset=0, roll='volume', adjustment='mul'):
        if False:
            i = 10
            return i + 15
        "Create a specifier for a continuous contract.\n\n        Parameters\n        ----------\n        root_symbol_str : str\n            The root symbol for the future chain.\n\n        offset : int, optional\n            The distance from the primary contract. Default is 0.\n\n        roll_style : str, optional\n            How rolls are determined. Default is 'volume'.\n\n        adjustment : str, optional\n            Method for adjusting lookback prices between rolls. Options are\n            'mul', 'add', and None. Default is 'mul'.\n\n        Returns\n        -------\n        continuous_future : zipline.assets.ContinuousFuture\n            The continuous future specifier.\n        "
        return self.asset_finder.create_continuous_future(root_symbol_str, offset, roll, adjustment)

    @api_method
    @preprocess(symbol_str=ensure_upper_case, country_code=optionally(ensure_upper_case))
    def symbol(self, symbol_str, country_code=None):
        if False:
            i = 10
            return i + 15
        'Lookup an Equity by its ticker symbol.\n\n        Parameters\n        ----------\n        symbol_str : str\n            The ticker symbol for the equity to lookup.\n        country_code : str or None, optional\n            A country to limit symbol searches to.\n\n        Returns\n        -------\n        equity : zipline.assets.Equity\n            The equity that held the ticker symbol on the current\n            symbol lookup date.\n\n        Raises\n        ------\n        SymbolNotFound\n            Raised when the symbols was not held on the current lookup date.\n\n        See Also\n        --------\n        :func:`zipline.api.set_symbol_lookup_date`\n        '
        _lookup_date = self._symbol_lookup_date if self._symbol_lookup_date is not None else self.sim_params.end_session
        return self.asset_finder.lookup_symbol(symbol_str, as_of_date=_lookup_date, country_code=country_code)

    @api_method
    def symbols(self, *args, **kwargs):
        if False:
            return 10
        'Lookup multuple Equities as a list.\n\n        Parameters\n        ----------\n        *args : iterable[str]\n            The ticker symbols to lookup.\n        country_code : str or None, optional\n            A country to limit symbol searches to.\n\n        Returns\n        -------\n        equities : list[zipline.assets.Equity]\n            The equities that held the given ticker symbols on the current\n            symbol lookup date.\n\n        Raises\n        ------\n        SymbolNotFound\n            Raised when one of the symbols was not held on the current\n            lookup date.\n\n        See Also\n        --------\n        :func:`zipline.api.set_symbol_lookup_date`\n        '
        return [self.symbol(identifier, **kwargs) for identifier in args]

    @api_method
    def sid(self, sid):
        if False:
            print('Hello World!')
        'Lookup an Asset by its unique asset identifier.\n\n        Parameters\n        ----------\n        sid : int\n            The unique integer that identifies an asset.\n\n        Returns\n        -------\n        asset : zipline.assets.Asset\n            The asset with the given ``sid``.\n\n        Raises\n        ------\n        SidsNotFound\n            When a requested ``sid`` does not map to any asset.\n        '
        return self.asset_finder.retrieve_asset(sid)

    @api_method
    @preprocess(symbol=ensure_upper_case)
    def future_symbol(self, symbol):
        if False:
            return 10
        "Lookup a futures contract with a given symbol.\n\n        Parameters\n        ----------\n        symbol : str\n            The symbol of the desired contract.\n\n        Returns\n        -------\n        future : zipline.assets.Future\n            The future that trades with the name ``symbol``.\n\n        Raises\n        ------\n        SymbolNotFound\n            Raised when no contract named 'symbol' is found.\n        "
        return self.asset_finder.lookup_future_symbol(symbol)

    def _calculate_order_value_amount(self, asset, value):
        if False:
            i = 10
            return i + 15
        '\n        Calculates how many shares/contracts to order based on the type of\n        asset being ordered.\n        '
        normalized_date = normalize_date(self.datetime)
        if normalized_date < asset.start_date:
            raise CannotOrderDelistedAsset(msg='Cannot order {0}, as it started trading on {1}.'.format(asset.symbol, asset.start_date))
        elif normalized_date > asset.end_date:
            raise CannotOrderDelistedAsset(msg='Cannot order {0}, as it stopped trading on {1}.'.format(asset.symbol, asset.end_date))
        else:
            last_price = self.trading_client.current_data.current(asset, 'price')
            if np.isnan(last_price):
                raise CannotOrderDelistedAsset(msg='Cannot order {0} on {1} as there is no last price for the security.'.format(asset.symbol, self.datetime))
        if tolerant_equals(last_price, 0):
            zero_message = "Price of 0 for {psid}; can't infer value".format(psid=asset)
            if self.logger:
                self.logger.debug(zero_message)
            return 0
        value_multiplier = asset.price_multiplier
        return value / (last_price * value_multiplier)

    def _can_order_asset(self, asset):
        if False:
            return 10
        if not isinstance(asset, Asset):
            raise UnsupportedOrderParameters(msg="Passing non-Asset argument to 'order()' is not supported. Use 'sid()' or 'symbol()' methods to look up an Asset.")
        if asset.auto_close_date:
            day = normalize_date(self.get_datetime())
            if day > min(asset.end_date, asset.auto_close_date):
                log.warn('Cannot place order for {0}, as it has de-listed. Any existing positions for this asset will be liquidated on {1}.'.format(asset.symbol, asset.auto_close_date))
                return False
        return True

    @api_method
    @disallowed_in_before_trading_start(OrderInBeforeTradingStart())
    def order(self, asset, amount, limit_price=None, stop_price=None, style=None):
        if False:
            return 10
        'Place an order for a fixed number of shares.\n\n        Parameters\n        ----------\n        asset : Asset\n            The asset to be ordered.\n        amount : int\n            The amount of shares to order. If ``amount`` is positive, this is\n            the number of shares to buy or cover. If ``amount`` is negative,\n            this is the number of shares to sell or short.\n        limit_price : float, optional\n            The limit price for the order.\n        stop_price : float, optional\n            The stop price for the order.\n        style : ExecutionStyle, optional\n            The execution style for the order.\n\n        Returns\n        -------\n        order_id : str or None\n            The unique identifier for this order, or None if no order was\n            placed.\n\n        Notes\n        -----\n        The ``limit_price`` and ``stop_price`` arguments provide shorthands for\n        passing common execution styles. Passing ``limit_price=N`` is\n        equivalent to ``style=LimitOrder(N)``. Similarly, passing\n        ``stop_price=M`` is equivalent to ``style=StopOrder(M)``, and passing\n        ``limit_price=N`` and ``stop_price=M`` is equivalent to\n        ``style=StopLimitOrder(N, M)``. It is an error to pass both a ``style``\n        and ``limit_price`` or ``stop_price``.\n\n        See Also\n        --------\n        :class:`zipline.finance.execution.ExecutionStyle`\n        :func:`zipline.api.order_value`\n        :func:`zipline.api.order_percent`\n        '
        if not self._can_order_asset(asset):
            return None
        (amount, style) = self._calculate_order(asset, amount, limit_price, stop_price, style)
        return self.blotter.order(asset, amount, style)

    def _calculate_order(self, asset, amount, limit_price=None, stop_price=None, style=None):
        if False:
            i = 10
            return i + 15
        amount = self.round_order(amount)
        self.validate_order_params(asset, amount, limit_price, stop_price, style)
        style = self.__convert_order_params_for_blotter(asset, limit_price, stop_price, style)
        return (amount, style)

    @staticmethod
    def round_order(amount):
        if False:
            i = 10
            return i + 15
        "\n        Convert number of shares to an integer.\n\n        By default, truncates to the integer share count that's either within\n        .0001 of amount or closer to zero.\n\n        E.g. 3.9999 -> 4.0; 5.5 -> 5.0; -5.5 -> -5.0\n        "
        return int(round_if_near_integer(amount))

    def validate_order_params(self, asset, amount, limit_price, stop_price, style):
        if False:
            i = 10
            return i + 15
        '\n        Helper method for validating parameters to the order API function.\n\n        Raises an UnsupportedOrderParameters if invalid arguments are found.\n        '
        if not self.initialized:
            raise OrderDuringInitialize(msg='order() can only be called from within handle_data()')
        if style:
            if limit_price:
                raise UnsupportedOrderParameters(msg='Passing both limit_price and style is not supported.')
            if stop_price:
                raise UnsupportedOrderParameters(msg='Passing both stop_price and style is not supported.')
        for control in self.trading_controls:
            control.validate(asset, amount, self.portfolio, self.get_datetime(), self.trading_client.current_data)

    @staticmethod
    def __convert_order_params_for_blotter(asset, limit_price, stop_price, style):
        if False:
            return 10
        '\n        Helper method for converting deprecated limit_price and stop_price\n        arguments into ExecutionStyle instances.\n\n        This function assumes that either style == None or (limit_price,\n        stop_price) == (None, None).\n        '
        if style:
            assert (limit_price, stop_price) == (None, None)
            return style
        if limit_price and stop_price:
            return StopLimitOrder(limit_price, stop_price, asset=asset)
        if limit_price:
            return LimitOrder(limit_price, asset=asset)
        if stop_price:
            return StopOrder(stop_price, asset=asset)
        else:
            return MarketOrder()

    @api_method
    @disallowed_in_before_trading_start(OrderInBeforeTradingStart())
    def order_value(self, asset, value, limit_price=None, stop_price=None, style=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Place an order for a fixed amount of money.\n\n        Equivalent to ``order(asset, value / data.current(asset, 'price'))``.\n\n        Parameters\n        ----------\n        asset : Asset\n            The asset to be ordered.\n        value : float\n            Amount of value of ``asset`` to be transacted. The number of shares\n            bought or sold will be equal to ``value / current_price``.\n        limit_price : float, optional\n            Limit price for the order.\n        stop_price : float, optional\n            Stop price for the order.\n        style : ExecutionStyle\n            The execution style for the order.\n\n        Returns\n        -------\n        order_id : str\n            The unique identifier for this order.\n\n        Notes\n        -----\n        See :func:`zipline.api.order` for more information about\n        ``limit_price``, ``stop_price``, and ``style``\n\n        See Also\n        --------\n        :class:`zipline.finance.execution.ExecutionStyle`\n        :func:`zipline.api.order`\n        :func:`zipline.api.order_percent`\n        "
        if not self._can_order_asset(asset):
            return None
        amount = self._calculate_order_value_amount(asset, value)
        return self.order(asset, amount, limit_price=limit_price, stop_price=stop_price, style=style)

    @property
    def recorded_vars(self):
        if False:
            i = 10
            return i + 15
        return copy(self._recorded_vars)

    def _sync_last_sale_prices(self, dt=None):
        if False:
            while True:
                i = 10
        'Sync the last sale prices on the metrics tracker to a given\n        datetime.\n\n        Parameters\n        ----------\n        dt : datetime\n            The time to sync the prices to.\n\n        Notes\n        -----\n        This call is cached by the datetime. Repeated calls in the same bar\n        are cheap.\n        '
        if dt is None:
            dt = self.datetime
        if dt != self._last_sync_time:
            self.metrics_tracker.sync_last_sale_prices(dt, self.data_portal)
            self._last_sync_time = dt

    @property
    def portfolio(self):
        if False:
            i = 10
            return i + 15
        self._sync_last_sale_prices()
        return self.metrics_tracker.portfolio

    @property
    def account(self):
        if False:
            print('Hello World!')
        self._sync_last_sale_prices()
        return self.metrics_tracker.account

    def set_logger(self, logger):
        if False:
            for i in range(10):
                print('nop')
        self.logger = logger

    def on_dt_changed(self, dt):
        if False:
            i = 10
            return i + 15
        '\n        Callback triggered by the simulation loop whenever the current dt\n        changes.\n\n        Any logic that should happen exactly once at the start of each datetime\n        group should happen here.\n        '
        self.datetime = dt
        self.blotter.set_date(dt)

    @api_method
    @preprocess(tz=coerce_string(pytz.timezone))
    @expect_types(tz=optional(tzinfo))
    def get_datetime(self, tz=None):
        if False:
            return 10
        '\n        Returns the current simulation datetime.\n\n        Parameters\n        ----------\n        tz : tzinfo or str, optional\n            The timezone to return the datetime in. This defaults to utc.\n\n        Returns\n        -------\n        dt : datetime\n            The current simulation datetime converted to ``tz``.\n        '
        dt = self.datetime
        assert dt.tzinfo == pytz.utc, 'Algorithm should have a utc datetime'
        if tz is not None:
            dt = dt.astimezone(tz)
        return dt

    @api_method
    def set_slippage(self, us_equities=None, us_futures=None):
        if False:
            i = 10
            return i + 15
        '\n        Set the slippage models for the simulation.\n\n        Parameters\n        ----------\n        us_equities : EquitySlippageModel\n            The slippage model to use for trading US equities.\n        us_futures : FutureSlippageModel\n            The slippage model to use for trading US futures.\n\n        Notes\n        -----\n        This function can only be called during\n        :func:`~zipline.api.initialize`.\n\n        See Also\n        --------\n        :class:`zipline.finance.slippage.SlippageModel`\n        '
        if self.initialized:
            raise SetSlippagePostInit()
        if us_equities is not None:
            if Equity not in us_equities.allowed_asset_types:
                raise IncompatibleSlippageModel(asset_type='equities', given_model=us_equities, supported_asset_types=us_equities.allowed_asset_types)
            self.blotter.slippage_models[Equity] = us_equities
        if us_futures is not None:
            if Future not in us_futures.allowed_asset_types:
                raise IncompatibleSlippageModel(asset_type='futures', given_model=us_futures, supported_asset_types=us_futures.allowed_asset_types)
            self.blotter.slippage_models[Future] = us_futures

    @api_method
    def set_commission(self, us_equities=None, us_futures=None):
        if False:
            print('Hello World!')
        'Sets the commission models for the simulation.\n\n        Parameters\n        ----------\n        us_equities : EquityCommissionModel\n            The commission model to use for trading US equities.\n        us_futures : FutureCommissionModel\n            The commission model to use for trading US futures.\n\n        Notes\n        -----\n        This function can only be called during\n        :func:`~zipline.api.initialize`.\n\n        See Also\n        --------\n        :class:`zipline.finance.commission.PerShare`\n        :class:`zipline.finance.commission.PerTrade`\n        :class:`zipline.finance.commission.PerDollar`\n        '
        if self.initialized:
            raise SetCommissionPostInit()
        if us_equities is not None:
            if Equity not in us_equities.allowed_asset_types:
                raise IncompatibleCommissionModel(asset_type='equities', given_model=us_equities, supported_asset_types=us_equities.allowed_asset_types)
            self.blotter.commission_models[Equity] = us_equities
        if us_futures is not None:
            if Future not in us_futures.allowed_asset_types:
                raise IncompatibleCommissionModel(asset_type='futures', given_model=us_futures, supported_asset_types=us_futures.allowed_asset_types)
            self.blotter.commission_models[Future] = us_futures

    @api_method
    def set_cancel_policy(self, cancel_policy):
        if False:
            return 10
        'Sets the order cancellation policy for the simulation.\n\n        Parameters\n        ----------\n        cancel_policy : CancelPolicy\n            The cancellation policy to use.\n\n        See Also\n        --------\n        :class:`zipline.api.EODCancel`\n        :class:`zipline.api.NeverCancel`\n        '
        if not isinstance(cancel_policy, CancelPolicy):
            raise UnsupportedCancelPolicy()
        if self.initialized:
            raise SetCancelPolicyPostInit()
        self.blotter.cancel_policy = cancel_policy

    @api_method
    def set_symbol_lookup_date(self, dt):
        if False:
            i = 10
            return i + 15
        'Set the date for which symbols will be resolved to their assets\n        (symbols may map to different firms or underlying assets at\n        different times)\n\n        Parameters\n        ----------\n        dt : datetime\n            The new symbol lookup date.\n        '
        try:
            self._symbol_lookup_date = pd.Timestamp(dt, tz='UTC')
        except ValueError:
            raise UnsupportedDatetimeFormat(input=dt, method='set_symbol_lookup_date')

    @property
    def data_frequency(self):
        if False:
            while True:
                i = 10
        return self.sim_params.data_frequency

    @data_frequency.setter
    def data_frequency(self, value):
        if False:
            for i in range(10):
                print('nop')
        assert value in ('daily', 'minute')
        self.sim_params.data_frequency = value

    @api_method
    @disallowed_in_before_trading_start(OrderInBeforeTradingStart())
    def order_percent(self, asset, percent, limit_price=None, stop_price=None, style=None):
        if False:
            return 10
        'Place an order in the specified asset corresponding to the given\n        percent of the current portfolio value.\n\n        Parameters\n        ----------\n        asset : Asset\n            The asset that this order is for.\n        percent : float\n            The percentage of the portfolio value to allocate to ``asset``.\n            This is specified as a decimal, for example: 0.50 means 50%.\n        limit_price : float, optional\n            The limit price for the order.\n        stop_price : float, optional\n            The stop price for the order.\n        style : ExecutionStyle\n            The execution style for the order.\n\n        Returns\n        -------\n        order_id : str\n            The unique identifier for this order.\n\n        Notes\n        -----\n        See :func:`zipline.api.order` for more information about\n        ``limit_price``, ``stop_price``, and ``style``\n\n        See Also\n        --------\n        :class:`zipline.finance.execution.ExecutionStyle`\n        :func:`zipline.api.order`\n        :func:`zipline.api.order_value`\n        '
        if not self._can_order_asset(asset):
            return None
        amount = self._calculate_order_percent_amount(asset, percent)
        return self.order(asset, amount, limit_price=limit_price, stop_price=stop_price, style=style)

    def _calculate_order_percent_amount(self, asset, percent):
        if False:
            while True:
                i = 10
        value = self.portfolio.portfolio_value * percent
        return self._calculate_order_value_amount(asset, value)

    @api_method
    @disallowed_in_before_trading_start(OrderInBeforeTradingStart())
    def order_target(self, asset, target, limit_price=None, stop_price=None, style=None):
        if False:
            for i in range(10):
                print('nop')
        "Place an order to adjust a position to a target number of shares. If\n        the position doesn't already exist, this is equivalent to placing a new\n        order. If the position does exist, this is equivalent to placing an\n        order for the difference between the target number of shares and the\n        current number of shares.\n\n        Parameters\n        ----------\n        asset : Asset\n            The asset that this order is for.\n        target : int\n            The desired number of shares of ``asset``.\n        limit_price : float, optional\n            The limit price for the order.\n        stop_price : float, optional\n            The stop price for the order.\n        style : ExecutionStyle\n            The execution style for the order.\n\n        Returns\n        -------\n        order_id : str\n            The unique identifier for this order.\n\n\n        Notes\n        -----\n        ``order_target`` does not take into account any open orders. For\n        example:\n\n        .. code-block:: python\n\n           order_target(sid(0), 10)\n           order_target(sid(0), 10)\n\n        This code will result in 20 shares of ``sid(0)`` because the first\n        call to ``order_target`` will not have been filled when the second\n        ``order_target`` call is made.\n\n        See :func:`zipline.api.order` for more information about\n        ``limit_price``, ``stop_price``, and ``style``\n\n        See Also\n        --------\n        :class:`zipline.finance.execution.ExecutionStyle`\n        :func:`zipline.api.order`\n        :func:`zipline.api.order_target_percent`\n        :func:`zipline.api.order_target_value`\n        "
        if not self._can_order_asset(asset):
            return None
        amount = self._calculate_order_target_amount(asset, target)
        return self.order(asset, amount, limit_price=limit_price, stop_price=stop_price, style=style)

    def _calculate_order_target_amount(self, asset, target):
        if False:
            i = 10
            return i + 15
        if asset in self.portfolio.positions:
            current_position = self.portfolio.positions[asset].amount
            target -= current_position
        return target

    @api_method
    @disallowed_in_before_trading_start(OrderInBeforeTradingStart())
    def order_target_value(self, asset, target, limit_price=None, stop_price=None, style=None):
        if False:
            while True:
                i = 10
        "Place an order to adjust a position to a target value. If\n        the position doesn't already exist, this is equivalent to placing a new\n        order. If the position does exist, this is equivalent to placing an\n        order for the difference between the target value and the\n        current value.\n        If the Asset being ordered is a Future, the 'target value' calculated\n        is actually the target exposure, as Futures have no 'value'.\n\n        Parameters\n        ----------\n        asset : Asset\n            The asset that this order is for.\n        target : float\n            The desired total value of ``asset``.\n        limit_price : float, optional\n            The limit price for the order.\n        stop_price : float, optional\n            The stop price for the order.\n        style : ExecutionStyle\n            The execution style for the order.\n\n        Returns\n        -------\n        order_id : str\n            The unique identifier for this order.\n\n        Notes\n        -----\n        ``order_target_value`` does not take into account any open orders. For\n        example:\n\n        .. code-block:: python\n\n           order_target_value(sid(0), 10)\n           order_target_value(sid(0), 10)\n\n        This code will result in 20 dollars of ``sid(0)`` because the first\n        call to ``order_target_value`` will not have been filled when the\n        second ``order_target_value`` call is made.\n\n        See :func:`zipline.api.order` for more information about\n        ``limit_price``, ``stop_price``, and ``style``\n\n        See Also\n        --------\n        :class:`zipline.finance.execution.ExecutionStyle`\n        :func:`zipline.api.order`\n        :func:`zipline.api.order_target`\n        :func:`zipline.api.order_target_percent`\n        "
        if not self._can_order_asset(asset):
            return None
        target_amount = self._calculate_order_value_amount(asset, target)
        amount = self._calculate_order_target_amount(asset, target_amount)
        return self.order(asset, amount, limit_price=limit_price, stop_price=stop_price, style=style)

    @api_method
    @disallowed_in_before_trading_start(OrderInBeforeTradingStart())
    def order_target_percent(self, asset, target, limit_price=None, stop_price=None, style=None):
        if False:
            i = 10
            return i + 15
        "Place an order to adjust a position to a target percent of the\n        current portfolio value. If the position doesn't already exist, this is\n        equivalent to placing a new order. If the position does exist, this is\n        equivalent to placing an order for the difference between the target\n        percent and the current percent.\n\n        Parameters\n        ----------\n        asset : Asset\n            The asset that this order is for.\n        target : float\n            The desired percentage of the portfolio value to allocate to\n            ``asset``. This is specified as a decimal, for example:\n            0.50 means 50%.\n        limit_price : float, optional\n            The limit price for the order.\n        stop_price : float, optional\n            The stop price for the order.\n        style : ExecutionStyle\n            The execution style for the order.\n\n        Returns\n        -------\n        order_id : str\n            The unique identifier for this order.\n\n        Notes\n        -----\n        ``order_target_value`` does not take into account any open orders. For\n        example:\n\n        .. code-block:: python\n\n           order_target_percent(sid(0), 10)\n           order_target_percent(sid(0), 10)\n\n        This code will result in 20% of the portfolio being allocated to sid(0)\n        because the first call to ``order_target_percent`` will not have been\n        filled when the second ``order_target_percent`` call is made.\n\n        See :func:`zipline.api.order` for more information about\n        ``limit_price``, ``stop_price``, and ``style``\n\n        See Also\n        --------\n        :class:`zipline.finance.execution.ExecutionStyle`\n        :func:`zipline.api.order`\n        :func:`zipline.api.order_target`\n        :func:`zipline.api.order_target_value`\n        "
        if not self._can_order_asset(asset):
            return None
        amount = self._calculate_order_target_percent_amount(asset, target)
        return self.order(asset, amount, limit_price=limit_price, stop_price=stop_price, style=style)

    def _calculate_order_target_percent_amount(self, asset, target):
        if False:
            return 10
        target_amount = self._calculate_order_percent_amount(asset, target)
        return self._calculate_order_target_amount(asset, target_amount)

    @api_method
    @expect_types(share_counts=pd.Series)
    @expect_dtypes(share_counts=int64_dtype)
    def batch_market_order(self, share_counts):
        if False:
            while True:
                i = 10
        'Place a batch market order for multiple assets.\n\n        Parameters\n        ----------\n        share_counts : pd.Series[Asset -> int]\n            Map from asset to number of shares to order for that asset.\n\n        Returns\n        -------\n        order_ids : pd.Index[str]\n            Index of ids for newly-created orders.\n        '
        style = MarketOrder()
        order_args = [(asset, amount, style) for (asset, amount) in iteritems(share_counts) if amount]
        return self.blotter.batch_order(order_args)

    @error_keywords(sid='Keyword argument `sid` is no longer supported for get_open_orders. Use `asset` instead.')
    @api_method
    def get_open_orders(self, asset=None):
        if False:
            for i in range(10):
                print('nop')
        'Retrieve all of the current open orders.\n\n        Parameters\n        ----------\n        asset : Asset\n            If passed and not None, return only the open orders for the given\n            asset instead of all open orders.\n\n        Returns\n        -------\n        open_orders : dict[list[Order]] or list[Order]\n            If no asset is passed this will return a dict mapping Assets\n            to a list containing all the open orders for the asset.\n            If an asset is passed then this will return a list of the open\n            orders for this asset.\n        '
        if asset is None:
            return {key: [order.to_api_obj() for order in orders] for (key, orders) in iteritems(self.blotter.open_orders) if orders}
        if asset in self.blotter.open_orders:
            orders = self.blotter.open_orders[asset]
            return [order.to_api_obj() for order in orders]
        return []

    @api_method
    def get_order(self, order_id):
        if False:
            print('Hello World!')
        'Lookup an order based on the order id returned from one of the\n        order functions.\n\n        Parameters\n        ----------\n        order_id : str\n            The unique identifier for the order.\n\n        Returns\n        -------\n        order : Order\n            The order object.\n        '
        if order_id in self.blotter.orders:
            return self.blotter.orders[order_id].to_api_obj()

    @api_method
    def cancel_order(self, order_param):
        if False:
            i = 10
            return i + 15
        'Cancel an open order.\n\n        Parameters\n        ----------\n        order_param : str or Order\n            The order_id or order object to cancel.\n        '
        order_id = order_param
        if isinstance(order_param, zipline.protocol.Order):
            order_id = order_param.id
        self.blotter.cancel(order_id)

    @api_method
    @require_initialized(HistoryInInitialize())
    def history(self, bar_count, frequency, field, ffill=True):
        if False:
            i = 10
            return i + 15
        'DEPRECATED: use ``data.history`` instead.\n        '
        warnings.warn('The `history` method is deprecated.  Use `data.history` instead.', category=ZiplineDeprecationWarning, stacklevel=4)
        return self.get_history_window(bar_count, frequency, self._calculate_universe(), field, ffill)

    def get_history_window(self, bar_count, frequency, assets, field, ffill):
        if False:
            for i in range(10):
                print('nop')
        if not self._in_before_trading_start:
            return self.data_portal.get_history_window(assets, self.datetime, bar_count, frequency, field, self.data_frequency, ffill)
        else:
            adjusted_dt = self.trading_calendar.previous_minute(self.datetime)
            window = self.data_portal.get_history_window(assets, adjusted_dt, bar_count, frequency, field, self.data_frequency, ffill)
            adjs = self.data_portal.get_adjustments(assets, field, adjusted_dt, self.datetime)
            window = window * adjs
            return window

    def register_account_control(self, control):
        if False:
            return 10
        '\n        Register a new AccountControl to be checked on each bar.\n        '
        if self.initialized:
            raise RegisterAccountControlPostInit()
        self.account_controls.append(control)

    def validate_account_controls(self):
        if False:
            return 10
        for control in self.account_controls:
            control.validate(self.portfolio, self.account, self.get_datetime(), self.trading_client.current_data)

    @api_method
    def set_max_leverage(self, max_leverage):
        if False:
            return 10
        'Set a limit on the maximum leverage of the algorithm.\n\n        Parameters\n        ----------\n        max_leverage : float\n            The maximum leverage for the algorithm. If not provided there will\n            be no maximum.\n        '
        control = MaxLeverage(max_leverage)
        self.register_account_control(control)

    @api_method
    def set_min_leverage(self, min_leverage, grace_period):
        if False:
            i = 10
            return i + 15
        'Set a limit on the minimum leverage of the algorithm.\n\n        Parameters\n        ----------\n        min_leverage : float\n            The minimum leverage for the algorithm.\n        grace_period : pd.Timedelta\n            The offset from the start date used to enforce a minimum leverage.\n        '
        deadline = self.sim_params.start_session + grace_period
        control = MinLeverage(min_leverage, deadline)
        self.register_account_control(control)

    def register_trading_control(self, control):
        if False:
            return 10
        '\n        Register a new TradingControl to be checked prior to order calls.\n        '
        if self.initialized:
            raise RegisterTradingControlPostInit()
        self.trading_controls.append(control)

    @api_method
    def set_max_position_size(self, asset=None, max_shares=None, max_notional=None, on_error='fail'):
        if False:
            print('Hello World!')
        "Set a limit on the number of shares and/or dollar value held for the\n        given sid. Limits are treated as absolute values and are enforced at\n        the time that the algo attempts to place an order for sid. This means\n        that it's possible to end up with more than the max number of shares\n        due to splits/dividends, and more than the max notional due to price\n        improvement.\n\n        If an algorithm attempts to place an order that would result in\n        increasing the absolute value of shares/dollar value exceeding one of\n        these limits, raise a TradingControlException.\n\n        Parameters\n        ----------\n        asset : Asset, optional\n            If provided, this sets the guard only on positions in the given\n            asset.\n        max_shares : int, optional\n            The maximum number of shares to hold for an asset.\n        max_notional : float, optional\n            The maximum value to hold for an asset.\n        "
        control = MaxPositionSize(asset=asset, max_shares=max_shares, max_notional=max_notional, on_error=on_error)
        self.register_trading_control(control)

    @api_method
    def set_max_order_size(self, asset=None, max_shares=None, max_notional=None, on_error='fail'):
        if False:
            for i in range(10):
                print('nop')
        'Set a limit on the number of shares and/or dollar value of any single\n        order placed for sid.  Limits are treated as absolute values and are\n        enforced at the time that the algo attempts to place an order for sid.\n\n        If an algorithm attempts to place an order that would result in\n        exceeding one of these limits, raise a TradingControlException.\n\n        Parameters\n        ----------\n        asset : Asset, optional\n            If provided, this sets the guard only on positions in the given\n            asset.\n        max_shares : int, optional\n            The maximum number of shares that can be ordered at one time.\n        max_notional : float, optional\n            The maximum value that can be ordered at one time.\n        '
        control = MaxOrderSize(asset=asset, max_shares=max_shares, max_notional=max_notional, on_error=on_error)
        self.register_trading_control(control)

    @api_method
    def set_max_order_count(self, max_count, on_error='fail'):
        if False:
            while True:
                i = 10
        'Set a limit on the number of orders that can be placed in a single\n        day.\n\n        Parameters\n        ----------\n        max_count : int\n            The maximum number of orders that can be placed on any single day.\n        '
        control = MaxOrderCount(on_error, max_count)
        self.register_trading_control(control)

    @api_method
    def set_do_not_order_list(self, restricted_list, on_error='fail'):
        if False:
            for i in range(10):
                print('nop')
        'Set a restriction on which assets can be ordered.\n\n        Parameters\n        ----------\n        restricted_list : container[Asset], SecurityList\n            The assets that cannot be ordered.\n        '
        if isinstance(restricted_list, SecurityList):
            warnings.warn('`set_do_not_order_list(security_lists.leveraged_etf_list)` is deprecated. Use `set_asset_restrictions(security_lists.restrict_leveraged_etfs)` instead.', category=ZiplineDeprecationWarning, stacklevel=2)
            restrictions = SecurityListRestrictions(restricted_list)
        else:
            warnings.warn('`set_do_not_order_list(container_of_assets)` is deprecated. Create a zipline.finance.asset_restrictions.StaticRestrictions object with a container of assets and use `set_asset_restrictions(StaticRestrictions(container_of_assets))` instead.', category=ZiplineDeprecationWarning, stacklevel=2)
            restrictions = StaticRestrictions(restricted_list)
        self.set_asset_restrictions(restrictions, on_error)

    @api_method
    @expect_types(restrictions=Restrictions, on_error=str)
    def set_asset_restrictions(self, restrictions, on_error='fail'):
        if False:
            i = 10
            return i + 15
        'Set a restriction on which assets can be ordered.\n\n        Parameters\n        ----------\n        restricted_list : Restrictions\n            An object providing information about restricted assets.\n\n        See Also\n        --------\n        zipline.finance.asset_restrictions.Restrictions\n        '
        control = RestrictedListOrder(on_error, restrictions)
        self.register_trading_control(control)
        self.restrictions |= restrictions

    @api_method
    def set_long_only(self, on_error='fail'):
        if False:
            print('Hello World!')
        'Set a rule specifying that this algorithm cannot take short\n        positions.\n        '
        self.register_trading_control(LongOnly(on_error))

    @api_method
    @require_not_initialized(AttachPipelineAfterInitialize())
    @expect_types(pipeline=Pipeline, name=string_types, chunks=(int, Iterable, type(None)))
    def attach_pipeline(self, pipeline, name, chunks=None, eager=True):
        if False:
            while True:
                i = 10
        'Register a pipeline to be computed at the start of each day.\n\n        Parameters\n        ----------\n        pipeline : Pipeline\n            The pipeline to have computed.\n        name : str\n            The name of the pipeline.\n        chunks : int or iterator, optional\n            The number of days to compute pipeline results for. Increasing\n            this number will make it longer to get the first results but\n            may improve the total runtime of the simulation. If an iterator\n            is passed, we will run in chunks based on values of the iterator.\n            Default is True.\n        eager : bool, optional\n            Whether or not to compute this pipeline prior to\n            before_trading_start.\n\n        Returns\n        -------\n        pipeline : Pipeline\n            Returns the pipeline that was attached unchanged.\n\n        See Also\n        --------\n        :func:`zipline.api.pipeline_output`\n        '
        if chunks is None:
            chunks = chain([5], repeat(126))
        elif isinstance(chunks, int):
            chunks = repeat(chunks)
        if name in self._pipelines:
            raise DuplicatePipelineName(name=name)
        self._pipelines[name] = AttachedPipeline(pipeline, iter(chunks), eager)
        return pipeline

    @api_method
    @require_initialized(PipelineOutputDuringInitialize())
    def pipeline_output(self, name):
        if False:
            i = 10
            return i + 15
        '\n        Get results of the pipeline attached by with name ``name``.\n\n        Parameters\n        ----------\n        name : str\n            Name of the pipeline from which to fetch results.\n\n        Returns\n        -------\n        results : pd.DataFrame\n            DataFrame containing the results of the requested pipeline for\n            the current simulation date.\n\n        Raises\n        ------\n        NoSuchPipeline\n            Raised when no pipeline with the name `name` has been registered.\n\n        See Also\n        --------\n        :func:`zipline.api.attach_pipeline`\n        :meth:`zipline.pipeline.engine.PipelineEngine.run_pipeline`\n        '
        try:
            (pipe, chunks, _) = self._pipelines[name]
        except KeyError:
            raise NoSuchPipeline(name=name, valid=list(self._pipelines.keys()))
        return self._pipeline_output(pipe, chunks, name)

    def _pipeline_output(self, pipeline, chunks, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Internal implementation of `pipeline_output`.\n        '
        today = normalize_date(self.get_datetime())
        try:
            data = self._pipeline_cache.get(name, today)
        except KeyError:
            (data, valid_until) = self.run_pipeline(pipeline, today, next(chunks))
            self._pipeline_cache.set(name, data, valid_until)
        try:
            return data.loc[today]
        except KeyError:
            return pd.DataFrame(index=[], columns=data.columns)

    def run_pipeline(self, pipeline, start_session, chunksize):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute `pipeline`, providing values for at least `start_date`.\n\n        Produces a DataFrame containing data for days between `start_date` and\n        `end_date`, where `end_date` is defined by:\n\n            `end_date = min(start_date + chunksize trading days,\n                            simulation_end)`\n\n        Returns\n        -------\n        (data, valid_until) : tuple (pd.DataFrame, pd.Timestamp)\n\n        See Also\n        --------\n        PipelineEngine.run_pipeline\n        '
        sessions = self.trading_calendar.all_sessions
        start_date_loc = sessions.get_loc(start_session)
        sim_end_session = self.sim_params.end_session
        end_loc = min(start_date_loc + chunksize, sessions.get_loc(sim_end_session))
        end_session = sessions[end_loc]
        return (self.engine.run_pipeline(pipeline, start_session, end_session), end_session)

    @staticmethod
    def default_pipeline_domain(calendar):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get a default pipeline domain for algorithms running on ``calendar``.\n\n        This will be used to infer a domain for pipelines that only use generic\n        datasets when running in the context of a TradingAlgorithm.\n        '
        return _DEFAULT_DOMAINS.get(calendar.name, domain.GENERIC)

    @staticmethod
    def default_fetch_csv_country_code(calendar):
        if False:
            print('Hello World!')
        '\n        Get a default country_code to use for fetch_csv symbol lookups.\n\n        This will be used to disambiguate symbol lookups for fetch_csv calls if\n        our asset db contains entries with the same ticker spread across\n        multiple\n        '
        return _DEFAULT_FETCH_CSV_COUNTRY_CODES.get(calendar.name)

    @classmethod
    def all_api_methods(cls):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a list of all the TradingAlgorithm API methods.\n        '
        return [fn for fn in itervalues(vars(cls)) if getattr(fn, 'is_api_method', False)]
_DEFAULT_DOMAINS = {d.calendar_name: d for d in domain.BUILT_IN_DOMAINS}
_DEFAULT_FETCH_CSV_COUNTRY_CODES = {d.calendar_name: d.country_code for d in domain.BUILT_IN_DOMAINS}
_DEFAULT_FETCH_CSV_COUNTRY_CODES['us_futures'] = 'US'