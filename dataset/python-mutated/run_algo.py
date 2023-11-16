import click
import os
import sys
import warnings
try:
    from pygments import highlight
    from pygments.lexers import PythonLexer
    from pygments.formatters import TerminalFormatter
    PYGMENTS = True
except ImportError:
    PYGMENTS = False
import logbook
import pandas as pd
import six
from toolz import concatv
from trading_calendars import get_calendar
from zipline.data import bundles
from zipline.data.benchmarks import get_benchmark_returns_from_file
from zipline.data.data_portal import DataPortal
from zipline.finance import metrics
from zipline.finance.trading import SimulationParameters
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.loaders import USEquityPricingLoader
import zipline.utils.paths as pth
from zipline.extensions import load
from zipline.errors import SymbolNotFound
from zipline.algorithm import TradingAlgorithm, NoBenchmark
from zipline.finance.blotter import Blotter
log = logbook.Logger(__name__)

class _RunAlgoError(click.ClickException, ValueError):
    """Signal an error that should have a different message if invoked from
    the cli.

    Parameters
    ----------
    pyfunc_msg : str
        The message that will be shown when called as a python function.
    cmdline_msg : str, optional
        The message that will be shown on the command line. If not provided,
        this will be the same as ``pyfunc_msg`
    """
    exit_code = 1

    def __init__(self, pyfunc_msg, cmdline_msg=None):
        if False:
            return 10
        if cmdline_msg is None:
            cmdline_msg = pyfunc_msg
        super(_RunAlgoError, self).__init__(cmdline_msg)
        self.pyfunc_msg = pyfunc_msg

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.pyfunc_msg

def _run(handle_data, initialize, before_trading_start, analyze, algofile, algotext, defines, data_frequency, capital_base, bundle, bundle_timestamp, start, end, output, trading_calendar, print_algo, metrics_set, local_namespace, environ, blotter, benchmark_spec):
    if False:
        while True:
            i = 10
    'Run a backtest for the given algorithm.\n\n    This is shared between the cli and :func:`zipline.run_algo`.\n    '
    bundle_data = bundles.load(bundle, environ, bundle_timestamp)
    if trading_calendar is None:
        trading_calendar = get_calendar('XNYS')
    if trading_calendar.session_distance(start, end) < 1:
        raise _RunAlgoError('There are no trading days between %s and %s' % (start.date(), end.date()))
    (benchmark_sid, benchmark_returns) = benchmark_spec.resolve(asset_finder=bundle_data.asset_finder, start_date=start, end_date=end)
    if algotext is not None:
        if local_namespace:
            ip = get_ipython()
            namespace = ip.user_ns
        else:
            namespace = {}
        for assign in defines:
            try:
                (name, value) = assign.split('=', 2)
            except ValueError:
                raise ValueError('invalid define %r, should be of the form name=value' % assign)
            try:
                namespace[name] = eval(value, namespace)
            except Exception as e:
                raise ValueError('failed to execute definition for name %r: %s' % (name, e))
    elif defines:
        raise _RunAlgoError('cannot pass define without `algotext`', "cannot pass '-D' / '--define' without '-t' / '--algotext'")
    else:
        namespace = {}
        if algofile is not None:
            algotext = algofile.read()
    if print_algo:
        if PYGMENTS:
            highlight(algotext, PythonLexer(), TerminalFormatter(), outfile=sys.stdout)
        else:
            click.echo(algotext)
    first_trading_day = bundle_data.equity_minute_bar_reader.first_trading_day
    data = DataPortal(bundle_data.asset_finder, trading_calendar=trading_calendar, first_trading_day=first_trading_day, equity_minute_reader=bundle_data.equity_minute_bar_reader, equity_daily_reader=bundle_data.equity_daily_bar_reader, adjustment_reader=bundle_data.adjustment_reader)
    pipeline_loader = USEquityPricingLoader.without_fx(bundle_data.equity_daily_bar_reader, bundle_data.adjustment_reader)

    def choose_loader(column):
        if False:
            print('Hello World!')
        if column in USEquityPricing.columns:
            return pipeline_loader
        raise ValueError('No PipelineLoader registered for column %s.' % column)
    if isinstance(metrics_set, six.string_types):
        try:
            metrics_set = metrics.load(metrics_set)
        except ValueError as e:
            raise _RunAlgoError(str(e))
    if isinstance(blotter, six.string_types):
        try:
            blotter = load(Blotter, blotter)
        except ValueError as e:
            raise _RunAlgoError(str(e))
    try:
        perf = TradingAlgorithm(namespace=namespace, data_portal=data, get_pipeline_loader=choose_loader, trading_calendar=trading_calendar, sim_params=SimulationParameters(start_session=start, end_session=end, trading_calendar=trading_calendar, capital_base=capital_base, data_frequency=data_frequency), metrics_set=metrics_set, blotter=blotter, benchmark_returns=benchmark_returns, benchmark_sid=benchmark_sid, **{'initialize': initialize, 'handle_data': handle_data, 'before_trading_start': before_trading_start, 'analyze': analyze} if algotext is None else {'algo_filename': getattr(algofile, 'name', '<algorithm>'), 'script': algotext}).run()
    except NoBenchmark:
        raise _RunAlgoError('No ``benchmark_spec`` was provided, and ``zipline.api.set_benchmark`` was not called in ``initialize``.', "Neither '--benchmark-symbol' nor '--benchmark-sid' was provided, and ``zipline.api.set_benchmark`` was not called in ``initialize``. Did you mean to pass '--no-benchmark'?")
    if output == '-':
        click.echo(str(perf))
    elif output != os.devnull:
        perf.to_pickle(output)
    return perf
_loaded_extensions = set()

def load_extensions(default, extensions, strict, environ, reload=False):
    if False:
        while True:
            i = 10
    'Load all of the given extensions. This should be called by run_algo\n    or the cli.\n\n    Parameters\n    ----------\n    default : bool\n        Load the default exension (~/.zipline/extension.py)?\n    extension : iterable[str]\n        The paths to the extensions to load. If the path ends in ``.py`` it is\n        treated as a script and executed. If it does not end in ``.py`` it is\n        treated as a module to be imported.\n    strict : bool\n        Should failure to load an extension raise. If this is false it will\n        still warn.\n    environ : mapping\n        The environment to use to find the default extension path.\n    reload : bool, optional\n        Reload any extensions that have already been loaded.\n    '
    if default:
        default_extension_path = pth.default_extension(environ=environ)
        pth.ensure_file(default_extension_path)
        extensions = concatv([default_extension_path], extensions)
    for ext in extensions:
        if ext in _loaded_extensions and (not reload):
            continue
        try:
            if ext.endswith('.py'):
                with open(ext) as f:
                    ns = {}
                    six.exec_(compile(f.read(), ext, 'exec'), ns, ns)
            else:
                __import__(ext)
        except Exception as e:
            if strict:
                raise
            warnings.warn('Failed to load extension: %r\n%s' % (ext, e), stacklevel=2)
        else:
            _loaded_extensions.add(ext)

def run_algorithm(start, end, initialize, capital_base, handle_data=None, before_trading_start=None, analyze=None, data_frequency='daily', bundle='quantopian-quandl', bundle_timestamp=None, trading_calendar=None, metrics_set='default', benchmark_returns=None, default_extension=True, extensions=(), strict_extensions=True, environ=os.environ, blotter='default'):
    if False:
        return 10
    "\n    Run a trading algorithm.\n\n    Parameters\n    ----------\n    start : datetime\n        The start date of the backtest.\n    end : datetime\n        The end date of the backtest..\n    initialize : callable[context -> None]\n        The initialize function to use for the algorithm. This is called once\n        at the very begining of the backtest and should be used to set up\n        any state needed by the algorithm.\n    capital_base : float\n        The starting capital for the backtest.\n    handle_data : callable[(context, BarData) -> None], optional\n        The handle_data function to use for the algorithm. This is called\n        every minute when ``data_frequency == 'minute'`` or every day\n        when ``data_frequency == 'daily'``.\n    before_trading_start : callable[(context, BarData) -> None], optional\n        The before_trading_start function for the algorithm. This is called\n        once before each trading day (after initialize on the first day).\n    analyze : callable[(context, pd.DataFrame) -> None], optional\n        The analyze function to use for the algorithm. This function is called\n        once at the end of the backtest and is passed the context and the\n        performance data.\n    data_frequency : {'daily', 'minute'}, optional\n        The data frequency to run the algorithm at.\n    bundle : str, optional\n        The name of the data bundle to use to load the data to run the backtest\n        with. This defaults to 'quantopian-quandl'.\n    bundle_timestamp : datetime, optional\n        The datetime to lookup the bundle data for. This defaults to the\n        current time.\n    trading_calendar : TradingCalendar, optional\n        The trading calendar to use for your backtest.\n    metrics_set : iterable[Metric] or str, optional\n        The set of metrics to compute in the simulation. If a string is passed,\n        resolve the set with :func:`zipline.finance.metrics.load`.\n    benchmark_returns : pd.Series, optional\n        Series of returns to use as the benchmark.\n    default_extension : bool, optional\n        Should the default zipline extension be loaded. This is found at\n        ``$ZIPLINE_ROOT/extension.py``\n    extensions : iterable[str], optional\n        The names of any other extensions to load. Each element may either be\n        a dotted module path like ``a.b.c`` or a path to a python file ending\n        in ``.py`` like ``a/b/c.py``.\n    strict_extensions : bool, optional\n        Should the run fail if any extensions fail to load. If this is false,\n        a warning will be raised instead.\n    environ : mapping[str -> str], optional\n        The os environment to use. Many extensions use this to get parameters.\n        This defaults to ``os.environ``.\n    blotter : str or zipline.finance.blotter.Blotter, optional\n        Blotter to use with this algorithm. If passed as a string, we look for\n        a blotter construction function registered with\n        ``zipline.extensions.register`` and call it with no parameters.\n        Default is a :class:`zipline.finance.blotter.SimulationBlotter` that\n        never cancels orders.\n\n    Returns\n    -------\n    perf : pd.DataFrame\n        The daily performance of the algorithm.\n\n    See Also\n    --------\n    zipline.data.bundles.bundles : The available data bundles.\n    "
    load_extensions(default_extension, extensions, strict_extensions, environ)
    benchmark_spec = BenchmarkSpec.from_returns(benchmark_returns)
    return _run(handle_data=handle_data, initialize=initialize, before_trading_start=before_trading_start, analyze=analyze, algofile=None, algotext=None, defines=(), data_frequency=data_frequency, capital_base=capital_base, bundle=bundle, bundle_timestamp=bundle_timestamp, start=start, end=end, output=os.devnull, trading_calendar=trading_calendar, print_algo=False, metrics_set=metrics_set, local_namespace=False, environ=environ, blotter=blotter, benchmark_spec=benchmark_spec)

class BenchmarkSpec(object):
    """
    Helper for different ways we can get benchmark data for the Zipline CLI and
    zipline.utils.run_algo.run_algorithm.

    Parameters
    ----------
    benchmark_returns : pd.Series, optional
        Series of returns to use as the benchmark.
    benchmark_file : str or file
        File containing a csv with `date` and `return` columns, to be read as
        the benchmark.
    benchmark_sid : int, optional
        Sid of the asset to use as a benchmark.
    benchmark_symbol : str, optional
        Symbol of the asset to use as a benchmark. Symbol will be looked up as
        of the end date of the backtest.
    no_benchmark : bool
        Flag indicating that no benchmark is configured. Benchmark-dependent
        metrics will be calculated using a dummy benchmark of all-zero returns.
    """

    def __init__(self, benchmark_returns, benchmark_file, benchmark_sid, benchmark_symbol, no_benchmark):
        if False:
            while True:
                i = 10
        self.benchmark_returns = benchmark_returns
        self.benchmark_file = benchmark_file
        self.benchmark_sid = benchmark_sid
        self.benchmark_symbol = benchmark_symbol
        self.no_benchmark = no_benchmark

    @classmethod
    def from_cli_params(cls, benchmark_sid, benchmark_symbol, benchmark_file, no_benchmark):
        if False:
            return 10
        return cls(benchmark_returns=None, benchmark_sid=benchmark_sid, benchmark_symbol=benchmark_symbol, benchmark_file=benchmark_file, no_benchmark=no_benchmark)

    @classmethod
    def from_returns(cls, benchmark_returns):
        if False:
            return 10
        return cls(benchmark_returns=benchmark_returns, benchmark_file=None, benchmark_sid=None, benchmark_symbol=None, no_benchmark=benchmark_returns is None)

    def resolve(self, asset_finder, start_date, end_date):
        if False:
            i = 10
            return i + 15
        '\n        Resolve inputs into values to be passed to TradingAlgorithm.\n\n        Returns a pair of ``(benchmark_sid, benchmark_returns)`` with at most\n        one non-None value. Both values may be None if no benchmark source has\n        been configured.\n\n        Parameters\n        ----------\n        asset_finder : zipline.assets.AssetFinder\n            Asset finder for the algorithm to be run.\n        start_date : pd.Timestamp\n            Start date of the algorithm to be run.\n        end_date : pd.Timestamp\n            End date of the algorithm to be run.\n\n        Returns\n        -------\n        benchmark_sid : int\n            Sid to use as benchmark.\n        benchmark_returns : pd.Series\n            Series of returns to use as benchmark.\n        '
        if self.benchmark_returns is not None:
            benchmark_sid = None
            benchmark_returns = self.benchmark_returns
        elif self.benchmark_file is not None:
            benchmark_sid = None
            benchmark_returns = get_benchmark_returns_from_file(self.benchmark_file)
        elif self.benchmark_sid is not None:
            benchmark_sid = self.benchmark_sid
            benchmark_returns = None
        elif self.benchmark_symbol is not None:
            try:
                asset = asset_finder.lookup_symbol(self.benchmark_symbol, as_of_date=end_date)
                benchmark_sid = asset.sid
                benchmark_returns = None
            except SymbolNotFound:
                raise _RunAlgoError('Symbol %r as a benchmark not found in this bundle.' % self.benchmark_symbol)
        elif self.no_benchmark:
            benchmark_sid = None
            benchmark_returns = self._zero_benchmark_returns(start_date=start_date, end_date=end_date)
        else:
            log.warn('No benchmark configured. Assuming algorithm calls set_benchmark.')
            log.warn('Pass --benchmark-sid, --benchmark-symbol, or --benchmark-file to set a source of benchmark returns.')
            log.warn('Pass --no-benchmark to use a dummy benchmark of zero returns.')
            benchmark_sid = None
            benchmark_returns = None
        return (benchmark_sid, benchmark_returns)

    @staticmethod
    def _zero_benchmark_returns(start_date, end_date):
        if False:
            for i in range(10):
                print('nop')
        return pd.Series(index=pd.date_range(start_date, end_date, tz='utc'), data=0.0)