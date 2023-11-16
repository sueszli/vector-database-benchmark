import warnings
import datetime
from datetime import timedelta
from functools import partial
from textwrap import dedent
from copy import deepcopy
import logbook
import toolz
from logbook import TestHandler, WARNING
from nose_parameterized import parameterized
from six import iteritems, itervalues, string_types
from six.moves import range
from testfixtures import TempDirectory
import numpy as np
import pandas as pd
import pytz
from pandas.core.common import PerformanceWarning
from trading_calendars import get_calendar, register_calendar
import zipline.api
from zipline.api import FixedSlippage
from zipline.assets import Equity, Future, Asset
from zipline.assets.continuous_futures import ContinuousFuture
from zipline.assets.synthetic import make_jagged_equity_info, make_simple_equity_info
from zipline.errors import AccountControlViolation, CannotOrderDelistedAsset, IncompatibleSlippageModel, RegisterTradingControlPostInit, ScheduleFunctionInvalidCalendar, SetCancelPolicyPostInit, SymbolNotFound, TradingControlViolation, UnsupportedCancelPolicy, UnsupportedDatetimeFormat, ZeroCapitalError
from zipline.finance.commission import PerShare, PerTrade
from zipline.finance.execution import LimitOrder
from zipline.finance.order import ORDER_STATUS
from zipline.finance.trading import SimulationParameters
from zipline.finance.asset_restrictions import Restriction, HistoricalRestrictions, StaticRestrictions, RESTRICTION_STATES
from zipline.finance.controls import AssetDateBounds
from zipline.testing import FakeDataPortal, create_daily_df_for_asset, create_data_portal_from_trade_history, create_minute_df_for_asset, make_test_handler, make_trade_data_for_asset_info, parameter_space, str_to_seconds, to_utc
from zipline.testing import RecordBatchBlotter
import zipline.testing.fixtures as zf
from zipline.test_algorithms import access_account_in_init, access_portfolio_in_init, api_algo, api_get_environment_algo, api_symbol_algo, handle_data_api, handle_data_noop, initialize_api, initialize_noop, noop_algo, record_float_magic, record_variables, call_with_kwargs, call_without_kwargs, call_with_bad_kwargs_current, call_with_bad_kwargs_history, bad_type_history_assets, bad_type_history_fields, bad_type_history_bar_count, bad_type_history_frequency, bad_type_history_assets_kwarg_list, bad_type_current_assets, bad_type_current_fields, bad_type_can_trade_assets, bad_type_is_stale_assets, bad_type_history_assets_kwarg, bad_type_history_fields_kwarg, bad_type_history_bar_count_kwarg, bad_type_history_frequency_kwarg, bad_type_current_assets_kwarg, bad_type_current_fields_kwarg, call_with_bad_kwargs_get_open_orders, call_with_good_kwargs_get_open_orders, call_with_no_kwargs_get_open_orders, empty_positions, no_handle_data
from zipline.testing.predicates import assert_equal
from zipline.utils.api_support import ZiplineAPI
from zipline.utils.context_tricks import CallbackManager, nop_context
from zipline.utils.events import date_rules, time_rules, Always, ComposedRule, Never, OncePerDay
import zipline.utils.factory as factory
_multiprocess_can_split_ = False

class TestRecord(zf.WithMakeAlgo, zf.ZiplineTestCase):
    ASSET_FINDER_EQUITY_SIDS = (133,)
    SIM_PARAMS_DATA_FREQUENCY = 'daily'
    DATA_PORTAL_USE_MINUTE_DATA = False

    def test_record_incr(self):
        if False:
            return 10

        def initialize(self):
            if False:
                while True:
                    i = 10
            self.incr = 0

        def handle_data(self, data):
            if False:
                return 10
            self.incr += 1
            self.record(incr=self.incr)
            name = 'name'
            self.record(name, self.incr)
            zipline.api.record(name, self.incr, 'name2', 2, name3=self.incr)
        output = self.run_algorithm(initialize=initialize, handle_data=handle_data)
        np.testing.assert_array_equal(output['incr'].values, range(1, len(output) + 1))
        np.testing.assert_array_equal(output['name'].values, range(1, len(output) + 1))
        np.testing.assert_array_equal(output['name2'].values, [2] * len(output))
        np.testing.assert_array_equal(output['name3'].values, range(1, len(output) + 1))

class TestMiscellaneousAPI(zf.WithMakeAlgo, zf.ZiplineTestCase):
    START_DATE = pd.Timestamp('2006-01-03', tz='UTC')
    END_DATE = pd.Timestamp('2006-01-04', tz='UTC')
    SIM_PARAMS_DATA_FREQUENCY = 'minute'
    sids = (1, 2)
    BENCHMARK_SID = None

    @classmethod
    def make_equity_info(cls):
        if False:
            for i in range(10):
                print('nop')
        return pd.concat((make_simple_equity_info(cls.sids, '2002-02-1', '2007-01-01'), pd.DataFrame.from_dict({3: {'symbol': 'PLAY', 'start_date': '2002-01-01', 'end_date': '2004-01-01', 'exchange': 'TEST'}, 4: {'symbol': 'PLAY', 'start_date': '2005-01-01', 'end_date': '2006-01-01', 'exchange': 'TEST'}}, orient='index')))

    @classmethod
    def make_futures_info(cls):
        if False:
            i = 10
            return i + 15
        return pd.DataFrame.from_dict({5: {'symbol': 'CLG06', 'root_symbol': 'CL', 'start_date': pd.Timestamp('2005-12-01', tz='UTC'), 'notice_date': pd.Timestamp('2005-12-20', tz='UTC'), 'expiration_date': pd.Timestamp('2006-01-20', tz='UTC'), 'exchange': 'TEST'}, 6: {'root_symbol': 'CL', 'symbol': 'CLK06', 'start_date': pd.Timestamp('2005-12-01', tz='UTC'), 'notice_date': pd.Timestamp('2006-03-20', tz='UTC'), 'expiration_date': pd.Timestamp('2006-04-20', tz='UTC'), 'exchange': 'TEST'}, 7: {'symbol': 'CLQ06', 'root_symbol': 'CL', 'start_date': pd.Timestamp('2005-12-01', tz='UTC'), 'notice_date': pd.Timestamp('2006-06-20', tz='UTC'), 'expiration_date': pd.Timestamp('2006-07-20', tz='UTC'), 'exchange': 'TEST'}, 8: {'symbol': 'CLX06', 'root_symbol': 'CL', 'start_date': pd.Timestamp('2006-02-01', tz='UTC'), 'notice_date': pd.Timestamp('2006-09-20', tz='UTC'), 'expiration_date': pd.Timestamp('2006-10-20', tz='UTC'), 'exchange': 'TEST'}}, orient='index')

    def test_cancel_policy_outside_init(self):
        if False:
            print('Hello World!')
        code = '\nfrom zipline.api import cancel_policy, set_cancel_policy\n\ndef initialize(algo):\n    pass\n\ndef handle_data(algo, data):\n    set_cancel_policy(cancel_policy.NeverCancel())\n'
        algo = self.make_algo(script=code)
        with self.assertRaises(SetCancelPolicyPostInit):
            algo.run()

    def test_cancel_policy_invalid_param(self):
        if False:
            i = 10
            return i + 15
        code = '\nfrom zipline.api import set_cancel_policy\n\ndef initialize(algo):\n    set_cancel_policy("foo")\n\ndef handle_data(algo, data):\n    pass\n'
        algo = self.make_algo(script=code)
        with self.assertRaises(UnsupportedCancelPolicy):
            algo.run()

    def test_zipline_api_resolves_dynamically(self):
        if False:
            while True:
                i = 10
        algo = self.make_algo(initialize=lambda context: None, handle_data=lambda context, data: None)
        for method in algo.all_api_methods():
            name = method.__name__
            sentinel = object()

            def fake_method(*args, **kwargs):
                if False:
                    while True:
                        i = 10
                return sentinel
            setattr(algo, name, fake_method)
            with ZiplineAPI(algo):
                self.assertIs(sentinel, getattr(zipline.api, name)())

    def test_sid_datetime(self):
        if False:
            while True:
                i = 10
        algo_text = '\nfrom zipline.api import sid, get_datetime\n\ndef initialize(context):\n    pass\n\ndef handle_data(context, data):\n    aapl_dt = data.current(sid(1), "last_traded")\n    assert_equal(aapl_dt, get_datetime())\n'
        self.run_algorithm(script=algo_text, namespace={'assert_equal': self.assertEqual})

    def test_datetime_bad_params(self):
        if False:
            i = 10
            return i + 15
        algo_text = '\nfrom zipline.api import get_datetime\nfrom pytz import timezone\n\ndef initialize(context):\n    pass\n\ndef handle_data(context, data):\n    get_datetime(timezone)\n'
        algo = self.make_algo(script=algo_text)
        with self.assertRaises(TypeError):
            algo.run()

    @parameterized.expand([(-1000, 'invalid_base'), (0, 'invalid_base')])
    def test_invalid_capital_base(self, cap_base, name):
        if False:
            print('Hello World!')
        "\n        Test that the appropriate error is being raised and orders aren't\n        filled for algos with capital base <= 0\n        "
        algo_text = '\ndef initialize(context):\n    pass\n\ndef handle_data(context, data):\n    order(sid(24), 1000)\n        '
        sim_params = SimulationParameters(start_session=pd.Timestamp('2006-01-03', tz='UTC'), end_session=pd.Timestamp('2006-01-06', tz='UTC'), capital_base=cap_base, data_frequency='minute', trading_calendar=self.trading_calendar)
        with self.assertRaises(ZeroCapitalError) as exc:
            self.make_algo(script=algo_text, sim_params=sim_params)
        error = exc.exception
        self.assertEqual(str(error), 'initial capital base must be greater than zero')

    def test_get_environment(self):
        if False:
            return 10
        expected_env = {'arena': 'backtest', 'data_frequency': 'minute', 'start': pd.Timestamp('2006-01-03 14:31:00+0000', tz='utc'), 'end': pd.Timestamp('2006-01-04 21:00:00+0000', tz='utc'), 'capital_base': 100000.0, 'platform': 'zipline'}

        def initialize(algo):
            if False:
                i = 10
                return i + 15
            self.assertEqual('zipline', algo.get_environment())
            self.assertEqual(expected_env, algo.get_environment('*'))

        def handle_data(algo, data):
            if False:
                i = 10
                return i + 15
            pass
        self.run_algorithm(initialize=initialize, handle_data=handle_data)

    def test_get_open_orders(self):
        if False:
            while True:
                i = 10

        def initialize(algo):
            if False:
                for i in range(10):
                    print('nop')
            algo.minute = 0

        def handle_data(algo, data):
            if False:
                return 10
            if algo.minute == 0:
                algo.order(algo.sid(1), 1)
                algo.order(algo.sid(2), 1, style=LimitOrder(0.01, asset=algo.sid(2)))
                algo.order(algo.sid(2), 1, style=LimitOrder(0.01, asset=algo.sid(2)))
                algo.order(algo.sid(2), 1, style=LimitOrder(0.01, asset=algo.sid(2)))
                all_orders = algo.get_open_orders()
                self.assertEqual(list(all_orders.keys()), [1, 2])
                self.assertEqual(all_orders[1], algo.get_open_orders(1))
                self.assertEqual(len(all_orders[1]), 1)
                self.assertEqual(all_orders[2], algo.get_open_orders(2))
                self.assertEqual(len(all_orders[2]), 3)
            if algo.minute == 1:
                all_orders = algo.get_open_orders()
                self.assertEqual(list(all_orders.keys()), [2])
                self.assertEqual([], algo.get_open_orders(1))
                orders_2 = algo.get_open_orders(2)
                self.assertEqual(all_orders[2], orders_2)
                self.assertEqual(len(all_orders[2]), 3)
                for order_ in orders_2:
                    algo.cancel_order(order_)
                all_orders = algo.get_open_orders()
                self.assertEqual(all_orders, {})
            algo.minute += 1
        self.run_algorithm(initialize=initialize, handle_data=handle_data)

    def test_schedule_function_custom_cal(self):
        if False:
            i = 10
            return i + 15
        algotext = '\nfrom zipline.api import (\n    schedule_function, get_datetime, time_rules, date_rules, calendars,\n)\n\ndef initialize(context):\n    schedule_function(\n        func=log_nyse_open,\n        date_rule=date_rules.every_day(),\n        time_rule=time_rules.market_open(),\n        calendar=calendars.US_EQUITIES,\n    )\n\n    schedule_function(\n        func=log_nyse_close,\n        date_rule=date_rules.every_day(),\n        time_rule=time_rules.market_close(),\n        calendar=calendars.US_EQUITIES,\n    )\n\n    context.nyse_opens = []\n    context.nyse_closes = []\n\ndef log_nyse_open(context, data):\n    context.nyse_opens.append(get_datetime())\n\ndef log_nyse_close(context, data):\n    context.nyse_closes.append(get_datetime())\n        '
        algo = self.make_algo(script=algotext, sim_params=self.make_simparams(trading_calendar=get_calendar('CMES')))
        algo.run()
        nyse = get_calendar('NYSE')
        for minute in algo.nyse_opens:
            session_label = nyse.minute_to_session_label(minute)
            session_open = nyse.session_open(session_label)
            self.assertEqual(session_open, minute)
        for minute in algo.nyse_closes:
            session_label = nyse.minute_to_session_label(minute)
            session_close = nyse.session_close(session_label)
            self.assertEqual(session_close - timedelta(minutes=1), minute)
        erroring_algotext = dedent("\n            from zipline.api import schedule_function\n            from trading_calendars import get_calendar\n\n            def initialize(context):\n                schedule_function(func=my_func, calendar=get_calendar('XNYS'))\n\n            def my_func(context, data):\n                pass\n            ")
        algo = self.make_algo(script=erroring_algotext, sim_params=self.make_simparams(trading_calendar=get_calendar('CMES')))
        with self.assertRaises(ScheduleFunctionInvalidCalendar):
            algo.run()

    def test_schedule_function(self):
        if False:
            while True:
                i = 10
        us_eastern = pytz.timezone('US/Eastern')

        def incrementer(algo, data):
            if False:
                return 10
            algo.func_called += 1
            curdt = algo.get_datetime().tz_convert(pytz.utc)
            self.assertEqual(curdt, us_eastern.localize(datetime.datetime.combine(curdt.date(), datetime.time(9, 31))))

        def initialize(algo):
            if False:
                while True:
                    i = 10
            algo.func_called = 0
            algo.days = 1
            algo.date = None
            algo.schedule_function(func=incrementer, date_rule=date_rules.every_day(), time_rule=time_rules.market_open())

        def handle_data(algo, data):
            if False:
                for i in range(10):
                    print('nop')
            if not algo.date:
                algo.date = algo.get_datetime().date()
            if algo.date < algo.get_datetime().date():
                algo.days += 1
                algo.date = algo.get_datetime().date()
        algo = self.make_algo(initialize=initialize, handle_data=handle_data)
        algo.run()
        self.assertEqual(algo.func_called, algo.days)

    def test_event_context(self):
        if False:
            print('Hello World!')
        expected_data = []
        collected_data_pre = []
        collected_data_post = []
        function_stack = []

        def pre(data):
            if False:
                while True:
                    i = 10
            function_stack.append(pre)
            collected_data_pre.append(data)

        def post(data):
            if False:
                i = 10
                return i + 15
            function_stack.append(post)
            collected_data_post.append(data)

        def initialize(context):
            if False:
                while True:
                    i = 10
            context.add_event(Always(), f)
            context.add_event(Always(), g)

        def handle_data(context, data):
            if False:
                for i in range(10):
                    print('nop')
            function_stack.append(handle_data)
            expected_data.append(data)

        def f(context, data):
            if False:
                while True:
                    i = 10
            function_stack.append(f)

        def g(context, data):
            if False:
                while True:
                    i = 10
            function_stack.append(g)
        algo = self.make_algo(initialize=initialize, handle_data=handle_data, create_event_context=CallbackManager(pre, post))
        algo.run()
        self.assertEqual(len(expected_data), 780)
        self.assertEqual(collected_data_pre, expected_data)
        self.assertEqual(collected_data_post, expected_data)
        self.assertEqual(len(function_stack), 3900, 'Incorrect number of functions called: %s != 3900' % len(function_stack))
        expected_functions = [pre, handle_data, f, g, post] * 97530
        for (n, (f, g)) in enumerate(zip(function_stack, expected_functions)):
            self.assertEqual(f, g, 'function at position %d was incorrect, expected %s but got %s' % (n, g.__name__, f.__name__))

    @parameterized.expand([('daily',), 'minute'])
    def test_schedule_function_rule_creation(self, mode):
        if False:
            i = 10
            return i + 15

        def nop(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return None
        self.sim_params.data_frequency = mode
        algo = self.make_algo(initialize=nop, handle_data=nop, sim_params=self.sim_params)
        algo.schedule_function(nop, time_rule=Never() & Always())
        event_rule = algo.event_manager._events[1].rule
        self.assertIsInstance(event_rule, OncePerDay)
        self.assertEqual(event_rule.cal, algo.trading_calendar)
        inner_rule = event_rule.rule
        self.assertIsInstance(inner_rule, ComposedRule)
        self.assertEqual(inner_rule.cal, algo.trading_calendar)
        first = inner_rule.first
        second = inner_rule.second
        composer = inner_rule.composer
        self.assertIsInstance(first, Always)
        self.assertEqual(first.cal, algo.trading_calendar)
        self.assertEqual(second.cal, algo.trading_calendar)
        if mode == 'daily':
            self.assertIsInstance(second, Always)
        else:
            self.assertIsInstance(second, ComposedRule)
            self.assertIsInstance(second.first, Never)
            self.assertEqual(second.first.cal, algo.trading_calendar)
            self.assertIsInstance(second.second, Always)
            self.assertEqual(second.second.cal, algo.trading_calendar)
        self.assertIs(composer, ComposedRule.lazy_and)

    def test_asset_lookup(self):
        if False:
            for i in range(10):
                print('nop')
        algo = self.make_algo()
        start_session = pd.Timestamp('2000-01-01', tz='UTC')
        algo.sim_params = algo.sim_params.create_new(start_session, pd.Timestamp('2001-12-01', tz='UTC'))
        with self.assertRaises(SymbolNotFound):
            algo.symbol('PLAY')
        with self.assertRaises(SymbolNotFound):
            algo.symbols('PLAY')
        algo.sim_params = algo.sim_params.create_new(start_session, pd.Timestamp('2002-12-01', tz='UTC'))
        list_result = algo.symbols('PLAY')
        self.assertEqual(3, list_result[0])
        algo.sim_params = algo.sim_params.create_new(start_session, pd.Timestamp('2004-12-01', tz='UTC'))
        self.assertEqual(3, algo.symbol('PLAY'))
        algo.sim_params = algo.sim_params.create_new(start_session, pd.Timestamp('2005-12-01', tz='UTC'))
        self.assertEqual(4, algo.symbol('PLAY'))
        algo.sim_params = algo.sim_params.create_new(start_session, pd.Timestamp('2006-12-01', tz='UTC'))
        self.assertEqual(4, algo.symbol('PLAY'))
        list_result = algo.symbols('PLAY')
        self.assertEqual(4, list_result[0])
        self.assertIsInstance(algo.sid(3), Equity)
        self.assertIsInstance(algo.sid(4), Equity)
        with self.assertRaises(TypeError):
            algo.symbol(1)
        with self.assertRaises(TypeError):
            algo.symbol((1,))
        with self.assertRaises(TypeError):
            algo.symbol({1})
        with self.assertRaises(TypeError):
            algo.symbol([1])
        with self.assertRaises(TypeError):
            algo.symbol({'foo': 'bar'})

    def test_future_symbol(self):
        if False:
            while True:
                i = 10
        ' Tests the future_symbol API function.\n        '
        algo = self.make_algo()
        algo.datetime = pd.Timestamp('2006-12-01', tz='UTC')
        cl = algo.future_symbol('CLG06')
        self.assertEqual(cl.sid, 5)
        self.assertEqual(cl.symbol, 'CLG06')
        self.assertEqual(cl.root_symbol, 'CL')
        self.assertEqual(cl.start_date, pd.Timestamp('2005-12-01', tz='UTC'))
        self.assertEqual(cl.notice_date, pd.Timestamp('2005-12-20', tz='UTC'))
        self.assertEqual(cl.expiration_date, pd.Timestamp('2006-01-20', tz='UTC'))
        with self.assertRaises(SymbolNotFound):
            algo.future_symbol('')
        with self.assertRaises(SymbolNotFound):
            algo.future_symbol('PLAY')
        with self.assertRaises(SymbolNotFound):
            algo.future_symbol('FOOBAR')
        with self.assertRaises(TypeError):
            algo.future_symbol(1)
        with self.assertRaises(TypeError):
            algo.future_symbol((1,))
        with self.assertRaises(TypeError):
            algo.future_symbol({1})
        with self.assertRaises(TypeError):
            algo.future_symbol([1])
        with self.assertRaises(TypeError):
            algo.future_symbol({'foo': 'bar'})

class TestSetSymbolLookupDate(zf.WithMakeAlgo, zf.ZiplineTestCase):
    START_DATE = pd.Timestamp('2006-01-03', tz='UTC')
    END_DATE = pd.Timestamp('2006-01-06', tz='UTC')
    SIM_PARAMS_START_DATE = pd.Timestamp('2006-01-04', tz='UTC')
    SIM_PARAMS_DATA_FREQUENCY = 'daily'
    DATA_PORTAL_USE_MINUTE_DATA = False
    BENCHMARK_SID = 3

    @classmethod
    def make_equity_info(cls):
        if False:
            i = 10
            return i + 15
        dates = pd.date_range(cls.START_DATE, cls.END_DATE)
        assert len(dates) == 4, 'Expected four dates.'
        cls.sids = [1, 2, 3]
        cls.asset_starts = [dates[0], dates[2]]
        cls.asset_ends = [dates[1], dates[3]]
        return pd.DataFrame.from_records([{'symbol': 'DUP', 'start_date': cls.asset_starts[0], 'end_date': cls.asset_ends[0], 'exchange': 'TEST', 'asset_name': 'FIRST'}, {'symbol': 'DUP', 'start_date': cls.asset_starts[1], 'end_date': cls.asset_ends[1], 'exchange': 'TEST', 'asset_name': 'SECOND'}, {'symbol': 'BENCH', 'start_date': cls.START_DATE, 'end_date': cls.END_DATE, 'exchange': 'TEST', 'asset_name': 'BENCHMARK'}], index=cls.sids)

    def test_set_symbol_lookup_date(self):
        if False:
            print('Hello World!')
        '\n        Test the set_symbol_lookup_date API method.\n        '
        set_symbol_lookup_date = zipline.api.set_symbol_lookup_date

        def initialize(context):
            if False:
                while True:
                    i = 10
            set_symbol_lookup_date(self.asset_ends[0])
            self.assertEqual(zipline.api.symbol('DUP').sid, self.sids[0])
            set_symbol_lookup_date(self.asset_ends[1])
            self.assertEqual(zipline.api.symbol('DUP').sid, self.sids[1])
            with self.assertRaises(UnsupportedDatetimeFormat):
                set_symbol_lookup_date('foobar')
        self.run_algorithm(initialize=initialize)

class TestPositions(zf.WithMakeAlgo, zf.ZiplineTestCase):
    START_DATE = pd.Timestamp('2006-01-03', tz='utc')
    END_DATE = pd.Timestamp('2006-01-06', tz='utc')
    SIM_PARAMS_CAPITAL_BASE = 1000
    ASSET_FINDER_EQUITY_SIDS = (1, 133)
    SIM_PARAMS_DATA_FREQUENCY = 'daily'

    @classmethod
    def make_equity_daily_bar_data(cls, country_code, sids):
        if False:
            i = 10
            return i + 15
        frame = pd.DataFrame({'open': [90, 95, 100, 105], 'high': [90, 95, 100, 105], 'low': [90, 95, 100, 105], 'close': [90, 95, 100, 105], 'volume': 100}, index=cls.equity_daily_bar_days)
        return ((sid, frame) for sid in sids)

    @classmethod
    def make_futures_info(cls):
        if False:
            for i in range(10):
                print('nop')
        return pd.DataFrame.from_dict({1000: {'symbol': 'CLF06', 'root_symbol': 'CL', 'start_date': cls.START_DATE, 'end_date': cls.END_DATE, 'auto_close_date': cls.END_DATE + cls.trading_calendar.day, 'exchange': 'CMES', 'multiplier': 100}}, orient='index')

    @classmethod
    def make_future_minute_bar_data(cls):
        if False:
            return 10
        trading_calendar = cls.trading_calendars[Future]
        sids = cls.asset_finder.futures_sids
        minutes = trading_calendar.minutes_for_sessions_in_range(cls.future_minute_bar_days[0], cls.future_minute_bar_days[-1])
        frame = pd.DataFrame({'open': 2.0, 'high': 2.0, 'low': 2.0, 'close': 2.0, 'volume': 100}, index=minutes)
        return ((sid, frame) for sid in sids)

    def test_portfolio_exited_position(self):
        if False:
            return 10

        def initialize(context, sids):
            if False:
                return 10
            context.ordered = False
            context.exited = False
            context.sids = sids

        def handle_data(context, data):
            if False:
                return 10
            if not context.ordered:
                for s in context.sids:
                    context.order(context.sid(s), 1)
                context.ordered = True
            if not context.exited:
                amounts = [pos.amount for pos in itervalues(context.portfolio.positions)]
                if len(amounts) > 0 and all([amount == 1 for amount in amounts]):
                    for stock in context.portfolio.positions:
                        context.order(context.sid(stock), -1)
                    context.exited = True
            context.record(num_positions=len(context.portfolio.positions))
        result = self.run_algorithm(initialize=initialize, handle_data=handle_data, sids=self.ASSET_FINDER_EQUITY_SIDS)
        expected_position_count = [0, 2, 0, 0]
        for (i, expected) in enumerate(expected_position_count):
            self.assertEqual(result.ix[i]['num_positions'], expected)

    def test_noop_orders(self):
        if False:
            i = 10
            return i + 15
        asset = self.asset_finder.retrieve_asset(1)

        def handle_data(algo, data):
            if False:
                print('Hello World!')
            algo.order(asset, 100, limit_price=1)
            algo.order(asset, 100, stop_price=10000000)
            algo.order(asset, 100, limit_price=10000000, stop_price=10000000)
            algo.order(asset, 100, limit_price=1, stop_price=1)
            algo.order(asset, -100, limit_price=1000000)
            algo.order(asset, -100, stop_price=1)
            algo.order(asset, -100, limit_price=1000000, stop_price=1000000)
            algo.order(asset, -100, limit_price=1, stop_price=1)
            algo.order(asset, 100, limit_price=1e-08)
            algo.order(asset, -100, stop_price=1e-08)
        daily_stats = self.run_algorithm(handle_data=handle_data)
        empty_positions = daily_stats.positions.map(lambda x: len(x) == 0)
        self.assertTrue(empty_positions.all())

    def test_position_weights(self):
        if False:
            while True:
                i = 10
        sids = (1, 133, 1000)
        (equity_1, equity_133, future_1000) = self.asset_finder.retrieve_all(sids)

        def initialize(algo, sids_and_amounts, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            algo.ordered = False
            algo.sids_and_amounts = sids_and_amounts
            algo.set_commission(us_equities=PerTrade(0), us_futures=PerTrade(0))
            algo.set_slippage(us_equities=FixedSlippage(0), us_futures=FixedSlippage(0))

        def handle_data(algo, data):
            if False:
                while True:
                    i = 10
            if not algo.ordered:
                for (s, amount) in algo.sids_and_amounts:
                    algo.order(algo.sid(s), amount)
                algo.ordered = True
            algo.record(position_weights=algo.portfolio.current_portfolio_weights)
        daily_stats = self.run_algorithm(sids_and_amounts=zip(sids, [2, -1, 1]), initialize=initialize, handle_data=handle_data)
        expected_position_weights = [pd.Series({}), pd.Series({equity_1: 190.0 / (190.0 - 95.0 + 905.0), equity_133: -95.0 / (190.0 - 95.0 + 905.0), future_1000: 200.0 / (190.0 - 95.0 + 905.0)}), pd.Series({equity_1: 200.0 / (200.0 - 100.0 + 905.0), equity_133: -100.0 / (200.0 - 100.0 + 905.0), future_1000: 200.0 / (200.0 - 100.0 + 905.0)}), pd.Series({equity_1: 210.0 / (210.0 - 105.0 + 905.0), equity_133: -105.0 / (210.0 - 105.0 + 905.0), future_1000: 200.0 / (210.0 - 105.0 + 905.0)})]
        for (i, expected) in enumerate(expected_position_weights):
            assert_equal(daily_stats.iloc[i]['position_weights'], expected)

class TestBeforeTradingStart(zf.WithMakeAlgo, zf.ZiplineTestCase):
    START_DATE = pd.Timestamp('2016-01-06', tz='utc')
    END_DATE = pd.Timestamp('2016-01-07', tz='utc')
    SIM_PARAMS_CAPITAL_BASE = 10000
    SIM_PARAMS_DATA_FREQUENCY = 'minute'
    EQUITY_DAILY_BAR_LOOKBACK_DAYS = EQUITY_MINUTE_BAR_LOOKBACK_DAYS = 1
    DATA_PORTAL_FIRST_TRADING_DAY = pd.Timestamp('2016-01-05', tz='UTC')
    EQUITY_MINUTE_BAR_START_DATE = pd.Timestamp('2016-01-05', tz='UTC')
    FUTURE_MINUTE_BAR_START_DATE = pd.Timestamp('2016-01-05', tz='UTC')
    data_start = ASSET_FINDER_EQUITY_START_DATE = pd.Timestamp('2016-01-05', tz='utc')
    SPLIT_ASSET_SID = 3
    ASSET_FINDER_EQUITY_SIDS = (1, 2, SPLIT_ASSET_SID)

    @classmethod
    def make_equity_minute_bar_data(cls):
        if False:
            i = 10
            return i + 15
        asset_minutes = cls.trading_calendar.minutes_in_range(cls.data_start, cls.END_DATE)
        minutes_count = len(asset_minutes)
        minutes_arr = np.arange(minutes_count) + 1
        split_data = pd.DataFrame({'open': minutes_arr + 1, 'high': minutes_arr + 2, 'low': minutes_arr - 1, 'close': minutes_arr, 'volume': 100 * minutes_arr}, index=asset_minutes)
        split_data.iloc[780:] = split_data.iloc[780:] / 2.0
        for sid in (1, 8554):
            yield (sid, create_minute_df_for_asset(cls.trading_calendar, cls.data_start, cls.END_DATE))
        yield (2, create_minute_df_for_asset(cls.trading_calendar, cls.data_start, cls.END_DATE, 50))
        yield (cls.SPLIT_ASSET_SID, split_data)

    @classmethod
    def make_splits_data(cls):
        if False:
            for i in range(10):
                print('nop')
        return pd.DataFrame.from_records([{'effective_date': str_to_seconds('2016-01-07'), 'ratio': 0.5, 'sid': cls.SPLIT_ASSET_SID}])

    @classmethod
    def make_equity_daily_bar_data(cls, country_code, sids):
        if False:
            print('Hello World!')
        for sid in sids:
            yield (sid, create_daily_df_for_asset(cls.trading_calendar, cls.data_start, cls.END_DATE))

    def test_data_in_bts_minute(self):
        if False:
            i = 10
            return i + 15
        algo_code = dedent('\n        from zipline.api import record, sid\n        def initialize(context):\n            context.history_values = []\n\n        def before_trading_start(context, data):\n            record(the_price1=data.current(sid(1), "price"))\n            record(the_high1=data.current(sid(1), "high"))\n            record(the_price2=data.current(sid(2), "price"))\n            record(the_high2=data.current(sid(2), "high"))\n\n            context.history_values.append(data.history(\n                [sid(1), sid(2)],\n                ["price", "high"],\n                60,\n                "1m"\n            ))\n\n        def handle_data(context, data):\n            pass\n        ')
        algo = self.make_algo(script=algo_code)
        results = algo.run()
        self.assertEqual(390, results.iloc[0].the_price1)
        self.assertEqual(392, results.iloc[0].the_high1)
        self.assertEqual(350, results.iloc[0].the_price2)
        self.assertTrue(np.isnan(results.iloc[0].the_high2))
        np.testing.assert_array_equal(range(331, 391), algo.history_values[0]['price'][1])
        np.testing.assert_array_equal(range(333, 393), algo.history_values[0]['high'][1])
        np.testing.assert_array_equal([300] * 19, algo.history_values[0]['price'][2][0:19])
        np.testing.assert_array_equal([350] * 40, algo.history_values[0]['price'][2][20:])
        np.testing.assert_array_equal(np.full(19, np.nan), algo.history_values[0]['high'][2][0:19])
        self.assertEqual(352, algo.history_values[0]['high'][2][19])
        np.testing.assert_array_equal(np.full(40, np.nan), algo.history_values[0]['high'][2][20:])

    def test_data_in_bts_daily(self):
        if False:
            print('Hello World!')
        algo_code = dedent('\n        from zipline.api import record, sid\n        def initialize(context):\n            context.history_values = []\n\n        def before_trading_start(context, data):\n            record(the_price1=data.current(sid(1), "price"))\n            record(the_high1=data.current(sid(1), "high"))\n            record(the_price2=data.current(sid(2), "price"))\n            record(the_high2=data.current(sid(2), "high"))\n\n            context.history_values.append(data.history(\n                [sid(1), sid(2)],\n                ["price", "high"],\n                1,\n                "1d",\n            ))\n\n        def handle_data(context, data):\n            pass\n        ')
        algo = self.make_algo(script=algo_code)
        results = algo.run()
        self.assertEqual(392, results.the_high1[0])
        self.assertEqual(390, results.the_price1[0])
        self.assertTrue(np.isnan(results.the_high2[0]))
        self.assertTrue(350, results.the_price2[0])
        self.assertEqual(392, algo.history_values[0]['high'][1][0])
        self.assertEqual(390, algo.history_values[0]['price'][1][0])
        self.assertEqual(352, algo.history_values[0]['high'][2][0])
        self.assertEqual(350, algo.history_values[0]['price'][2][0])

    def test_portfolio_bts(self):
        if False:
            return 10
        algo_code = dedent('\n        from zipline.api import order, sid, record\n\n        def initialize(context):\n            context.ordered = False\n            context.hd_portfolio = context.portfolio\n\n        def before_trading_start(context, data):\n            bts_portfolio = context.portfolio\n\n            # Assert that the portfolio in BTS is the same as the last\n            # portfolio in handle_data\n            assert (context.hd_portfolio == bts_portfolio)\n            record(pos_value=bts_portfolio.positions_value)\n\n        def handle_data(context, data):\n            if not context.ordered:\n                order(sid(1), 1)\n                context.ordered = True\n            context.hd_portfolio = context.portfolio\n        ')
        algo = self.make_algo(script=algo_code)
        results = algo.run()
        self.assertEqual(results.pos_value.iloc[0], 0)
        self.assertEqual(results.pos_value.iloc[1], 780)

    def test_account_bts(self):
        if False:
            while True:
                i = 10
        algo_code = dedent('\n        from zipline.api import order, sid, record, set_slippage, slippage\n\n        def initialize(context):\n            context.ordered = False\n            context.hd_account = context.account\n            set_slippage(slippage.VolumeShareSlippage())\n\n        def before_trading_start(context, data):\n            bts_account = context.account\n\n            # Assert that the account in BTS is the same as the last account\n            # in handle_data\n            assert (context.hd_account == bts_account)\n            record(port_value=context.account.equity_with_loan)\n\n        def handle_data(context, data):\n            if not context.ordered:\n                order(sid(1), 1)\n                context.ordered = True\n            context.hd_account = context.account\n        ')
        algo = self.make_algo(script=algo_code)
        results = algo.run()
        self.assertEqual(results.port_value.iloc[0], 10000)
        self.assertAlmostEqual(results.port_value.iloc[1], 10000 + 780 - 392 - 0, places=2)

    def test_portfolio_bts_with_overnight_split(self):
        if False:
            i = 10
            return i + 15
        algo_code = dedent("\n        from zipline.api import order, sid, record\n\n        def initialize(context):\n            context.ordered = False\n            context.hd_portfolio = context.portfolio\n\n        def before_trading_start(context, data):\n            bts_portfolio = context.portfolio\n            # Assert that the portfolio in BTS is the same as the last\n            # portfolio in handle_data, except for the positions\n            for k in bts_portfolio.__dict__:\n                if k != 'positions':\n                    assert (context.hd_portfolio.__dict__[k]\n                            == bts_portfolio.__dict__[k])\n            record(pos_value=bts_portfolio.positions_value)\n            record(pos_amount=bts_portfolio.positions[sid(3)].amount)\n            record(\n                last_sale_price=bts_portfolio.positions[sid(3)].last_sale_price\n            )\n\n        def handle_data(context, data):\n            if not context.ordered:\n                order(sid(3), 1)\n                context.ordered = True\n            context.hd_portfolio = context.portfolio\n        ")
        results = self.run_algorithm(script=algo_code)
        self.assertEqual(results.pos_value.iloc[0], 0)
        self.assertEqual(results.pos_value.iloc[1], 780)
        self.assertEqual(results.pos_amount.iloc[0], 0)
        self.assertEqual(results.pos_amount.iloc[1], 2)
        self.assertEqual(results.last_sale_price.iloc[0], 0)
        self.assertEqual(results.last_sale_price.iloc[1], 390)

    def test_account_bts_with_overnight_split(self):
        if False:
            return 10
        algo_code = dedent('\n        from zipline.api import order, sid, record, set_slippage, slippage\n\n        def initialize(context):\n            context.ordered = False\n            context.hd_account = context.account\n            set_slippage(slippage.VolumeShareSlippage())\n\n\n        def before_trading_start(context, data):\n            bts_account = context.account\n            # Assert that the account in BTS is the same as the last account\n            # in handle_data\n            assert (context.hd_account == bts_account)\n            record(port_value=bts_account.equity_with_loan)\n\n        def handle_data(context, data):\n            if not context.ordered:\n                order(sid(1), 1)\n                context.ordered = True\n            context.hd_account = context.account\n        ')
        results = self.run_algorithm(script=algo_code)
        self.assertEqual(results.port_value.iloc[0], 10000)
        self.assertAlmostEqual(results.port_value.iloc[1], 10000 + 780 - 392 - 0, places=2)

class TestAlgoScript(zf.WithMakeAlgo, zf.ZiplineTestCase):
    START_DATE = pd.Timestamp('2006-01-03', tz='utc')
    END_DATE = pd.Timestamp('2006-12-31', tz='utc')
    SIM_PARAMS_DATA_FREQUENCY = 'daily'
    DATA_PORTAL_USE_MINUTE_DATA = False
    EQUITY_DAILY_BAR_LOOKBACK_DAYS = 5
    STRING_TYPE_NAMES = [s.__name__ for s in string_types]
    STRING_TYPE_NAMES_STRING = ', '.join(STRING_TYPE_NAMES)
    ASSET_TYPE_NAME = Asset.__name__
    CONTINUOUS_FUTURE_NAME = ContinuousFuture.__name__
    ASSET_OR_STRING_TYPE_NAMES = ', '.join([ASSET_TYPE_NAME] + STRING_TYPE_NAMES)
    ASSET_OR_STRING_OR_CF_TYPE_NAMES = ', '.join([ASSET_TYPE_NAME, CONTINUOUS_FUTURE_NAME] + STRING_TYPE_NAMES)
    ARG_TYPE_TEST_CASES = (('history__assets', (bad_type_history_assets, ASSET_OR_STRING_OR_CF_TYPE_NAMES, True)), ('history__fields', (bad_type_history_fields, STRING_TYPE_NAMES_STRING, True)), ('history__bar_count', (bad_type_history_bar_count, 'int', False)), ('history__frequency', (bad_type_history_frequency, STRING_TYPE_NAMES_STRING, False)), ('current__assets', (bad_type_current_assets, ASSET_OR_STRING_OR_CF_TYPE_NAMES, True)), ('current__fields', (bad_type_current_fields, STRING_TYPE_NAMES_STRING, True)), ('is_stale__assets', (bad_type_is_stale_assets, 'Asset', True)), ('can_trade__assets', (bad_type_can_trade_assets, 'Asset', True)), ('history_kwarg__assets', (bad_type_history_assets_kwarg, ASSET_OR_STRING_OR_CF_TYPE_NAMES, True)), ('history_kwarg_bad_list__assets', (bad_type_history_assets_kwarg_list, ASSET_OR_STRING_OR_CF_TYPE_NAMES, True)), ('history_kwarg__fields', (bad_type_history_fields_kwarg, STRING_TYPE_NAMES_STRING, True)), ('history_kwarg__bar_count', (bad_type_history_bar_count_kwarg, 'int', False)), ('history_kwarg__frequency', (bad_type_history_frequency_kwarg, STRING_TYPE_NAMES_STRING, False)), ('current_kwarg__assets', (bad_type_current_assets_kwarg, ASSET_OR_STRING_OR_CF_TYPE_NAMES, True)), ('current_kwarg__fields', (bad_type_current_fields_kwarg, STRING_TYPE_NAMES_STRING, True)))
    sids = (0, 1, 3, 133)
    BENCHMARK_SID = None

    @classmethod
    def make_equity_info(cls):
        if False:
            while True:
                i = 10
        register_calendar('TEST', get_calendar('NYSE'), force=True)
        data = make_simple_equity_info(cls.sids, cls.START_DATE, cls.END_DATE)
        data.loc[3, 'symbol'] = 'TEST'
        return data

    @classmethod
    def make_equity_daily_bar_data(cls, country_code, sids):
        if False:
            print('Hello World!')
        cal = cls.trading_calendars[Equity]
        sessions = cal.sessions_in_range(cls.START_DATE, cls.END_DATE)
        frame = pd.DataFrame({'close': 10.0, 'high': 10.5, 'low': 9.5, 'open': 10.0, 'volume': 100}, index=sessions)
        for sid in sids:
            yield (sid, frame)

    def test_noop(self):
        if False:
            while True:
                i = 10
        self.run_algorithm(initialize=initialize_noop, handle_data=handle_data_noop)

    def test_noop_string(self):
        if False:
            while True:
                i = 10
        self.run_algorithm(script=noop_algo)

    def test_no_handle_data(self):
        if False:
            i = 10
            return i + 15
        self.run_algorithm(script=no_handle_data)

    def test_api_calls(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_algorithm(initialize=initialize_api, handle_data=handle_data_api)

    def test_api_calls_string(self):
        if False:
            i = 10
            return i + 15
        self.run_algorithm(script=api_algo)

    def test_api_get_environment(self):
        if False:
            i = 10
            return i + 15
        platform = 'zipline'
        algo = self.make_algo(script=api_get_environment_algo, platform=platform)
        algo.run()
        self.assertEqual(algo.environment, platform)

    def test_api_symbol(self):
        if False:
            print('Hello World!')
        self.run_algorithm(script=api_symbol_algo)

    def test_fixed_slippage(self):
        if False:
            return 10
        test_algo = self.make_algo(script='\nfrom zipline.api import (slippage,\n                         commission,\n                         set_slippage,\n                         set_commission,\n                         order,\n                         record,\n                         sid)\n\ndef initialize(context):\n    model = slippage.FixedSlippage(spread=0.10)\n    set_slippage(model)\n    set_commission(commission.PerTrade(100.00))\n    context.count = 1\n    context.incr = 0\n\ndef handle_data(context, data):\n    if context.incr < context.count:\n        order(sid(0), -1000)\n    record(price=data.current(sid(0), "price"))\n\n    context.incr += 1')
        results = test_algo.run()
        all_txns = [val for sublist in results['transactions'].tolist() for val in sublist]
        self.assertEqual(len(all_txns), 1)
        txn = all_txns[0]
        expected_spread = 0.05
        expected_price = test_algo.recorded_vars['price'] - expected_spread
        self.assertEqual(expected_price, txn['price'])
        self.assertEqual(9850, results.capital_used[1])
        self.assertEqual(100, results['orders'].iloc[1][0]['commission'])

    @parameterized.expand([('no_minimum_commission', 0), ('default_minimum_commission', 0), ('alternate_minimum_commission', 2)])
    def test_volshare_slippage(self, name, minimum_commission):
        if False:
            print('Hello World!')
        tempdir = TempDirectory()
        try:
            if name == 'default_minimum_commission':
                commission_line = 'set_commission(commission.PerShare(0.02))'
            else:
                commission_line = 'set_commission(commission.PerShare(0.02, min_trade_cost={0}))'.format(minimum_commission)
            trades = factory.create_daily_trade_source([0], self.sim_params, self.asset_finder, self.trading_calendar)
            data_portal = create_data_portal_from_trade_history(self.asset_finder, self.trading_calendar, tempdir, self.sim_params, {0: trades})
            test_algo = self.make_algo(data_portal=data_portal, script='\nfrom zipline.api import *\n\ndef initialize(context):\n    model = slippage.VolumeShareSlippage(\n                            volume_limit=.3,\n                            price_impact=0.05\n                       )\n    set_slippage(model)\n    {0}\n\n    context.count = 2\n    context.incr = 0\n\ndef handle_data(context, data):\n    if context.incr < context.count:\n        # order small lots to be sure the\n        # order will fill in a single transaction\n        order(sid(0), 5000)\n    record(price=data.current(sid(0), "price"))\n    record(volume=data.current(sid(0), "volume"))\n    record(incr=context.incr)\n    context.incr += 1\n    '.format(commission_line))
            results = test_algo.run()
            all_txns = [val for sublist in results['transactions'].tolist() for val in sublist]
            self.assertEqual(len(all_txns), 67)
            all_orders = list(toolz.concat(results['orders']))
            if minimum_commission == 0:
                for order_ in all_orders:
                    self.assertAlmostEqual(order_['filled'] * 0.02, order_['commission'])
            else:
                for order_ in all_orders:
                    if order_['filled'] > 0:
                        self.assertAlmostEqual(max(order_['filled'] * 0.02, minimum_commission), order_['commission'])
                    else:
                        self.assertEqual(0, order_['commission'])
        finally:
            tempdir.cleanup()

    def test_incorrectly_set_futures_slippage_model(self):
        if False:
            for i in range(10):
                print('nop')
        code = dedent("\n            from zipline.api import set_slippage, slippage\n\n            class MySlippage(slippage.FutureSlippageModel):\n                def process_order(self, data, order):\n                    return data.current(order.asset, 'price'), order.amount\n\n            def initialize(context):\n                set_slippage(MySlippage())\n            ")
        test_algo = self.make_algo(script=code)
        with self.assertRaises(IncompatibleSlippageModel):
            test_algo.run()

    def test_algo_record_vars(self):
        if False:
            return 10
        test_algo = self.make_algo(script=record_variables)
        results = test_algo.run()
        for i in range(1, 252):
            self.assertEqual(results.iloc[i - 1]['incr'], i)

    def test_algo_record_nan(self):
        if False:
            print('Hello World!')
        test_algo = self.make_algo(script=record_float_magic % 'nan')
        results = test_algo.run()
        for i in range(1, 252):
            self.assertTrue(np.isnan(results.iloc[i - 1]['data']))

    def test_batch_market_order_matches_multiple_manual_orders(self):
        if False:
            i = 10
            return i + 15
        share_counts = pd.Series([50, 100])
        multi_blotter = RecordBatchBlotter()
        multi_test_algo = self.make_algo(script=dedent('                from collections import OrderedDict\n                from six import iteritems\n\n                from zipline.api import sid, order\n\n\n                def initialize(context):\n                    context.assets = [sid(0), sid(3)]\n                    context.placed = False\n\n                def handle_data(context, data):\n                    if not context.placed:\n                        it = zip(context.assets, {share_counts})\n                        for asset, shares in it:\n                            order(asset, shares)\n\n                        context.placed = True\n\n            ').format(share_counts=list(share_counts)), blotter=multi_blotter)
        multi_stats = multi_test_algo.run()
        self.assertFalse(multi_blotter.order_batch_called)
        batch_blotter = RecordBatchBlotter()
        batch_test_algo = self.make_algo(script=dedent('                import pandas as pd\n\n                from zipline.api import sid, batch_market_order\n\n\n                def initialize(context):\n                    context.assets = [sid(0), sid(3)]\n                    context.placed = False\n\n                def handle_data(context, data):\n                    if not context.placed:\n                        orders = batch_market_order(pd.Series(\n                            index=context.assets, data={share_counts}\n                        ))\n                        assert len(orders) == 2,                             "len(orders) was %s but expected 2" % len(orders)\n                        for o in orders:\n                            assert o is not None, "An order is None"\n\n                        context.placed = True\n\n            ').format(share_counts=list(share_counts)), blotter=batch_blotter)
        batch_stats = batch_test_algo.run()
        self.assertTrue(batch_blotter.order_batch_called)
        for stats in (multi_stats, batch_stats):
            stats.orders = stats.orders.apply(lambda orders: [toolz.dissoc(o, 'id') for o in orders])
            stats.transactions = stats.transactions.apply(lambda txns: [toolz.dissoc(txn, 'order_id') for txn in txns])
        assert_equal(multi_stats, batch_stats)

    def test_batch_market_order_filters_null_orders(self):
        if False:
            while True:
                i = 10
        share_counts = [50, 0]
        batch_blotter = RecordBatchBlotter()
        batch_test_algo = self.make_algo(script=dedent('                import pandas as pd\n\n                from zipline.api import sid, batch_market_order\n\n                def initialize(context):\n                    context.assets = [sid(0), sid(3)]\n                    context.placed = False\n\n                def handle_data(context, data):\n                    if not context.placed:\n                        orders = batch_market_order(pd.Series(\n                            index=context.assets, data={share_counts}\n                        ))\n                        assert len(orders) == 1,                             "len(orders) was %s but expected 1" % len(orders)\n                        for o in orders:\n                            assert o is not None, "An order is None"\n\n                        context.placed = True\n\n            ').format(share_counts=share_counts), blotter=batch_blotter)
        batch_test_algo.run()
        self.assertTrue(batch_blotter.order_batch_called)

    def test_order_dead_asset(self):
        if False:
            while True:
                i = 10
        params = SimulationParameters(start_session=pd.Timestamp('2007-01-03', tz='UTC'), end_session=pd.Timestamp('2007-01-05', tz='UTC'), trading_calendar=self.trading_calendar)
        self.run_algorithm(script='\nfrom zipline.api import order, sid\n\ndef initialize(context):\n    pass\n\ndef handle_data(context, data):\n    order(sid(0), 10)\n        ')
        for order_str in ['order_value', 'order_percent']:
            test_algo = self.make_algo(script='\nfrom zipline.api import order_percent, order_value, sid\n\ndef initialize(context):\n    pass\n\ndef handle_data(context, data):\n    {0}(sid(0), 10)\n        '.format(order_str), sim_params=params)
        with self.assertRaises(CannotOrderDelistedAsset):
            test_algo.run()

    def test_portfolio_in_init(self):
        if False:
            i = 10
            return i + 15
        "\n        Test that accessing portfolio in init doesn't break.\n        "
        self.run_algorithm(script=access_portfolio_in_init)

    def test_account_in_init(self):
        if False:
            print('Hello World!')
        "\n        Test that accessing account in init doesn't break.\n        "
        self.run_algorithm(script=access_account_in_init)

    def test_without_kwargs(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that api methods on the data object can be called with positional\n        arguments.\n        '
        params = SimulationParameters(start_session=pd.Timestamp('2006-01-10', tz='UTC'), end_session=pd.Timestamp('2006-01-11', tz='UTC'), trading_calendar=self.trading_calendar)
        self.run_algorithm(sim_params=params, script=call_without_kwargs)

    def test_good_kwargs(self):
        if False:
            print('Hello World!')
        '\n        Test that api methods on the data object can be called with keyword\n        arguments.\n        '
        params = SimulationParameters(start_session=pd.Timestamp('2006-01-10', tz='UTC'), end_session=pd.Timestamp('2006-01-11', tz='UTC'), trading_calendar=self.trading_calendar)
        self.run_algorithm(script=call_with_kwargs, sim_params=params)

    @parameterized.expand([('history', call_with_bad_kwargs_history), ('current', call_with_bad_kwargs_current)])
    def test_bad_kwargs(self, name, algo_text):
        if False:
            return 10
        '\n        Test that api methods on the data object called with bad kwargs return\n        a meaningful TypeError that we create, rather than an unhelpful cython\n        error\n        '
        algo = self.make_algo(script=algo_text)
        with self.assertRaises(TypeError) as cm:
            algo.run()
        self.assertEqual("%s() got an unexpected keyword argument 'blahblah'" % name, cm.exception.args[0])

    @parameterized.expand(ARG_TYPE_TEST_CASES)
    def test_arg_types(self, name, inputs):
        if False:
            return 10
        keyword = name.split('__')[1]
        algo = self.make_algo(script=inputs[0])
        with self.assertRaises(TypeError) as cm:
            algo.run()
        expected = 'Expected %s argument to be of type %s%s' % (keyword, 'or iterable of type ' if inputs[2] else '', inputs[1])
        self.assertEqual(expected, cm.exception.args[0])

    def test_empty_asset_list_to_history(self):
        if False:
            while True:
                i = 10
        params = SimulationParameters(start_session=pd.Timestamp('2006-01-10', tz='UTC'), end_session=pd.Timestamp('2006-01-11', tz='UTC'), trading_calendar=self.trading_calendar)
        self.run_algorithm(script=dedent('\n                def initialize(context):\n                    pass\n\n                def handle_data(context, data):\n                    data.history([], "price", 5, \'1d\')\n                '), sim_params=params)

    @parameterized.expand([('bad_kwargs', call_with_bad_kwargs_get_open_orders), ('good_kwargs', call_with_good_kwargs_get_open_orders), ('no_kwargs', call_with_no_kwargs_get_open_orders)])
    def test_get_open_orders_kwargs(self, name, script):
        if False:
            return 10
        algo = self.make_algo(script=script)
        if name == 'bad_kwargs':
            with self.assertRaises(TypeError) as cm:
                algo.run()
                self.assertEqual('Keyword argument `sid` is no longer supported for get_open_orders. Use `asset` instead.', cm.exception.args[0])
        else:
            algo.run()

    def test_empty_positions(self):
        if False:
            return 10
        "\n        Test that when we try context.portfolio.positions[stock] on a stock\n        for which we have no positions, we return a Position with values 0\n        (but more importantly, we don't crash) and don't save this Position\n        to the user-facing dictionary PositionTracker._positions_store\n        "
        results = self.run_algorithm(script=empty_positions)
        num_positions = results.num_positions
        amounts = results.amounts
        self.assertTrue(all(num_positions == 0))
        self.assertTrue(all(amounts == 0))

    def test_schedule_function_time_rule_positionally_misplaced(self):
        if False:
            print('Hello World!')
        '\n        Test that when a user specifies a time rule for the date_rule argument,\n        but no rule in the time_rule argument\n        (e.g. schedule_function(func, <time_rule>)), we assume that means\n        assign a time rule but no date rule\n        '
        sim_params = factory.create_simulation_parameters(start=pd.Timestamp('2006-01-12', tz='UTC'), end=pd.Timestamp('2006-01-13', tz='UTC'), data_frequency='minute')
        algocode = dedent('\n        from zipline.api import time_rules, schedule_function\n\n        def do_at_open(context, data):\n            context.done_at_open.append(context.get_datetime())\n\n        def do_at_close(context, data):\n            context.done_at_close.append(context.get_datetime())\n\n        def initialize(context):\n            context.done_at_open = []\n            context.done_at_close = []\n            schedule_function(do_at_open, time_rules.market_open())\n            schedule_function(do_at_close, time_rules.market_close())\n\n        def handle_data(algo, data):\n            pass\n        ')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('ignore', PerformanceWarning)
            algo = self.make_algo(script=algocode, sim_params=sim_params)
            algo.run()
            self.assertEqual(len(w), 2)
            for (i, warning) in enumerate(w):
                self.assertIsInstance(warning.message, UserWarning)
                self.assertEqual(warning.message.args[0], 'Got a time rule for the second positional argument date_rule. You should use keyword argument time_rule= when calling schedule_function without specifying a date_rule')
                self.assertEqual(warning.lineno, 13 + i)
        self.assertEqual(algo.done_at_open, [pd.Timestamp('2006-01-12 14:31:00', tz='UTC'), pd.Timestamp('2006-01-13 14:31:00', tz='UTC')])
        self.assertEqual(algo.done_at_close, [pd.Timestamp('2006-01-12 20:59:00', tz='UTC'), pd.Timestamp('2006-01-13 20:59:00', tz='UTC')])

class TestCapitalChanges(zf.WithMakeAlgo, zf.ZiplineTestCase):
    START_DATE = pd.Timestamp('2006-01-03', tz='UTC')
    END_DATE = pd.Timestamp('2006-01-09', tz='UTC')
    sids = ASSET_FINDER_EQUITY_SIDS = (0, 1)
    DAILY_SID = 0
    MINUTELY_SID = 1
    BENCHMARK_SID = None

    @classmethod
    def make_equity_minute_bar_data(cls):
        if False:
            return 10
        minutes = cls.trading_calendar.minutes_in_range(cls.START_DATE, cls.END_DATE)
        closes = np.arange(100, 100 + len(minutes), 1)
        opens = closes
        highs = closes + 5
        lows = closes - 5
        frame = pd.DataFrame(index=minutes, data={'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': 10000})
        yield (cls.MINUTELY_SID, frame)

    @classmethod
    def make_equity_daily_bar_data(cls, country_code, sids):
        if False:
            i = 10
            return i + 15
        days = cls.trading_calendar.sessions_in_range(cls.START_DATE, cls.END_DATE)
        closes = np.arange(10.0, 10.0 + len(days), 1.0)
        opens = closes
        highs = closes + 0.5
        lows = closes - 0.5
        frame = pd.DataFrame(index=days, data={'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': 10000})
        yield (cls.DAILY_SID, frame)

    @parameterized.expand([('target', 151000.0), ('delta', 50000.0)])
    def test_capital_changes_daily_mode(self, change_type, value):
        if False:
            return 10
        capital_changes = {pd.Timestamp('2006-01-06', tz='UTC'): {'type': change_type, 'value': value}}
        algocode = '\nfrom zipline.api import set_slippage, set_commission, slippage, commission,     schedule_function, time_rules, order, sid\n\ndef initialize(context):\n    set_slippage(slippage.FixedSlippage(spread=0))\n    set_commission(commission.PerShare(0, 0))\n    schedule_function(order_stuff, time_rule=time_rules.market_open())\n\ndef order_stuff(context, data):\n    order(sid(0), 1000)\n'
        algo = self.make_algo(script=algocode, capital_changes=capital_changes, sim_params=SimulationParameters(start_session=self.START_DATE, end_session=self.END_DATE, trading_calendar=self.nyse_calendar))
        gen = algo.get_generator()
        results = list(gen)
        cumulative_perf = [r['cumulative_perf'] for r in results if 'cumulative_perf' in r]
        daily_perf = [r['daily_perf'] for r in results if 'daily_perf' in r]
        capital_change_packets = [r['capital_change'] for r in results if 'capital_change' in r]
        self.assertEqual(len(capital_change_packets), 1)
        self.assertEqual(capital_change_packets[0], {'date': pd.Timestamp('2006-01-06', tz='UTC'), 'type': 'cash', 'target': 151000.0 if change_type == 'target' else None, 'delta': 50000.0})
        expected_daily = {}
        expected_capital_changes = np.array([0.0, 0.0, 0.0, 50000.0, 0.0])
        expected_daily['returns'] = np.array([0.0, 0.0, (100000.0 + 1000.0) / 100000.0 - 1.0, (151000.0 + 2000.0) / 151000.0 - 1.0, (153000.0 + 3000.0) / 153000.0 - 1.0])
        expected_daily['pnl'] = np.array([0.0, 0.0, 1000.0, 2000.0, 3000.0])
        expected_daily['capital_used'] = np.array([0.0, -11000.0, -12000.0, -13000.0, -14000.0])
        expected_daily['ending_cash'] = np.array([100000.0] * 5) + np.cumsum(expected_capital_changes) + np.cumsum(expected_daily['capital_used'])
        expected_daily['starting_cash'] = expected_daily['ending_cash'] - expected_daily['capital_used']
        expected_daily['starting_value'] = np.array([0.0, 0.0, 11000.0, 24000.0, 39000.0])
        expected_daily['ending_value'] = expected_daily['starting_value'] + expected_daily['pnl'] - expected_daily['capital_used']
        expected_daily['portfolio_value'] = expected_daily['ending_value'] + expected_daily['ending_cash']
        stats = ['returns', 'pnl', 'capital_used', 'starting_cash', 'ending_cash', 'starting_value', 'ending_value', 'portfolio_value']
        expected_cumulative = {'returns': np.cumprod(expected_daily['returns'] + 1) - 1, 'pnl': np.cumsum(expected_daily['pnl']), 'capital_used': np.cumsum(expected_daily['capital_used']), 'starting_cash': np.repeat(expected_daily['starting_cash'][0:1], 5), 'ending_cash': expected_daily['ending_cash'], 'starting_value': np.repeat(expected_daily['starting_value'][0:1], 5), 'ending_value': expected_daily['ending_value'], 'portfolio_value': expected_daily['portfolio_value']}
        for stat in stats:
            np.testing.assert_array_almost_equal(np.array([perf[stat] for perf in daily_perf]), expected_daily[stat], err_msg='daily ' + stat)
            np.testing.assert_array_almost_equal(np.array([perf[stat] for perf in cumulative_perf]), expected_cumulative[stat], err_msg='cumulative ' + stat)
        self.assertEqual(algo.capital_change_deltas, {pd.Timestamp('2006-01-06', tz='UTC'): 50000.0})

    @parameterized.expand([('interday_target', [('2006-01-04', 2388.0)]), ('interday_delta', [('2006-01-04', 1000.0)]), ('intraday_target', [('2006-01-04 17:00', 2184.0), ('2006-01-04 18:00', 2804.0)]), ('intraday_delta', [('2006-01-04 17:00', 500.0), ('2006-01-04 18:00', 500.0)])])
    def test_capital_changes_minute_mode_daily_emission(self, change, values):
        if False:
            return 10
        (change_loc, change_type) = change.split('_')
        sim_params = SimulationParameters(start_session=pd.Timestamp('2006-01-03', tz='UTC'), end_session=pd.Timestamp('2006-01-05', tz='UTC'), data_frequency='minute', capital_base=1000.0, trading_calendar=self.nyse_calendar)
        capital_changes = {pd.Timestamp(datestr, tz='UTC'): {'type': change_type, 'value': value} for (datestr, value) in values}
        algocode = '\nfrom zipline.api import set_slippage, set_commission, slippage, commission,     schedule_function, time_rules, order, sid\n\ndef initialize(context):\n    set_slippage(slippage.FixedSlippage(spread=0))\n    set_commission(commission.PerShare(0, 0))\n    schedule_function(order_stuff, time_rule=time_rules.market_open())\n\ndef order_stuff(context, data):\n    order(sid(1), 1)\n'
        algo = self.make_algo(script=algocode, sim_params=sim_params, capital_changes=capital_changes)
        gen = algo.get_generator()
        results = list(gen)
        cumulative_perf = [r['cumulative_perf'] for r in results if 'cumulative_perf' in r]
        daily_perf = [r['daily_perf'] for r in results if 'daily_perf' in r]
        capital_change_packets = [r['capital_change'] for r in results if 'capital_change' in r]
        self.assertEqual(len(capital_change_packets), len(capital_changes))
        expected = [{'date': pd.Timestamp(val[0], tz='UTC'), 'type': 'cash', 'target': val[1] if change_type == 'target' else None, 'delta': 1000.0 if len(values) == 1 else 500.0} for val in values]
        self.assertEqual(capital_change_packets, expected)
        expected_daily = {}
        expected_capital_changes = np.array([0.0, 1000.0, 0.0])
        if change_loc == 'intraday':
            day2_return = (1388.0 + 149.0 + 147.0) / 1388.0 * (2184.0 + 60.0 + 60.0) / 2184.0 * (2804.0 + 181.0 + 181.0) / 2804.0 - 1.0
        else:
            day2_return = (2388.0 + 390.0 + 388.0) / 2388.0 - 1
        expected_daily['returns'] = np.array([(1000.0 + 489 - 101) / 1000.0 - 1.0, day2_return, (3166.0 + 390.0 + 390.0 + 388.0) / 3166.0 - 1.0])
        expected_daily['pnl'] = np.array([388.0, 390.0 + 388.0, 390.0 + 390.0 + 388.0])
        expected_daily['capital_used'] = np.array([-101.0, -491.0, -881.0])
        expected_daily['ending_cash'] = np.array([1000.0] * 3) + np.cumsum(expected_capital_changes) + np.cumsum(expected_daily['capital_used'])
        expected_daily['starting_cash'] = expected_daily['ending_cash'] - expected_daily['capital_used']
        if change_loc == 'intraday':
            expected_daily['starting_cash'] -= expected_capital_changes
        expected_daily['starting_value'] = np.array([0.0, 489.0, 879.0 * 2])
        expected_daily['ending_value'] = expected_daily['starting_value'] + expected_daily['pnl'] - expected_daily['capital_used']
        expected_daily['portfolio_value'] = expected_daily['ending_value'] + expected_daily['ending_cash']
        stats = ['returns', 'pnl', 'capital_used', 'starting_cash', 'ending_cash', 'starting_value', 'ending_value', 'portfolio_value']
        expected_cumulative = {'returns': np.cumprod(expected_daily['returns'] + 1) - 1, 'pnl': np.cumsum(expected_daily['pnl']), 'capital_used': np.cumsum(expected_daily['capital_used']), 'starting_cash': np.repeat(expected_daily['starting_cash'][0:1], 3), 'ending_cash': expected_daily['ending_cash'], 'starting_value': np.repeat(expected_daily['starting_value'][0:1], 3), 'ending_value': expected_daily['ending_value'], 'portfolio_value': expected_daily['portfolio_value']}
        for stat in stats:
            np.testing.assert_array_almost_equal(np.array([perf[stat] for perf in daily_perf]), expected_daily[stat])
            np.testing.assert_array_almost_equal(np.array([perf[stat] for perf in cumulative_perf]), expected_cumulative[stat])
        if change_loc == 'interday':
            self.assertEqual(algo.capital_change_deltas, {pd.Timestamp('2006-01-04', tz='UTC'): 1000.0})
        else:
            self.assertEqual(algo.capital_change_deltas, {pd.Timestamp('2006-01-04 17:00', tz='UTC'): 500.0, pd.Timestamp('2006-01-04 18:00', tz='UTC'): 500.0})

    @parameterized.expand([('interday_target', [('2006-01-04', 2388.0)]), ('interday_delta', [('2006-01-04', 1000.0)]), ('intraday_target', [('2006-01-04 17:00', 2184.0), ('2006-01-04 18:00', 2804.0)]), ('intraday_delta', [('2006-01-04 17:00', 500.0), ('2006-01-04 18:00', 500.0)])])
    def test_capital_changes_minute_mode_minute_emission(self, change, values):
        if False:
            i = 10
            return i + 15
        (change_loc, change_type) = change.split('_')
        sim_params = SimulationParameters(start_session=pd.Timestamp('2006-01-03', tz='UTC'), end_session=pd.Timestamp('2006-01-05', tz='UTC'), data_frequency='minute', emission_rate='minute', capital_base=1000.0, trading_calendar=self.nyse_calendar)
        capital_changes = {pd.Timestamp(val[0], tz='UTC'): {'type': change_type, 'value': val[1]} for val in values}
        algocode = '\nfrom zipline.api import set_slippage, set_commission, slippage, commission,     schedule_function, time_rules, order, sid\n\ndef initialize(context):\n    set_slippage(slippage.FixedSlippage(spread=0))\n    set_commission(commission.PerShare(0, 0))\n    schedule_function(order_stuff, time_rule=time_rules.market_open())\n\ndef order_stuff(context, data):\n    order(sid(1), 1)\n'
        algo = self.make_algo(script=algocode, sim_params=sim_params, capital_changes=capital_changes)
        gen = algo.get_generator()
        results = list(gen)
        cumulative_perf = [r['cumulative_perf'] for r in results if 'cumulative_perf' in r]
        minute_perf = [r['minute_perf'] for r in results if 'minute_perf' in r]
        daily_perf = [r['daily_perf'] for r in results if 'daily_perf' in r]
        capital_change_packets = [r['capital_change'] for r in results if 'capital_change' in r]
        self.assertEqual(len(capital_change_packets), len(capital_changes))
        expected = [{'date': pd.Timestamp(val[0], tz='UTC'), 'type': 'cash', 'target': val[1] if change_type == 'target' else None, 'delta': 1000.0 if len(values) == 1 else 500.0} for val in values]
        self.assertEqual(capital_change_packets, expected)
        expected_minute = {}
        capital_changes_after_start = np.array([0.0] * 1170)
        if change_loc == 'intraday':
            capital_changes_after_start[539:599] = 500.0
            capital_changes_after_start[599:780] = 1000.0
        expected_minute['pnl'] = np.array([0.0] * 1170)
        expected_minute['pnl'][:2] = 0.0
        expected_minute['pnl'][2:392] = 1.0
        expected_minute['pnl'][392:782] = 2.0
        expected_minute['pnl'][782:] = 3.0
        for (start, end) in ((0, 390), (390, 780), (780, 1170)):
            expected_minute['pnl'][start:end] = np.cumsum(expected_minute['pnl'][start:end])
        expected_minute['capital_used'] = np.concatenate(([0.0] * 1, [-101.0] * 389, [0.0] * 1, [-491.0] * 389, [0.0] * 1, [-881.0] * 389))
        day2adj = 0.0 if change_loc == 'intraday' else 1000.0
        expected_minute['starting_cash'] = np.concatenate(([1000.0] * 390, [1000.0 - 101.0 + day2adj] * 390, [1000.0 - 101.0 - 491.0 + 1000] * 390))
        expected_minute['ending_cash'] = expected_minute['starting_cash'] + expected_minute['capital_used'] + capital_changes_after_start
        expected_minute['starting_value'] = np.concatenate(([0.0] * 390, [489.0] * 390, [879.0 * 2] * 390))
        expected_minute['ending_value'] = expected_minute['starting_value'] + expected_minute['pnl'] - expected_minute['capital_used']
        expected_minute['portfolio_value'] = expected_minute['ending_value'] + expected_minute['ending_cash']
        expected_minute['returns'] = expected_minute['pnl'] / (expected_minute['starting_value'] + expected_minute['starting_cash'])
        if change_loc == 'intraday':
            prev_subperiod_return = expected_minute['returns'][538]
            cur_subperiod_pnl = expected_minute['pnl'][539:599] - expected_minute['pnl'][538]
            cur_subperiod_starting_value = np.array([expected_minute['ending_value'][538]] * 60)
            cur_subperiod_starting_cash = np.array([expected_minute['ending_cash'][538] + 500] * 60)
            cur_subperiod_returns = cur_subperiod_pnl / (cur_subperiod_starting_value + cur_subperiod_starting_cash)
            expected_minute['returns'][539:599] = (cur_subperiod_returns + 1.0) * (prev_subperiod_return + 1.0) - 1.0
            prev_subperiod_return = expected_minute['returns'][598]
            cur_subperiod_pnl = expected_minute['pnl'][599:780] - expected_minute['pnl'][598]
            cur_subperiod_starting_value = np.array([expected_minute['ending_value'][598]] * 181)
            cur_subperiod_starting_cash = np.array([expected_minute['ending_cash'][598] + 500] * 181)
            cur_subperiod_returns = cur_subperiod_pnl / (cur_subperiod_starting_value + cur_subperiod_starting_cash)
            expected_minute['returns'][599:780] = (cur_subperiod_returns + 1.0) * (prev_subperiod_return + 1.0) - 1.0
        expected_daily = {k: np.array([v[389], v[779], v[1169]]) for (k, v) in iteritems(expected_minute)}
        stats = ['pnl', 'capital_used', 'starting_cash', 'ending_cash', 'starting_value', 'ending_value', 'portfolio_value', 'returns']
        expected_cumulative = deepcopy(expected_minute)
        expected_cumulative['returns'][390:] = (expected_cumulative['returns'][390:] + 1) * (expected_daily['returns'][0] + 1) - 1
        expected_cumulative['returns'][780:] = (expected_cumulative['returns'][780:] + 1) * (expected_daily['returns'][1] + 1) - 1
        expected_cumulative['pnl'][390:] += expected_daily['pnl'][0]
        expected_cumulative['pnl'][780:] += expected_daily['pnl'][1]
        expected_cumulative['capital_used'][390:] += expected_daily['capital_used'][0]
        expected_cumulative['capital_used'][780:] += expected_daily['capital_used'][1]
        expected_cumulative['starting_cash'] = np.repeat(expected_daily['starting_cash'][0:1], 1170)
        expected_cumulative['starting_value'] = np.repeat(expected_daily['starting_value'][0:1], 1170)
        for stat in stats:
            for i in (390, 781, 1172):
                expected_cumulative[stat] = np.insert(expected_cumulative[stat], i, expected_cumulative[stat][i - 1])
        for stat in stats:
            np.testing.assert_array_almost_equal(np.array([perf[stat] for perf in minute_perf]), expected_minute[stat])
            np.testing.assert_array_almost_equal(np.array([perf[stat] for perf in daily_perf]), expected_daily[stat])
            np.testing.assert_array_almost_equal(np.array([perf[stat] for perf in cumulative_perf]), expected_cumulative[stat])
        if change_loc == 'interday':
            self.assertEqual(algo.capital_change_deltas, {pd.Timestamp('2006-01-04', tz='UTC'): 1000.0})
        else:
            self.assertEqual(algo.capital_change_deltas, {pd.Timestamp('2006-01-04 17:00', tz='UTC'): 500.0, pd.Timestamp('2006-01-04 18:00', tz='UTC'): 500.0})

class TestGetDatetime(zf.WithMakeAlgo, zf.ZiplineTestCase):
    SIM_PARAMS_DATA_FREQUENCY = 'minute'
    START_DATE = to_utc('2014-01-02 9:31')
    END_DATE = to_utc('2014-01-03 9:31')
    ASSET_FINDER_EQUITY_SIDS = (0, 1)
    BENCHMARK_SID = None

    @parameterized.expand([('default', None), ('utc', 'UTC'), ('us_east', 'US/Eastern')])
    def test_get_datetime(self, name, tz):
        if False:
            for i in range(10):
                print('nop')
        algo = dedent('\n            import pandas as pd\n            from zipline.api import get_datetime\n\n            def initialize(context):\n                context.tz = {tz} or \'UTC\'\n                context.first_bar = True\n\n            def handle_data(context, data):\n                dt = get_datetime({tz})\n                if dt.tz.zone != context.tz:\n                    raise ValueError("Mismatched Zone")\n\n                if context.first_bar:\n                    if dt.tz_convert("US/Eastern").hour != 9:\n                        raise ValueError("Mismatched Hour")\n                    elif dt.tz_convert("US/Eastern").minute != 31:\n                        raise ValueError("Mismatched Minute")\n\n                    context.first_bar = False\n            '.format(tz=repr(tz)))
        algo = self.make_algo(script=algo)
        algo.run()
        self.assertFalse(algo.first_bar)

class TestTradingControls(zf.WithMakeAlgo, zf.ZiplineTestCase):
    START_DATE = pd.Timestamp('2006-01-03', tz='utc')
    END_DATE = pd.Timestamp('2006-01-06', tz='utc')
    sid = 133
    sids = ASSET_FINDER_EQUITY_SIDS = (133, 134)
    SIM_PARAMS_DATA_FREQUENCY = 'daily'
    DATA_PORTAL_USE_MINUTE_DATA = True

    @classmethod
    def init_class_fixtures(cls):
        if False:
            i = 10
            return i + 15
        super(TestTradingControls, cls).init_class_fixtures()
        cls.asset = cls.asset_finder.retrieve_asset(cls.sid)
        cls.another_asset = cls.asset_finder.retrieve_asset(134)

    def _check_algo(self, algo, expected_order_count, expected_exc):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(expected_exc) if expected_exc else nop_context:
            algo.run()
        self.assertEqual(algo.order_count, expected_order_count)

    def check_algo_succeeds(self, algo, order_count=4):
        if False:
            return 10
        self._check_algo(algo, order_count, None)

    def check_algo_fails(self, algo, order_count):
        if False:
            print('Hello World!')
        self._check_algo(algo, order_count, TradingControlViolation)

    def test_set_max_position_size(self):
        if False:
            return 10

        def initialize(self, asset, max_shares, max_notional):
            if False:
                while True:
                    i = 10
            self.set_slippage(FixedSlippage())
            self.order_count = 0
            self.set_max_position_size(asset=asset, max_shares=max_shares, max_notional=max_notional)

        def handle_data(algo, data):
            if False:
                i = 10
                return i + 15
            algo.order(algo.sid(self.sid), 1)
            algo.order_count += 1
        algo = self.make_algo(asset=self.asset, max_shares=10, max_notional=500.0, initialize=initialize, handle_data=handle_data)
        self.check_algo_succeeds(algo)

        def handle_data(algo, data):
            if False:
                return 10
            algo.order(algo.sid(self.sid), 3)
            algo.order_count += 1
        algo = self.make_algo(asset=self.asset, max_shares=10, max_notional=500.0, initialize=initialize, handle_data=handle_data)
        self.check_algo_fails(algo, 3)

        def handle_data(algo, data):
            if False:
                return 10
            algo.order(algo.sid(self.sid), 3)
            algo.order_count += 1
        algo = self.make_algo(asset=self.asset, max_shares=10, max_notional=67.0, initialize=initialize, handle_data=handle_data)
        self.check_algo_fails(algo, 2)

        def handle_data(algo, data):
            if False:
                for i in range(10):
                    print('nop')
            algo.order(algo.sid(self.sid), 10000)
            algo.order_count += 1
        algo = self.make_algo(asset=self.another_asset, max_shares=10, max_notional=67.0, initialize=initialize, handle_data=handle_data)
        self.check_algo_succeeds(algo)

        def handle_data(algo, data):
            if False:
                return 10
            algo.order(algo.sid(self.sid), 10000)
            algo.order_count += 1
        algo = self.make_algo(max_shares=10, max_notional=61.0, asset=None, initialize=initialize, handle_data=handle_data)
        self.check_algo_fails(algo, 0)

    def test_set_asset_restrictions(self):
        if False:
            return 10

        def initialize(algo, sid, restrictions, on_error):
            if False:
                for i in range(10):
                    print('nop')
            algo.order_count = 0
            algo.set_asset_restrictions(restrictions, on_error)

        def handle_data(algo, data):
            if False:
                print('Hello World!')
            algo.could_trade = data.can_trade(algo.sid(self.sid))
            algo.order(algo.sid(self.sid), 100)
            algo.order_count += 1
        rlm = HistoricalRestrictions([Restriction(self.sid, self.sim_params.start_session, RESTRICTION_STATES.FROZEN)])
        algo = self.make_algo(sid=self.sid, restrictions=rlm, on_error='fail', initialize=initialize, handle_data=handle_data)
        self.check_algo_fails(algo, 0)
        self.assertFalse(algo.could_trade)
        rlm = StaticRestrictions([self.sid])
        algo = self.make_algo(sid=self.sid, restrictions=rlm, on_error='fail', initialize=initialize, handle_data=handle_data)
        self.check_algo_fails(algo, 0)
        self.assertFalse(algo.could_trade)
        algo = self.make_algo(sid=self.sid, restrictions=rlm, on_error='log', initialize=initialize, handle_data=handle_data)
        with make_test_handler(self) as log_catcher:
            self.check_algo_succeeds(algo)
        logs = [r.message for r in log_catcher.records]
        self.assertIn('Order for 100 shares of Equity(133 [A]) at 2006-01-03 21:00:00+00:00 violates trading constraint RestrictedListOrder({})', logs)
        self.assertFalse(algo.could_trade)
        rlm = HistoricalRestrictions([Restriction(sid, self.sim_params.start_session, RESTRICTION_STATES.FROZEN) for sid in [134, 135, 136]])
        algo = self.make_algo(sid=self.sid, restrictions=rlm, on_error='fail', initialize=initialize, handle_data=handle_data)
        self.check_algo_succeeds(algo)
        self.assertTrue(algo.could_trade)

    @parameterized.expand([('order_first_restricted_sid', 0), ('order_second_restricted_sid', 1)])
    def test_set_multiple_asset_restrictions(self, name, to_order_idx):
        if False:
            i = 10
            return i + 15

        def initialize(algo, restrictions1, restrictions2, on_error):
            if False:
                while True:
                    i = 10
            algo.order_count = 0
            algo.set_asset_restrictions(restrictions1, on_error)
            algo.set_asset_restrictions(restrictions2, on_error)

        def handle_data(algo, data):
            if False:
                while True:
                    i = 10
            algo.could_trade1 = data.can_trade(algo.sid(self.sids[0]))
            algo.could_trade2 = data.can_trade(algo.sid(self.sids[1]))
            algo.order(algo.sid(self.sids[to_order_idx]), 100)
            algo.order_count += 1
        rl1 = StaticRestrictions([self.sids[0]])
        rl2 = StaticRestrictions([self.sids[1]])
        algo = self.make_algo(restrictions1=rl1, restrictions2=rl2, initialize=initialize, handle_data=handle_data, on_error='fail')
        self.check_algo_fails(algo, 0)
        self.assertFalse(algo.could_trade1)
        self.assertFalse(algo.could_trade2)

    def test_set_do_not_order_list(self):
        if False:
            print('Hello World!')

        def initialize(self, restricted_list):
            if False:
                print('Hello World!')
            self.order_count = 0
            self.set_do_not_order_list(restricted_list, on_error='fail')

        def handle_data(algo, data):
            if False:
                return 10
            algo.could_trade = data.can_trade(algo.sid(self.sid))
            algo.order(algo.sid(self.sid), 100)
            algo.order_count += 1
        rlm = [self.sid]
        algo = self.make_algo(restricted_list=rlm, initialize=initialize, handle_data=handle_data)
        self.check_algo_fails(algo, 0)
        self.assertFalse(algo.could_trade)

    def test_set_max_order_size(self):
        if False:
            for i in range(10):
                print('nop')

        def initialize(algo, asset, max_shares, max_notional):
            if False:
                print('Hello World!')
            algo.order_count = 0
            algo.set_max_order_size(asset=asset, max_shares=max_shares, max_notional=max_notional)

        def handle_data(algo, data):
            if False:
                print('Hello World!')
            algo.order(algo.sid(self.sid), 1)
            algo.order_count += 1
        algo = self.make_algo(initialize=initialize, handle_data=handle_data, asset=self.asset, max_shares=10, max_notional=500.0)
        self.check_algo_succeeds(algo)

        def handle_data(algo, data):
            if False:
                i = 10
                return i + 15
            algo.order(algo.sid(self.sid), algo.order_count + 1)
            algo.order_count += 1
        algo = self.make_algo(initialize=initialize, handle_data=handle_data, asset=self.asset, max_shares=3, max_notional=500.0)
        self.check_algo_fails(algo, 3)

        def handle_data(algo, data):
            if False:
                while True:
                    i = 10
            algo.order(algo.sid(self.sid), algo.order_count + 1)
            algo.order_count += 1
        algo = self.make_algo(initialize=initialize, handle_data=handle_data, asset=self.asset, max_shares=10, max_notional=40.0)
        self.check_algo_fails(algo, 3)

        def handle_data(algo, data):
            if False:
                print('Hello World!')
            algo.order(algo.sid(self.sid), 10000)
            algo.order_count += 1
        algo = self.make_algo(initialize=initialize, handle_data=handle_data, asset=self.another_asset, max_shares=1, max_notional=1.0)
        self.check_algo_succeeds(algo)

        def handle_data(algo, data):
            if False:
                i = 10
                return i + 15
            algo.order(algo.sid(self.sid), 10000)
            algo.order_count += 1
        algo = self.make_algo(initialize=initialize, handle_data=handle_data, asset=None, max_shares=1, max_notional=1.0)
        self.check_algo_fails(algo, 0)

    def test_set_max_order_count(self):
        if False:
            i = 10
            return i + 15

        def initialize(algo, count):
            if False:
                print('Hello World!')
            algo.order_count = 0
            algo.set_max_order_count(count)

        def handle_data(algo, data):
            if False:
                while True:
                    i = 10
            for i in range(5):
                algo.order(self.asset, 1)
                algo.order_count += 1
        algo = self.make_algo(count=3, initialize=initialize, handle_data=handle_data)
        with self.assertRaises(TradingControlViolation):
            algo.run()
        self.assertEqual(algo.order_count, 3)

    def test_set_max_order_count_minutely(self):
        if False:
            return 10
        sim_params = self.make_simparams(data_frequency='minute')

        def initialize(algo, max_orders_per_day):
            if False:
                return 10
            algo.minute_count = 0
            algo.order_count = 0
            algo.set_max_order_count(max_orders_per_day)

        def handle_data(algo, data):
            if False:
                return 10
            if algo.minute_count == 0 or algo.minute_count == 100:
                for i in range(5):
                    algo.order(self.asset, 1)
                    algo.order_count += 1
            algo.minute_count += 1
        algo = self.make_algo(initialize=initialize, handle_data=handle_data, max_orders_per_day=9, sim_params=sim_params)
        with self.assertRaises(TradingControlViolation):
            algo.run()
        self.assertEqual(algo.order_count, 9)

        def handle_data(algo, data):
            if False:
                for i in range(10):
                    print('nop')
            if algo.minute_count % 390 == 0:
                for i in range(5):
                    algo.order(self.asset, 1)
                    algo.order_count += 1
            algo.minute_count += 1
        algo = self.make_algo(initialize=initialize, handle_data=handle_data, max_orders_per_day=5, sim_params=sim_params)
        algo.run()
        self.assertEqual(algo.order_count, 20)

    def test_long_only(self):
        if False:
            for i in range(10):
                print('nop')

        def initialize(algo):
            if False:
                for i in range(10):
                    print('nop')
            algo.order_count = 0
            algo.set_long_only()

        def handle_data(algo, data):
            if False:
                print('Hello World!')
            algo.order(algo.sid(self.sid), -1)
            algo.order_count += 1
        algo = self.make_algo(initialize=initialize, handle_data=handle_data)
        self.check_algo_fails(algo, 0)

        def handle_data(algo, data):
            if False:
                while True:
                    i = 10
            if algo.order_count % 2 == 0:
                algo.order(algo.sid(self.sid), 1)
            else:
                algo.order(algo.sid(self.sid), -1)
            algo.order_count += 1
        algo = self.make_algo(initialize=initialize, handle_data=handle_data)
        self.check_algo_succeeds(algo)

        def handle_data(algo, data):
            if False:
                return 10
            amounts = [1, 1, 1, -3]
            algo.order(algo.sid(self.sid), amounts[algo.order_count])
            algo.order_count += 1
        algo = self.make_algo(initialize=initialize, handle_data=handle_data)
        self.check_algo_succeeds(algo)

        def handle_data(algo, data):
            if False:
                return 10
            amounts = [1, 1, 1, -4]
            algo.order(algo.sid(self.sid), amounts[algo.order_count])
            algo.order_count += 1
        algo = self.make_algo(initialize=initialize, handle_data=handle_data)
        self.check_algo_fails(algo, 3)

    def test_register_post_init(self):
        if False:
            while True:
                i = 10

        def initialize(algo):
            if False:
                for i in range(10):
                    print('nop')
            algo.initialized = True

        def handle_data(algo, data):
            if False:
                return 10
            with self.assertRaises(RegisterTradingControlPostInit):
                algo.set_max_position_size(self.sid, 1, 1)
            with self.assertRaises(RegisterTradingControlPostInit):
                algo.set_max_order_size(self.sid, 1, 1)
            with self.assertRaises(RegisterTradingControlPostInit):
                algo.set_max_order_count(1)
            with self.assertRaises(RegisterTradingControlPostInit):
                algo.set_long_only()
        self.run_algorithm(initialize=initialize, handle_data=handle_data)

class TestAssetDateBounds(zf.WithMakeAlgo, zf.ZiplineTestCase):
    START_DATE = pd.Timestamp('2014-01-02', tz='UTC')
    END_DATE = pd.Timestamp('2014-01-03', tz='UTC')
    SIM_PARAMS_START_DATE = END_DATE
    SIM_PARAMS_DATA_FREQUENCY = 'daily'
    DATA_PORTAL_USE_MINUTE_DATA = False
    BENCHMARK_SID = 3

    @classmethod
    def make_equity_info(cls):
        if False:
            print('Hello World!')
        T = partial(pd.Timestamp, tz='UTC')
        return pd.DataFrame.from_records([{'sid': 1, 'symbol': 'OLD', 'start_date': T('1990'), 'end_date': T('1991'), 'exchange': 'TEST'}, {'sid': 2, 'symbol': 'NEW', 'start_date': T('2017'), 'end_date': T('2018'), 'exchange': 'TEST'}, {'sid': 3, 'symbol': 'GOOD', 'start_date': cls.START_DATE, 'end_date': cls.END_DATE, 'exchange': 'TEST'}])

    def test_asset_date_bounds(self):
        if False:
            return 10

        def initialize(algo):
            if False:
                return 10
            algo.ran = False
            algo.register_trading_control(AssetDateBounds(on_error='fail'))

        def handle_data(algo, data):
            if False:
                return 10
            algo.order(algo.sid(3), 1)
            with self.assertRaises(TradingControlViolation):
                algo.order(algo.sid(1), 1)
            with self.assertRaises(TradingControlViolation):
                algo.order(algo.sid(2), 1)
            algo.ran = True
        algo = self.make_algo(initialize=initialize, handle_data=handle_data)
        algo.run()
        self.assertTrue(algo.ran)

class TestAccountControls(zf.WithMakeAlgo, zf.ZiplineTestCase):
    START_DATE = pd.Timestamp('2006-01-03', tz='utc')
    END_DATE = pd.Timestamp('2006-01-06', tz='utc')
    (sidint,) = ASSET_FINDER_EQUITY_SIDS = (133,)
    BENCHMARK_SID = None
    SIM_PARAMS_DATA_FREQUENCY = 'daily'
    DATA_PORTAL_USE_MINUTE_DATA = False

    @classmethod
    def make_equity_daily_bar_data(cls, country_code, sids):
        if False:
            print('Hello World!')
        frame = pd.DataFrame(data={'close': [10.0, 10.0, 11.0, 11.0], 'open': [10.0, 10.0, 11.0, 11.0], 'low': [9.5, 9.5, 10.45, 10.45], 'high': [10.5, 10.5, 11.55, 11.55], 'volume': [100, 100, 100, 300]}, index=cls.equity_daily_bar_days)
        yield (cls.sidint, frame)

    def _check_algo(self, algo, expected_exc):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(expected_exc) if expected_exc else nop_context:
            algo.run()

    def check_algo_succeeds(self, algo):
        if False:
            i = 10
            return i + 15
        self._check_algo(algo, None)

    def check_algo_fails(self, algo):
        if False:
            return 10
        self._check_algo(algo, AccountControlViolation)

    def test_set_max_leverage(self):
        if False:
            for i in range(10):
                print('nop')

        def initialize(algo, max_leverage):
            if False:
                return 10
            algo.set_max_leverage(max_leverage=max_leverage)

        def handle_data(algo, data):
            if False:
                return 10
            algo.order(algo.sid(self.sidint), 1)
            algo.record(latest_time=algo.get_datetime())
        algo = self.make_algo(initialize=initialize, handle_data=handle_data, max_leverage=0)
        self.check_algo_fails(algo)
        self.assertEqual(algo.recorded_vars['latest_time'], pd.Timestamp('2006-01-04 21:00:00', tz='UTC'))

        def handle_data(algo, data):
            if False:
                while True:
                    i = 10
            algo.order(algo.sid(self.sidint), 1)
        algo = self.make_algo(initialize=initialize, handle_data=handle_data, max_leverage=1)
        self.check_algo_succeeds(algo)

    def test_set_min_leverage(self):
        if False:
            return 10

        def initialize(algo, min_leverage, grace_period):
            if False:
                print('Hello World!')
            algo.set_min_leverage(min_leverage=min_leverage, grace_period=grace_period)

        def handle_data(algo, data):
            if False:
                print('Hello World!')
            algo.order_target_percent(algo.sid(self.sidint), 0.5)
            algo.record(latest_time=algo.get_datetime())

        def make_algo(min_leverage, grace_period):
            if False:
                while True:
                    i = 10
            return self.make_algo(initialize=initialize, handle_data=handle_data, min_leverage=min_leverage, grace_period=grace_period)
        offset = pd.Timedelta('10 days')
        algo = make_algo(min_leverage=1, grace_period=offset)
        self.check_algo_succeeds(algo)
        offset = pd.Timedelta('1 days')
        algo = make_algo(min_leverage=1, grace_period=offset)
        self.check_algo_fails(algo)
        self.assertEqual(algo.recorded_vars['latest_time'], pd.Timestamp('2006-01-04 21:00:00', tz='UTC'))
        offset = pd.Timedelta('2 days')
        algo = make_algo(min_leverage=1, grace_period=offset)
        self.check_algo_fails(algo)
        self.assertEqual(algo.recorded_vars['latest_time'], pd.Timestamp('2006-01-05 21:00:00', tz='UTC'))
        algo = make_algo(min_leverage=0.0001, grace_period=offset)
        self.check_algo_succeeds(algo)

class TestFuturesAlgo(zf.WithMakeAlgo, zf.ZiplineTestCase):
    START_DATE = pd.Timestamp('2016-01-06', tz='utc')
    END_DATE = pd.Timestamp('2016-01-07', tz='utc')
    FUTURE_MINUTE_BAR_START_DATE = pd.Timestamp('2016-01-05', tz='UTC')
    SIM_PARAMS_DATA_FREQUENCY = 'minute'
    TRADING_CALENDAR_STRS = ('us_futures',)
    TRADING_CALENDAR_PRIMARY_CAL = 'us_futures'
    BENCHMARK_SID = None

    @classmethod
    def make_futures_info(cls):
        if False:
            return 10
        return pd.DataFrame.from_dict({1: {'symbol': 'CLG16', 'root_symbol': 'CL', 'start_date': pd.Timestamp('2015-12-01', tz='UTC'), 'notice_date': pd.Timestamp('2016-01-20', tz='UTC'), 'expiration_date': pd.Timestamp('2016-02-19', tz='UTC'), 'auto_close_date': pd.Timestamp('2016-01-18', tz='UTC'), 'exchange': 'TEST'}}, orient='index')

    def test_futures_history(self):
        if False:
            while True:
                i = 10
        algo_code = dedent("\n            from datetime import time\n            from zipline.api import (\n                date_rules,\n                get_datetime,\n                schedule_function,\n                sid,\n                time_rules,\n            )\n\n            def initialize(context):\n                context.history_values = []\n\n                schedule_function(\n                    make_history_call,\n                    date_rules.every_day(),\n                    time_rules.market_open(),\n                )\n\n                schedule_function(\n                    check_market_close_time,\n                    date_rules.every_day(),\n                    time_rules.market_close(),\n                )\n\n            def make_history_call(context, data):\n                # Ensure that the market open is 6:31am US/Eastern.\n                open_time = get_datetime().tz_convert('US/Eastern').time()\n                assert open_time == time(6, 31)\n                context.history_values.append(\n                    data.history(sid(1), 'close', 5, '1m'),\n                )\n\n            def check_market_close_time(context, data):\n                # Ensure that this function is called at 4:59pm US/Eastern.\n                # By default, `market_close()` uses an offset of 1 minute.\n                close_time = get_datetime().tz_convert('US/Eastern').time()\n                assert close_time == time(16, 59)\n            ")
        algo = self.make_algo(script=algo_code, trading_calendar=get_calendar('us_futures'))
        algo.run()
        np.testing.assert_array_equal(algo.history_values[0].index, pd.date_range('2016-01-06 6:27', '2016-01-06 6:31', freq='min', tz='US/Eastern'))
        np.testing.assert_array_equal(algo.history_values[1].index, pd.date_range('2016-01-07 6:27', '2016-01-07 6:31', freq='min', tz='US/Eastern'))
        np.testing.assert_array_equal(algo.history_values[0].values, list(map(float, range(2196, 2201))))
        np.testing.assert_array_equal(algo.history_values[1].values, list(map(float, range(3636, 3641))))

    @staticmethod
    def algo_with_slippage(slippage_model):
        if False:
            while True:
                i = 10
        return dedent("\n            from zipline.api import (\n                commission,\n                order,\n                set_commission,\n                set_slippage,\n                sid,\n                slippage,\n                get_datetime,\n            )\n\n            def initialize(context):\n                commission_model = commission.PerFutureTrade(0)\n                set_commission(us_futures=commission_model)\n                slippage_model = slippage.{model}\n                set_slippage(us_futures=slippage_model)\n                context.ordered = False\n\n            def handle_data(context, data):\n                if not context.ordered:\n                    order(sid(1), 10)\n                    context.ordered = True\n                    context.order_price = data.current(sid(1), 'price')\n            ").format(model=slippage_model)

    def test_fixed_future_slippage(self):
        if False:
            while True:
                i = 10
        algo_code = self.algo_with_slippage('FixedSlippage(spread=0.10)')
        algo = self.make_algo(script=algo_code, trading_calendar=get_calendar('us_futures'))
        results = algo.run()
        all_txns = [val for sublist in results['transactions'].tolist() for val in sublist]
        self.assertEqual(len(all_txns), 1)
        txn = all_txns[0]
        expected_spread = 0.05
        expected_price = algo.order_price + 1 + expected_spread
        self.assertEqual(txn['price'], expected_price)
        self.assertEqual(results['orders'][0][0]['commission'], 0.0)

    def test_volume_contract_slippage(self):
        if False:
            for i in range(10):
                print('nop')
        algo_code = self.algo_with_slippage('VolumeShareSlippage(volume_limit=0.05, price_impact=0.1)')
        algo = self.make_algo(script=algo_code, trading_calendar=get_calendar('us_futures'))
        results = algo.run()
        self.assertEqual(results['orders'][0][0]['commission'], 0.0)
        all_txns = [val for sublist in results['transactions'].tolist() for val in sublist]
        self.assertEqual(len(all_txns), 2)
        for (i, txn) in enumerate(all_txns):
            order_price = algo.order_price + i + 1
            expected_impact = order_price * 0.1 * 0.05 ** 2
            expected_price = order_price + expected_impact
            self.assertEqual(txn['price'], expected_price)

class TestAnalyzeAPIMethod(zf.WithMakeAlgo, zf.ZiplineTestCase):
    START_DATE = pd.Timestamp('2016-01-05', tz='utc')
    END_DATE = pd.Timestamp('2016-01-05', tz='utc')
    SIM_PARAMS_DATA_FREQUENCY = 'daily'
    DATA_PORTAL_USE_MINUTE_DATA = False

    def test_analyze_called(self):
        if False:
            while True:
                i = 10
        self.perf_ref = None

        def initialize(context):
            if False:
                print('Hello World!')
            pass

        def handle_data(context, data):
            if False:
                while True:
                    i = 10
            pass

        def analyze(context, perf):
            if False:
                i = 10
                return i + 15
            self.perf_ref = perf
        algo = self.make_algo(initialize=initialize, handle_data=handle_data, analyze=analyze)
        results = algo.run()
        self.assertIs(results, self.perf_ref)

class TestOrderCancelation(zf.WithMakeAlgo, zf.ZiplineTestCase):
    START_DATE = pd.Timestamp('2016-01-05', tz='utc')
    END_DATE = pd.Timestamp('2016-01-07', tz='utc')
    ASSET_FINDER_EQUITY_SIDS = (1,)
    ASSET_FINDER_EQUITY_SYMBOLS = ('ASSET1',)
    BENCHMARK_SID = None
    code = dedent('\n        from zipline.api import (\n            sid, order, set_slippage, slippage, VolumeShareSlippage,\n            set_cancel_policy, cancel_policy, EODCancel\n        )\n\n\n        def initialize(context):\n            set_slippage(\n                slippage.VolumeShareSlippage(\n                    volume_limit=1,\n                    price_impact=0\n                )\n            )\n\n            {0}\n            context.ordered = False\n\n\n        def handle_data(context, data):\n            if not context.ordered:\n                order(sid(1), {1})\n                context.ordered = True\n        ')

    @classmethod
    def make_equity_minute_bar_data(cls):
        if False:
            i = 10
            return i + 15
        asset_minutes = cls.trading_calendar.minutes_for_sessions_in_range(cls.START_DATE, cls.END_DATE)
        minutes_count = len(asset_minutes)
        minutes_arr = np.arange(1, 1 + minutes_count)
        yield (1, pd.DataFrame({'open': minutes_arr + 1, 'high': minutes_arr + 2, 'low': minutes_arr - 1, 'close': minutes_arr, 'volume': np.full(minutes_count, 1.0)}, index=asset_minutes))

    @classmethod
    def make_equity_daily_bar_data(cls, country_code, sids):
        if False:
            print('Hello World!')
        yield (1, pd.DataFrame({'open': np.full(3, 1, dtype=np.float64), 'high': np.full(3, 1, dtype=np.float64), 'low': np.full(3, 1, dtype=np.float64), 'close': np.full(3, 1, dtype=np.float64), 'volume': np.full(3, 1, dtype=np.float64)}, index=cls.equity_daily_bar_days))

    def prep_algo(self, cancelation_string, data_frequency='minute', amount=1000, minute_emission=False):
        if False:
            for i in range(10):
                print('nop')
        code = self.code.format(cancelation_string, amount)
        return self.make_algo(script=code, sim_params=self.make_simparams(data_frequency=data_frequency, emission_rate='minute' if minute_emission else 'daily'))

    @parameter_space(direction=[1, -1], minute_emission=[True, False])
    def test_eod_order_cancel_minute(self, direction, minute_emission):
        if False:
            while True:
                i = 10
        '\n        Test that EOD order cancel works in minute mode for both shorts and\n        longs, and both daily emission and minute emission\n        '
        algo = self.prep_algo('set_cancel_policy(cancel_policy.EODCancel())', amount=np.copysign(1000, direction), minute_emission=minute_emission)
        log_catcher = TestHandler()
        with log_catcher:
            results = algo.run()
            for daily_positions in results.positions:
                self.assertEqual(1, len(daily_positions))
                self.assertEqual(np.copysign(389, direction), daily_positions[0]['amount'])
                self.assertEqual(1, results.positions[0][0]['sid'])
            np.testing.assert_array_equal([1, 0, 0], list(map(len, results.orders)))
            np.testing.assert_array_equal([389, 0, 0], list(map(len, results.transactions)))
            the_order = results.orders[0][0]
            self.assertEqual(ORDER_STATUS.CANCELLED, the_order['status'])
            self.assertEqual(np.copysign(389, direction), the_order['filled'])
            warnings = [record for record in log_catcher.records if record.level == WARNING]
            self.assertEqual(1, len(warnings))
            if direction == 1:
                self.assertEqual('Your order for 1000 shares of ASSET1 has been partially filled. 389 shares were successfully purchased. 611 shares were not filled by the end of day and were canceled.', str(warnings[0].message))
            elif direction == -1:
                self.assertEqual('Your order for -1000 shares of ASSET1 has been partially filled. 389 shares were successfully sold. 611 shares were not filled by the end of day and were canceled.', str(warnings[0].message))

    def test_default_cancelation_policy(self):
        if False:
            print('Hello World!')
        algo = self.prep_algo('')
        log_catcher = TestHandler()
        with log_catcher:
            results = algo.run()
            np.testing.assert_array_equal([1, 1, 1], list(map(len, results.orders)))
            np.testing.assert_array_equal([389, 390, 221], list(map(len, results.transactions)))
            self.assertFalse(log_catcher.has_warnings)

    def test_eod_order_cancel_daily(self):
        if False:
            return 10
        algo = self.prep_algo('set_cancel_policy(cancel_policy.EODCancel())', 'daily')
        log_catcher = TestHandler()
        with log_catcher:
            results = algo.run()
            np.testing.assert_array_equal([1, 1, 1], list(map(len, results.orders)))
            np.testing.assert_array_equal([0, 1, 1], list(map(len, results.transactions)))
            self.assertFalse(log_catcher.has_warnings)

class TestDailyEquityAutoClose(zf.WithMakeAlgo, zf.ZiplineTestCase):
    """
    Tests if delisted equities are properly removed from a portfolio holding
    positions in said equities.
    """
    START_DATE = pd.Timestamp('2015-01-05', tz='UTC')
    END_DATE = pd.Timestamp('2015-01-13', tz='UTC')
    SIM_PARAMS_DATA_FREQUENCY = 'daily'
    DATA_PORTAL_USE_MINUTE_DATA = False
    BENCHMARK_SID = None

    @classmethod
    def init_class_fixtures(cls):
        if False:
            print('Hello World!')
        super(TestDailyEquityAutoClose, cls).init_class_fixtures()
        cls.assets = cls.asset_finder.retrieve_all(cls.asset_finder.equities_sids)

    @classmethod
    def make_equity_info(cls):
        if False:
            while True:
                i = 10
        cls.test_days = cls.trading_calendar.sessions_in_range(cls.START_DATE, cls.END_DATE)
        assert len(cls.test_days) == 7, 'Number of days in test changed!'
        cls.first_asset_expiration = cls.test_days[2]
        cls.asset_info = make_jagged_equity_info(num_assets=3, start_date=cls.test_days[0], first_end=cls.first_asset_expiration, frequency=cls.trading_calendar.day, periods_between_ends=2, auto_close_delta=2 * cls.trading_calendar.day)
        return cls.asset_info

    @classmethod
    def make_equity_daily_bar_data(cls, country_code, sids):
        if False:
            for i in range(10):
                print('nop')
        cls.daily_data = make_trade_data_for_asset_info(dates=cls.test_days, asset_info=cls.asset_info, price_start=10, price_step_by_sid=10, price_step_by_date=1, volume_start=100, volume_step_by_sid=100, volume_step_by_date=10)
        return cls.daily_data.items()

    def daily_prices_on_tick(self, row):
        if False:
            print('Hello World!')
        return [trades.iloc[row].close for trades in itervalues(self.daily_data)]

    def final_daily_price(self, asset):
        if False:
            return 10
        return self.daily_data[asset.sid].loc[asset.end_date].close

    def default_initialize(self):
        if False:
            return 10
        '\n        Initialize function shared between test algos.\n        '

        def initialize(context):
            if False:
                i = 10
                return i + 15
            context.ordered = False
            context.set_commission(PerShare(0, 0))
            context.set_slippage(FixedSlippage(spread=0))
            context.num_positions = []
            context.cash = []
        return initialize

    def default_handle_data(self, assets, order_size):
        if False:
            i = 10
            return i + 15
        '\n        Handle data function shared between test algos.\n        '

        def handle_data(context, data):
            if False:
                return 10
            if not context.ordered:
                for asset in assets:
                    context.order(asset, order_size)
                context.ordered = True
            context.cash.append(context.portfolio.cash)
            context.num_positions.append(len(context.portfolio.positions))
        return handle_data

    @parameter_space(order_size=[10, -10], capital_base=[1, 100000], __fail_fast=True)
    def test_daily_delisted_equities(self, order_size, capital_base):
        if False:
            return 10
        '\n        Make sure that after an equity gets delisted, our portfolio holds the\n        correct number of equities and correct amount of cash.\n        '
        assets = self.assets
        final_prices = {asset.sid: self.final_daily_price(asset) for asset in assets}
        initial_fill_prices = self.daily_prices_on_tick(1)
        cost_basis = sum(initial_fill_prices) * order_size
        fp0 = final_prices[0]
        fp1 = final_prices[1]
        algo = self.make_algo(initialize=self.default_initialize(), handle_data=self.default_handle_data(assets, order_size), sim_params=self.make_simparams(capital_base=capital_base, data_frequency='daily'))
        output = algo.run()
        initial_cash = capital_base
        after_fills = initial_cash - cost_basis
        after_first_auto_close = after_fills + fp0 * order_size
        after_second_auto_close = after_first_auto_close + fp1 * order_size
        expected_cash = [initial_cash, after_fills, after_fills, after_fills, after_first_auto_close, after_first_auto_close, after_second_auto_close]
        expected_num_positions = [0, 3, 3, 3, 2, 2, 1]
        self.assertEqual(expected_cash, list(output['ending_cash']))
        expected_cash.insert(3, after_fills)
        self.assertEqual(algo.cash, expected_cash[:-1])
        if order_size > 0:
            self.assertEqual(expected_num_positions, list(output['longs_count']))
            self.assertEqual([0] * len(self.test_days), list(output['shorts_count']))
        else:
            self.assertEqual(expected_num_positions, list(output['shorts_count']))
            self.assertEqual([0] * len(self.test_days), list(output['longs_count']))
        expected_num_positions.insert(3, 3)
        self.assertEqual(algo.num_positions, expected_num_positions[:-1])
        transactions = output['transactions']
        initial_fills = transactions.iloc[1]
        self.assertEqual(len(initial_fills), len(assets))
        last_minute_of_session = self.trading_calendar.session_close(self.test_days[1])
        for (asset, txn) in zip(assets, initial_fills):
            self.assertDictContainsSubset({'amount': order_size, 'commission': None, 'dt': last_minute_of_session, 'price': initial_fill_prices[asset], 'sid': asset}, txn)
            self.assertIsInstance(txn['order_id'], str)

        def transactions_for_date(date):
            if False:
                for i in range(10):
                    print('nop')
            return transactions.iloc[self.test_days.get_loc(date)]
        (first_auto_close_transaction,) = transactions_for_date(assets[0].auto_close_date)
        self.assertEqual(first_auto_close_transaction, {'amount': -order_size, 'commission': None, 'dt': self.trading_calendar.session_close(assets[0].auto_close_date), 'price': fp0, 'sid': assets[0], 'order_id': None})
        (second_auto_close_transaction,) = transactions_for_date(assets[1].auto_close_date)
        self.assertEqual(second_auto_close_transaction, {'amount': -order_size, 'commission': None, 'dt': self.trading_calendar.session_close(assets[1].auto_close_date), 'price': fp1, 'sid': assets[1], 'order_id': None})

    def test_cancel_open_orders(self):
        if False:
            while True:
                i = 10
        '\n        Test that any open orders for an equity that gets delisted are\n        canceled.  Unless an equity is auto closed, any open orders for that\n        equity will persist indefinitely.\n        '
        assets = self.assets
        first_asset_end_date = assets[0].end_date
        first_asset_auto_close_date = assets[0].auto_close_date

        def initialize(context):
            if False:
                print('Hello World!')
            pass

        def handle_data(context, data):
            if False:
                i = 10
                return i + 15
            assert context.portfolio.cash == context.portfolio.starting_cash
            today_session = self.trading_calendar.minute_to_session_label(context.get_datetime())
            day_after_auto_close = self.trading_calendar.next_session_label(first_asset_auto_close_date)
            if today_session == first_asset_end_date:
                assert len(context.get_open_orders()) == 0
                context.order(context.sid(0), 10)
                assert len(context.get_open_orders()) == 1
            elif today_session == first_asset_auto_close_date:
                assert len(context.get_open_orders()) == 1
            elif today_session == day_after_auto_close:
                assert len(context.get_open_orders()) == 0
        algo = self.make_algo(initialize=initialize, handle_data=handle_data, sim_params=self.make_simparams(data_frequency='daily'))
        results = algo.run()
        orders = results['orders']

        def orders_for_date(date):
            if False:
                i = 10
                return i + 15
            return orders.iloc[self.test_days.get_loc(date)]
        original_open_orders = orders_for_date(first_asset_end_date)
        assert len(original_open_orders) == 1
        last_close_for_asset = algo.trading_calendar.session_close(first_asset_end_date)
        self.assertDictContainsSubset({'amount': 10, 'commission': 0.0, 'created': last_close_for_asset, 'dt': last_close_for_asset, 'sid': assets[0], 'status': ORDER_STATUS.OPEN, 'filled': 0}, original_open_orders[0])
        orders_after_auto_close = orders_for_date(first_asset_auto_close_date)
        assert len(orders_after_auto_close) == 1
        self.assertDictContainsSubset({'amount': 10, 'commission': 0.0, 'created': last_close_for_asset, 'dt': algo.trading_calendar.session_close(first_asset_auto_close_date), 'sid': assets[0], 'status': ORDER_STATUS.CANCELLED, 'filled': 0}, orders_after_auto_close[0])

class TestMinutelyEquityAutoClose(zf.WithMakeAlgo, zf.ZiplineTestCase):
    START_DATE = pd.Timestamp('2015-01-05', tz='UTC')
    END_DATE = pd.Timestamp('2015-01-13', tz='UTC')
    BENCHMARK_SID = None

    @classmethod
    def init_class_fixtures(cls):
        if False:
            print('Hello World!')
        super(TestMinutelyEquityAutoClose, cls).init_class_fixtures()
        cls.assets = cls.asset_finder.retrieve_all(cls.asset_finder.equities_sids)

    @classmethod
    def make_equity_info(cls):
        if False:
            return 10
        cls.test_days = cls.trading_calendar.sessions_in_range(cls.START_DATE, cls.END_DATE)
        cls.test_minutes = cls.trading_calendar.minutes_for_sessions_in_range(cls.START_DATE, cls.END_DATE)
        cls.first_asset_expiration = cls.test_days[2]
        cls.asset_info = make_jagged_equity_info(num_assets=3, start_date=cls.test_days[0], first_end=cls.first_asset_expiration, frequency=cls.trading_calendar.day, periods_between_ends=2, auto_close_delta=1 * cls.trading_calendar.day)
        return cls.asset_info

    @classmethod
    def make_equity_minute_bar_data(cls):
        if False:
            print('Hello World!')
        cls.minute_data = make_trade_data_for_asset_info(dates=cls.test_minutes, asset_info=cls.asset_info, price_start=10, price_step_by_sid=10, price_step_by_date=1, volume_start=100, volume_step_by_sid=100, volume_step_by_date=10)
        return cls.minute_data.items()

    def minute_prices_on_tick(self, row):
        if False:
            print('Hello World!')
        return [trades.iloc[row].close for trades in itervalues(self.minute_data)]

    def final_minute_price(self, asset):
        if False:
            while True:
                i = 10
        return self.minute_data[asset.sid].loc[self.trading_calendar.session_close(asset.end_date)].close

    def default_initialize(self):
        if False:
            i = 10
            return i + 15
        '\n        Initialize function shared between test algos.\n        '

        def initialize(context):
            if False:
                print('Hello World!')
            context.ordered = False
            context.set_commission(PerShare(0, 0))
            context.set_slippage(FixedSlippage(spread=0))
            context.num_positions = []
            context.cash = []
        return initialize

    def default_handle_data(self, assets, order_size):
        if False:
            return 10
        '\n        Handle data function shared between test algos.\n        '

        def handle_data(context, data):
            if False:
                while True:
                    i = 10
            if not context.ordered:
                for asset in assets:
                    context.order(asset, order_size)
                context.ordered = True
            context.cash.append(context.portfolio.cash)
            context.num_positions.append(len(context.portfolio.positions))
        return handle_data

    def test_minutely_delisted_equities(self):
        if False:
            print('Hello World!')
        assets = self.assets
        final_prices = {asset.sid: self.final_minute_price(asset) for asset in assets}
        backtest_minutes = self.minute_data[0].index.tolist()
        order_size = 10
        capital_base = 100000
        algo = self.make_algo(initialize=self.default_initialize(), handle_data=self.default_handle_data(assets, order_size), sim_params=self.make_simparams(capital_base=capital_base, data_frequency='minute'))
        output = algo.run()
        initial_fill_prices = self.minute_prices_on_tick(1)
        cost_basis = sum(initial_fill_prices) * order_size
        fp0 = final_prices[0]
        fp1 = final_prices[1]
        initial_cash = capital_base
        after_fills = initial_cash - cost_basis
        after_first_auto_close = after_fills + fp0 * order_size
        after_second_auto_close = after_first_auto_close + fp1 * order_size
        expected_cash = [initial_cash]
        expected_position_counts = [0]
        expected_cash.extend([after_fills] * (389 + 390 + 390 + 390))
        expected_position_counts.extend([3] * (389 + 390 + 390 + 390))
        expected_cash.extend([after_first_auto_close] * (390 + 390))
        expected_position_counts.extend([2] * (390 + 390))
        expected_cash.extend([after_second_auto_close] * 390)
        expected_position_counts.extend([1] * 390)
        self.assertEqual(len(algo.cash), len(expected_cash))
        self.assertEqual(algo.cash, expected_cash)
        self.assertEqual(list(output['ending_cash']), [after_fills, after_fills, after_fills, after_first_auto_close, after_first_auto_close, after_second_auto_close, after_second_auto_close])
        self.assertEqual(algo.num_positions, expected_position_counts)
        self.assertEqual(list(output['longs_count']), [3, 3, 3, 2, 2, 1, 1])
        transactions = output['transactions']
        initial_fills = transactions.iloc[0]
        self.assertEqual(len(initial_fills), len(assets))
        for (asset, txn) in zip(assets, initial_fills):
            self.assertDictContainsSubset({'amount': order_size, 'commission': None, 'dt': backtest_minutes[1], 'price': initial_fill_prices[asset], 'sid': asset}, txn)
            self.assertIsInstance(txn['order_id'], str)

        def transactions_for_date(date):
            if False:
                while True:
                    i = 10
            return transactions.iloc[self.test_days.get_loc(date)]
        (first_auto_close_transaction,) = transactions_for_date(assets[0].auto_close_date)
        self.assertEqual(first_auto_close_transaction, {'amount': -order_size, 'commission': None, 'dt': algo.trading_calendar.session_close(assets[0].auto_close_date), 'price': fp0, 'sid': assets[0], 'order_id': None})
        (second_auto_close_transaction,) = transactions_for_date(assets[1].auto_close_date)
        self.assertEqual(second_auto_close_transaction, {'amount': -order_size, 'commission': None, 'dt': algo.trading_calendar.session_close(assets[1].auto_close_date), 'price': fp1, 'sid': assets[1], 'order_id': None})

class TestOrderAfterDelist(zf.WithMakeAlgo, zf.ZiplineTestCase):
    start = pd.Timestamp('2016-01-05', tz='utc')
    day_1 = pd.Timestamp('2016-01-06', tz='utc')
    day_4 = pd.Timestamp('2016-01-11', tz='utc')
    end = pd.Timestamp('2016-01-15', tz='utc')
    BENCHMARK_SID = None

    @classmethod
    def make_equity_info(cls):
        if False:
            for i in range(10):
                print('nop')
        return pd.DataFrame.from_dict({1: {'start_date': cls.start, 'end_date': cls.day_1, 'auto_close_date': cls.day_4, 'symbol': 'ASSET1', 'exchange': 'TEST'}, 2: {'start_date': cls.start, 'end_date': cls.day_4, 'auto_close_date': cls.day_1, 'symbol': 'ASSET2', 'exchange': 'TEST'}}, orient='index')

    def init_instance_fixtures(self):
        if False:
            while True:
                i = 10
        super(TestOrderAfterDelist, self).init_instance_fixtures()
        self.data_portal = FakeDataPortal(self.asset_finder)

    @parameterized.expand([('auto_close_after_end_date', 1), ('auto_close_before_end_date', 2)])
    def test_order_in_quiet_period(self, name, sid):
        if False:
            while True:
                i = 10
        asset = self.asset_finder.retrieve_asset(sid)
        algo_code = dedent('\n        from zipline.api import (\n            sid,\n            order,\n            order_value,\n            order_percent,\n            order_target,\n            order_target_percent,\n            order_target_value\n        )\n\n        def initialize(context):\n            pass\n\n        def handle_data(context, data):\n            order(sid({sid}), 1)\n            order_value(sid({sid}), 100)\n            order_percent(sid({sid}), 0.5)\n            order_target(sid({sid}), 50)\n            order_target_percent(sid({sid}), 0.5)\n            order_target_value(sid({sid}), 50)\n        ').format(sid=sid)
        algo = self.make_algo(script=algo_code, sim_params=SimulationParameters(start_session=pd.Timestamp('2016-01-06', tz='UTC'), end_session=pd.Timestamp('2016-01-07', tz='UTC'), trading_calendar=self.trading_calendar, data_frequency='minute'))
        with make_test_handler(self) as log_catcher:
            algo.run()
            warnings = [r for r in log_catcher.records if r.level == logbook.WARNING]
            self.assertEqual(6 * 390, len(warnings))
            for w in warnings:
                expected_message = 'Cannot place order for ASSET{sid}, as it has de-listed. Any existing positions for this asset will be liquidated on {date}.'.format(sid=sid, date=asset.auto_close_date)
                self.assertEqual(expected_message, w.message)

class AlgoInputValidationTestCase(zf.WithMakeAlgo, zf.ZiplineTestCase):

    def test_reject_passing_both_api_methods_and_script(self):
        if False:
            return 10
        script = dedent('\n            def initialize(context):\n                pass\n\n            def handle_data(context, data):\n                pass\n\n            def before_trading_start(context, data):\n                pass\n\n            def analyze(context, results):\n                pass\n            ')
        for method in ('initialize', 'handle_data', 'before_trading_start', 'analyze'):
            with self.assertRaises(ValueError):
                self.make_algo(script=script, **{method: lambda *args, **kwargs: None})