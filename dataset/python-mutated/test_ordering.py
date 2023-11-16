from nose_parameterized import parameterized
import pandas as pd
from zipline.algorithm import TradingAlgorithm
import zipline.api as api
import zipline.errors as ze
from zipline.finance.execution import StopLimitOrder
import zipline.testing.fixtures as zf
from zipline.testing.predicates import assert_equal
import zipline.test_algorithms as zta

def T(s):
    if False:
        return 10
    return pd.Timestamp(s, tz='UTC')

class TestOrderMethods(zf.WithConstantEquityMinuteBarData, zf.WithConstantFutureMinuteBarData, zf.WithMakeAlgo, zf.ZiplineTestCase):
    START_DATE = T('2006-01-03')
    END_DATE = T('2006-01-06')
    SIM_PARAMS_START_DATE = T('2006-01-04')
    ASSET_FINDER_EQUITY_SIDS = (1,)
    EQUITY_DAILY_BAR_SOURCE_FROM_MINUTE = True
    FUTURE_DAILY_BAR_SOURCE_FROM_MINUTE = True
    EQUITY_MINUTE_CONSTANT_LOW = 2.0
    EQUITY_MINUTE_CONSTANT_OPEN = 2.0
    EQUITY_MINUTE_CONSTANT_CLOSE = 2.0
    EQUITY_MINUTE_CONSTANT_HIGH = 2.0
    EQUITY_MINUTE_CONSTANT_VOLUME = 10000.0
    FUTURE_MINUTE_CONSTANT_LOW = 2.0
    FUTURE_MINUTE_CONSTANT_OPEN = 2.0
    FUTURE_MINUTE_CONSTANT_CLOSE = 2.0
    FUTURE_MINUTE_CONSTANT_HIGH = 2.0
    FUTURE_MINUTE_CONSTANT_VOLUME = 10000.0
    SIM_PARAMS_CAPITAL_BASE = 10000

    @classmethod
    def make_futures_info(cls):
        if False:
            while True:
                i = 10
        return pd.DataFrame.from_dict({2: {'multiplier': 10, 'symbol': 'F', 'exchange': 'TEST'}}, orient='index')

    @classmethod
    def init_class_fixtures(cls):
        if False:
            return 10
        super(TestOrderMethods, cls).init_class_fixtures()
        cls.EQUITY = cls.asset_finder.retrieve_asset(1)
        cls.FUTURE = cls.asset_finder.retrieve_asset(2)

    @parameterized.expand([('order', 1), ('order_value', 1000), ('order_target', 1), ('order_target_value', 1000), ('order_percent', 1), ('order_target_percent', 1)])
    def test_cannot_order_in_before_trading_start(self, order_method, amount):
        if False:
            while True:
                i = 10
        algotext = '\nfrom zipline.api import sid, {order_func}\n\ndef initialize(context):\n    context.asset = sid(1)\n\ndef before_trading_start(context, data):\n    {order_func}(context.asset, {arg})\n     '.format(order_func=order_method, arg=amount)
        algo = self.make_algo(script=algotext)
        with self.assertRaises(ze.OrderInBeforeTradingStart):
            algo.run()

    @parameterized.expand([('order', 5000), ('order_value', 10000), ('order_percent', 1)])
    def test_order_equity_non_targeted(self, order_method, amount):
        if False:
            print('Hello World!')
        algotext = '\nimport zipline.api as api\n\ndef initialize(context):\n    api.set_slippage(api.slippage.FixedSlippage(spread=0.0))\n    api.set_commission(api.commission.PerShare(0))\n\n    context.equity = api.sid(1)\n\n    api.schedule_function(\n        func=do_order,\n        date_rule=api.date_rules.every_day(),\n        time_rule=api.time_rules.market_open(),\n    )\n\ndef do_order(context, data):\n    context.ordered = True\n    api.{order_func}(context.equity, {arg})\n     '.format(order_func=order_method, arg=amount)
        result = self.run_algorithm(script=algotext)
        for orders in result.orders.values:
            assert_equal(len(orders), 1)
            assert_equal(orders[0]['amount'], 5000)
            assert_equal(orders[0]['sid'], self.EQUITY)
        for (i, positions) in enumerate(result.positions.values, start=1):
            assert_equal(len(positions), 1)
            assert_equal(positions[0]['amount'], 5000.0 * i)
            assert_equal(positions[0]['sid'], self.EQUITY)

    @parameterized.expand([('order_target', 5000), ('order_target_value', 10000), ('order_target_percent', 1)])
    def test_order_equity_targeted(self, order_method, amount):
        if False:
            i = 10
            return i + 15
        algotext = '\nimport zipline.api as api\n\ndef initialize(context):\n    api.set_slippage(api.slippage.FixedSlippage(spread=0.0))\n    api.set_commission(api.commission.PerShare(0))\n\n    context.equity = api.sid(1)\n\n    api.schedule_function(\n        func=do_order,\n        date_rule=api.date_rules.every_day(),\n        time_rule=api.time_rules.market_open(),\n    )\n\ndef do_order(context, data):\n    context.ordered = True\n    api.{order_func}(context.equity, {arg})\n     '.format(order_func=order_method, arg=amount)
        result = self.run_algorithm(script=algotext)
        assert_equal([len(ords) for ords in result.orders], [1, 0, 0, 0])
        order = result.orders.iloc[0][0]
        assert_equal(order['amount'], 5000)
        assert_equal(order['sid'], self.EQUITY)
        for positions in result.positions.values:
            assert_equal(len(positions), 1)
            assert_equal(positions[0]['amount'], 5000.0)
            assert_equal(positions[0]['sid'], self.EQUITY)

    @parameterized.expand([('order', 500), ('order_value', 10000), ('order_percent', 1)])
    def test_order_future_non_targeted(self, order_method, amount):
        if False:
            i = 10
            return i + 15
        algotext = '\nimport zipline.api as api\n\ndef initialize(context):\n    api.set_slippage(us_futures=api.slippage.FixedSlippage(spread=0.0))\n    api.set_commission(us_futures=api.commission.PerTrade(0.0))\n\n    context.future = api.sid(2)\n\n    api.schedule_function(\n        func=do_order,\n        date_rule=api.date_rules.every_day(),\n        time_rule=api.time_rules.market_open(),\n    )\n\ndef do_order(context, data):\n    context.ordered = True\n    api.{order_func}(context.future, {arg})\n     '.format(order_func=order_method, arg=amount)
        result = self.run_algorithm(script=algotext)
        for orders in result.orders.values:
            assert_equal(len(orders), 1)
            assert_equal(orders[0]['amount'], 500)
            assert_equal(orders[0]['sid'], self.FUTURE)
        for (i, positions) in enumerate(result.positions.values, start=1):
            assert_equal(len(positions), 1)
            assert_equal(positions[0]['amount'], 500.0 * i)
            assert_equal(positions[0]['sid'], self.FUTURE)

    @parameterized.expand([('order_target', 500), ('order_target_value', 10000), ('order_target_percent', 1)])
    def test_order_future_targeted(self, order_method, amount):
        if False:
            return 10
        algotext = '\nimport zipline.api as api\n\ndef initialize(context):\n    api.set_slippage(us_futures=api.slippage.FixedSlippage(spread=0.0))\n    api.set_commission(us_futures=api.commission.PerTrade(0.0))\n\n    context.future = api.sid(2)\n\n    api.schedule_function(\n        func=do_order,\n        date_rule=api.date_rules.every_day(),\n        time_rule=api.time_rules.market_open(),\n    )\n\ndef do_order(context, data):\n    context.ordered = True\n    api.{order_func}(context.future, {arg})\n     '.format(order_func=order_method, arg=amount)
        result = self.run_algorithm(script=algotext)
        assert_equal([len(ords) for ords in result.orders], [1, 0, 0, 0])
        order = result.orders.iloc[0][0]
        assert_equal(order['amount'], 500)
        assert_equal(order['sid'], self.FUTURE)
        for positions in result.positions.values:
            assert_equal(len(positions), 1)
            assert_equal(positions[0]['amount'], 500.0)
            assert_equal(positions[0]['sid'], self.FUTURE)

    @parameterized.expand([(api.order, 5000), (api.order_value, 10000), (api.order_percent, 1.0), (api.order_target, 5000), (api.order_target_value, 10000), (api.order_target_percent, 1.0)])
    def test_order_method_style_forwarding(self, order_method, order_param):
        if False:
            while True:
                i = 10

        def initialize(context):
            if False:
                while True:
                    i = 10
            api.schedule_function(func=do_order, date_rule=api.date_rules.every_day(), time_rule=api.time_rules.market_open())

        def do_order(context, data):
            if False:
                i = 10
                return i + 15
            assert len(context.portfolio.positions.keys()) == 0
            order_method(self.EQUITY, order_param, style=StopLimitOrder(10, 10, asset=self.EQUITY))
            assert len(context.blotter.open_orders[self.EQUITY]) == 1
            result = context.blotter.open_orders[self.EQUITY][0]
            assert result.limit == 10
            assert result.stop == 10
        self.run_algorithm(initialize=initialize, sim_params=self.sim_params.create_new(start_session=self.END_DATE, end_session=self.END_DATE))

class TestOrderMethodsDailyFrequency(zf.WithMakeAlgo, zf.ZiplineTestCase):
    START_DATE = T('2006-01-03')
    END_DATE = T('2006-01-06')
    SIM_PARAMS_START_DATE = T('2006-01-04')
    ASSET_FINDER_EQUITY_SIDS = (1,)
    SIM_PARAMS_DATA_FREQUENCY = 'daily'
    DATA_PORTAL_USE_MINUTE_DATA = False

    def test_invalid_order_parameters(self):
        if False:
            while True:
                i = 10
        self.run_algorithm(algo_class=zta.InvalidOrderAlgorithm, sids=[1])

    def test_cant_order_in_initialize(self):
        if False:
            i = 10
            return i + 15
        algotext = '\nfrom zipline.api import (sid, order)\n\ndef initialize(context):\n    order(sid(1), 10)'
        algo = self.make_algo(script=algotext)
        with self.assertRaises(ze.OrderDuringInitialize):
            algo.run()

class TestOrderRounding(zf.ZiplineTestCase):

    def test_order_rounding(self):
        if False:
            print('Hello World!')
        answer_key = [(0, 0), (10, 10), (1.1, 1), (1.5, 1), (1.9998, 1), (1.99991, 2)]
        for (input, answer) in answer_key:
            self.assertEqual(answer, TradingAlgorithm.round_order(input))
            self.assertEqual(-1 * answer, TradingAlgorithm.round_order(-1 * input))