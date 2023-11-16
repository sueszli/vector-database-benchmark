from textwrap import dedent
from nose_parameterized import parameterized
from pandas import DataFrame
from zipline.assets import Equity, Future
from zipline.errors import IncompatibleCommissionModel
from zipline.finance.commission import CommissionModel, EquityCommissionModel, FutureCommissionModel, PerContract, PerDollar, PerFutureTrade, PerShare, PerTrade
from zipline.finance.order import Order
from zipline.finance.transaction import Transaction
from zipline.testing import ZiplineTestCase
from zipline.testing.fixtures import WithAssetFinder, WithMakeAlgo

class CommissionUnitTests(WithAssetFinder, ZiplineTestCase):
    ASSET_FINDER_EQUITY_SIDS = (1, 2)

    @classmethod
    def make_futures_info(cls):
        if False:
            for i in range(10):
                print('nop')
        return DataFrame({'sid': [1000, 1001], 'root_symbol': ['CL', 'FV'], 'symbol': ['CLF07', 'FVF07'], 'start_date': [cls.START_DATE, cls.START_DATE], 'end_date': [cls.END_DATE, cls.END_DATE], 'notice_date': [cls.END_DATE, cls.END_DATE], 'expiration_date': [cls.END_DATE, cls.END_DATE], 'multiplier': [500, 500], 'exchange': ['CMES', 'CMES']})

    def generate_order_and_txns(self, sid, order_amount, fill_amounts):
        if False:
            while True:
                i = 10
        asset1 = self.asset_finder.retrieve_asset(sid)
        order = Order(dt=None, asset=asset1, amount=order_amount)
        txn1 = Transaction(asset=asset1, amount=fill_amounts[0], dt=None, price=100, order_id=order.id)
        txn2 = Transaction(asset=asset1, amount=fill_amounts[1], dt=None, price=101, order_id=order.id)
        txn3 = Transaction(asset=asset1, amount=fill_amounts[2], dt=None, price=102, order_id=order.id)
        return (order, [txn1, txn2, txn3])

    def verify_per_trade_commissions(self, model, expected_commission, sid, order_amount=None, fill_amounts=None):
        if False:
            return 10
        fill_amounts = fill_amounts or [230, 170, 100]
        order_amount = order_amount or sum(fill_amounts)
        (order, txns) = self.generate_order_and_txns(sid, order_amount, fill_amounts)
        self.assertEqual(expected_commission, model.calculate(order, txns[0]))
        order.commission = expected_commission
        self.assertEqual(0, model.calculate(order, txns[1]))
        self.assertEqual(0, model.calculate(order, txns[2]))

    def test_allowed_asset_types(self):
        if False:
            print('Hello World!')

        class MyEquitiesModel(EquityCommissionModel):

            def calculate(self, order, transaction):
                if False:
                    print('Hello World!')
                return 0
        self.assertEqual(MyEquitiesModel.allowed_asset_types, (Equity,))

        class MyFuturesModel(FutureCommissionModel):

            def calculate(self, order, transaction):
                if False:
                    while True:
                        i = 10
                return 0
        self.assertEqual(MyFuturesModel.allowed_asset_types, (Future,))

        class MyMixedModel(EquityCommissionModel, FutureCommissionModel):

            def calculate(self, order, transaction):
                if False:
                    for i in range(10):
                        print('nop')
                return 0
        self.assertEqual(MyMixedModel.allowed_asset_types, (Equity, Future))

        class MyMixedModel(CommissionModel):

            def calculate(self, order, transaction):
                if False:
                    print('Hello World!')
                return 0
        self.assertEqual(MyMixedModel.allowed_asset_types, (Equity, Future))
        SomeType = type('SomeType', (object,), {})

        class MyCustomModel(EquityCommissionModel, FutureCommissionModel):
            allowed_asset_types = (SomeType,)

            def calculate(self, order, transaction):
                if False:
                    i = 10
                    return i + 15
                return 0
        self.assertEqual(MyCustomModel.allowed_asset_types, (SomeType,))

    def test_per_trade(self):
        if False:
            i = 10
            return i + 15
        model = PerTrade(cost=10)
        self.verify_per_trade_commissions(model, expected_commission=10, sid=1)
        model = PerFutureTrade(cost=10)
        self.verify_per_trade_commissions(model, expected_commission=10, sid=1000)
        model = PerFutureTrade(cost={'CL': 5, 'FV': 10})
        self.verify_per_trade_commissions(model, expected_commission=5, sid=1000)
        self.verify_per_trade_commissions(model, expected_commission=10, sid=1001)

    def test_per_share_no_minimum(self):
        if False:
            print('Hello World!')
        model = PerShare(cost=0.0075, min_trade_cost=None)
        fill_amounts = [230, 170, 100]
        (order, txns) = self.generate_order_and_txns(sid=1, order_amount=500, fill_amounts=fill_amounts)
        expected_commissions = [1.725, 1.275, 0.75]
        for (fill_amount, expected_commission, txn) in zip(fill_amounts, expected_commissions, txns):
            commission = model.calculate(order, txn)
            self.assertAlmostEqual(expected_commission, commission)
            order.filled += fill_amount
            order.commission += commission

    def test_per_share_shrinking_position(self):
        if False:
            for i in range(10):
                print('nop')
        model = PerShare(cost=0.0075, min_trade_cost=None)
        fill_amounts = [-230, -170, -100]
        (order, txns) = self.generate_order_and_txns(sid=1, order_amount=-500, fill_amounts=fill_amounts)
        expected_commissions = [1.725, 1.275, 0.75]
        for (fill_amount, expected_commission, txn) in zip(fill_amounts, expected_commissions, txns):
            commission = model.calculate(order, txn)
            self.assertAlmostEqual(expected_commission, commission)
            order.filled += fill_amount
            order.commission += commission

    def verify_per_unit_commissions(self, model, commission_totals, sid, order_amount=None, fill_amounts=None):
        if False:
            print('Hello World!')
        fill_amounts = fill_amounts or [230, 170, 100]
        order_amount = order_amount or sum(fill_amounts)
        (order, txns) = self.generate_order_and_txns(sid, order_amount, fill_amounts)
        for (i, commission_total) in enumerate(commission_totals):
            order.commission += model.calculate(order, txns[i])
            self.assertAlmostEqual(commission_total, order.commission)
            order.filled += txns[i].amount

    def test_per_contract_no_minimum(self):
        if False:
            return 10
        model = PerContract(cost=0.01, exchange_fee=0.3, min_trade_cost=None)
        self.verify_per_unit_commissions(model=model, commission_totals=[2.6, 4.3, 5.3], sid=1000, order_amount=500, fill_amounts=[230, 170, 100])
        model = PerContract(cost={'CL': 0.01, 'FV': 0.0075}, exchange_fee={'CL': 0.3, 'FV': 0.5}, min_trade_cost=None)
        self.verify_per_unit_commissions(model, [2.6, 4.3, 5.3], sid=1000)
        self.verify_per_unit_commissions(model, [2.225, 3.5, 4.25], sid=1001)

    def test_per_share_with_minimum(self):
        if False:
            for i in range(10):
                print('nop')
        self.verify_per_unit_commissions(PerShare(cost=0.0075, min_trade_cost=1), commission_totals=[1.725, 3, 3.75], sid=1)
        self.verify_per_unit_commissions(PerShare(cost=0.0075, min_trade_cost=2.5), commission_totals=[2.5, 3, 3.75], sid=1)
        self.verify_per_unit_commissions(PerShare(cost=0.0075, min_trade_cost=3.5), commission_totals=[3.5, 3.5, 3.75], sid=1)
        self.verify_per_unit_commissions(PerShare(cost=0.0075, min_trade_cost=5.5), commission_totals=[5.5, 5.5, 5.5], sid=1)

    def test_per_contract_with_minimum(self):
        if False:
            return 10
        self.verify_per_unit_commissions(PerContract(cost=0.01, exchange_fee=0.3, min_trade_cost=1), commission_totals=[2.6, 4.3, 5.3], sid=1000)
        self.verify_per_unit_commissions(PerContract(cost=0.01, exchange_fee=0.3, min_trade_cost=3), commission_totals=[3.0, 4.3, 5.3], sid=1000)
        self.verify_per_unit_commissions(PerContract(cost=0.01, exchange_fee=0.3, min_trade_cost=5), commission_totals=[5.0, 5.0, 5.3], sid=1000)
        self.verify_per_unit_commissions(PerContract(cost=0.01, exchange_fee=0.3, min_trade_cost=7), commission_totals=[7.0, 7.0, 7.0], sid=1000)

    def test_per_dollar(self):
        if False:
            i = 10
            return i + 15
        model = PerDollar(cost=0.0015)
        (order, txns) = self.generate_order_and_txns(sid=1, order_amount=500, fill_amounts=[230, 170, 100])
        self.assertAlmostEqual(34.5, model.calculate(order, txns[0]))
        self.assertAlmostEqual(25.755, model.calculate(order, txns[1]))
        self.assertAlmostEqual(15.3, model.calculate(order, txns[2]))

class CommissionAlgorithmTests(WithMakeAlgo, ZiplineTestCase):
    SIM_PARAMS_DATA_FREQUENCY = 'daily'
    DATA_PORTAL_USE_MINUTE_DATA = True
    (sidint,) = ASSET_FINDER_EQUITY_SIDS = (133,)
    code = dedent('\n        from zipline.api import (\n            sid, order, set_slippage, slippage, FixedSlippage,\n            set_commission, commission\n        )\n\n        def initialize(context):\n            # for these tests, let us take out the entire bar with no price\n            # impact\n            set_slippage(\n                us_equities=slippage.VolumeShareSlippage(1.0, 0),\n                us_futures=slippage.VolumeShareSlippage(1.0, 0),\n            )\n\n            {commission}\n            context.ordered = False\n\n\n        def handle_data(context, data):\n            if not context.ordered:\n                order(sid({sid}), {amount})\n                context.ordered = True\n        ')

    @classmethod
    def make_futures_info(cls):
        if False:
            while True:
                i = 10
        return DataFrame({'sid': [1000, 1001], 'root_symbol': ['CL', 'FV'], 'symbol': ['CLF07', 'FVF07'], 'start_date': [cls.START_DATE, cls.START_DATE], 'end_date': [cls.END_DATE, cls.END_DATE], 'notice_date': [cls.END_DATE, cls.END_DATE], 'expiration_date': [cls.END_DATE, cls.END_DATE], 'multiplier': [500, 500], 'exchange': ['CMES', 'CMES']})

    @classmethod
    def make_equity_daily_bar_data(cls, country_code, sids):
        if False:
            for i in range(10):
                print('nop')
        sessions = cls.trading_calendar.sessions_in_range(cls.START_DATE, cls.END_DATE)
        for sid in sids:
            yield (sid, DataFrame(index=sessions, data={'open': 10.0, 'high': 10.0, 'low': 10.0, 'close': 10.0, 'volume': 100.0}))

    def get_results(self, algo_code):
        if False:
            return 10
        return self.run_algorithm(script=algo_code)

    def test_per_trade(self):
        if False:
            i = 10
            return i + 15
        results = self.get_results(self.code.format(commission='set_commission(commission.PerTrade(1))', sid=133, amount=300))
        for orders in results.orders[1:4]:
            self.assertEqual(1, orders[0]['commission'])
        self.verify_capital_used(results, [-1001, -1000, -1000])

    def test_futures_per_trade(self):
        if False:
            for i in range(10):
                print('nop')
        results = self.get_results(self.code.format(commission='set_commission(us_futures=commission.PerFutureTrade(1))', sid=1000, amount=10))
        self.assertEqual(results.orders[1][0]['commission'], 1.0)
        self.assertEqual(results.capital_used[1], -1.0)

    def test_per_share_no_minimum(self):
        if False:
            return 10
        results = self.get_results(self.code.format(commission='set_commission(commission.PerShare(0.05, None))', sid=133, amount=300))
        for (i, orders) in enumerate(results.orders[1:4]):
            self.assertEqual((i + 1) * 5, orders[0]['commission'])
        self.verify_capital_used(results, [-1005, -1005, -1005])

    def test_per_share_with_minimum(self):
        if False:
            return 10
        results = self.get_results(self.code.format(commission='set_commission(commission.PerShare(0.05, 3))', sid=133, amount=300))
        for (i, orders) in enumerate(results.orders[1:4]):
            self.assertEqual((i + 1) * 5, orders[0]['commission'])
        self.verify_capital_used(results, [-1005, -1005, -1005])
        results = self.get_results(self.code.format(commission='set_commission(commission.PerShare(0.05, 8))', sid=133, amount=300))
        self.assertEqual(8, results.orders[1][0]['commission'])
        self.assertEqual(10, results.orders[2][0]['commission'])
        self.assertEqual(15, results.orders[3][0]['commission'])
        self.verify_capital_used(results, [-1008, -1002, -1005])
        results = self.get_results(self.code.format(commission='set_commission(commission.PerShare(0.05, 12))', sid=133, amount=300))
        self.assertEqual(12, results.orders[1][0]['commission'])
        self.assertEqual(12, results.orders[2][0]['commission'])
        self.assertEqual(15, results.orders[3][0]['commission'])
        self.verify_capital_used(results, [-1012, -1000, -1003])
        results = self.get_results(self.code.format(commission='set_commission(commission.PerShare(0.05, 18))', sid=133, amount=300))
        self.assertEqual(18, results.orders[1][0]['commission'])
        self.assertEqual(18, results.orders[2][0]['commission'])
        self.assertEqual(18, results.orders[3][0]['commission'])
        self.verify_capital_used(results, [-1018, -1000, -1000])

    @parameterized.expand([(None, 1.8), (1, 1.8), (3, 3.0)])
    def test_per_contract(self, min_trade_cost, expected_commission):
        if False:
            for i in range(10):
                print('nop')
        results = self.get_results(self.code.format(commission='set_commission(us_futures=commission.PerContract(cost=0.05, exchange_fee=1.3, min_trade_cost={}))'.format(min_trade_cost), sid=1000, amount=10))
        self.assertEqual(results.orders[1][0]['commission'], expected_commission)
        self.assertEqual(results.capital_used[1], -expected_commission)

    def test_per_dollar(self):
        if False:
            while True:
                i = 10
        results = self.get_results(self.code.format(commission='set_commission(commission.PerDollar(0.01))', sid=133, amount=300))
        for (i, orders) in enumerate(results.orders[1:4]):
            self.assertEqual((i + 1) * 10, orders[0]['commission'])
        self.verify_capital_used(results, [-1010, -1010, -1010])

    def test_incorrectly_set_futures_model(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(IncompatibleCommissionModel):
            self.get_results(self.code.format(commission='set_commission(commission.PerContract(0, 0))', sid=1000, amount=10))

    def verify_capital_used(self, results, values):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(values[0], results.capital_used[1])
        self.assertEqual(values[1], results.capital_used[2])
        self.assertEqual(values[2], results.capital_used[3])