from nose_parameterized import parameterized
from six.moves import range
import pandas as pd
from zipline.errors import BadOrderParameters
from zipline.finance.execution import LimitOrder, MarketOrder, StopLimitOrder, StopOrder
from zipline.testing.fixtures import WithLogger, ZiplineTestCase, WithConstantFutureMinuteBarData
from zipline.testing.predicates import assert_equal

class ExecutionStyleTestCase(WithConstantFutureMinuteBarData, WithLogger, ZiplineTestCase):
    """
    Tests for zipline ExecutionStyle classes.
    """

    class ArbitraryObject:

        def __str__(self):
            if False:
                for i in range(10):
                    print('nop')
            return 'This should yield a bad order error when\n            passed as a stop or limit price.'
    epsilon = 1e-06
    INVALID_PRICES = [(-1,), (-1.0,), (0 - epsilon,), (float('nan'),), (float('inf'),), (ArbitraryObject(),)]
    EXPECTED_PRICE_ROUNDING = [(0.0, 0.0, 0.0), (0.0005, 0.0, 0.0), (1.0005, 1.0, 1.0), (1.0005 + epsilon, 1.0, 1.01), (1.0095 - epsilon, 1.0, 1.01), (1.0095, 1.01, 1.01), (0.01, 0.01, 0.01)]
    smaller_epsilon = 1e-08
    EXPECTED_PRECISION_ROUNDING = [(0.0, 0.0, 0.0), (0.0005, 0.0005, 0.0005), (5e-05, 0.0, 0.0001), (5e-06, 0.0, 0.0), (1.000005, 1.0, 1.0), (1.000005 + smaller_epsilon, 1.0, 1.0001), (1.000095 - smaller_epsilon, 1.0, 1.0001), (1.000095, 1.0001, 1.0001), (0.01, 0.01, 0.01)]
    EXPECTED_CUSTOM_TICK_SIZE_ROUNDING = [(0.0, 0.0, 0.0), (0.0005, 0.0, 0.0), (1.0025, 1.0, 1.0), (1.0025 + epsilon, 1.0, 1.05), (1.0475 - epsilon, 1.0, 1.05), (1.0475, 1.05, 1.05), (0.05, 0.05, 0.05)]
    EXPECTED_PRICE_ROUNDING += [(x + delta, y + delta, z + delta) for (x, y, z) in EXPECTED_PRICE_ROUNDING for delta in range(1, 10)]
    EXPECTED_PRECISION_ROUNDING += [(x + delta, y + delta, z + delta) for (x, y, z) in EXPECTED_PRECISION_ROUNDING for delta in range(1, 10)]
    EXPECTED_CUSTOM_TICK_SIZE_ROUNDING += [(x + delta, y + delta, z + delta) for (x, y, z) in EXPECTED_CUSTOM_TICK_SIZE_ROUNDING for delta in range(1, 10)]
    FINAL_PARAMETER_SET = [(x, y, z, 1) for (x, y, z) in EXPECTED_PRICE_ROUNDING] + [(x, y, z, 2) for (x, y, z) in EXPECTED_PRECISION_ROUNDING] + [(x, y, z, 3) for (x, y, z) in EXPECTED_CUSTOM_TICK_SIZE_ROUNDING]

    @classmethod
    def make_futures_info(cls):
        if False:
            for i in range(10):
                print('nop')
        return pd.DataFrame.from_dict({1: {'multiplier': 100, 'tick_size': 0.01, 'symbol': 'F1', 'exchange': 'TEST'}, 2: {'multiplier': 100, 'tick_size': 0.0001, 'symbol': 'F2', 'exchange': 'TEST'}, 3: {'multiplier': 100, 'tick_size': 0.05, 'symbol': 'F3', 'exchange': 'TEST'}}, orient='index')

    @classmethod
    def init_class_fixtures(cls):
        if False:
            for i in range(10):
                print('nop')
        super(ExecutionStyleTestCase, cls).init_class_fixtures()

    @parameterized.expand(INVALID_PRICES)
    def test_invalid_prices(self, price):
        if False:
            i = 10
            return i + 15
        '\n        Test that execution styles throw appropriate exceptions upon receipt\n        of an invalid price field.\n        '
        with self.assertRaises(BadOrderParameters):
            LimitOrder(price)
        with self.assertRaises(BadOrderParameters):
            StopOrder(price)
        for (lmt, stp) in [(price, 1), (1, price), (price, price)]:
            with self.assertRaises(BadOrderParameters):
                StopLimitOrder(lmt, stp)

    def test_market_order_prices(self):
        if False:
            while True:
                i = 10
        '\n        Basic unit tests for the MarketOrder class.\n        '
        style = MarketOrder()
        assert_equal(style.get_limit_price(_is_buy=True), None)
        assert_equal(style.get_limit_price(_is_buy=False), None)
        assert_equal(style.get_stop_price(_is_buy=True), None)
        assert_equal(style.get_stop_price(_is_buy=False), None)

    @parameterized.expand(FINAL_PARAMETER_SET)
    def test_limit_order_prices(self, price, expected_limit_buy_or_stop_sell, expected_limit_sell_or_stop_buy, asset):
        if False:
            while True:
                i = 10
        '\n        Test price getters for the LimitOrder class.\n        '
        style = LimitOrder(price, asset=self.asset_finder.retrieve_asset(asset))
        assert_equal(expected_limit_buy_or_stop_sell, style.get_limit_price(is_buy=True))
        assert_equal(expected_limit_sell_or_stop_buy, style.get_limit_price(is_buy=False))
        assert_equal(None, style.get_stop_price(_is_buy=True))
        assert_equal(None, style.get_stop_price(_is_buy=False))

    @parameterized.expand(FINAL_PARAMETER_SET)
    def test_stop_order_prices(self, price, expected_limit_buy_or_stop_sell, expected_limit_sell_or_stop_buy, asset):
        if False:
            return 10
        '\n        Test price getters for StopOrder class. Note that the expected rounding\n        direction for stop prices is the reverse of that for limit prices.\n        '
        style = StopOrder(price, asset=self.asset_finder.retrieve_asset(asset))
        assert_equal(None, style.get_limit_price(_is_buy=False))
        assert_equal(None, style.get_limit_price(_is_buy=True))
        assert_equal(expected_limit_buy_or_stop_sell, style.get_stop_price(is_buy=False))
        assert_equal(expected_limit_sell_or_stop_buy, style.get_stop_price(is_buy=True))

    @parameterized.expand(FINAL_PARAMETER_SET)
    def test_stop_limit_order_prices(self, price, expected_limit_buy_or_stop_sell, expected_limit_sell_or_stop_buy, asset):
        if False:
            print('Hello World!')
        '\n        Test price getters for StopLimitOrder class. Note that the expected\n        rounding direction for stop prices is the reverse of that for limit\n        prices.\n        '
        style = StopLimitOrder(price, price + 1, asset=self.asset_finder.retrieve_asset(asset))
        assert_equal(expected_limit_buy_or_stop_sell, style.get_limit_price(is_buy=True))
        assert_equal(expected_limit_sell_or_stop_buy, style.get_limit_price(is_buy=False))
        assert_equal(expected_limit_buy_or_stop_sell + 1, style.get_stop_price(is_buy=False))
        assert_equal(expected_limit_sell_or_stop_buy + 1, style.get_stop_price(is_buy=True))