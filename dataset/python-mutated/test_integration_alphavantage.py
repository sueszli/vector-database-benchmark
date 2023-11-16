from ..alpha_vantage.alphavantage import AlphaVantage
from ..alpha_vantage.timeseries import TimeSeries
from ..alpha_vantage.techindicators import TechIndicators
from ..alpha_vantage.cryptocurrencies import CryptoCurrencies
from ..alpha_vantage.foreignexchange import ForeignExchange
from pandas import DataFrame as df
import os
import unittest
import timeit
import time

class TestAlphaVantage(unittest.TestCase):
    """
        Test data request different implementations
    """
    _API_KEY_TEST = os.environ['API_KEY']
    _API_EQ_NAME_TEST = 'MSFT'
    _RAPIDAPI_KEY_TEST = os.getenv('RAPIDAPI_KEY')

    def setUp(self):
        if False:
            print('Hello World!')
        '\n        Wait some time before running each call again.\n        '
        time.sleep(1)

    def _assert_result_is_format(self, func, output_format='json', **args):
        if False:
            print('Hello World!')
        'Check that the data and meta data object are dictionaries\n\n        Keyword arguments\n        func -- the function to assert its format\n        output_format -- the format of the call\n        **args -- The parameters for the call\n        '
        stime = timeit.default_timer()
        (data, meta_data) = func(**args)
        elapsed = timeit.default_timer() - stime
        print('Function: {} - Format: {} - Took: {}'.format(func.__name__, output_format, elapsed))
        if output_format == 'json':
            self.assertIsInstance(data, dict, 'Result Data must be a dictionary')
            if meta_data is not None:
                self.assertIsInstance(meta_data, dict, 'Result Meta Data must be a                 dictionary')
        elif output_format == 'pandas':
            self.assertIsInstance(data, df, 'Result Data must be a pandas data frame')
            if meta_data is not None:
                self.assertIsInstance(meta_data, dict, 'Result Meta Data must be a                 dictionary')

    def test_key_none(self):
        if False:
            print('Hello World!')
        'Raise an error when a key has not been given\n        '
        try:
            AlphaVantage()
            self.fail(msg='A None api key must raise an error')
        except ValueError:
            self.assertTrue(True)

    def test_rapidapi_key_with_get_daily(self):
        if False:
            while True:
                i = 10
        'RapidAPI calls must return the same data as non-rapidapi calls\n        '
        ts_rapidapi = TimeSeries(key=TestAlphaVantage._RAPIDAPI_KEY_TEST, rapidapi=True)
        ts = TimeSeries(key=TestAlphaVantage._API_KEY_TEST)
        (rapidapi_data, _) = ts_rapidapi.get_daily(symbol=TestAlphaVantage._API_EQ_NAME_TEST)
        (data, _) = ts.get_daily(symbol=TestAlphaVantage._API_EQ_NAME_TEST)
        self.assertTrue(rapidapi_data == data)

    def test_get_daily_is_format(self):
        if False:
            i = 10
            return i + 15
        'Result must be a dictionary containing the json data\n        '
        ts = TimeSeries(key=TestAlphaVantage._API_KEY_TEST)
        self._assert_result_is_format(ts.get_daily, symbol=TestAlphaVantage._API_EQ_NAME_TEST)
        ts = TimeSeries(key=TestAlphaVantage._API_KEY_TEST, output_format='pandas')
        self._assert_result_is_format(ts.get_daily, output_format='pandas', symbol=TestAlphaVantage._API_EQ_NAME_TEST)

    def test_get_daily_adjusted_is_format(self):
        if False:
            while True:
                i = 10
        'Result must be a dictionary containing the json data\n        '
        ts = TimeSeries(key=TestAlphaVantage._API_KEY_TEST)
        self._assert_result_is_format(ts.get_daily_adjusted, symbol=TestAlphaVantage._API_EQ_NAME_TEST)
        ts = TimeSeries(key=TestAlphaVantage._API_KEY_TEST, output_format='pandas')
        self._assert_result_is_format(ts.get_daily_adjusted, output_format='pandas', symbol=TestAlphaVantage._API_EQ_NAME_TEST)

    def test_get_sma_is_format(self):
        if False:
            i = 10
            return i + 15
        'Result must be a dictionary containing the json data\n        '
        ti = TechIndicators(key=TestAlphaVantage._API_KEY_TEST)
        self._assert_result_is_format(ti.get_sma, symbol=TestAlphaVantage._API_EQ_NAME_TEST)
        ti = TechIndicators(key=TestAlphaVantage._API_KEY_TEST, output_format='pandas')
        self._assert_result_is_format(ti.get_sma, output_format='pandas', symbol=TestAlphaVantage._API_EQ_NAME_TEST)

    def test_get_ema_is_format(self):
        if False:
            for i in range(10):
                print('nop')
        'Result must be a dictionary containing the json data\n        '
        ti = TechIndicators(key=TestAlphaVantage._API_KEY_TEST)
        self._assert_result_is_format(ti.get_ema, symbol=TestAlphaVantage._API_EQ_NAME_TEST)
        ti = TechIndicators(key=TestAlphaVantage._API_KEY_TEST, output_format='pandas')
        self._assert_result_is_format(ti.get_ema, output_format='pandas', symbol=TestAlphaVantage._API_EQ_NAME_TEST)

    def test_get_currency_exchange_rate(self):
        if False:
            return 10
        'Test that we get a dictionary containing json data\n        '
        cc = ForeignExchange(key=TestAlphaVantage._API_KEY_TEST)
        self._assert_result_is_format(cc.get_currency_exchange_rate, output_format='json', from_currency='USD', to_currency='BTC')

    def test_get_currency_exchange_intraday_json(self):
        if False:
            i = 10
            return i + 15
        'Test that we get a dictionary containing json data\n        '
        fe = ForeignExchange(key=TestAlphaVantage._API_KEY_TEST)
        self._assert_result_is_format(fe.get_currency_exchange_intraday, output_format='json', from_symbol='EUR', to_symbol='USD', interval='1min')

    def test_get_currency_exchange_intraday_pandas(self):
        if False:
            print('Hello World!')
        'Test that we get a dictionary containing pandas data\n        '
        fe = ForeignExchange(key=TestAlphaVantage._API_KEY_TEST, output_format='pandas')
        self._assert_result_is_format(fe.get_currency_exchange_intraday, output_format='pandas', from_symbol='USD', to_symbol='JPY', interval='5min')

    def test_get_currency_exchange_daily_json(self):
        if False:
            print('Hello World!')
        'Test that we get a dictionary containing json data\n        '
        fe = ForeignExchange(key=TestAlphaVantage._API_KEY_TEST)
        self._assert_result_is_format(fe.get_currency_exchange_daily, output_format='json', from_symbol='EUR', to_symbol='USD')

    def test_get_currency_exchange_daily_pandas(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that we get a dictionary containing pandas data\n        '
        fe = ForeignExchange(key=TestAlphaVantage._API_KEY_TEST, output_format='pandas')
        self._assert_result_is_format(fe.get_currency_exchange_daily, output_format='pandas', from_symbol='USD', to_symbol='JPY')

    def test_get_currency_exchange_weekly_json(self):
        if False:
            while True:
                i = 10
        'Test that we get a dictionary containing json data\n        '
        fe = ForeignExchange(key=TestAlphaVantage._API_KEY_TEST)
        self._assert_result_is_format(fe.get_currency_exchange_weekly, output_format='json', from_symbol='EUR', to_symbol='USD', outputsize='full')

    def test_get_currency_exchange_weekly_pandas(self):
        if False:
            while True:
                i = 10
        'Test that we get a dictionary containing pandas data\n        '
        fe = ForeignExchange(key=TestAlphaVantage._API_KEY_TEST, output_format='pandas')
        self._assert_result_is_format(fe.get_currency_exchange_weekly, output_format='pandas', from_symbol='USD', to_symbol='JPY')

    def test_get_currency_exchange_monthly_json(self):
        if False:
            i = 10
            return i + 15
        'Test that we get a dictionary containing json data\n        '
        fe = ForeignExchange(key=TestAlphaVantage._API_KEY_TEST)
        self._assert_result_is_format(fe.get_currency_exchange_monthly, output_format='json', from_symbol='EUR', to_symbol='USD')

    def test_get_currency_exchange_monthly_pandas(self):
        if False:
            i = 10
            return i + 15
        'Test that we get a dictionary containing pandas data\n        '
        fe = ForeignExchange(key=TestAlphaVantage._API_KEY_TEST, output_format='pandas')
        self._assert_result_is_format(fe.get_currency_exchange_monthly, output_format='pandas', from_symbol='USD', to_symbol='JPY', outputsize='full')

    def test_get_digital_currency_weekly(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that we get a dictionary containing json data\n        '
        cc = CryptoCurrencies(key=TestAlphaVantage._API_KEY_TEST)
        self._assert_result_is_format(cc.get_digital_currency_weekly, output_format='json', symbol='BTC', market='CNY')
        cc = CryptoCurrencies(key=TestAlphaVantage._API_KEY_TEST, output_format='pandas')
        self._assert_result_is_format(cc.get_digital_currency_weekly, output_format='pandas', symbol='BTC', market='CNY')