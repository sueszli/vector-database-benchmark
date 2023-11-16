from ..alpha_vantage.alphavantage import AlphaVantage
from ..alpha_vantage.timeseries import TimeSeries
from ..alpha_vantage.techindicators import TechIndicators
from ..alpha_vantage.sectorperformance import SectorPerformances
from ..alpha_vantage.foreignexchange import ForeignExchange
from ..alpha_vantage.fundamentaldata import FundamentalData
from pandas import DataFrame as df, Timestamp
import unittest
import sys
import collections
from os import path
import requests_mock

class TestAlphaVantage(unittest.TestCase):
    _API_KEY_TEST = 'test'
    _API_EQ_NAME_TEST = 'MSFT'

    @staticmethod
    def get_file_from_url(url):
        if False:
            print('Hello World!')
        '\n            Return the file name used for testing, found in the test data folder\n            formed using the original url\n        '
        tmp = url
        for ch in [':', '/', '.', '?', '=', '&', ',']:
            if ch in tmp:
                tmp = tmp.replace(ch, '_')
        path_dir = path.join(path.dirname(path.abspath(__file__)), 'test_data/')
        return path.join(path.join(path_dir, tmp))

    def test_key_none(self):
        if False:
            for i in range(10):
                print('nop')
        'Raise an error when a key has not been given\n        '
        try:
            AlphaVantage()
            self.fail(msg='A None api key must raise an error')
        except ValueError:
            self.assertTrue(True)

    @requests_mock.Mocker()
    def test_handle_api_call(self, mock_request):
        if False:
            i = 10
            return i + 15
        ' Test that api call returns a json file as requested\n        '
        av = AlphaVantage(key=TestAlphaVantage._API_KEY_TEST)
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=MSFT&interval=1min&apikey=test'
        path_file = self.get_file_from_url('mock_time_series')
        with open(path_file) as f:
            mock_request.get(url, text=f.read())
            data = av._handle_api_call(url)
            self.assertIsInstance(data, dict, 'Result Data must be a dictionary')

    @requests_mock.Mocker()
    def test_rapidapi_key(self, mock_request):
        if False:
            i = 10
            return i + 15
        ' Test that the rapidAPI key calls the rapidAPI endpoint\n        '
        ts = TimeSeries(key=TestAlphaVantage._API_KEY_TEST, rapidapi=True)
        url = 'https://alpha-vantage.p.rapidapi.com/query?function=TIME_SERIES_INTRADAY&symbol=MSFT&interval=1min&outputsize=full&datatype=json'
        path_file = self.get_file_from_url('mock_time_series')
        with open(path_file) as f:
            mock_request.get(url, text=f.read())
            (data, _) = ts.get_intraday('MSFT', interval='1min', outputsize='full')
            self.assertIsInstance(data, dict, 'Result Data must be a dictionary')

    @requests_mock.Mocker()
    def test_time_series_intraday(self, mock_request):
        if False:
            print('Hello World!')
        ' Test that api call returns a json file as requested\n        '
        ts = TimeSeries(key=TestAlphaVantage._API_KEY_TEST)
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=MSFT&interval=1min&outputsize=full&apikey=test&datatype=json'
        path_file = self.get_file_from_url('mock_time_series')
        with open(path_file) as f:
            mock_request.get(url, text=f.read())
            (data, _) = ts.get_intraday('MSFT', interval='1min', outputsize='full')
            self.assertIsInstance(data, dict, 'Result Data must be a dictionary')

    @requests_mock.Mocker()
    def test_time_series_intraday_pandas(self, mock_request):
        if False:
            for i in range(10):
                print('nop')
        ' Test that api call returns a json file as requested\n        '
        ts = TimeSeries(key=TestAlphaVantage._API_KEY_TEST, output_format='pandas')
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=MSFT&interval=1min&outputsize=full&apikey=test&datatype=json'
        path_file = self.get_file_from_url('mock_time_series')
        with open(path_file) as f:
            mock_request.get(url, text=f.read())
            (data, _) = ts.get_intraday('MSFT', interval='1min', outputsize='full')
            self.assertIsInstance(data, df, 'Result Data must be a pandas data frame')

    @requests_mock.Mocker()
    def test_time_series_intraday_date_indexing(self, mock_request):
        if False:
            while True:
                i = 10
        ' Test that api call returns a pandas data frame with a date as index\n        '
        ts = TimeSeries(key=TestAlphaVantage._API_KEY_TEST, output_format='pandas', indexing_type='date')
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=MSFT&interval=1min&outputsize=full&apikey=test&datatype=json'
        path_file = self.get_file_from_url('mock_time_series')
        with open(path_file) as f:
            mock_request.get(url, text=f.read())
            (data, _) = ts.get_intraday('MSFT', interval='1min', outputsize='full')
            if ts.indexing_type == 'date':
                assert isinstance(data.index[0], Timestamp)
            elif sys.version_info[0] == 3:
                assert isinstance(data.index[0], str)
            else:
                assert isinstance(data.index[0], basestring)

    @requests_mock.Mocker()
    def test_time_series_intraday_date_integer(self, mock_request):
        if False:
            return 10
        ' Test that api call returns a pandas data frame with an integer as index\n        '
        ts = TimeSeries(key=TestAlphaVantage._API_KEY_TEST, output_format='pandas', indexing_type='integer')
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=MSFT&interval=1min&outputsize=full&apikey=test&datatype=json'
        path_file = self.get_file_from_url('mock_time_series')
        with open(path_file) as f:
            mock_request.get(url, text=f.read())
            (data, _) = ts.get_intraday('MSFT', interval='1min', outputsize='full')
            assert type(data.index[0]) == int

    @requests_mock.Mocker()
    def test_time_series_intraday_extended(self, mock_request):
        if False:
            for i in range(10):
                print('nop')
        ' Test that api call returns a csv-reader as requested\n        '
        ts = TimeSeries(key=TestAlphaVantage._API_KEY_TEST, output_format='csv')
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=MSFT&interval=1min&slice=year1month1&adjusted=True&apikey=test&datatype=csv'
        path_file = self.get_file_from_url('mock_time_series_extended')
        with open(path_file) as f:
            mock_request.get(url, text=f.read())
            (data, _) = ts.get_intraday_extended('MSFT', interval='1min')
            self.assertIsInstance(data, collections.Iterator, 'Result Data must implement Iterator-interface')

    @requests_mock.Mocker()
    def test_technical_indicator_sma_python3(self, mock_request):
        if False:
            i = 10
            return i + 15
        ' Test that api call returns a json file as requested\n        '
        ti = TechIndicators(key=TestAlphaVantage._API_KEY_TEST)
        url = 'https://www.alphavantage.co/query?function=SMA&symbol=MSFT&interval=15min&time_period=10&series_type=close&apikey=test'
        path_file = self.get_file_from_url('mock_technical_indicator')
        with open(path_file) as f:
            mock_request.get(url, text=f.read())
            (data, _) = ti.get_sma('MSFT', interval='15min', time_period=10, series_type='close')
            self.assertIsInstance(data, dict, 'Result Data must be a dictionary')

    @requests_mock.Mocker()
    def test_technical_indicator_sma_pandas(self, mock_request):
        if False:
            print('Hello World!')
        ' Test that api call returns a json file as requested\n        '
        ti = TechIndicators(key=TestAlphaVantage._API_KEY_TEST, output_format='pandas')
        url = 'https://www.alphavantage.co/query?function=SMA&symbol=MSFT&interval=15min&time_period=10&series_type=close&apikey=test'
        path_file = self.get_file_from_url('mock_technical_indicator')
        with open(path_file) as f:
            mock_request.get(url, text=f.read())
            (data, _) = ti.get_sma('MSFT', interval='15min', time_period=10, series_type='close')
            self.assertIsInstance(data, df, 'Result Data must be a pandas data frame')

    @requests_mock.Mocker()
    def test_sector_perfomance_python3(self, mock_request):
        if False:
            for i in range(10):
                print('nop')
        ' Test that api call returns a json file as requested\n        '
        sp = SectorPerformances(key=TestAlphaVantage._API_KEY_TEST)
        url = 'https://www.alphavantage.co/query?function=SECTOR&apikey=test'
        path_file = self.get_file_from_url('mock_sector')
        with open(path_file) as f:
            mock_request.get(url, text=f.read())
            (data, _) = sp.get_sector()
            self.assertIsInstance(data, dict, 'Result Data must be a dictionary')

    @requests_mock.Mocker()
    def test_sector_perfomance_pandas(self, mock_request):
        if False:
            i = 10
            return i + 15
        ' Test that api call returns a json file as requested\n        '
        sp = SectorPerformances(key=TestAlphaVantage._API_KEY_TEST, output_format='pandas')
        url = 'https://www.alphavantage.co/query?function=SECTOR&apikey=test'
        path_file = self.get_file_from_url('mock_sector')
        with open(path_file) as f:
            mock_request.get(url, text=f.read())
            (data, _) = sp.get_sector()
            self.assertIsInstance(data, df, 'Result Data must be a pandas data frame')

    @requests_mock.Mocker()
    def test_foreign_exchange(self, mock_request):
        if False:
            return 10
        ' Test that api call returns a json file as requested\n        '
        fe = ForeignExchange(key=TestAlphaVantage._API_KEY_TEST)
        url = 'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency=BTC&to_currency=CNY&apikey=test'
        path_file = self.get_file_from_url('mock_foreign_exchange')
        with open(path_file) as f:
            mock_request.get(url, text=f.read())
            (data, _) = fe.get_currency_exchange_rate(from_currency='BTC', to_currency='CNY')
            self.assertIsInstance(data, dict, 'Result Data must be a dictionary')

    @requests_mock.Mocker()
    def test_fundamental_data(self, mock_request):
        if False:
            print('Hello World!')
        'Test that api call returns a json file as requested\n        '
        fd = FundamentalData(key=TestAlphaVantage._API_KEY_TEST)
        url = 'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol=IBM&apikey=test'
        path_file = self.get_file_from_url('mock_fundamental_data')
        with open(path_file) as f:
            mock_request.get(url, text=f.read())
            (data, _) = fd.get_income_statement_annual(symbol='IBM')
            self.assertIsInstance(data, df, 'Result Data must be a pandas data frame')

    @requests_mock.Mocker()
    def test_company_overview(self, mock_request):
        if False:
            for i in range(10):
                print('nop')
        'Test that api call returns a json file as requested\n        '
        fd = FundamentalData(key=TestAlphaVantage._API_KEY_TEST)
        url = 'https://www.alphavantage.co/query?function=OVERVIEW&symbol=IBM&apikey=test'
        path_file = self.get_file_from_url('mock_company_overview')
        with open(path_file) as f:
            mock_request.get(url, text=f.read())
            (data, _) = fd.get_company_overview(symbol='IBM')
            self.assertIsInstance(data, dict, 'Result Data must be a dictionary')