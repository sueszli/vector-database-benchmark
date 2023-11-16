from nose_parameterized import parameterized
import pandas as pd
import numpy as np
from mock import patch
from zipline.errors import UnsupportedOrderParameters
from zipline.sources.requests_csv import mask_requests_args
from zipline.utils import factory
from zipline.testing import FetcherDataPortal
from zipline.testing.fixtures import WithResponses, WithMakeAlgo, ZiplineTestCase
from .resources.fetcher_inputs.fetcher_test_data import AAPL_CSV_DATA, AAPL_IBM_CSV_DATA, AAPL_MINUTE_CSV_DATA, CPIAUCSL_DATA, FETCHER_ALTERNATE_COLUMN_HEADER, FETCHER_UNIVERSE_DATA, FETCHER_UNIVERSE_DATA_TICKER_COLUMN, MULTI_SIGNAL_CSV_DATA, NON_ASSET_FETCHER_UNIVERSE_DATA, PALLADIUM_DATA, NFLX_DATA

class FetcherTestCase(WithResponses, WithMakeAlgo, ZiplineTestCase):
    START_DATE = pd.Timestamp('2006-01-03', tz='utc')
    END_DATE = pd.Timestamp('2006-12-29', tz='utc')
    SIM_PARAMS_DATA_FREQUENCY = 'daily'
    DATA_PORTAL_USE_MINUTE_DATA = False
    BENCHMARK_SID = None

    @classmethod
    def make_equity_info(cls):
        if False:
            for i in range(10):
                print('nop')
        return pd.DataFrame.from_dict({24: {'start_date': pd.Timestamp('2006-01-01', tz='UTC'), 'end_date': pd.Timestamp('2007-01-01', tz='UTC'), 'symbol': 'AAPL', 'exchange': 'nasdaq'}, 3766: {'start_date': pd.Timestamp('2006-01-01', tz='UTC'), 'end_date': pd.Timestamp('2007-01-01', tz='UTC'), 'symbol': 'IBM', 'exchange': 'nasdaq'}, 5061: {'start_date': pd.Timestamp('2006-01-01', tz='UTC'), 'end_date': pd.Timestamp('2007-01-01', tz='UTC'), 'symbol': 'MSFT', 'exchange': 'nasdaq'}, 14848: {'start_date': pd.Timestamp('2006-01-01', tz='UTC'), 'end_date': pd.Timestamp('2007-01-01', tz='UTC'), 'symbol': 'YHOO', 'exchange': 'nasdaq'}, 25317: {'start_date': pd.Timestamp('2006-01-01', tz='UTC'), 'end_date': pd.Timestamp('2007-01-01', tz='UTC'), 'symbol': 'DELL', 'exchange': 'nasdaq'}, 13: {'start_date': pd.Timestamp('2006-01-01', tz='UTC'), 'end_date': pd.Timestamp('2010-01-01', tz='UTC'), 'symbol': 'NFLX', 'exchange': 'nasdaq'}, 9999999: {'start_date': pd.Timestamp('2006-01-01', tz='UTC'), 'end_date': pd.Timestamp('2007-01-01', tz='UTC'), 'symbol': 'AAPL', 'exchange': 'non_us_exchange'}}, orient='index')

    @classmethod
    def make_exchanges_info(cls, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return pd.DataFrame.from_records([{'exchange': 'nasdaq', 'country_code': 'US'}, {'exchange': 'non_us_exchange', 'country_code': 'CA'}])

    def run_algo(self, code, sim_params=None):
        if False:
            print('Hello World!')
        if sim_params is None:
            sim_params = self.sim_params
        test_algo = self.make_algo(script=code, sim_params=sim_params, data_portal=FetcherDataPortal(self.asset_finder, self.trading_calendar))
        results = test_algo.run()
        return results

    def test_minutely_fetcher(self):
        if False:
            for i in range(10):
                print('nop')
        self.responses.add(self.responses.GET, 'https://fake.urls.com/aapl_minute_csv_data.csv', body=AAPL_MINUTE_CSV_DATA, content_type='text/csv')
        sim_params = factory.create_simulation_parameters(start=pd.Timestamp('2006-01-03', tz='UTC'), end=pd.Timestamp('2006-01-10', tz='UTC'), emission_rate='minute', data_frequency='minute')
        test_algo = self.make_algo(script='\nfrom zipline.api import fetch_csv, record, sid\n\ndef initialize(context):\n    fetch_csv(\'https://fake.urls.com/aapl_minute_csv_data.csv\')\n\ndef handle_data(context, data):\n    record(aapl_signal=data.current(sid(24), "signal"))\n', sim_params=sim_params)
        gen = test_algo.get_generator()
        perf_packets = list(gen)
        signal = [result['minute_perf']['recorded_vars']['aapl_signal'] for result in perf_packets if 'minute_perf' in result]
        self.assertEqual(6 * 390, len(signal))
        np.testing.assert_array_equal([np.NaN] * 390, signal[0:390])
        np.testing.assert_array_equal([2] * 390, signal[390:780])
        np.testing.assert_array_equal([3] * 780, signal[780:1560])
        np.testing.assert_array_equal([4] * 780, signal[1560:])

    def test_fetch_csv_with_multi_symbols(self):
        if False:
            for i in range(10):
                print('nop')
        self.responses.add(self.responses.GET, 'https://fake.urls.com/multi_signal_csv_data.csv', body=MULTI_SIGNAL_CSV_DATA, content_type='text/csv')
        results = self.run_algo('\nfrom zipline.api import fetch_csv, record, sid\n\ndef initialize(context):\n    fetch_csv(\'https://fake.urls.com/multi_signal_csv_data.csv\')\n    context.stocks = [sid(3766), sid(25317)]\n\ndef handle_data(context, data):\n    record(ibm_signal=data.current(sid(3766), "signal"))\n    record(dell_signal=data.current(sid(25317), "signal"))\n    ')
        self.assertEqual(5, results['ibm_signal'].iloc[-1])
        self.assertEqual(5, results['dell_signal'].iloc[-1])

    def test_fetch_csv_with_pure_signal_file(self):
        if False:
            return 10
        self.responses.add(self.responses.GET, 'https://fake.urls.com/cpiaucsl_data.csv', body=CPIAUCSL_DATA, content_type='text/csv')
        results = self.run_algo('\nfrom zipline.api import fetch_csv, sid, record\n\ndef clean(df):\n    return df.rename(columns={\'Value\':\'cpi\', \'Date\':\'date\'})\n\ndef initialize(context):\n    fetch_csv(\n        \'https://fake.urls.com/cpiaucsl_data.csv\',\n        symbol=\'urban\',\n        pre_func=clean,\n        date_format=\'%Y-%m-%d\'\n        )\n    context.stocks = [sid(3766), sid(25317)]\n\ndef handle_data(context, data):\n\n    cur_cpi = data.current("urban", "cpi")\n    record(cpi=cur_cpi)\n            ')
        self.assertEqual(results['cpi'][-1], 203.1)

    def test_algo_fetch_csv(self):
        if False:
            print('Hello World!')
        self.responses.add(self.responses.GET, 'https://fake.urls.com/aapl_csv_data.csv', body=AAPL_CSV_DATA, content_type='text/csv')
        results = self.run_algo('\nfrom zipline.api import fetch_csv, record, sid\n\ndef normalize(df):\n    df[\'scaled\'] = df[\'signal\'] * 10\n    return df\n\ndef initialize(context):\n    fetch_csv(\'https://fake.urls.com/aapl_csv_data.csv\',\n            post_func=normalize)\n    context.checked_name = False\n\ndef handle_data(context, data):\n    record(\n        signal=data.current(sid(24), "signal"),\n        scaled=data.current(sid(24), "scaled"),\n        price=data.current(sid(24), "price"))\n        ')
        self.assertEqual(5, results['signal'][-1])
        self.assertEqual(50, results['scaled'][-1])
        self.assertEqual(24, results['price'][-1])

    def test_algo_fetch_csv_with_extra_symbols(self):
        if False:
            print('Hello World!')
        self.responses.add(self.responses.GET, 'https://fake.urls.com/aapl_ibm_csv_data.csv', body=AAPL_IBM_CSV_DATA, content_type='text/csv')
        results = self.run_algo('\nfrom zipline.api import fetch_csv, record, sid\n\ndef normalize(df):\n    df[\'scaled\'] = df[\'signal\'] * 10\n    return df\n\ndef initialize(context):\n    fetch_csv(\'https://fake.urls.com/aapl_ibm_csv_data.csv\',\n            post_func=normalize,\n            mask=True)\n\ndef handle_data(context, data):\n    record(\n        signal=data.current(sid(24),"signal"),\n        scaled=data.current(sid(24), "scaled"),\n        price=data.current(sid(24), "price"))\n            ')
        self.assertEqual(5, results['signal'][-1])
        self.assertEqual(50, results['scaled'][-1])
        self.assertEqual(24, results['price'][-1])

    @parameterized.expand([('unspecified', ''), ('none', 'usecols=None'), ('without date', "usecols=['Value']"), ('with date', "usecols=('Value', 'Date')")])
    def test_usecols(self, testname, usecols):
        if False:
            while True:
                i = 10
        self.responses.add(self.responses.GET, 'https://fake.urls.com/cpiaucsl_data.csv', body=CPIAUCSL_DATA, content_type='text/csv')
        code = '\nfrom zipline.api import fetch_csv, sid, record\n\ndef clean(df):\n    return df.rename(columns={{\'Value\':\'cpi\'}})\n\ndef initialize(context):\n    fetch_csv(\n        \'https://fake.urls.com/cpiaucsl_data.csv\',\n        symbol=\'urban\',\n        pre_func=clean,\n        date_column=\'Date\',\n        date_format=\'%Y-%m-%d\',{usecols}\n        )\n    context.stocks = [sid(3766), sid(25317)]\n\ndef handle_data(context, data):\n    data.current("urban", "cpi")\n        '
        results = self.run_algo(code.format(usecols=usecols))
        self.assertEqual(len(results), 251)

    def test_sources_merge_custom_ticker(self):
        if False:
            print('Hello World!')
        requests_kwargs = {}

        def capture_kwargs(zelf, url, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            requests_kwargs.update(mask_requests_args(url, kwargs).requests_kwargs)
            return PALLADIUM_DATA
        with patch('zipline.sources.requests_csv.PandasRequestsCSV.fetch_url', new=capture_kwargs):
            results = self.run_algo('\nfrom zipline.api import fetch_csv, record, sid\n\ndef rename_col(df):\n    df = df.rename(columns={\'New York 15:00\': \'price\'})\n    df = df.fillna(method=\'ffill\')\n    return df[[\'price\', \'sid\']]\n\ndef initialize(context):\n    fetch_csv(\'https://dl.dropbox.com/u/16705795/PALL.csv\',\n        date_column=\'Date\',\n        symbol=\'palladium\',\n        post_func=rename_col,\n        date_format=\'%Y-%m-%d\'\n        )\n    context.stock = sid(24)\n\ndef handle_data(context, data):\n    record(palladium=data.current("palladium", "price"))\n    record(aapl=data.current(context.stock, "price"))\n        ')
            np.testing.assert_array_equal([24] * 251, results['aapl'])
            self.assertEqual(337, results['palladium'].iloc[-1])
            expected = {'allow_redirects': False, 'stream': True, 'timeout': 30.0}
            self.assertEqual(expected, requests_kwargs)

    @parameterized.expand([('symbol', FETCHER_UNIVERSE_DATA, None), ('arglebargle', FETCHER_UNIVERSE_DATA_TICKER_COLUMN, FETCHER_ALTERNATE_COLUMN_HEADER)])
    def test_fetcher_universe(self, name, data, column_name):
        if False:
            for i in range(10):
                print('nop')
        with patch('zipline.sources.requests_csv.PandasRequestsCSV.fetch_url', new=lambda *a, **k: data):
            sim_params = factory.create_simulation_parameters(start=pd.Timestamp('2006-01-09', tz='UTC'), end=pd.Timestamp('2006-01-11', tz='UTC'))
            algocode = '\nfrom pandas import Timestamp\nfrom zipline.api import fetch_csv, record, sid, get_datetime\nfrom zipline.utils.pandas_utils import normalize_date\n\ndef initialize(context):\n    fetch_csv(\n        \'https://dl.dropbox.com/u/16705795/dtoc_history.csv\',\n        date_format=\'%m/%d/%Y\'{token}\n    )\n    context.expected_sids = {{\n        Timestamp(\'2006-01-09 00:00:00+0000\', tz=\'UTC\'):[24, 3766, 5061],\n        Timestamp(\'2006-01-10 00:00:00+0000\', tz=\'UTC\'):[24, 3766, 5061],\n        Timestamp(\'2006-01-11 00:00:00+0000\', tz=\'UTC\'):[24, 3766, 5061, 14848]\n    }}\n    context.bar_count = 0\n\ndef handle_data(context, data):\n    expected = context.expected_sids[normalize_date(get_datetime())]\n    actual = data.fetcher_assets\n    for stk in expected:\n        if stk not in actual:\n            raise Exception(\n                "{{stk}} is missing on dt={{dt}}".format(\n                    stk=stk, dt=get_datetime()))\n\n    record(sid_count=len(actual))\n    record(bar_count=context.bar_count)\n    context.bar_count += 1\n            '
            replacement = ''
            if column_name:
                replacement = ",symbol_column='%s'\n" % column_name
            real_algocode = algocode.format(token=replacement)
            results = self.run_algo(real_algocode, sim_params=sim_params)
            self.assertEqual(len(results), 3)
            self.assertEqual(3, results['sid_count'].iloc[0])
            self.assertEqual(3, results['sid_count'].iloc[1])
            self.assertEqual(4, results['sid_count'].iloc[2])

    def test_fetcher_universe_non_security_return(self):
        if False:
            return 10
        self.responses.add(self.responses.GET, 'https://fake.urls.com/bad_fetcher_universe_data.csv', body=NON_ASSET_FETCHER_UNIVERSE_DATA, content_type='text/csv')
        sim_params = factory.create_simulation_parameters(start=pd.Timestamp('2006-01-09', tz='UTC'), end=pd.Timestamp('2006-01-10', tz='UTC'))
        self.run_algo('\nfrom zipline.api import fetch_csv\n\ndef initialize(context):\n    fetch_csv(\n        \'https://fake.urls.com/bad_fetcher_universe_data.csv\',\n        date_format=\'%m/%d/%Y\'\n    )\n\ndef handle_data(context, data):\n    if len(data.fetcher_assets) > 0:\n        raise Exception("Shouldn\'t be any assets in fetcher_assets!")\n            ', sim_params=sim_params)

    def test_order_against_data(self):
        if False:
            return 10
        self.responses.add(self.responses.GET, 'https://fake.urls.com/palladium_data.csv', body=PALLADIUM_DATA, content_type='text/csv')
        with self.assertRaises(UnsupportedOrderParameters):
            self.run_algo("\nfrom zipline.api import fetch_csv, order, sid\n\ndef rename_col(df):\n    return df.rename(columns={'New York 15:00': 'price'})\n\ndef initialize(context):\n    fetch_csv('https://fake.urls.com/palladium_data.csv',\n        date_column='Date',\n        symbol='palladium',\n        post_func=rename_col,\n        date_format='%Y-%m-%d'\n        )\n    context.stock = sid(24)\n\ndef handle_data(context, data):\n    order('palladium', 100)\n            ")

    def test_fetcher_universe_minute(self):
        if False:
            for i in range(10):
                print('nop')
        self.responses.add(self.responses.GET, 'https://fake.urls.com/fetcher_universe_data.csv', body=FETCHER_UNIVERSE_DATA, content_type='text/csv')
        sim_params = factory.create_simulation_parameters(start=pd.Timestamp('2006-01-09', tz='UTC'), end=pd.Timestamp('2006-01-11', tz='UTC'), data_frequency='minute')
        results = self.run_algo('\nfrom pandas import Timestamp\nfrom zipline.api import fetch_csv, record, get_datetime\n\ndef initialize(context):\n    fetch_csv(\n        \'https://fake.urls.com/fetcher_universe_data.csv\',\n        date_format=\'%m/%d/%Y\'\n    )\n    context.expected_sids = {\n        Timestamp(\'2006-01-09 00:00:00+0000\', tz=\'UTC\'):[24, 3766, 5061],\n        Timestamp(\'2006-01-10 00:00:00+0000\', tz=\'UTC\'):[24, 3766, 5061],\n        Timestamp(\'2006-01-11 00:00:00+0000\', tz=\'UTC\'):[24, 3766, 5061, 14848]\n    }\n    context.bar_count = 0\n\ndef handle_data(context, data):\n    expected = context.expected_sids[get_datetime().replace(hour=0, minute=0)]\n    actual = data.fetcher_assets\n    for stk in expected:\n        if stk not in actual:\n            raise Exception("{stk} is missing".format(stk=stk))\n\n    record(sid_count=len(actual))\n    record(bar_count=context.bar_count)\n    context.bar_count += 1\n        ', sim_params=sim_params)
        self.assertEqual(3, len(results))
        self.assertEqual(3, results['sid_count'].iloc[0])
        self.assertEqual(3, results['sid_count'].iloc[1])
        self.assertEqual(4, results['sid_count'].iloc[2])

    def test_fetcher_in_before_trading_start(self):
        if False:
            return 10
        self.responses.add(self.responses.GET, 'https://fake.urls.com/fetcher_nflx_data.csv', body=NFLX_DATA, content_type='text/csv')
        sim_params = factory.create_simulation_parameters(start=pd.Timestamp('2013-06-13', tz='UTC'), end=pd.Timestamp('2013-11-15', tz='UTC'), data_frequency='minute')
        results = self.run_algo("\nfrom zipline.api import fetch_csv, record, symbol\n\ndef initialize(context):\n    fetch_csv('https://fake.urls.com/fetcher_nflx_data.csv',\n               date_column = 'Settlement Date',\n               date_format = '%m/%d/%y')\n    context.stock = symbol('NFLX')\n\ndef before_trading_start(context, data):\n    record(Short_Interest = data.current(context.stock, 'dtc'))\n", sim_params=sim_params)
        values = results['Short_Interest']
        np.testing.assert_array_equal(values[0:33], np.full(33, np.nan))
        np.testing.assert_array_almost_equal(values[33:44], [1.690317] * 11)
        np.testing.assert_array_almost_equal(values[44:55], [2.811858] * 11)
        np.testing.assert_array_almost_equal(values[55:64], [2.50233] * 9)
        np.testing.assert_array_almost_equal(values[64:75], [2.550829] * 11)
        np.testing.assert_array_almost_equal(values[75:], [2.64484] * 35)

    def test_fetcher_bad_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.responses.add(self.responses.GET, 'https://fake.urls.com/fetcher_nflx_data.csv', body=NFLX_DATA, content_type='text/csv')
        sim_params = factory.create_simulation_parameters(start=pd.Timestamp('2013-06-12', tz='UTC'), end=pd.Timestamp('2013-06-14', tz='UTC'), data_frequency='minute')
        results = self.run_algo("\nfrom zipline.api import fetch_csv, symbol\nimport numpy as np\n\ndef initialize(context):\n    fetch_csv('https://fake.urls.com/fetcher_nflx_data.csv',\n               date_column = 'Settlement Date',\n               date_format = '%m/%d/%y')\n    context.nflx = symbol('NFLX')\n    context.aapl = symbol('AAPL', country_code='US')\n\ndef handle_data(context, data):\n    assert np.isnan(data.current(context.nflx, 'invalid_column'))\n    assert np.isnan(data.current(context.aapl, 'invalid_column'))\n    assert np.isnan(data.current(context.aapl, 'dtc'))\n", sim_params=sim_params)
        self.assertEqual(3, len(results))