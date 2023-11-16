import requests
import os
from functools import wraps
import inspect
import sys
import re
try:
    import pandas
    _PANDAS_FOUND = True
except ImportError:
    _PANDAS_FOUND = False
import csv

class AlphaVantage(object):
    """ Base class where the decorators and base function for the other
    classes of this python wrapper will inherit from.
    """
    _ALPHA_VANTAGE_API_URL = 'https://www.alphavantage.co/query?'
    _ALPHA_VANTAGE_MATH_MAP = ['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'T3', 'KAMA', 'MAMA']
    _ALPHA_VANTAGE_DIGITAL_CURRENCY_LIST = 'https://www.alphavantage.co/digital_currency_list/'
    _RAPIDAPI_URL = 'https://alpha-vantage.p.rapidapi.com/query?'

    def __init__(self, key=None, output_format='json', treat_info_as_error=True, indexing_type='date', proxy=None, rapidapi=False):
        if False:
            i = 10
            return i + 15
        " Initialize the class\n\n        Keyword Arguments:\n            key:  Alpha Vantage api key\n            retries:  Maximum amount of retries in case of faulty connection or\n                server not able to answer the call.\n            treat_info_as_error: Treat information from the api as errors\n            output_format:  Either 'json', 'pandas' os 'csv'\n            indexing_type: Either 'date' to use the default date string given\n            by the alpha vantage api call or 'integer' if you just want an\n            integer indexing on your dataframe. Only valid, when the\n            output_format is 'pandas'\n            proxy: Dictionary mapping protocol or protocol and hostname to\n            the URL of the proxy.\n            rapidapi: Boolean describing whether or not the API key is\n            through the RapidAPI platform or not\n        "
        if key is None:
            key = os.getenv('ALPHAVANTAGE_API_KEY')
        if not key or not isinstance(key, str):
            raise ValueError('The AlphaVantage API key must be provided either through the key parameter or through the environment variable ALPHAVANTAGE_API_KEY. Get a free key from the alphavantage website: https://www.alphavantage.co/support/#api-key')
        self.headers = {}
        if rapidapi:
            self.headers = {'x-rapidapi-host': 'alpha-vantage.p.rapidapi.com', 'x-rapidapi-key': key}
        self.rapidapi = rapidapi
        self.key = key
        self.output_format = output_format
        if self.output_format == 'pandas' and (not _PANDAS_FOUND):
            raise ValueError('The pandas library was not found, therefore can not be used as an output format, please install manually')
        self.treat_info_as_error = treat_info_as_error
        self._append_type = True
        self.indexing_type = indexing_type
        self.proxy = proxy or {}

    @classmethod
    def _call_api_on_func(cls, func):
        if False:
            while True:
                i = 10
        ' Decorator for forming the api call with the arguments of the\n        function, it works by taking the arguments given to the function\n        and building the url to call the api on it\n\n        Keyword Arguments:\n            func:  The function to be decorated\n        '
        if sys.version_info[0] < 3:
            argspec = inspect.getargspec(func)
        else:
            argspec = inspect.getfullargspec(func)
        try:
            positional_count = len(argspec.args) - len(argspec.defaults)
            defaults = dict(zip(argspec.args[positional_count:], argspec.defaults))
        except TypeError:
            if argspec.args:
                positional_count = len(argspec.args)
                defaults = {}
            elif argspec.defaults:
                positional_count = 0
                defaults = argspec.defaults

        @wraps(func)
        def _call_wrapper(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            used_kwargs = kwargs.copy()
            used_kwargs.update(zip(argspec.args[positional_count:], args[positional_count:]))
            used_kwargs.update({k: used_kwargs.get(k, d) for (k, d) in defaults.items()})
            (function_name, data_key, meta_data_key) = func(self, *args, **kwargs)
            base_url = AlphaVantage._RAPIDAPI_URL if self.rapidapi else AlphaVantage._ALPHA_VANTAGE_API_URL
            url = '{}function={}'.format(base_url, function_name)
            for (idx, arg_name) in enumerate(argspec.args[1:]):
                try:
                    arg_value = args[idx]
                except IndexError:
                    arg_value = used_kwargs[arg_name]
                if 'matype' in arg_name and arg_value:
                    arg_value = self.map_to_matype(arg_value)
                if arg_value:
                    if isinstance(arg_value, tuple) or isinstance(arg_value, list):
                        arg_value = ','.join(arg_value)
                    url = '{}&{}={}'.format(url, arg_name, arg_value)
            if 'json' in self.output_format.lower() or 'csv' in self.output_format.lower():
                oformat = self.output_format.lower()
            elif 'pandas' in self.output_format.lower():
                oformat = 'json'
            else:
                raise ValueError('Output format: {} not recognized, only json,pandas and csv are supported'.format(self.output_format.lower()))
            apikey_parameter = '' if self.rapidapi else '&apikey={}'.format(self.key)
            if self._append_type:
                url = '{}{}&datatype={}'.format(url, apikey_parameter, oformat)
            else:
                url = '{}{}'.format(url, apikey_parameter)
            return (self._handle_api_call(url), data_key, meta_data_key)
        return _call_wrapper

    @classmethod
    def _output_format_sector(cls, func, override=None):
        if False:
            for i in range(10):
                print('nop')
        ' Decorator in charge of giving the output its right format, either\n        json or pandas (replacing the % for usable floats, range 0-1.0)\n\n        Keyword Arguments:\n            func: The function to be decorated\n            override: Override the internal format of the call, default None\n        Returns:\n            A decorator for the format sector api call\n        '

        @wraps(func)
        def _format_wrapper(self, *args, **kwargs):
            if False:
                print('Hello World!')
            (json_response, data_key, meta_data_key) = func(self, *args, **kwargs)
            if isinstance(data_key, list):
                data = {key: {k: self.percentage_to_float(v) for (k, v) in json_response[key].items()} for key in data_key}
            else:
                data = json_response[data_key]
            meta_data = json_response[meta_data_key]
            if override is None:
                output_format = self.output_format.lower()
            elif 'json' or 'pandas' in override.lower():
                output_format = override.lower()
            if output_format == 'json':
                return (data, meta_data)
            elif output_format == 'pandas':
                data_pandas = pandas.DataFrame.from_dict(data, orient='columns')
                col_names = [re.sub('\\d+.', '', name).strip(' ') for name in list(data_pandas)]
                data_pandas.columns = col_names
                return (data_pandas, meta_data)
            else:
                raise ValueError('Format: {} is not supported'.format(self.output_format))
        return _format_wrapper

    @classmethod
    def _output_format(cls, func, override=None):
        if False:
            return 10
        ' Decorator in charge of giving the output its right format, either\n        json or pandas\n\n        Keyword Arguments:\n            func:  The function to be decorated\n            override:  Override the internal format of the call, default None\n        '

        @wraps(func)
        def _format_wrapper(self, *args, **kwargs):
            if False:
                print('Hello World!')
            (call_response, data_key, meta_data_key) = func(self, *args, **kwargs)
            if 'json' in self.output_format.lower() or 'pandas' in self.output_format.lower():
                if data_key is not None:
                    data = call_response[data_key]
                else:
                    data = call_response
                if meta_data_key is not None:
                    meta_data = call_response[meta_data_key]
                else:
                    meta_data = None
                if override is None:
                    output_format = self.output_format.lower()
                elif 'json' or 'pandas' in override.lower():
                    output_format = override.lower()
                if output_format == 'json':
                    if isinstance(data, list):
                        if not data:
                            data_pandas = pandas.DataFrame()
                        else:
                            data_array = []
                            for val in data:
                                data_array.append([v for (_, v) in val.items()])
                            data_pandas = pandas.DataFrame(data_array, columns=[k for (k, _) in data[0].items()])
                        return (data_pandas, meta_data)
                    else:
                        return (data, meta_data)
                elif output_format == 'pandas':
                    if isinstance(data, list):
                        if not data:
                            data_pandas = pandas.DataFrame()
                        else:
                            data_array = []
                            for val in data:
                                data_array.append([v for (_, v) in val.items()])
                            data_pandas = pandas.DataFrame(data_array, columns=[k for (k, _) in data[0].items()])
                    else:
                        try:
                            data_pandas = pandas.DataFrame.from_dict(data, orient='index', dtype='float')
                        except ValueError:
                            data = {data_key: data}
                            data_pandas = pandas.DataFrame.from_dict(data, orient='index', dtype='object')
                            return (data_pandas, meta_data)
                    if 'integer' in self.indexing_type:
                        data_pandas.reset_index(level=0, inplace=True)
                        data_pandas.index.name = 'index'
                    else:
                        data_pandas.index.name = 'date'
                        data_pandas.index = pandas.to_datetime(data_pandas.index)
                    return (data_pandas, meta_data)
            elif 'csv' in self.output_format.lower():
                return (call_response, None)
            else:
                raise ValueError('Format: {} is not supported'.format(self.output_format))
        return _format_wrapper

    def set_proxy(self, proxy=None):
        if False:
            while True:
                i = 10
        ' Set a new proxy configuration\n\n        Keyword Arguments:\n            proxy: Dictionary mapping protocol or protocol and hostname to\n            the URL of the proxy.\n        '
        self.proxy = proxy or {}

    def map_to_matype(self, matype):
        if False:
            while True:
                i = 10
        ' Convert to the alpha vantage math type integer. It returns an\n        integer correspondent to the type of math to apply to a function. It\n        raises ValueError if an integer greater than the supported math types\n        is given.\n\n        Keyword Arguments:\n            matype:  The math type of the alpha vantage api. It accepts\n            integers or a string representing the math type.\n\n                * 0 = Simple Moving Average (SMA),\n                * 1 = Exponential Moving Average (EMA),\n                * 2 = Weighted Moving Average (WMA),\n                * 3 = Double Exponential Moving Average (DEMA),\n                * 4 = Triple Exponential Moving Average (TEMA),\n                * 5 = Triangular Moving Average (TRIMA),\n                * 6 = T3 Moving Average,\n                * 7 = Kaufman Adaptive Moving Average (KAMA),\n                * 8 = MESA Adaptive Moving Average (MAMA)\n        '
        try:
            value = int(matype)
            if abs(value) > len(AlphaVantage._ALPHA_VANTAGE_MATH_MAP):
                raise ValueError('The value {} is not supported'.format(value))
        except ValueError:
            value = AlphaVantage._ALPHA_VANTAGE_MATH_MAP.index(matype)
        return value

    def _handle_api_call(self, url):
        if False:
            print('Hello World!')
        ' Handle the return call from the  api and return a data and meta_data\n        object. It raises a ValueError on problems\n\n        Keyword Arguments:\n            url:  The url of the service\n            data_key:  The key for getting the data from the jso object\n            meta_data_key:  The key for getting the meta data information out\n            of the json object\n        '
        response = requests.get(url, proxies=self.proxy, headers=self.headers)
        if 'json' in self.output_format.lower() or 'pandas' in self.output_format.lower():
            json_response = response.json()
            if not json_response:
                raise ValueError('Error getting data from the api, no return was given.')
            elif 'Error Message' in json_response:
                raise ValueError(json_response['Error Message'])
            elif 'Information' in json_response and self.treat_info_as_error:
                raise ValueError(json_response['Information'])
            elif 'Note' in json_response and self.treat_info_as_error:
                raise ValueError(json_response['Note'])
            return json_response
        else:
            csv_response = csv.reader(response.text.splitlines())
            if not csv_response:
                raise ValueError('Error getting data from the api, no return was given.')
            return csv_response