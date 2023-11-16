from .alphavantage import AlphaVantage as av

class TechIndicators(av):
    """This class implements all the technical indicator api calls
    """

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Inherit AlphaVantage base class with its default arguments\n        '
        super(TechIndicators, self).__init__(*args, **kwargs)
        self._append_type = False
        if self.output_format.lower() == 'csv':
            raise ValueError('Output format {} is not comatible with the TechIndicators class'.format(self.output_format.lower()))

    @av._output_format
    @av._call_api_on_func
    def get_sma(self, symbol, interval='daily', time_period=20, series_type='close'):
        if False:
            for i in range(10):
                print('nop')
        " Return simple moving average time series in two json objects as data and\n        meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  How many data points to average (default 20)\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n        "
        _FUNCTION_KEY = 'SMA'
        return (_FUNCTION_KEY, 'Technical Analysis: SMA', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_ema(self, symbol, interval='daily', time_period=20, series_type='close'):
        if False:
            while True:
                i = 10
        " Return exponential moving average time series in two json objects\n        as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  How many data points to average (default 20)\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n        "
        _FUNCTION_KEY = 'EMA'
        return (_FUNCTION_KEY, 'Technical Analysis: EMA', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_wma(self, symbol, interval='daily', time_period=20, series_type='close'):
        if False:
            return 10
        " Return weighted moving average time series in two json objects\n        as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  How many data points to average (default 20)\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n        "
        _FUNCTION_KEY = 'WMA'
        return (_FUNCTION_KEY, 'Technical Analysis: WMA', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_dema(self, symbol, interval='daily', time_period=20, series_type='close'):
        if False:
            return 10
        " Return double exponential moving average time series in two json\n        objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  How many data points to average (default 20)\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n        "
        _FUNCTION_KEY = 'DEMA'
        return (_FUNCTION_KEY, 'Technical Analysis: DEMA', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_tema(self, symbol, interval='daily', time_period=20, series_type='close'):
        if False:
            for i in range(10):
                print('nop')
        " Return triple exponential moving average time series in two json\n        objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  How many data points to average (default 20)\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n        "
        _FUNCTION_KEY = 'TEMA'
        return (_FUNCTION_KEY, 'Technical Analysis: TEMA', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_trima(self, symbol, interval='daily', time_period=20, series_type='close'):
        if False:
            for i in range(10):
                print('nop')
        " Return triangular moving average time series in two json\n        objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  How many data points to average (default 20)\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n        "
        _FUNCTION_KEY = 'TRIMA'
        return (_FUNCTION_KEY, 'Technical Analysis: TRIMA', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_kama(self, symbol, interval='daily', time_period=20, series_type='close'):
        if False:
            return 10
        " Return Kaufman adaptative moving average time series in two json\n        objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  How many data points to average (default 20)\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n        "
        _FUNCTION_KEY = 'KAMA'
        return (_FUNCTION_KEY, 'Technical Analysis: KAMA', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_mama(self, symbol, interval='daily', series_type='close', fastlimit=None, slowlimit=None):
        if False:
            i = 10
            return i + 15
        " Return MESA adaptative moving average time series in two json\n        objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n            fastlimit:  Positive floats for the fast limit are accepted\n                (default=None)\n            slowlimit:  Positive floats for the slow limit are accepted\n                (default=None)\n        "
        _FUNCTION_KEY = 'MAMA'
        return (_FUNCTION_KEY, 'Technical Analysis: MAMA', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_vwap(self, symbol, interval='5min'):
        if False:
            print('Hello World!')
        " Returns the volume weighted average price (VWAP) for intraday time series.\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min' \n                (default 5min)\n        "
        _FUNCTION_KEY = 'VWAP'
        return (_FUNCTION_KEY, 'Technical Analysis: VWAP', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_t3(self, symbol, interval='daily', time_period=20, series_type='close'):
        if False:
            i = 10
            return i + 15
        " Return triple exponential moving average time series in two json\n        objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  How many data points to average (default 20)\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n        "
        _FUNCTION_KEY = 'T3'
        return (_FUNCTION_KEY, 'Technical Analysis: T3', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_macd(self, symbol, interval='daily', series_type='close', fastperiod=None, slowperiod=None, signalperiod=None):
        if False:
            while True:
                i = 10
        " Return the moving average convergence/divergence time series in two\n        json objects as data and meta_data. It raises ValueError when problems\n        arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily'\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n            fastperiod:  Positive integers are accepted (default=None)\n            slowperiod:  Positive integers are accepted (default=None)\n            signalperiod:  Positive integers are accepted (default=None)\n        "
        _FUNCTION_KEY = 'MACD'
        return (_FUNCTION_KEY, 'Technical Analysis: MACD', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_macdext(self, symbol, interval='daily', series_type='close', fastperiod=None, slowperiod=None, signalperiod=None, fastmatype=None, slowmatype=None, signalmatype=None):
        if False:
            return 10
        " Return the moving average convergence/divergence time series in two\n        json objects as data and meta_data. It raises ValueError when problems\n        arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n            fastperiod:  Positive integers are accepted (default=None)\n            slowperiod:  Positive integers are accepted (default=None)\n            signalperiod:  Positive integers are accepted (default=None)\n            fastmatype:  Moving average type for the faster moving average.\n                By default, fastmatype=0. Integers 0 - 8 are accepted\n                (check  down the mappings) or the string containing the math type can\n                also be used.\n            slowmatype:  Moving average type for the slower moving average.\n                By default, slowmatype=0. Integers 0 - 8 are accepted\n                (check down the mappings) or the string containing the math type can\n                also be used.\n            signalmatype:  Moving average type for the signal moving average.\n                By default, signalmatype=0. Integers 0 - 8 are accepted\n                (check down the mappings) or the string containing the math type can\n                also be used.\n\n                * 0 = Simple Moving Average (SMA),\n                * 1 = Exponential Moving Average (EMA),\n                * 2 = Weighted Moving Average (WMA),\n                * 3 = Double Exponential Moving Average (DEMA),\n                * 4 = Triple Exponential Moving Average (TEMA),\n                * 5 = Triangular Moving Average (TRIMA),\n                * 6 = T3 Moving Average,\n                * 7 = Kaufman Adaptive Moving Average (KAMA),\n                * 8 = MESA Adaptive Moving Average (MAMA)\n        "
        _FUNCTION_KEY = 'MACDEXT'
        return (_FUNCTION_KEY, 'Technical Analysis: MACDEXT', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_stoch(self, symbol, interval='daily', fastkperiod=None, slowkperiod=None, slowdperiod=None, slowkmatype=None, slowdmatype=None):
        if False:
            print('Hello World!')
        " Return the stochatic oscillator values in two\n        json objects as data and meta_data. It raises ValueError when problems\n        arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            fastkperiod:  The time period of the fastk moving average. Positive\n                integers are accepted (default=None)\n            slowkperiod:  The time period of the slowk moving average. Positive\n                integers are accepted (default=None)\n            slowdperiod: The time period of the slowd moving average. Positive\n                integers are accepted (default=None)\n            slowkmatype:  Moving average type for the slowk moving average.\n                By default, fastmatype=0. Integers 0 - 8 are accepted\n                (check  down the mappings) or the string containing the math type can\n                also be used.\n            slowdmatype:  Moving average type for the slowd moving average.\n                By default, slowmatype=0. Integers 0 - 8 are accepted\n                (check down the mappings) or the string containing the math type can\n                also be used.\n\n                * 0 = Simple Moving Average (SMA),\n                * 1 = Exponential Moving Average (EMA),\n                * 2 = Weighted Moving Average (WMA),\n                * 3 = Double Exponential Moving Average (DEMA),\n                * 4 = Triple Exponential Moving Average (TEMA),\n                * 5 = Triangular Moving Average (TRIMA),\n                * 6 = T3 Moving Average,\n                * 7 = Kaufman Adaptive Moving Average (KAMA),\n                * 8 = MESA Adaptive Moving Average (MAMA)\n        "
        _FUNCTION_KEY = 'STOCH'
        return (_FUNCTION_KEY, 'Technical Analysis: STOCH', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_stochf(self, symbol, interval='daily', fastkperiod=None, fastdperiod=None, fastdmatype=None):
        if False:
            print('Hello World!')
        " Return the stochatic oscillator values in two\n        json objects as data and meta_data. It raises ValueError when problems\n        arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            fastkperiod:  The time period of the fastk moving average. Positive\n                integers are accepted (default=None)\n            fastdperiod:  The time period of the fastd moving average. Positive\n                integers are accepted (default=None)\n            fastdmatype:  Moving average type for the fastdmatype moving average.\n                By default, fastmatype=0. Integers 0 - 8 are accepted\n                (check  down the mappings) or the string containing the math type can\n                also be used.\n\n                * 0 = Simple Moving Average (SMA),\n                * 1 = Exponential Moving Average (EMA),\n                * 2 = Weighted Moving Average (WMA),\n                * 3 = Double Exponential Moving Average (DEMA),\n                * 4 = Triple Exponential Moving Average (TEMA),\n                * 5 = Triangular Moving Average (TRIMA),\n                * 6 = T3 Moving Average,\n                * 7 = Kaufman Adaptive Moving Average (KAMA),\n                * 8 = MESA Adaptive Moving Average (MAMA)\n        "
        _FUNCTION_KEY = 'STOCHF'
        return (_FUNCTION_KEY, 'Technical Analysis: STOCHF', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_rsi(self, symbol, interval='daily', time_period=20, series_type='close'):
        if False:
            while True:
                i = 10
        " Return the relative strength index time series in two json\n        objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  How many data points to average (default 20)\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n        "
        _FUNCTION_KEY = 'RSI'
        return (_FUNCTION_KEY, 'Technical Analysis: RSI', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_stochrsi(self, symbol, interval='daily', time_period=20, series_type='close', fastkperiod=None, fastdperiod=None, fastdmatype=None):
        if False:
            print('Hello World!')
        " Return the stochatic relative strength index in two\n        json objects as data and meta_data. It raises ValueError when problems\n        arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  How many data points to average (default 20)\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n            fastkperiod:  The time period of the fastk moving average. Positive\n                integers are accepted (default=None)\n            fastdperiod:  The time period of the fastd moving average. Positive\n                integers are accepted (default=None)\n            fastdmatype:  Moving average type for the fastdmatype moving average.\n                By default, fastmatype=0. Integers 0 - 8 are accepted\n                (check  down the mappings) or the string containing the math type can\n                also be used.\n\n                * 0 = Simple Moving Average (SMA),\n                * 1 = Exponential Moving Average (EMA),\n                * 2 = Weighted Moving Average (WMA),\n                * 3 = Double Exponential Moving Average (DEMA),\n                * 4 = Triple Exponential Moving Average (TEMA),\n                * 5 = Triangular Moving Average (TRIMA),\n                * 6 = T3 Moving Average,\n                * 7 = Kaufman Adaptive Moving Average (KAMA),\n                * 8 = MESA Adaptive Moving Average (MAMA)\n        "
        _FUNCTION_KEY = 'STOCHRSI'
        return (_FUNCTION_KEY, 'Technical Analysis: STOCHRSI', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_willr(self, symbol, interval='daily', time_period=20):
        if False:
            while True:
                i = 10
        " Return the Williams' %R (WILLR) values in two json objects as data\n        and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  How many data points to average (default 20)\n        "
        _FUNCTION_KEY = 'WILLR'
        return (_FUNCTION_KEY, 'Technical Analysis: WILLR', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_adx(self, symbol, interval='daily', time_period=20):
        if False:
            for i in range(10):
                print('nop')
        " Return  the average directional movement index values in two json\n        objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  How many data points to average (default 20)\n        "
        _FUNCTION_KEY = 'ADX'
        return (_FUNCTION_KEY, 'Technical Analysis: ADX', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_adxr(self, symbol, interval='daily', time_period=20):
        if False:
            print('Hello World!')
        " Return  the average directional movement index  rating in two json\n        objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  How many data points to average (default 20)\n        "
        _FUNCTION_KEY = 'ADXR'
        return (_FUNCTION_KEY, 'Technical Analysis: ADXR', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_apo(self, symbol, interval='daily', series_type='close', fastperiod=None, slowperiod=None, matype=None):
        if False:
            return 10
        " Return the absolute price oscillator values in two\n        json objects as data and meta_data. It raises ValueError when problems\n        arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default '60min)'\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n            fastperiod:  Positive integers are accepted (default=None)\n            slowperiod:  Positive integers are accepted (default=None)\n            matype    :  Moving average type. By default, fastmatype=0.\n                Integers 0 - 8 are accepted (check  down the mappings) or the string\n                containing the math type can also be used.\n\n                * 0 = Simple Moving Average (SMA),\n                * 1 = Exponential Moving Average (EMA),\n                * 2 = Weighted Moving Average (WMA),\n                * 3 = Double Exponential Moving Average (DEMA),\n                * 4 = Triple Exponential Moving Average (TEMA),\n                * 5 = Triangular Moving Average (TRIMA),\n                * 6 = T3 Moving Average,\n                * 7 = Kaufman Adaptive Moving Average (KAMA),\n                * 8 = MESA Adaptive Moving Average (MAMA)\n        "
        _FUNCTION_KEY = 'APO'
        return (_FUNCTION_KEY, 'Technical Analysis: APO', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_ppo(self, symbol, interval='daily', series_type='close', fastperiod=None, slowperiod=None, matype=None):
        if False:
            for i in range(10):
                print('nop')
        " Return the percentage price oscillator values in two\n        json objects as data and meta_data. It raises ValueError when problems\n        arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily'\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n            fastperiod:  Positive integers are accepted (default=None)\n            slowperiod:  Positive integers are accepted (default=None)\n            matype    :  Moving average type. By default, fastmatype=0.\n                Integers 0 - 8 are accepted (check  down the mappings) or the string\n                containing the math type can also be used.\n\n                * 0 = Simple Moving Average (SMA),\n                * 1 = Exponential Moving Average (EMA),\n                * 2 = Weighted Moving Average (WMA),\n                * 3 = Double Exponential Moving Average (DEMA),\n                * 4 = Triple Exponential Moving Average (TEMA),\n                * 5 = Triangular Moving Average (TRIMA),\n                * 6 = T3 Moving Average,\n                * 7 = Kaufman Adaptive Moving Average (KAMA),\n                * 8 = MESA Adaptive Moving Average (MAMA)\n        "
        _FUNCTION_KEY = 'PPO'
        return (_FUNCTION_KEY, 'Technical Analysis: PPO', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_mom(self, symbol, interval='daily', time_period=20, series_type='close'):
        if False:
            for i in range(10):
                print('nop')
        " Return the momentum values in two json\n        objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  How many data points to average (default 20)\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n        "
        _FUNCTION_KEY = 'MOM'
        return (_FUNCTION_KEY, 'Technical Analysis: MOM', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_bop(self, symbol, interval='daily', time_period=20):
        if False:
            print('Hello World!')
        " Return the balance of power values in two json\n        objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  How many data points to average (default 20)\n        "
        _FUNCTION_KEY = 'BOP'
        return (_FUNCTION_KEY, 'Technical Analysis: BOP', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_cci(self, symbol, interval='daily', time_period=20):
        if False:
            print('Hello World!')
        " Return the commodity channel index values  in two json\n        objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  How many data points to average (default 20)\n        "
        _FUNCTION_KEY = 'CCI'
        return (_FUNCTION_KEY, 'Technical Analysis: CCI', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_cmo(self, symbol, interval='daily', time_period=20, series_type='close'):
        if False:
            print('Hello World!')
        " Return the Chande momentum oscillator in two json\n        objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  How many data points to average (default 20)\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n        "
        _FUNCTION_KEY = 'CMO'
        return (_FUNCTION_KEY, 'Technical Analysis: CMO', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_roc(self, symbol, interval='daily', time_period=20, series_type='close'):
        if False:
            print('Hello World!')
        " Return the rate of change values in two json\n        objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  How many data points to average (default 20)\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n        "
        _FUNCTION_KEY = 'ROC'
        return (_FUNCTION_KEY, 'Technical Analysis: ROC', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_rocr(self, symbol, interval='daily', time_period=20, series_type='close'):
        if False:
            while True:
                i = 10
        " Return the rate of change ratio values in two json\n        objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  How many data points to average (default 20)\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n        "
        _FUNCTION_KEY = 'ROCR'
        return (_FUNCTION_KEY, 'Technical Analysis: ROCR', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_aroon(self, symbol, interval='daily', time_period=20, series_type='close'):
        if False:
            while True:
                i = 10
        " Return the aroon values in two json\n        objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  How many data points to average (default 20)\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n        "
        _FUNCTION_KEY = 'AROON'
        return (_FUNCTION_KEY, 'Technical Analysis: AROON', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_aroonosc(self, symbol, interval='daily', time_period=20, series_type='close'):
        if False:
            for i in range(10):
                print('nop')
        " Return the aroon oscillator values in two json\n        objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  How many data points to average (default 20)\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n        "
        _FUNCTION_KEY = 'AROONOSC'
        return (_FUNCTION_KEY, 'Technical Analysis: AROONOSC', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_mfi(self, symbol, interval='daily', time_period=20, series_type='close'):
        if False:
            for i in range(10):
                print('nop')
        " Return the money flow index values in two json\n        objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  How many data points to average (default 20)\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n        "
        _FUNCTION_KEY = 'MFI'
        return (_FUNCTION_KEY, 'Technical Analysis: MFI', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_trix(self, symbol, interval='daily', time_period=20, series_type='close'):
        if False:
            i = 10
            return i + 15
        " Return the1-day rate of change of a triple smooth exponential\n        moving average in two json objects as data and meta_data.\n        It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  How many data points to average (default 20)\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n        "
        _FUNCTION_KEY = 'TRIX'
        return (_FUNCTION_KEY, 'Technical Analysis: TRIX', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_ultosc(self, symbol, interval='daily', timeperiod1=None, timeperiod2=None, timeperiod3=None):
        if False:
            i = 10
            return i + 15
        " Return the ultimate oscillaror values in two json objects as\n        data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            timeperiod1:  The first time period indicator. Positive integers are\n                accepted. By default, timeperiod1=7\n            timeperiod2:  The first time period indicator. Positive integers are\n                accepted. By default, timeperiod2=14\n            timeperiod3:  The first time period indicator. Positive integers are\n                accepted. By default, timeperiod3=28\n        "
        _FUNCTION_KEY = 'ULTOSC'
        return (_FUNCTION_KEY, 'Technical Analysis: ULTOSC', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_dx(self, symbol, interval='daily', time_period=20, series_type='close'):
        if False:
            while True:
                i = 10
        " Return the directional movement index values in two json objects as\n        data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  How many data points to average (default 20)\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n        "
        _FUNCTION_KEY = 'DX'
        return (_FUNCTION_KEY, 'Technical Analysis: DX', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_minus_di(self, symbol, interval='daily', time_period=20):
        if False:
            i = 10
            return i + 15
        " Return the minus directional indicator values in two json\n        objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  How many data points to average (default 20)\n        "
        _FUNCTION_KEY = 'MINUS_DI'
        return (_FUNCTION_KEY, 'Technical Analysis: MINUS_DI', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_plus_di(self, symbol, interval='daily', time_period=20):
        if False:
            return 10
        " Return the plus directional indicator values in two json\n        objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  How many data points to average (default 20)\n        "
        _FUNCTION_KEY = 'PLUS_DI'
        return (_FUNCTION_KEY, 'Technical Analysis: PLUS_DI', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_minus_dm(self, symbol, interval='daily', time_period=20):
        if False:
            for i in range(10):
                print('nop')
        " Return the minus directional movement values in two json\n        objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n        "
        _FUNCTION_KEY = 'MINUS_DM'
        return (_FUNCTION_KEY, 'Technical Analysis: MINUS_DM', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_plus_dm(self, symbol, interval='daily', time_period=20):
        if False:
            for i in range(10):
                print('nop')
        " Return the plus directional movement values in two json\n        objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n        "
        _FUNCTION_KEY = 'PLUS_DM'
        return (_FUNCTION_KEY, 'Technical Analysis: PLUS_DM', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_bbands(self, symbol, interval='daily', time_period=20, series_type='close', nbdevup=None, nbdevdn=None, matype=None):
        if False:
            print('Hello World!')
        " Return the bollinger bands values in two\n        json objects as data and meta_data. It raises ValueError when problems\n        arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  Number of data points used to calculate each BBANDS value.\n                Positive integers are accepted (e.g., time_period=60, time_period=200)\n                (default=20)\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n            nbdevup:  The standard deviation multiplier of the upper band. Positive\n                integers are accepted as default (default=2)\n            nbdevdn:  The standard deviation multiplier of the lower band. Positive\n                integers are accepted as default (default=2)\n            matype :  Moving average type. By default, matype=0.\n                Integers 0 - 8 are accepted (check  down the mappings) or the string\n                containing the math type can also be used.\n\n                * 0 = Simple Moving Average (SMA),\n                * 1 = Exponential Moving Average (EMA),\n                * 2 = Weighted Moving Average (WMA),\n                * 3 = Double Exponential Moving Average (DEMA),\n                * 4 = Triple Exponential Moving Average (TEMA),\n                * 5 = Triangular Moving Average (TRIMA),\n                * 6 = T3 Moving Average,\n                * 7 = Kaufman Adaptive Moving Average (KAMA),\n                * 8 = MESA Adaptive Moving Average (MAMA)\n        "
        _FUNCTION_KEY = 'BBANDS'
        return (_FUNCTION_KEY, 'Technical Analysis: BBANDS', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_midpoint(self, symbol, interval='daily', time_period=20, series_type='close'):
        if False:
            print('Hello World!')
        " Return the midpoint values in two json objects as\n        data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  How many data points to average (default 20)\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n        "
        _FUNCTION_KEY = 'MIDPOINT'
        return (_FUNCTION_KEY, 'Technical Analysis: MIDPOINT', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_midprice(self, symbol, interval='daily', time_period=20):
        if False:
            return 10
        " Return the midprice values in two json objects as\n        data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  How many data points to average (default 20)\n        "
        _FUNCTION_KEY = 'MIDPRICE'
        return (_FUNCTION_KEY, 'Technical Analysis: MIDPRICE', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_sar(self, symbol, interval='daily', acceleration=None, maximum=None):
        if False:
            while True:
                i = 10
        " Return the midprice values in two json objects as\n        data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            acceleration:  The acceleration factor. Positive floats are accepted (\n                default 0.01)\n            maximum:  The acceleration factor maximum value. Positive floats\n                are accepted (default 0.20 )\n        "
        _FUNCTION_KEY = 'SAR'
        return (_FUNCTION_KEY, 'Technical Analysis: SAR', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_trange(self, symbol, interval='daily'):
        if False:
            while True:
                i = 10
        " Return the true range values in two json\n        objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n        "
        _FUNCTION_KEY = 'TRANGE'
        return (_FUNCTION_KEY, 'Technical Analysis: TRANGE', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_atr(self, symbol, interval='daily', time_period=20):
        if False:
            while True:
                i = 10
        " Return the average true range values in two json objects as\n        data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  How many data points to average (default 20)\n        "
        _FUNCTION_KEY = 'ATR'
        return (_FUNCTION_KEY, 'Technical Analysis: ATR', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_natr(self, symbol, interval='daily', time_period=20):
        if False:
            i = 10
            return i + 15
        " Return the normalized average true range values in two json objects\n        as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            time_period:  How many data points to average (default 20)\n        "
        _FUNCTION_KEY = 'NATR'
        return (_FUNCTION_KEY, 'Technical Analysis: NATR', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_ad(self, symbol, interval='daily'):
        if False:
            return 10
        " Return the Chaikin A/D line values in two json\n        objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n        "
        _FUNCTION_KEY = 'AD'
        return (_FUNCTION_KEY, 'Technical Analysis: Chaikin A/D', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_adosc(self, symbol, interval='daily', fastperiod=None, slowperiod=None):
        if False:
            for i in range(10):
                print('nop')
        " Return the Chaikin A/D oscillator values in two\n        json objects as data and meta_data. It raises ValueError when problems\n        arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily'\n            fastperiod:  Positive integers are accepted (default=None)\n            slowperiod:  Positive integers are accepted (default=None)\n        "
        _FUNCTION_KEY = 'ADOSC'
        return (_FUNCTION_KEY, 'Technical Analysis: ADOSC', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_obv(self, symbol, interval='daily'):
        if False:
            for i in range(10):
                print('nop')
        " Return the on balance volume values in two json\n        objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n        "
        _FUNCTION_KEY = 'OBV'
        return (_FUNCTION_KEY, 'Technical Analysis: OBV', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_ht_trendline(self, symbol, interval='daily', series_type='close'):
        if False:
            return 10
        " Return the Hilbert transform, instantaneous trendline values in two\n        json objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n        "
        _FUNCTION_KEY = 'HT_TRENDLINE'
        return (_FUNCTION_KEY, 'Technical Analysis: HT_TRENDLINE', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_ht_sine(self, symbol, interval='daily', series_type='close'):
        if False:
            i = 10
            return i + 15
        " Return the Hilbert transform, sine wave values in two\n        json objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            series_type:  The desired price type in the time series. Four types\n            are supported: 'close', 'open', 'high', 'low' (default 'close')\n        "
        _FUNCTION_KEY = 'HT_SINE'
        return (_FUNCTION_KEY, 'Technical Analysis: HT_SINE', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_ht_trendmode(self, symbol, interval='daily', series_type='close'):
        if False:
            while True:
                i = 10
        " Return the Hilbert transform, trend vs cycle mode in two\n        json objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n        "
        _FUNCTION_KEY = 'HT_TRENDMODE'
        return (_FUNCTION_KEY, 'Technical Analysis: HT_TRENDMODE', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_ht_dcperiod(self, symbol, interval='daily', series_type='close'):
        if False:
            while True:
                i = 10
        " Return the Hilbert transform, dominant cycle period in two\n        json objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n        "
        _FUNCTION_KEY = 'HT_DCPERIOD'
        return (_FUNCTION_KEY, 'Technical Analysis: HT_DCPERIOD', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_ht_dcphase(self, symbol, interval='daily', series_type='close'):
        if False:
            while True:
                i = 10
        " Return the Hilbert transform, dominant cycle phase in two\n        json objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n        "
        _FUNCTION_KEY = 'HT_DCPHASE'
        return (_FUNCTION_KEY, 'Technical Analysis: HT_DCPHASE', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_ht_phasor(self, symbol, interval='daily', series_type='close'):
        if False:
            print('Hello World!')
        " Return the Hilbert transform, phasor components in two\n        json objects as data and meta_data. It raises ValueError when problems arise\n\n        Keyword Arguments:\n            symbol:  the symbol for the equity we want to get its data\n            interval:  time interval between two conscutive values,\n                supported values are '1min', '5min', '15min', '30min', '60min', 'daily',\n                'weekly', 'monthly' (default 'daily')\n            series_type:  The desired price type in the time series. Four types\n                are supported: 'close', 'open', 'high', 'low' (default 'close')\n        "
        _FUNCTION_KEY = 'HT_PHASOR'
        return (_FUNCTION_KEY, 'Technical Analysis: HT_PHASOR', 'Meta Data')