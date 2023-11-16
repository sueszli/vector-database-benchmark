import locale
from urllib.parse import urlencode
from libqtile.log_utils import logger
from libqtile.widget.generic_poll_text import GenPollUrl

class StockTicker(GenPollUrl):
    """
    A stock ticker widget, based on the alphavantage API. Users must acquire an
    API key from https://www.alphavantage.co/support/#api-key

    The widget defaults to the TIME_SERIES_INTRADAY API function (i.e. stock
    symbols), but arbitrary Alpha Vantage API queries can be made by passing
    extra arguments to the constructor.

    ::

        # Display AMZN
        widget.StockTicker(apikey=..., symbol="AMZN")

        # Display BTC
        widget.StockTicker(
            apikey=..., function="DIGITAL_CURRENCY_INTRADAY", symbol="BTC", market="USD"
        )
    """
    defaults = [('interval', '1min', 'The default latency to query'), ('func', 'TIME_SERIES_INTRADAY', 'The default API function to query'), ('function', 'TIME_SERIES_INTRADAY', 'DEPRECATED: Use `func`.')]

    def __init__(self, **config):
        if False:
            return 10
        if 'function' in config:
            logger.warning('`function` parameter is deprecated. Please rename to `func`')
            config['func'] = config.pop('function')
        GenPollUrl.__init__(self, **config)
        self.add_defaults(StockTicker.defaults)
        self.sign = locale.localeconv()['currency_symbol']
        self.query = {'interval': self.interval, 'outputsize': 'compact', 'function': self.func}
        for (k, v) in config.items():
            self.query[k] = v

    @property
    def url(self):
        if False:
            while True:
                i = 10
        url = 'https://www.alphavantage.co/query?' + urlencode(self.query)
        return url

    def parse(self, body):
        if False:
            for i in range(10):
                print('nop')
        last = None
        for (k, v) in body['Meta Data'].items():
            if 'Last Refreshed' in k:
                last = v
        other = None
        for (k, v) in body.items():
            if k != 'Meta Data':
                other = v
                break
        price = None
        for (k, v) in other[last].items():
            if 'price' in k or 'close' in k:
                price = '{:0.2f}'.format(float(v))
                break
        return '{symbol}: {sign}{price}'.format(symbol=self.symbol, sign=self.sign, price=price)