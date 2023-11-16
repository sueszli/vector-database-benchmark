from .alphavantage import AlphaVantage as av

class CryptoCurrencies(av):
    """This class implements all the crypto currencies api calls
    """

    @av._output_format
    @av._call_api_on_func
    def get_digital_currency_daily(self, symbol, market):
        if False:
            return 10
        ' Returns  the daily historical time series for a digital currency\n        (e.g., BTC) traded on a specific market (e.g., CNY/Chinese Yuan),\n        refreshed daily at midnight (UTC). Prices and volumes are quoted in\n        both the market-specific currency and USD..\n\n        Keyword Arguments:\n            symbol: The digital/crypto currency of your choice. It can be any\n            of the currencies in the digital currency list. For example:\n            symbol=BTC.\n            market: The exchange market of your choice. It can be any of the\n            market in the market list. For example: market=CNY.\n        '
        _FUNCTION_KEY = 'DIGITAL_CURRENCY_DAILY'
        return (_FUNCTION_KEY, 'Time Series (Digital Currency Daily)', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_digital_currency_weekly(self, symbol, market):
        if False:
            return 10
        ' Returns  the weekly historical time series for a digital currency\n        (e.g., BTC) traded on a specific market (e.g., CNY/Chinese Yuan),\n        refreshed daily at midnight (UTC). Prices and volumes are quoted in\n        both the market-specific currency and USD..\n\n        Keyword Arguments:\n            symbol: The digital/crypto currency of your choice. It can be any\n            of the currencies in the digital currency list. For example:\n            symbol=BTC.\n            market: The exchange market of your choice. It can be any of the\n            market in the market list. For example: market=CNY.\n        '
        _FUNCTION_KEY = 'DIGITAL_CURRENCY_WEEKLY'
        return (_FUNCTION_KEY, 'Time Series (Digital Currency Weekly)', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_digital_currency_monthly(self, symbol, market):
        if False:
            print('Hello World!')
        ' Returns  the monthly historical time series for a digital currency\n        (e.g., BTC) traded on a specific market (e.g., CNY/Chinese Yuan),\n        refreshed daily at midnight (UTC). Prices and volumes are quoted in\n        both the market-specific currency and USD..\n\n        Keyword Arguments:\n            symbol: The digital/crypto currency of your choice. It can be any\n            of the currencies in the digital currency list. For example:\n            symbol=BTC.\n            market: The exchange market of your choice. It can be any of the\n            market in the market list. For example: market=CNY.\n        '
        _FUNCTION_KEY = 'DIGITAL_CURRENCY_MONTHLY'
        return (_FUNCTION_KEY, 'Time Series (Digital Currency Monthly)', 'Meta Data')

    @av._output_format
    @av._call_api_on_func
    def get_digital_currency_exchange_rate(self, from_currency, to_currency):
        if False:
            for i in range(10):
                print('nop')
        ' Returns the realtime exchange rate for any pair of digital\n        currency (e.g., BTC) or physical currency (e.g., USD).\n        Keyword Arguments:\n            from_currency: The currency you would like to get the exchange rate\n            for. It can either be a physical currency or digital/crypto currency.\n            For example: from_currency=USD or from_currency=BTC.\n            to_currency: The destination currency for the exchange rate.\n            It can either be a physical currency or digital/crypto currency.\n            For example: to_currency=USD or to_currency=BTC.\n        '
        _FUNCTION_KEY = 'CURRENCY_EXCHANGE_RATE'
        return (_FUNCTION_KEY, 'Realtime Currency Exchange Rate', None)

    @av._output_format
    @av._call_api_on_func
    def get_digital_crypto_rating(self, symbol):
        if False:
            return 10
        ' Returns the Fundamental Crypto Asset Score for a digital currency\n        (e.g., BTC), and when it was last updated.\n\n        Keyword Arguments:\n            symbol: The digital/crypto currency of your choice. It can be any\n            of the currencies in the digital currency list. For example:\n            symbol=BTC.\n        '
        _FUNCTION_KEY = 'CRYPTO_RATING'
        return (_FUNCTION_KEY, 'Crypto Rating (FCAS)', None)