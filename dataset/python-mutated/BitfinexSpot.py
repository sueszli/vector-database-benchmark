import requests
import jesse.helpers as jh
from jesse import exceptions
from jesse.modes.import_candles_mode.drivers.interface import CandleExchange
from jesse.enums import exchanges
from .bitfinex_utils import timeframe_to_interval

class BitfinexSpot(CandleExchange):

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(name=exchanges.BITFINEX_SPOT, count=1440, rate_limit_per_second=1, backup_exchange_class=None)
        self.endpoint = 'https://api-pub.bitfinex.com/v2/candles'

    def get_starting_time(self, symbol: str) -> int:
        if False:
            i = 10
            return i + 15
        dashless_symbol = jh.dashless_symbol(symbol)
        if symbol == 'BTC-USD':
            return jh.date_to_timestamp('2015-08-01')
        elif symbol == 'ETH-USD':
            return jh.date_to_timestamp('2016-01-01')
        payload = {'sort': 1, 'limit': 5000}
        response = requests.get(f'{self.endpoint}/trade:1D:t{dashless_symbol}/hist', params=payload)
        self.validate_response(response)
        data = response.json()
        if not len(data):
            raise exceptions.SymbolNotFound(f"No candle exists for {symbol} in Bitfinex. You're probably misspelling the symbol name.")
        first_timestamp = int(data[0][0])
        return first_timestamp + 60000 * 1440

    def fetch(self, symbol: str, start_timestamp: int, timeframe: str) -> list:
        if False:
            print('Hello World!')
        end_timestamp = start_timestamp + (self.count - 1) * 60000 * jh.timeframe_to_one_minutes(timeframe)
        interval = timeframe_to_interval(timeframe)
        payload = {'start': start_timestamp, 'end': end_timestamp, 'limit': self.count, 'sort': 1}
        dashless_symbol = jh.dashless_symbol(symbol)
        response = requests.get(f'{self.endpoint}/trade:{interval}:t{dashless_symbol}/hist', params=payload)
        self.validate_response(response)
        data = response.json()
        return [{'id': jh.generate_unique_id(), 'exchange': self.name, 'symbol': symbol, 'timeframe': timeframe, 'timestamp': d[0], 'open': d[1], 'close': d[2], 'high': d[3], 'low': d[4], 'volume': d[5]} for d in data]