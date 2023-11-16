import time
import dateparser
import pytz
import json
from datetime import datetime
from binance.client import Client

def date_to_milliseconds(date_str):
    if False:
        return 10
    'Convert UTC date to milliseconds\n\n    If using offset strings add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"\n\n    See dateparse docs for formats http://dateparser.readthedocs.io/en/latest/\n\n    :param date_str: date in readable format, i.e. "January 01, 2018", "11 hours ago UTC", "now UTC"\n    :type date_str: str\n    '
    epoch = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
    d = dateparser.parse(date_str)
    if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
        d = d.replace(tzinfo=pytz.utc)
    return int((d - epoch).total_seconds() * 1000.0)

def interval_to_milliseconds(interval):
    if False:
        print('Hello World!')
    'Convert a Binance interval string to milliseconds\n\n    :param interval: Binance interval string 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w\n    :type interval: str\n\n    :return:\n         None if unit not one of m, h, d or w\n         None if string not in correct format\n         int value of interval in milliseconds\n    '
    ms = None
    seconds_per_unit = {'m': 60, 'h': 60 * 60, 'd': 24 * 60 * 60, 'w': 7 * 24 * 60 * 60}
    unit = interval[-1]
    if unit in seconds_per_unit:
        try:
            ms = int(interval[:-1]) * seconds_per_unit[unit] * 1000
        except ValueError:
            pass
    return ms

def get_historical_klines(symbol, interval, start_str, end_str=None):
    if False:
        for i in range(10):
            print('nop')
    'Get Historical Klines from Binance\n\n    See dateparse docs for valid start and end string formats http://dateparser.readthedocs.io/en/latest/\n\n    If using offset strings for dates add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"\n\n    :param symbol: Name of symbol pair e.g BNBBTC\n    :type symbol: str\n    :param interval: Biannce Kline interval\n    :type interval: str\n    :param start_str: Start date string in UTC format\n    :type start_str: str\n    :param end_str: optional - end date string in UTC format\n    :type end_str: str\n\n    :return: list of OHLCV values\n\n    '
    client = Client('', '')
    output_data = []
    limit = 500
    timeframe = interval_to_milliseconds(interval)
    start_ts = date_to_milliseconds(start_str)
    end_ts = None
    if end_str:
        end_ts = date_to_milliseconds(end_str)
    idx = 0
    symbol_existed = False
    while True:
        temp_data = client.get_klines(symbol=symbol, interval=interval, limit=limit, startTime=start_ts, endTime=end_ts)
        if not symbol_existed and len(temp_data):
            symbol_existed = True
        if symbol_existed:
            output_data += temp_data
            start_ts = temp_data[len(temp_data) - 1][0] + timeframe
        else:
            start_ts += timeframe
        idx += 1
        if len(temp_data) < limit:
            break
        if idx % 3 == 0:
            time.sleep(1)
    return output_data
symbol = 'ETHBTC'
start = '1 Dec, 2017'
end = '1 Jan, 2018'
interval = Client.KLINE_INTERVAL_30MINUTE
klines = get_historical_klines(symbol, interval, start, end)
with open('Binance_{}_{}_{}-{}.json'.format(symbol, interval, date_to_milliseconds(start), date_to_milliseconds(end)), 'w') as f:
    f.write(json.dumps(klines))