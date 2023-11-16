from __future__ import print_function
import collections
import time
from datetime import datetime
import pandas as pd
import ystockquote
from arctic import Arctic

def get_stock_history(ticker, start_date, end_date):
    if False:
        print('Hello World!')
    data = ystockquote.get_historical_prices(ticker, start_date, end_date)
    df = pd.DataFrame(collections.OrderedDict(sorted(data.items()))).T
    df = df.convert_objects(convert_numeric=True)
    return df
arctic = Arctic('localhost')
arctic.delete_library('jblackburn.stocks')
arctic.initialize_library('jblackburn.stocks')
arctic.list_libraries()
stocks = arctic['jblackburn.stocks']
aapl = get_stock_history('aapl', '2015-01-01', '2015-02-01')
aapl
stocks.write('aapl', aapl, metadata={'source': 'YAHOO'})
stocks.read('aapl').data['Adj Close'].plot()
stocks.read('aapl').metadata
stocks.read('aapl').version
aapl = get_stock_history('aapl', '2015-02-01', '2015-03-01')
stocks.append('aapl', aapl)
stocks.read('aapl').data
stocks.list_symbols()
stocks.list_versions('aapl')
stocks.read('aapl', as_of=1).data.ix[-1]
stocks.read('aapl', as_of=2).data.ix[-1]
stocks.snapshot('snap')
stocks.read('aapl', as_of='snap').data.ix[-1]
lib = arctic['nyse']

def load_all_stock_history_NYSE():
    if False:
        i = 10
        return i + 15
    nyse = pd.read_csv('/users/is/jblackburn/git/arctic/howtos/nyse.csv')
    stocks = [x.split('/')[0] for x in nyse['Ticker']]
    print(len(stocks), ' symbols')
    for (i, stock) in enumerate(stocks):
        try:
            now = datetime.now()
            data = get_stock_history('aapl', '1980-01-01', '2015-07-07')
            lib.write(stock, data)
            print('loaded data for: ', stock, datetime.now() - now)
        except Exception as e:
            print('Failed for ', stock, str(e))
print(len(lib.list_symbols()), ' NYSE symbols loaded')

def read_all_data_from_lib(lib):
    if False:
        i = 10
        return i + 15
    start = time.time()
    rows_read = 0
    for s in lib.list_symbols():
        rows_read += len(lib.read(s).data)
    print('Symbols: %s Rows: %s  Time: %s  Rows/s: %s' % (len(lib.list_symbols()), rows_read, time.time() - start, rows_read / (time.time() - start)))
read_all_data_from_lib(lib)