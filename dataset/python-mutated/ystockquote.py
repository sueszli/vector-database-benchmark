try:
    from urllib.request import Request, urlopen
    from urllib.parse import urlencode
except ImportError:
    from urllib2 import Request, urlopen
    from urllib import urlencode

def _request(symbol, stat):
    if False:
        while True:
            i = 10
    url = 'http://finance.yahoo.com/d/quotes.csv?s=%s&f=%s' % (symbol, stat)
    req = Request(url)
    resp = urlopen(req)
    return str(resp.read().decode('utf-8').strip())

def get_all(symbol):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get all available quote data for the given ticker symbol.\n    Returns a dictionary.\n    '
    values = _request(symbol, 'l1c1va2xj1b4j4dyekjm3m4rr5p5p6s7').split(',')
    return dict(price=values[0], change=values[1], volume=values[2], avg_daily_volume=values[3], stock_exchange=values[4], market_cap=values[5], book_value=values[6], ebitda=values[7], dividend_per_share=values[8], dividend_yield=values[9], earnings_per_share=values[10], fifty_two_week_high=values[11], fifty_two_week_low=values[12], fifty_day_moving_avg=values[13], two_hundred_day_moving_avg=values[14], price_earnings_ratio=values[15], price_earnings_growth_ratio=values[16], price_sales_ratio=values[17], price_book_ratio=values[18], short_ratio=values[19])

def get_price(symbol):
    if False:
        print('Hello World!')
    return _request(symbol, 'l1')

def get_change(symbol):
    if False:
        i = 10
        return i + 15
    return _request(symbol, 'c1')

def get_volume(symbol):
    if False:
        while True:
            i = 10
    return _request(symbol, 'v')

def get_avg_daily_volume(symbol):
    if False:
        return 10
    return _request(symbol, 'a2')

def get_stock_exchange(symbol):
    if False:
        i = 10
        return i + 15
    return _request(symbol, 'x')

def get_market_cap(symbol):
    if False:
        return 10
    return _request(symbol, 'j1')

def get_book_value(symbol):
    if False:
        for i in range(10):
            print('nop')
    return _request(symbol, 'b4')

def get_ebitda(symbol):
    if False:
        while True:
            i = 10
    return _request(symbol, 'j4')

def get_dividend_per_share(symbol):
    if False:
        return 10
    return _request(symbol, 'd')

def get_dividend_yield(symbol):
    if False:
        return 10
    return _request(symbol, 'y')

def get_earnings_per_share(symbol):
    if False:
        i = 10
        return i + 15
    return _request(symbol, 'e')

def get_52_week_high(symbol):
    if False:
        while True:
            i = 10
    return _request(symbol, 'k')

def get_52_week_low(symbol):
    if False:
        return 10
    return _request(symbol, 'j')

def get_50day_moving_avg(symbol):
    if False:
        while True:
            i = 10
    return _request(symbol, 'm3')

def get_200day_moving_avg(symbol):
    if False:
        while True:
            i = 10
    return _request(symbol, 'm4')

def get_price_earnings_ratio(symbol):
    if False:
        for i in range(10):
            print('nop')
    return _request(symbol, 'r')

def get_price_earnings_growth_ratio(symbol):
    if False:
        while True:
            i = 10
    return _request(symbol, 'r5')

def get_price_sales_ratio(symbol):
    if False:
        while True:
            i = 10
    return _request(symbol, 'p5')

def get_price_book_ratio(symbol):
    if False:
        for i in range(10):
            print('nop')
    return _request(symbol, 'p6')

def get_short_ratio(symbol):
    if False:
        return 10
    return _request(symbol, 's7')

def get_historical_prices(symbol, start_date, end_date):
    if False:
        while True:
            i = 10
    "\n    Get historical prices for the given ticker symbol.\n    Date format is 'YYYY-MM-DD'\n    Returns a nested list (first item is list of column headers).\n    "
    params = urlencode({'s': symbol, 'a': int(start_date[5:7]) - 1, 'b': int(start_date[8:10]), 'c': int(start_date[0:4]), 'd': int(end_date[5:7]) - 1, 'e': int(end_date[8:10]), 'f': int(end_date[0:4]), 'g': 'd', 'ignore': '.csv'})
    url = 'http://ichart.yahoo.com/table.csv?%s' % params
    req = Request(url)
    resp = urlopen(req)
    content = str(resp.read().decode('utf-8').strip())
    days = content.splitlines()
    return [day.split(',') for day in days]