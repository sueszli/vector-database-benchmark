from zipline.api import order, symbol
from zipline.finance import commission, slippage
stocks = ['AAPL', 'MSFT']

def initialize(context):
    if False:
        for i in range(10):
            print('nop')
    context.has_ordered = False
    context.stocks = stocks
    context.set_commission(commission.PerShare(cost=0.0075, min_trade_cost=1.0))
    context.set_slippage(slippage.VolumeShareSlippage())

def handle_data(context, data):
    if False:
        while True:
            i = 10
    if not context.has_ordered:
        for stock in context.stocks:
            order(symbol(stock), 100)
        context.has_ordered = True

def _test_args():
    if False:
        for i in range(10):
            print('nop')
    "Extra arguments to use when zipline's automated tests run this example.\n    "
    import pandas as pd
    return {'start': pd.Timestamp('2008', tz='utc'), 'end': pd.Timestamp('2013', tz='utc')}