from zipline.api import order, record, symbol
from zipline.finance import commission, slippage

def initialize(context):
    if False:
        while True:
            i = 10
    context.asset = symbol('AAPL')
    context.set_commission(commission.PerShare(cost=0.0075, min_trade_cost=1.0))
    context.set_slippage(slippage.VolumeShareSlippage())

def handle_data(context, data):
    if False:
        while True:
            i = 10
    order(context.asset, 10)
    record(AAPL=data.current(context.asset, 'price'))

def analyze(context=None, results=None):
    if False:
        for i in range(10):
            print('nop')
    import matplotlib.pyplot as plt
    ax1 = plt.subplot(211)
    results.portfolio_value.plot(ax=ax1)
    ax1.set_ylabel('Portfolio value (USD)')
    ax2 = plt.subplot(212, sharex=ax1)
    results.AAPL.plot(ax=ax2)
    ax2.set_ylabel('AAPL price (USD)')
    plt.gcf().set_size_inches(18, 8)
    plt.show()

def _test_args():
    if False:
        print('Hello World!')
    "Extra arguments to use when zipline's automated tests run this example.\n    "
    import pandas as pd
    return {'start': pd.Timestamp('2014-01-01', tz='utc'), 'end': pd.Timestamp('2014-11-01', tz='utc')}