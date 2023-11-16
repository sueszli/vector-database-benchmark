"""
A simple Pipeline algorithm that longs the top 3 stocks by RSI and shorts
the bottom 3 each day.
"""
from six import viewkeys
from zipline.api import attach_pipeline, date_rules, order_target_percent, pipeline_output, record, schedule_function
from zipline.finance import commission, slippage
from zipline.pipeline import Pipeline
from zipline.pipeline.factors import RSI

def make_pipeline():
    if False:
        print('Hello World!')
    rsi = RSI()
    return Pipeline(columns={'longs': rsi.top(3), 'shorts': rsi.bottom(3)})

def rebalance(context, data):
    if False:
        for i in range(10):
            print('nop')
    pipeline_data = context.pipeline_data
    all_assets = pipeline_data.index
    longs = all_assets[pipeline_data.longs]
    shorts = all_assets[pipeline_data.shorts]
    record(universe_size=len(all_assets))
    one_third = 1.0 / 3.0
    for asset in longs:
        order_target_percent(asset, one_third)
    for asset in shorts:
        order_target_percent(asset, -one_third)
    portfolio_assets = longs | shorts
    positions = context.portfolio.positions
    for asset in viewkeys(positions) - set(portfolio_assets):
        if data.can_trade(asset):
            order_target_percent(asset, 0)

def initialize(context):
    if False:
        print('Hello World!')
    attach_pipeline(make_pipeline(), 'my_pipeline')
    schedule_function(rebalance, date_rules.every_day())
    context.set_commission(commission.PerShare(cost=0.0075, min_trade_cost=1.0))
    context.set_slippage(slippage.VolumeShareSlippage())

def before_trading_start(context, data):
    if False:
        for i in range(10):
            print('nop')
    context.pipeline_data = pipeline_output('my_pipeline')

def _test_args():
    if False:
        for i in range(10):
            print('nop')
    "\n    Extra arguments to use when zipline's automated tests run this example.\n\n    Notes for testers:\n\n    Gross leverage should be roughly 2.0 on every day except the first.\n    Net leverage should be roughly 2.0 on every day except the first.\n\n    Longs Count should always be 3 after the first day.\n    Shorts Count should be 3 after the first day, except on 2013-10-30, when it\n    dips to 2 for a day because DELL is delisted.\n    "
    import pandas as pd
    return {'start': pd.Timestamp('2013-10-07', tz='utc'), 'end': pd.Timestamp('2013-11-30', tz='utc'), 'capital_base': 100000}