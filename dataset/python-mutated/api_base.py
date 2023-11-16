from __future__ import division
import types
from datetime import date, datetime
from typing import Callable, List, Optional, Union, Iterable
import pandas as pd
import numpy as np
import six
from rqalpha.apis import names
from rqalpha.environment import Environment
from rqalpha.core.execution_context import ExecutionContext
from rqalpha.utils import is_valid_price
from rqalpha.utils.exception import RQInvalidArgument
from rqalpha.utils.i18n import gettext as _
from rqalpha.utils.arg_checker import apply_rules, verify_that
from rqalpha.api import export_as_api
from rqalpha.utils.logger import user_log as logger, user_system_log, user_print
from rqalpha.model.instrument import Instrument
from rqalpha.model.tick import TickObject
from rqalpha.const import EXECUTION_PHASE, ORDER_STATUS, SIDE, POSITION_EFFECT, ORDER_TYPE, MATCHING_TYPE, RUN_TYPE, POSITION_DIRECTION, DEFAULT_ACCOUNT_TYPE
from rqalpha.model.order import Order, MarketOrder, LimitOrder, OrderStyle, VWAPOrder, TWAPOrder
from rqalpha.core.events import EVENT, Event
from rqalpha.core.strategy_context import StrategyContext
from rqalpha.portfolio.position import Position
export_as_api(logger, name='logger')
export_as_api(user_print, name='print')
export_as_api(LimitOrder, name='LimitOrder')
export_as_api(MarketOrder, name='MarketOrder')
export_as_api(VWAPOrder, name='VWAPOrder')
export_as_api(TWAPOrder, name='TWAPOrder')
export_as_api(ORDER_STATUS, name='ORDER_STATUS')
export_as_api(SIDE, name='SIDE')
export_as_api(POSITION_EFFECT, name='POSITION_EFFECT')
export_as_api(POSITION_DIRECTION, name='POSITION_DIRECTION')
export_as_api(ORDER_TYPE, name='ORDER_TYPE')
export_as_api(RUN_TYPE, name='RUN_TYPE')
export_as_api(MATCHING_TYPE, name='MATCHING_TYPE')
export_as_api(EVENT, name='EVENT')

def assure_instrument(id_or_ins):
    if False:
        return 10
    if isinstance(id_or_ins, Instrument):
        return id_or_ins
    elif isinstance(id_or_ins, six.string_types):
        return Environment.get_instance().data_proxy.instrument(id_or_ins)
    else:
        raise RQInvalidArgument(_(u'unsupported order_book_id type'))

def assure_order_book_id(id_or_ins):
    if False:
        while True:
            i = 10
    return assure_instrument(id_or_ins).order_book_id

def cal_style(price, style, price_or_style=None):
    if False:
        for i in range(10):
            print('nop')
    if price_or_style is None:
        if price:
            price_or_style = price
        if style:
            price_or_style = style
    if price_or_style is None:
        return MarketOrder()
    if not isinstance(price_or_style, (int, float, OrderStyle)):
        raise RQInvalidArgument(f'price or style or price_or_style type no support. {price_or_style}')
    if isinstance(price_or_style, OrderStyle):
        return price_or_style
    return LimitOrder(price_or_style)

def calc_open_close_style(price, style, price_or_style):
    if False:
        print('Hello World!')
    if isinstance(price_or_style, tuple):
        _length = len(price_or_style)
        if _length == 0:
            (o, c) = (None, None)
        elif _length == 1:
            (o, c) = (price_or_style[0], price_or_style[0])
        else:
            (o, c) = (price_or_style[0], price_or_style[1])
        open_style = cal_style(price, style, o)
        close_style = cal_style(price, style, c)
    else:
        open_style = cal_style(price, style, price_or_style)
        close_style = open_style
    return (open_style, close_style)

@export_as_api
@ExecutionContext.enforce_phase(EXECUTION_PHASE.BEFORE_TRADING, EXECUTION_PHASE.OPEN_AUCTION, EXECUTION_PHASE.ON_BAR, EXECUTION_PHASE.ON_TICK, EXECUTION_PHASE.AFTER_TRADING, EXECUTION_PHASE.SCHEDULED)
def get_open_orders():
    if False:
        i = 10
        return i + 15
    '\n    获取当日未成交订单数据\n    '
    broker = Environment.get_instance().broker
    return [o for o in broker.get_open_orders() if o.position_effect != POSITION_EFFECT.EXERCISE]

@export_as_api
@apply_rules(verify_that('id_or_ins').is_valid_instrument(), verify_that('amount').is_number().is_greater_than(0), verify_that('side').is_in([SIDE.BUY, SIDE.SELL]))
def submit_order(id_or_ins, amount, side, price=None, position_effect=None):
    if False:
        print('Hello World!')
    "\n    通用下单函数，策略可以通过该函数自由选择参数下单。\n\n    :param id_or_ins: 下单标的物\n    :param amount: 下单量，需为正数\n    :param side: 多空方向\n    :param price: 下单价格，默认为None，表示市价单\n    :param position_effect: 开平方向，交易股票不需要该参数\n    :example:\n\n    .. code-block:: python\n\n        # 购买 2000 股的平安银行股票，并以市价单发送：\n        submit_order('000001.XSHE', 2000, SIDE.BUY)\n        # 平 10 份 RB1812 多方向的今仓，并以 4000 的价格发送限价单\n        submit_order('RB1812', 10, SIDE.SELL, price=4000, position_effect=POSITION_EFFECT.CLOSE_TODAY)\n\n    "
    order_book_id = assure_order_book_id(id_or_ins)
    env = Environment.get_instance()
    if env.config.base.run_type != RUN_TYPE.BACKTEST and env.get_instrument(order_book_id).type == 'Future':
        if '88' in order_book_id:
            raise RQInvalidArgument(_(u'Main Future contracts[88] are not supported in paper trading.'))
        if '99' in order_book_id:
            raise RQInvalidArgument(_(u'Index Future contracts[99] are not supported in paper trading.'))
    style = cal_style(price, None)
    market_price = env.get_last_price(order_book_id)
    if not is_valid_price(market_price):
        user_system_log.warn(_(u'Order Creation Failed: [{order_book_id}] No market data').format(order_book_id=order_book_id))
        return
    amount = int(amount)
    order = Order.__from_create__(order_book_id=order_book_id, quantity=amount, side=side, style=style, position_effect=position_effect)
    if order.type == ORDER_TYPE.MARKET:
        order.set_frozen_price(market_price)
    if env.can_submit_order(order):
        env.broker.submit_order(order)
        return order

@export_as_api
@ExecutionContext.enforce_phase(EXECUTION_PHASE.BEFORE_TRADING, EXECUTION_PHASE.OPEN_AUCTION, EXECUTION_PHASE.ON_BAR, EXECUTION_PHASE.ON_TICK, EXECUTION_PHASE.AFTER_TRADING, EXECUTION_PHASE.SCHEDULED, EXECUTION_PHASE.GLOBAL)
@apply_rules(verify_that('order').is_instance_of(Order))
def cancel_order(order):
    if False:
        for i in range(10):
            print('nop')
    '\n    撤单\n\n    :param order: 需要撤销的order对象\n    '
    env = Environment.get_instance()
    if env.can_cancel_order(order):
        env.broker.cancel_order(order)
    return order

@export_as_api
@ExecutionContext.enforce_phase(EXECUTION_PHASE.ON_INIT, EXECUTION_PHASE.BEFORE_TRADING, EXECUTION_PHASE.OPEN_AUCTION, EXECUTION_PHASE.OPEN_AUCTION, EXECUTION_PHASE.ON_BAR, EXECUTION_PHASE.ON_TICK, EXECUTION_PHASE.AFTER_TRADING, EXECUTION_PHASE.SCHEDULED)
@apply_rules(verify_that('id_or_symbols').are_valid_instruments())
def update_universe(id_or_symbols):
    if False:
        return 10
    "\n    该方法用于更新现在关注的证券的集合（e.g.：股票池）。PS：会在下一个bar事件触发时候产生（新的关注的股票池更新）效果。并且update_universe会是覆盖（overwrite）的操作而不是在已有的股票池的基础上进行增量添加。比如已有的股票池为['000001.XSHE', '000024.XSHE']然后调用了update_universe(['000030.XSHE'])之后，股票池就会变成000030.XSHE一个股票了，随后的数据更新也只会跟踪000030.XSHE这一个股票了。\n\n    :param id_or_symbols: 标的物\n\n    "
    if isinstance(id_or_symbols, (six.string_types, Instrument)):
        id_or_symbols = [id_or_symbols]
    order_book_ids = set((assure_order_book_id(order_book_id) for order_book_id in id_or_symbols))
    if order_book_ids != Environment.get_instance().get_universe():
        Environment.get_instance().update_universe(order_book_ids)

@export_as_api
@ExecutionContext.enforce_phase(EXECUTION_PHASE.ON_INIT, EXECUTION_PHASE.BEFORE_TRADING, EXECUTION_PHASE.OPEN_AUCTION, EXECUTION_PHASE.ON_BAR, EXECUTION_PHASE.ON_TICK, EXECUTION_PHASE.AFTER_TRADING, EXECUTION_PHASE.SCHEDULED)
@apply_rules(verify_that('id_or_symbols').are_valid_instruments())
def subscribe(id_or_symbols):
    if False:
        i = 10
        return i + 15
    '\n    订阅合约行情。\n\n    在日级别回测中不需要订阅合约。\n\n    在分钟回测中，若策略只设置了股票账户则不需要订阅合约；若设置了期货账户，则需要订阅策略关注的期货合约，框架会根据订阅的期货合约品种触发对应交易时间的 handle_bar。为了方便起见，也可以以直接订阅主力连续合约。\n\n    在 tick 回测中，策略需要订阅每一个关注的股票/期货合约，框架会根据订阅池触发对应标的的 handle_tick。\n\n    :param id_or_symbols: 标的物\n\n    '
    current_universe = Environment.get_instance().get_universe()
    if isinstance(id_or_symbols, six.string_types):
        order_book_id = instruments(id_or_symbols).order_book_id
        current_universe.add(order_book_id)
    elif isinstance(id_or_symbols, Instrument):
        current_universe.add(id_or_symbols.order_book_id)
    elif isinstance(id_or_symbols, Iterable):
        for item in id_or_symbols:
            current_universe.add(assure_order_book_id(item))
    else:
        raise RQInvalidArgument(_(u'unsupported order_book_id type'))
    verify_that('id_or_symbols')._are_valid_instruments('subscribe', id_or_symbols)
    Environment.get_instance().update_universe(current_universe)

@export_as_api
@ExecutionContext.enforce_phase(EXECUTION_PHASE.ON_INIT, EXECUTION_PHASE.BEFORE_TRADING, EXECUTION_PHASE.OPEN_AUCTION, EXECUTION_PHASE.ON_BAR, EXECUTION_PHASE.ON_TICK, EXECUTION_PHASE.AFTER_TRADING, EXECUTION_PHASE.SCHEDULED)
@apply_rules(verify_that('id_or_symbols').are_valid_instruments())
def unsubscribe(id_or_symbols):
    if False:
        while True:
            i = 10
    '\n    取消订阅合约行情。取消订阅会导致合约池内合约的减少，如果当前合约池中没有任何合约，则策略直接退出。\n\n    :param id_or_symbols: 标的物\n\n    '
    current_universe = Environment.get_instance().get_universe()
    if isinstance(id_or_symbols, six.string_types):
        order_book_id = instruments(id_or_symbols).order_book_id
        current_universe.discard(order_book_id)
    elif isinstance(id_or_symbols, Instrument):
        current_universe.discard(id_or_symbols.order_book_id)
    elif isinstance(id_or_symbols, Iterable):
        for item in id_or_symbols:
            i = assure_order_book_id(item)
            current_universe.discard(i)
    else:
        raise RQInvalidArgument(_(u'unsupported order_book_id type'))
    Environment.get_instance().update_universe(current_universe)

@export_as_api
@ExecutionContext.enforce_phase(EXECUTION_PHASE.ON_INIT, EXECUTION_PHASE.BEFORE_TRADING, EXECUTION_PHASE.OPEN_AUCTION, EXECUTION_PHASE.ON_BAR, EXECUTION_PHASE.ON_TICK, EXECUTION_PHASE.AFTER_TRADING, EXECUTION_PHASE.SCHEDULED)
@apply_rules(verify_that('date').is_valid_date(ignore_none=True), verify_that('tenor').is_in(names.VALID_TENORS, ignore_none=True))
def get_yield_curve(date=None, tenor=None):
    if False:
        while True:
            i = 10
    "\n    获取某个国家市场指定日期的收益率曲线水平。\n\n    数据为2002年至今的中债国债收益率曲线，来源于中央国债登记结算有限责任公司。\n\n    :param date: 查询日期，默认为策略当前日期前一天\n    :param tenor: 标准期限，'0S' - 隔夜，'1M' - 1个月，'1Y' - 1年，默认为全部期限\n\n    :example:\n\n    ..  code-block:: python3\n        :linenos:\n\n        [In]\n        get_yield_curve('20130104')\n\n        [Out]\n                        0S      1M      2M      3M      6M      9M      1Y      2Y          2013-01-04  0.0196  0.0253  0.0288  0.0279  0.0280  0.0283  0.0292  0.0310\n\n                        3Y      4Y   ...        6Y      7Y      8Y      9Y     10Y          2013-01-04  0.0314  0.0318   ...    0.0342  0.0350  0.0353  0.0357  0.0361\n        ...\n    "
    env = Environment.get_instance()
    trading_date = env.trading_dt.date()
    yesterday = env.data_proxy.get_previous_trading_date(trading_date)
    if date is None:
        date = yesterday
    else:
        date = pd.Timestamp(date)
        if date > yesterday:
            raise RQInvalidArgument('get_yield_curve: {} >= now({})'.format(date, yesterday))
    return env.data_proxy.get_yield_curve(start_date=date, end_date=date, tenor=tenor)

@export_as_api
@ExecutionContext.enforce_phase(EXECUTION_PHASE.BEFORE_TRADING, EXECUTION_PHASE.OPEN_AUCTION, EXECUTION_PHASE.ON_BAR, EXECUTION_PHASE.ON_TICK, EXECUTION_PHASE.AFTER_TRADING, EXECUTION_PHASE.SCHEDULED)
@apply_rules(verify_that('order_book_id', pre_check=True).is_listed_instrument(), verify_that('bar_count').is_instance_of(int).is_greater_than(0), verify_that('frequency', pre_check=True).is_valid_frequency(), verify_that('fields').are_valid_fields(names.VALID_HISTORY_FIELDS, ignore_none=True), verify_that('skip_suspended').is_instance_of(bool), verify_that('include_now').is_instance_of(bool), verify_that('adjust_type').is_in({'pre', 'none', 'post'}))
def history_bars(order_book_id, bar_count, frequency, fields=None, skip_suspended=True, include_now=False, adjust_type='pre'):
    if False:
        while True:
            i = 10
    "\n    获取指定合约的历史 k 线行情，同时支持日以及分钟历史数据。不能在init中调用。\n\n    日回测获取分钟历史数据：不支持\n\n    日回测获取日历史数据\n\n    =========================   ===================================================\n    调用时间                      返回数据\n    =========================   ===================================================\n    T日before_trading            T-1日day bar\n    T日handle_bar                T日day bar\n    =========================   ===================================================\n\n    分钟回测获取日历史数据\n\n    =========================   ===================================================\n    调用时间                      返回数据\n    =========================   ===================================================\n    T日before_trading            T-1日day bar\n    T日handle_bar                T-1日day bar\n    =========================   ===================================================\n\n    分钟回测获取分钟历史数据\n\n    =========================   ===================================================\n    调用时间                      返回数据\n    =========================   ===================================================\n    T日before_trading            T-1日最后一个minute bar\n    T日handle_bar                T日当前minute bar\n    =========================   ===================================================\n\n    :param order_book_id: 合约代码\n    :param bar_count: 获取的历史数据数量，必填项\n    :param frequency: 获取数据什么样的频率进行。'1d'、'1m' 和 '1w' 分别表示每日、每分钟和每周，必填项\n    :param fields: 返回数据字段。必填项。见下方列表。\n    :param skip_suspended: 是否跳过停牌数据\n    :param include_now: 是否包含当前数据\n    :param adjust_type: 复权类型，默认为前复权 pre；可选 pre, none, post\n\n    =========================   ===================================================\n    fields                      字段名\n    =========================   ===================================================\n    datetime                    时间戳\n    open                        开盘价\n    high                        最高价\n    low                         最低价\n    close                       收盘价\n    volume                      成交量\n    total_turnover              成交额\n    open_interest               持仓量（期货专用）\n    basis_spread                期现差（股指期货专用）\n    settlement                  结算价（期货日线专用）\n    prev_settlement             结算价（期货日线专用）\n    =========================   ===================================================\n\n    :example:\n\n    获取最近5天的日线收盘价序列（策略当前日期为20160706）:\n\n    ..  code-block:: python3\n        :linenos:\n\n        [In]\n        logger.info(history_bars('000002.XSHE', 5, '1d', 'close'))\n        [Out]\n        [ 8.69  8.7   8.71  8.81  8.81]\n    "
    order_book_id = assure_order_book_id(order_book_id)
    env = Environment.get_instance()
    dt = env.calendar_dt
    if frequency[-1] == 'm' and env.config.base.frequency == '1d':
        raise RQInvalidArgument('can not get minute history in day back test')
    if frequency[-1] == 'd' and frequency != '1d':
        raise RQInvalidArgument('invalid frequency')
    if adjust_type not in {'pre', 'post', 'none'}:
        raise RuntimeError('invalid adjust_type')
    if frequency == '1d':
        sys_frequency = Environment.get_instance().config.base.frequency
        if sys_frequency in ['1m', 'tick'] and (not include_now) and (ExecutionContext.phase() != EXECUTION_PHASE.AFTER_TRADING) or ExecutionContext.phase() in (EXECUTION_PHASE.BEFORE_TRADING, EXECUTION_PHASE.OPEN_AUCTION):
            dt = env.data_proxy.get_previous_trading_date(env.trading_dt.date())
            include_now = False
        if sys_frequency == '1d':
            include_now = False
    if fields is None:
        fields = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    return env.data_proxy.history_bars(order_book_id, bar_count, frequency, fields, dt, skip_suspended=skip_suspended, include_now=include_now, adjust_type=adjust_type, adjust_orig=env.trading_dt)

@export_as_api
@apply_rules(verify_that('order_book_id', pre_check=True).is_listed_instrument(), verify_that('count').is_instance_of(int).is_greater_than(0))
def history_ticks(order_book_id, count):
    if False:
        i = 10
        return i + 15
    '\n    获取指定合约历史（不晚于当前时间的）tick 对象，仅支持在 tick 级别的策略（回测、模拟交易、实盘）中调用。\n\n    :param order_book_id: 合约代码\n    :param count: 获取的 tick 数量\n\n    '
    env = Environment.get_instance()
    sys_frequency = env.config.base.frequency
    if sys_frequency == '1d':
        raise RuntimeError('history_ticks does not support day bar backtest.')
    order_book_id = assure_order_book_id(order_book_id)
    dt = env.calendar_dt
    return env.data_proxy.history_ticks(order_book_id, count, dt)

@export_as_api
@ExecutionContext.enforce_phase(EXECUTION_PHASE.ON_INIT, EXECUTION_PHASE.BEFORE_TRADING, EXECUTION_PHASE.OPEN_AUCTION, EXECUTION_PHASE.ON_BAR, EXECUTION_PHASE.ON_TICK, EXECUTION_PHASE.AFTER_TRADING, EXECUTION_PHASE.SCHEDULED)
@apply_rules(verify_that('type').are_valid_fields(names.VALID_INSTRUMENT_TYPES, ignore_none=True), verify_that('date').is_valid_date(ignore_none=True))
def all_instruments(type=None, date=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    获取某个国家市场的所有合约信息。使用者可以通过这一方法很快地对合约信息有一个快速了解，目前仅支持中国市场。\n\n    :param type: 需要查询合约类型，例如：type='CS'代表股票。默认是所有类型\n    :param date: 查询时间点\n\n    其中type参数传入的合约类型和对应的解释如下：\n\n    =========================   ===================================================\n    合约类型                      说明\n    =========================   ===================================================\n    CS                          Common Stock, 即股票\n    ETF                         Exchange Traded Fund, 即交易所交易基金\n    LOF                         Listed Open-Ended Fund，即上市型开放式基金\n    INDX                        Index, 即指数\n    Future                      Futures，即期货，包含股指、国债和商品期货\n    =========================   ===================================================\n\n    "
    env = Environment.get_instance()
    if date is None:
        dt = env.trading_dt
    else:
        dt = pd.Timestamp(date).to_pydatetime()
        dt = min(dt, env.trading_dt)
    if type is not None:
        if isinstance(type, six.string_types):
            type = [type]
        types = set()
        for t in type:
            if t == 'Stock':
                types.add('CS')
            elif t == 'Fund':
                types.update(['ETF', 'LOF'])
            else:
                types.add(t)
    else:
        types = None
    result = env.data_proxy.all_instruments(types, dt)
    if types is not None and len(types) == 1:
        data = []
        for i in result:
            instrument_dic = {k: v for (k, v) in i.__dict__.items() if not k.startswith('_')}
            data.append(instrument_dic)
        return pd.DataFrame(data)
    return pd.DataFrame([[i.order_book_id, i.symbol, i.type, i.listed_date, i.de_listed_date] for i in result], columns=['order_book_id', 'symbol', 'type', 'listed_date', 'de_listed_date'])

@export_as_api
@ExecutionContext.enforce_phase(EXECUTION_PHASE.ON_INIT, EXECUTION_PHASE.BEFORE_TRADING, EXECUTION_PHASE.OPEN_AUCTION, EXECUTION_PHASE.ON_BAR, EXECUTION_PHASE.ON_TICK, EXECUTION_PHASE.AFTER_TRADING, EXECUTION_PHASE.SCHEDULED)
@apply_rules(verify_that('id_or_symbols').is_instance_of((str, Iterable)))
def instruments(id_or_symbols):
    if False:
        while True:
            i = 10
    "\n    获取某个国家市场内一个或多个合约的详细信息。目前仅支持中国市场。\n\n    :param id_or_symbols: 合约代码或者合约代码列表\n\n    :example:\n\n    *   获取单一股票合约的详细信息:\n\n        ..  code-block:: python3\n            :linenos:\n\n            [In]instruments('000001.XSHE')\n            [Out]\n            Instrument(order_book_id=000001.XSHE, symbol=平安银行, abbrev_symbol=PAYH, listed_date=19910403, de_listed_date=null, board_type=MainBoard, sector_code_name=金融, sector_code=Financials, round_lot=100, exchange=XSHE, special_type=Normal, status=Active)\n\n    *   获取多个股票合约的详细信息:\n\n        ..  code-block:: python3\n            :linenos:\n\n            [In]instruments(['000001.XSHE', '000024.XSHE'])\n            [Out]\n            [Instrument(order_book_id=000001.XSHE, symbol=平安银行, abbrev_symbol=PAYH, listed_date=19910403, de_listed_date=null, board_type=MainBoard, sector_code_name=金融, sector_code=Financials, round_lot=100, exchange=XSHE, special_type=Normal, status=Active), Instrument(order_book_id=000024.XSHE, symbol=招商地产, abbrev_symbol=ZSDC, listed_date=19930607, de_listed_date=null, board_type=MainBoard, sector_code_name=金融, sector_code=Financials, round_lot=100, exchange=XSHE, special_type=Normal, status=Active)]\n\n    *   获取合约已上市天数:\n\n        ..  code-block:: python\n            :linenos:\n\n            instruments('000001.XSHE').days_from_listed()\n\n    *   获取合约距离到期天数:\n\n        ..  code-block:: python\n            :linenos:\n\n            instruments('IF1701').days_to_expire()\n    "
    return Environment.get_instance().data_proxy.instruments(id_or_symbols)

@export_as_api
@apply_rules(verify_that('start_date').is_valid_date(ignore_none=False), verify_that('end_date').is_valid_date(ignore_none=False))
def get_trading_dates(start_date, end_date):
    if False:
        print('Hello World!')
    '\n    获取某个国家市场的交易日列表（起止日期加入判断）。目前仅支持中国市场。\n\n    :param start_date: 开始日期\n    :param end_date: 结束如期\n\n    '
    return Environment.get_instance().data_proxy.get_trading_dates(start_date, end_date)

@export_as_api
@apply_rules(verify_that('date').is_valid_date(ignore_none=False), verify_that('n').is_instance_of(int).is_greater_or_equal_than(1))
def get_previous_trading_date(date, n=1):
    if False:
        for i in range(10):
            print('nop')
    "\n    获取指定日期的之前的第 n 个交易日。\n\n    :param date: 指定日期\n    :param n:\n\n    :example:\n\n    ..  code-block:: python3\n        :linenos:\n\n        [In]get_previous_trading_date(date='2016-05-02')\n        [Out]\n        [datetime.date(2016, 4, 29)]\n    "
    return Environment.get_instance().data_proxy.get_previous_trading_date(date, n)

@export_as_api
@apply_rules(verify_that('date').is_valid_date(ignore_none=False), verify_that('n').is_instance_of(int).is_greater_or_equal_than(1))
def get_next_trading_date(date, n=1):
    if False:
        while True:
            i = 10
    "\n    获取指定日期之后的第 n 个交易日\n\n    :param date: 指定日期\n    :param n:\n\n    :example:\n\n    ..  code-block:: python3\n        :linenos:\n\n        [In]get_next_trading_date(date='2016-05-01')\n        [Out]\n        [datetime.date(2016, 5, 3)]\n    "
    return Environment.get_instance().data_proxy.get_next_trading_date(date, n)

@export_as_api
@ExecutionContext.enforce_phase(EXECUTION_PHASE.BEFORE_TRADING, EXECUTION_PHASE.OPEN_AUCTION, EXECUTION_PHASE.ON_BAR, EXECUTION_PHASE.ON_TICK, EXECUTION_PHASE.AFTER_TRADING, EXECUTION_PHASE.SCHEDULED)
@apply_rules(verify_that('id_or_symbol').is_valid_instrument())
def current_snapshot(id_or_symbol):
    if False:
        return 10
    "\n    获得当前市场快照数据。只能在日内交易阶段调用，获取当日调用时点的市场快照数据。\n    市场快照数据记录了每日从开盘到当前的数据信息，可以理解为一个动态的day bar数据。\n    在目前分钟回测中，快照数据为当日所有分钟线累积而成，一般情况下，最后一个分钟线获取到的快照数据应当与当日的日线行情保持一致。\n    需要注意，在实盘模拟中，该函数返回的是调用当时的市场快照情况，所以在同一个handle_bar中不同时点调用可能返回的数据不同。\n    如果当日截止到调用时候对应股票没有任何成交，那么snapshot中的close, high, low, last几个价格水平都将以0表示。\n\n    :param d_or_symbol: 合约代码或简称\n\n    :example:\n\n    在handle_bar中调用该函数，假设策略当前时间是20160104 09:33:\n\n    ..  code-block:: python3\n        :linenos:\n\n        [In]\n        logger.info(current_snapshot('000001.XSHE'))\n        [Out]\n        2016-01-04 09:33:00.00  INFO\n        Snapshot(order_book_id: '000001.XSHE', datetime: datetime.datetime(2016, 1, 4, 9, 33), open: 10.0, high: 10.025, low: 9.9667, last: 9.9917, volume: 2050320, total_turnover: 20485195, prev_close: 9.99)\n    "
    env = Environment.get_instance()
    frequency = env.config.base.frequency
    order_book_id = assure_order_book_id(id_or_symbol)
    dt = env.calendar_dt
    if env.config.base.run_type == RUN_TYPE.BACKTEST:
        if ExecutionContext.phase() == EXECUTION_PHASE.BEFORE_TRADING:
            dt = env.data_proxy.get_previous_trading_date(env.trading_dt.date())
            return env.data_proxy.current_snapshot(order_book_id, '1d', dt)
        elif ExecutionContext.phase() == EXECUTION_PHASE.AFTER_TRADING:
            return env.data_proxy.current_snapshot(order_book_id, '1d', dt)
    return env.data_proxy.current_snapshot(order_book_id, frequency, dt)

@export_as_api
def get_positions():
    if False:
        for i in range(10):
            print('nop')
    '\n    获取所有持仓对象列表。\n\n    :example:\n\n    ..  code-block:: python3\n\n        [In] get_positions()\n        [Out]\n        [StockPosition(order_book_id=000001.XSHE, direction=LONG, quantity=1000, market_value=19520.0, trading_pnl=0.0, position_pnl=0),\n        StockPosition(order_book_id=RB2112, direction=SHORT, quantity=2, market_value=-111580.0, trading_pnl=0.0, position_pnl=0)]\n\n    '
    portfolio = Environment.get_instance().portfolio
    return portfolio.get_positions()

@export_as_api
@apply_rules(verify_that('direction').is_in([POSITION_DIRECTION.LONG, POSITION_DIRECTION.SHORT]))
def get_position(order_book_id, direction=POSITION_DIRECTION.LONG):
    if False:
        while True:
            i = 10
    "\n    获取某个标的的持仓对象。\n\n    :param order_book_id: 标的编号\n    :param direction: 持仓方向\n\n    :example:\n\n    ..  code-block:: python3\n\n        [In] get_position('000001.XSHE', POSITION_DIRECTION.LONG)\n        [Out]\n        StockPosition(order_book_id=000001.XSHE, direction=LONG, quantity=268600, market_value=4995960.0, trading_pnl=0.0, position_pnl=0)\n\n    "
    portfolio = Environment.get_instance().portfolio
    return portfolio.get_position(order_book_id, direction)

@export_as_api
@ExecutionContext.enforce_phase(EXECUTION_PHASE.ON_INIT)
@apply_rules(verify_that('event_type').is_instance_of(EVENT), verify_that('handler').is_instance_of(types.FunctionType))
def subscribe_event(event_type, handler):
    if False:
        while True:
            i = 10
    '\n    订阅框架内部事件，注册事件处理函数\n\n    :param event_type: 事件类型\n    :param handler: 处理函数\n\n    '
    env = Environment.get_instance()
    user_strategy = env.user_strategy
    env.event_bus.add_listener(event_type, user_strategy.wrap_user_event_handler(handler), user=True)

@export_as_api
def symbol(order_book_id, sep=', '):
    if False:
        while True:
            i = 10
    if isinstance(order_book_id, six.string_types):
        return '{}[{}]'.format(order_book_id, Environment.get_instance().get_instrument(order_book_id).symbol)
    else:
        s = sep.join((symbol(item) for item in order_book_id))
        return s

@export_as_api
@ExecutionContext.enforce_phase(EXECUTION_PHASE.OPEN_AUCTION, EXECUTION_PHASE.ON_BAR, EXECUTION_PHASE.ON_TICK, EXECUTION_PHASE.SCHEDULED, EXECUTION_PHASE.GLOBAL)
def deposit(account_type: str, amount: float, receiving_days: int=0):
    if False:
        while True:
            i = 10
    '\n    入金（增加账户资金）\n\n    :param account_type: 账户类型\n    :param amount: 入金金额\n    :param receiving_days: 入金到账天数，0 表示立刻到账，1 表示资金在下一个交易日盘前到账\n    :return: None\n    '
    env = Environment.get_instance()
    return env.portfolio.deposit_withdraw(account_type, amount, receiving_days)

@export_as_api
@ExecutionContext.enforce_phase(EXECUTION_PHASE.OPEN_AUCTION, EXECUTION_PHASE.ON_BAR, EXECUTION_PHASE.ON_TICK, EXECUTION_PHASE.SCHEDULED, EXECUTION_PHASE.GLOBAL)
@apply_rules(verify_that('account_type').is_in(DEFAULT_ACCOUNT_TYPE), verify_that('amount').is_number())
def withdraw(account_type, amount):
    if False:
        return 10
    '\n    出金（减少账户资金）\n\n    :param account_type: 账户类型\n    :param amount: 减少金额\n    :return: None\n    '
    env = Environment.get_instance()
    return env.portfolio.deposit_withdraw(account_type, amount * -1)

@export_as_api
@ExecutionContext.enforce_phase(EXECUTION_PHASE.OPEN_AUCTION, EXECUTION_PHASE.ON_BAR, EXECUTION_PHASE.ON_TICK, EXECUTION_PHASE.SCHEDULED, EXECUTION_PHASE.GLOBAL)
@apply_rules(verify_that('account_type').is_in(DEFAULT_ACCOUNT_TYPE), verify_that('amount', pre_check=True).is_instance_of((int, float)).is_greater_than(0))
def finance(amount, account_type=DEFAULT_ACCOUNT_TYPE.STOCK):
    if False:
        for i in range(10):
            print('nop')
    '\n    融资\n\n    :param amount: 融资金额\n    :param account_type: 融资账户\n    :return: None\n    '
    env = Environment.get_instance()
    return env.portfolio.finance_repay(amount, account_type)

@export_as_api
@ExecutionContext.enforce_phase(EXECUTION_PHASE.OPEN_AUCTION, EXECUTION_PHASE.ON_BAR, EXECUTION_PHASE.ON_TICK, EXECUTION_PHASE.SCHEDULED)
@apply_rules(verify_that('account_type').is_in(DEFAULT_ACCOUNT_TYPE), verify_that('amount', pre_check=True).is_instance_of((int, float)).is_greater_than(0))
def repay(amount, account_type=DEFAULT_ACCOUNT_TYPE.STOCK):
    if False:
        return 10
    '\n    还款\n\n    :param amount: 还款金额\n    :param account_type: 还款账户\n    :return: None\n    '
    env = Environment.get_instance()
    return env.portfolio.finance_repay(amount * -1, account_type)