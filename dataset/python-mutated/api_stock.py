import datetime
from decimal import Decimal, getcontext
from itertools import chain
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from rqalpha.api import export_as_api
from rqalpha.apis.api_abstract import order, order_percent, order_shares, order_target_percent, order_target_value, order_to, order_value, common_rules, TUPLE_PRICE_OR_STYLE_TYPE, PRICE_OR_STYLE_TYPE
from rqalpha.apis.api_base import assure_instrument, assure_order_book_id, cal_style, calc_open_close_style
from rqalpha.const import DEFAULT_ACCOUNT_TYPE, EXECUTION_PHASE, INSTRUMENT_TYPE, ORDER_TYPE, POSITION_DIRECTION, POSITION_EFFECT, SIDE
from rqalpha.core.execution_context import ExecutionContext
from rqalpha.environment import Environment
from rqalpha.mod.rqalpha_mod_sys_risk.validators.cash_validator import is_cash_enough
from rqalpha.model.instrument import IndustryCode as industry_code
from rqalpha.model.instrument import IndustryCodeItem, Instrument
from rqalpha.model.instrument import SectorCode as sector_code
from rqalpha.model.instrument import SectorCodeItem
from rqalpha.model.order import LimitOrder, MarketOrder, Order, OrderStyle, ALGO_ORDER_STYLES
from rqalpha.utils import INST_TYPE_IN_STOCK_ACCOUNT, is_valid_price
from rqalpha.utils.arg_checker import apply_rules, verify_that
from rqalpha.utils.datetime_func import to_date
from rqalpha.utils.exception import RQInvalidArgument
from rqalpha.utils.i18n import gettext as _
from rqalpha.utils.logger import user_system_log
getcontext().prec = 10
export_as_api(industry_code, name='industry_code')
export_as_api(sector_code, name='sector_code')
KSH_MIN_AMOUNT = 200

def _get_account_position_ins(id_or_ins):
    if False:
        i = 10
        return i + 15
    ins = assure_instrument(id_or_ins)
    try:
        account = Environment.get_instance().portfolio.accounts[DEFAULT_ACCOUNT_TYPE.STOCK]
    except KeyError:
        raise KeyError(_(u'order_book_id: {order_book_id} needs stock account, please set and try again!').format(order_book_id=ins.order_book_id))
    position = account.get_position(ins.order_book_id, POSITION_DIRECTION.LONG)
    return (account, position, ins)

def _round_order_quantity(ins, quantity) -> int:
    if False:
        print('Hello World!')
    if ins.type == 'CS' and ins.board_type == 'KSH':
        return 0 if abs(quantity) < KSH_MIN_AMOUNT else int(quantity)
    else:
        round_lot = ins.round_lot
        try:
            return int(Decimal(quantity) / Decimal(round_lot)) * round_lot
        except ValueError:
            raise

def _get_order_style_price(order_book_id, style):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(style, LimitOrder):
        return style.get_limit_price()
    env = Environment.get_instance()
    if isinstance(style, MarketOrder):
        return env.data_proxy.get_last_price(order_book_id)
    if isinstance(style, ALGO_ORDER_STYLES):
        (price, _) = env.data_proxy.get_algo_bar(order_book_id, style, env.calendar_dt)
        return price
    raise RuntimeError(f'no support {style} order style')

def _submit_order(ins, amount, side, position_effect, style, current_quantity, auto_switch_order_value):
    if False:
        for i in range(10):
            print('nop')
    env = Environment.get_instance()
    if isinstance(style, LimitOrder) and np.isnan(style.get_limit_price()):
        raise RQInvalidArgument(_(u'Limit order price should not be nan.'))
    price = env.data_proxy.get_last_price(ins.order_book_id)
    if not is_valid_price(price):
        user_system_log.warn(_(u'Order Creation Failed: [{order_book_id}] No market data').format(order_book_id=ins.order_book_id))
        return
    if side == SIDE.BUY and current_quantity != -amount or (side == SIDE.SELL and current_quantity != abs(amount)):
        amount = _round_order_quantity(ins, amount)
    if amount == 0:
        user_system_log.warn(_(u'Order Creation Failed: 0 order quantity, order_book_id={order_book_id}').format(order_book_id=ins.order_book_id))
        return
    order = Order.__from_create__(ins.order_book_id, abs(amount), side, style, position_effect)
    if side == SIDE.BUY and auto_switch_order_value:
        (account, position, ins) = _get_account_position_ins(ins)
        if not is_cash_enough(env, order, account.cash):
            user_system_log.warn(_('insufficient cash, use all remaining cash({}) to create order').format(account.cash))
            return _order_value(account, position, ins, account.cash, style)
    return env.submit_order(order)

def _order_shares(ins, amount, style, quantity, auto_switch_order_value):
    if False:
        return 10
    (side, position_effect) = (SIDE.BUY, POSITION_EFFECT.OPEN) if amount > 0 else (SIDE.SELL, POSITION_EFFECT.CLOSE)
    return _submit_order(ins, amount, side, position_effect, style, quantity, auto_switch_order_value)

def _order_value(account, position, ins, cash_amount, style):
    if False:
        i = 10
        return i + 15
    env = Environment.get_instance()
    if cash_amount > 0:
        cash_amount = min(cash_amount, account.cash)
    if isinstance(style, LimitOrder):
        price = style.get_limit_price()
    else:
        price = env.data_proxy.get_last_price(ins.order_book_id)
        if not is_valid_price(price):
            user_system_log.warn(_(u'Order Creation Failed: [{order_book_id}] No market data').format(order_book_id=ins.order_book_id))
            return
    amount = int(Decimal(cash_amount) / Decimal(price))
    round_lot = int(ins.round_lot)
    if cash_amount > 0:
        amount = _round_order_quantity(ins, amount)
        while amount > 0:
            expected_transaction_cost = env.get_order_transaction_cost(Order.__from_create__(ins.order_book_id, amount, SIDE.BUY, LimitOrder(price), POSITION_EFFECT.OPEN))
            if amount * price + expected_transaction_cost <= cash_amount:
                break
            amount -= round_lot
        else:
            user_system_log.warn(_(u'Order Creation Failed: 0 order quantity, order_book_id={order_book_id}').format(order_book_id=ins.order_book_id))
            return
    if amount < 0:
        amount = max(amount, -position.closable)
    return _order_shares(ins, amount, style, position.quantity, auto_switch_order_value=False)

@order_shares.register(INST_TYPE_IN_STOCK_ACCOUNT)
def stock_order_shares(id_or_ins, amount, price_or_style=None, price=None, style=None):
    if False:
        for i in range(10):
            print('nop')
    auto_switch_order_value = Environment.get_instance().config.mod.sys_accounts.auto_switch_order_value
    (account, position, ins) = _get_account_position_ins(id_or_ins)
    return _order_shares(assure_instrument(id_or_ins), amount, cal_style(price, style, price_or_style), position.quantity, auto_switch_order_value)

@order_value.register(INST_TYPE_IN_STOCK_ACCOUNT)
def stock_order_value(id_or_ins, cash_amount, price_or_style=None, price=None, style=None):
    if False:
        print('Hello World!')
    (account, position, ins) = _get_account_position_ins(id_or_ins)
    return _order_value(account, position, ins, cash_amount, cal_style(price, style, price_or_style))

@order_percent.register(INST_TYPE_IN_STOCK_ACCOUNT)
def stock_order_percent(id_or_ins, percent, price_or_style=None, price=None, style=None):
    if False:
        i = 10
        return i + 15
    (account, position, ins) = _get_account_position_ins(id_or_ins)
    return _order_value(account, position, ins, account.total_value * percent, cal_style(price, style, price_or_style))

@order_target_value.register(INST_TYPE_IN_STOCK_ACCOUNT)
def stock_order_target_value(id_or_ins, cash_amount, price_or_style=None, price=None, style=None):
    if False:
        return 10
    (account, position, ins) = _get_account_position_ins(id_or_ins)
    (open_style, close_style) = calc_open_close_style(price, style, price_or_style)
    if cash_amount == 0:
        return _submit_order(ins, position.closable, SIDE.SELL, POSITION_EFFECT.CLOSE, close_style, position.quantity, False)
    _delta = cash_amount - position.market_value
    _style = open_style if _delta > 0 else close_style
    return _order_value(account, position, ins, _delta, _style)

@order_target_percent.register(INST_TYPE_IN_STOCK_ACCOUNT)
def stock_order_target_percent(id_or_ins, percent, price_or_style=None, price=None, style=None):
    if False:
        return 10
    (account, position, ins) = _get_account_position_ins(id_or_ins)
    (open_style, close_style) = calc_open_close_style(price, style, price_or_style)
    if percent == 0:
        return _submit_order(ins, position.closable, SIDE.SELL, POSITION_EFFECT.CLOSE, close_style, position.quantity, False)
    _delta = account.total_value * percent - position.market_value
    _style = open_style if _delta > 0 else close_style
    return _order_value(account, position, ins, _delta, _style)

@order.register(INST_TYPE_IN_STOCK_ACCOUNT)
def stock_order(order_book_id, quantity, price_or_style=None, price=None, style=None):
    if False:
        return 10
    result_order = stock_order_shares(order_book_id, quantity, price, style, price_or_style)
    if result_order:
        return [result_order]
    return []

@order_to.register(INST_TYPE_IN_STOCK_ACCOUNT)
def stock_order_to(order_book_id, quantity, price_or_style=None, price=None, style=None):
    if False:
        i = 10
        return i + 15
    position = Environment.get_instance().portfolio.get_position(order_book_id, POSITION_DIRECTION.LONG)
    (open_style, close_style) = calc_open_close_style(price, style, price_or_style)
    quantity = quantity - position.quantity
    _style = open_style if quantity > 0 else close_style
    result_order = stock_order_shares(order_book_id, quantity, price, _style, price_or_style)
    if result_order:
        return [result_order]
    return []

@export_as_api
@ExecutionContext.enforce_phase(EXECUTION_PHASE.OPEN_AUCTION, EXECUTION_PHASE.ON_BAR, EXECUTION_PHASE.ON_TICK, EXECUTION_PHASE.SCHEDULED, EXECUTION_PHASE.GLOBAL)
@apply_rules(verify_that('id_or_ins').is_valid_stock(), verify_that('amount').is_number(), *common_rules)
def order_lots(id_or_ins, amount, price_or_style=None, price=None, style=None):
    if False:
        i = 10
        return i + 15
    "\n    指定手数发送买/卖单。如有需要落单类型当做一个参量传入，如果忽略掉落单类型，那么默认是市价单（market order）。\n\n    :param id_or_ins: 下单标的物\n    :param int amount: 下单量, 正数代表买入，负数代表卖出。将会根据一手xx股来向下调整到一手的倍数，比如中国A股就是调整成100股的倍数。\n    :param price_or_style: 默认为None，表示市价单，可设置价格，表示限价单，也可以直接设置订单类型，有如下选项：MarketOrder、LimitOrder、\n                            TWAPOrder、VWAPOrder\n\n    :example:\n\n    .. code-block:: python\n\n        #买入20手的平安银行股票，并且发送市价单：\n        order_lots('000001.XSHE', 20)\n        #买入10手平安银行股票，并且发送限价单，价格为￥10：\n        order_lots('000001.XSHE', 10, price_or_style=LimitOrder(10))\n\n    "
    auto_switch_order_value = Environment.get_instance().config.mod.sys_accounts.auto_switch_order_value
    (account, position, ins) = _get_account_position_ins(id_or_ins)
    return _order_shares(ins, amount * int(ins.round_lot), cal_style(price, style, price_or_style), position.quantity, auto_switch_order_value)
ORDER_TARGET_PORTFOLIO_SUPPORTED_INS_TYPES = {INSTRUMENT_TYPE.CS, INSTRUMENT_TYPE.ETF, INSTRUMENT_TYPE.LOF, INSTRUMENT_TYPE.INDX, INSTRUMENT_TYPE.CONVERTIBLE}

@export_as_api
@ExecutionContext.enforce_phase(EXECUTION_PHASE.OPEN_AUCTION, EXECUTION_PHASE.ON_BAR, EXECUTION_PHASE.ON_TICK, EXECUTION_PHASE.SCHEDULED, EXECUTION_PHASE.GLOBAL)
def order_target_portfolio(target_portfolio: Dict[str, float], price_or_styles: Dict[str, TUPLE_PRICE_OR_STYLE_TYPE]=dict({})) -> List[Order]:
    if False:
        return 10
    "\n    批量调整股票仓位至目标权重。注意：股票账户中未出现在 target_portfolio 中的资产将被平仓！\n\n    该 API 的参数 target_portfolio 为字典，key 为 order_book_id 或 instrument，value 为权重。\n    此时将根据参数 price_or_styles 中设置的价格来计算目标持仓数量并调仓。\n\n    :param target_portfolio: 目标权重字典，key 为 order_book_id，value 为权重。\n    :param price_or_styles: 目标下单价格字典，key 为 order_book_id, value 为价格或订单类型或订单类型和价格组成的 tuple\n\n    :example:\n\n    .. code-block:: python\n\n        # 调整仓位，以使平安银行和万科 A 的持仓占比分别达到 10% 和 15%, 同时发送市价单\n        order_target_portfolio({\n            '000001.XSHE': 0.1,\n            '000002.XSHE': 0.15\n        })\n\n        # 调整仓位，分别以 14 和 26 元发出限价单，目标是使平安银行和万科 A 的持仓占比分别达到 10% 和 15%\n        order_target_portfolio({\n            '000001.XSHE': 0.1,\n            '000002.XSHE': 0.15\n        }, {\n            '000001.XSHE': 14,\n            '000002.XSHE': 26,\n        })\n\n        # 调整仓位，使平安银行和万科 A 的持仓占比分别达到 10% 和 15%。\n        # 其中平安银行的平仓价为 14 元，开仓价为 15 元；万科 A 的平仓价为 26 元，开仓价为 27 元。\n        order_target_portfolio({\n            '000001.XSHE': 0.1,\n            '000002.XSHE': 0.15\n        }, {\n            '000001.XSHE': (15, 14),\n            '000002.XSHE': (27, 26)\n        })\n\n    "
    env = Environment.get_instance()
    target: Dict[str, Tuple[float, float, float, float]] = {}
    for (id_or_ins, percent) in target_portfolio.items():
        ins = assure_instrument(id_or_ins)
        if not ins:
            raise RQInvalidArgument(_('function order_target_portfolio: invalid keys of target_portfolio, expected order_book_ids or Instrument objects, got {} (type: {})').format(id_or_ins, type(id_or_ins)))
        if ins.type not in ORDER_TARGET_PORTFOLIO_SUPPORTED_INS_TYPES:
            raise RQInvalidArgument(_('function order_target_portfolio: invalid instrument type, excepted CS/ETF/LOF/INDX, got {}').format(ins.order_book_id))
        order_book_id = ins.order_book_id
        last_price = env.data_proxy.get_last_price(order_book_id)
        if not is_valid_price(last_price):
            user_system_log.warn(_(u'Order Creation Failed: [{order_book_id}] No market data').format(order_book_id=order_book_id))
            continue
        price_or_style = price_or_styles.get(ins.order_book_id)
        (open_style, close_style) = calc_open_close_style(price=None, style=None, price_or_style=price_or_style)
        if percent < 0:
            raise RQInvalidArgument(_('function order_target_portfolio: invalid values of target_portfolio, excepted float between 0 and 1, got {} (key: {})').format(percent, id_or_ins))
        target[order_book_id] = (percent, open_style, close_style, last_price)
    total_percent = sum((p for (p, *__) in target.values()))
    if total_percent > 1 and (not np.isclose(total_percent, 1)):
        raise RQInvalidArgument(_('total percent should be lower than 1, current: {}').format(total_percent))
    account = env.portfolio.accounts[DEFAULT_ACCOUNT_TYPE.STOCK]
    current_quantities = {p.order_book_id: p.quantity for p in account.get_positions() if p.direction == POSITION_DIRECTION.LONG}
    for (order_book_id, quantity) in current_quantities.items():
        if order_book_id not in target:
            env.submit_order(Order.__from_create__(order_book_id, quantity, SIDE.SELL, MarketOrder(), POSITION_EFFECT.CLOSE))
    account_value = account.total_value
    (close_orders, open_orders) = ([], [])
    for (order_book_id, (target_percent, open_style, close_style, last_price)) in target.items():
        open_price = _get_order_style_price(order_book_id, open_style)
        close_price = _get_order_style_price(order_book_id, close_style)
        if not (is_valid_price(close_price) and is_valid_price(open_price)):
            user_system_log.warn(_('Adjust position of {id_or_ins} Failed: Invalid close/open price {close_price}/{open_price}').format(id_or_ins=order_book_id, close_price=close_price, open_price=open_price))
            continue
        delta_quantity = account_value * target_percent / close_price - current_quantities.get(order_book_id, 0)
        delta_quantity = _round_order_quantity(env.data_proxy.instrument(order_book_id), delta_quantity)
        if delta_quantity == 0:
            continue
        elif delta_quantity > 0:
            (quantity, side, position_effect) = (delta_quantity, SIDE.BUY, POSITION_EFFECT.OPEN)
            order_list = open_orders
            target_style = open_style
        else:
            (quantity, side, position_effect) = (abs(delta_quantity), SIDE.SELL, POSITION_EFFECT.CLOSE)
            order_list = close_orders
            target_style = close_style
        order = Order.__from_create__(order_book_id, quantity, side, target_style, position_effect)
        if isinstance(target_style, MarketOrder):
            order.set_frozen_price(last_price)
        order_list.append(order)
    return list((env.submit_order(o) for o in chain(close_orders, open_orders)))

@export_as_api
@ExecutionContext.enforce_phase(EXECUTION_PHASE.ON_INIT, EXECUTION_PHASE.BEFORE_TRADING, EXECUTION_PHASE.OPEN_AUCTION, EXECUTION_PHASE.ON_BAR, EXECUTION_PHASE.ON_TICK, EXECUTION_PHASE.AFTER_TRADING, EXECUTION_PHASE.SCHEDULED)
@apply_rules(verify_that('order_book_id').is_valid_instrument(), verify_that('count').is_greater_than(0))
def is_suspended(order_book_id, count=1):
    if False:
        print('Hello World!')
    '\n    判断某只股票是否全天停牌。\n\n    :param order_book_id: 某只股票的代码或股票代码，可传入单只股票的order_book_id, symbol\n    :param count: 回溯获取的数据个数。默认为当前能够获取到的最近的数据\n\n    '
    dt = Environment.get_instance().calendar_dt.date()
    order_book_id = assure_order_book_id(order_book_id)
    return Environment.get_instance().data_proxy.is_suspended(order_book_id, dt, count)

@export_as_api
@ExecutionContext.enforce_phase(EXECUTION_PHASE.ON_INIT, EXECUTION_PHASE.BEFORE_TRADING, EXECUTION_PHASE.OPEN_AUCTION, EXECUTION_PHASE.ON_BAR, EXECUTION_PHASE.ON_TICK, EXECUTION_PHASE.AFTER_TRADING, EXECUTION_PHASE.SCHEDULED)
@apply_rules(verify_that('order_book_id').is_valid_instrument())
def is_st_stock(order_book_id, count=1):
    if False:
        while True:
            i = 10
    "\n    判断股票在一段时间内是否为ST股（包括ST与*ST）。\n\n    ST股是有退市风险因此风险比较大的股票，很多时候您也会希望判断自己使用的股票是否是'ST'股来避开这些风险大的股票。另外，我们目前的策略比赛也禁止了使用'ST'股。\n\n    :param order_book_id: 某只股票的代码，可传入单只股票的order_book_id, symbol\n    :param count: 回溯获取的数据个数。默认为当前能够获取到的最近的数据\n    "
    dt = Environment.get_instance().calendar_dt.date()
    order_book_id = assure_order_book_id(order_book_id)
    return Environment.get_instance().data_proxy.is_st_stock(order_book_id, dt, count)

@export_as_api
@ExecutionContext.enforce_phase(EXECUTION_PHASE.ON_INIT, EXECUTION_PHASE.BEFORE_TRADING, EXECUTION_PHASE.OPEN_AUCTION, EXECUTION_PHASE.ON_BAR, EXECUTION_PHASE.ON_TICK, EXECUTION_PHASE.AFTER_TRADING, EXECUTION_PHASE.SCHEDULED)
@apply_rules(verify_that('code').is_instance_of((str, IndustryCodeItem)))
def industry(code):
    if False:
        while True:
            i = 10
    '\n    获得属于某一行业的所有股票列表。\n\n    :param code: 行业名称或行业代码。例如，农业可填写industry_code.A01 或 \'A01\'\n\n    我们目前使用的行业分类来自于中国国家统计局的 `国民经济行业分类 <http://www.stats.gov.cn/tjsj/tjbz/hyflbz/>`_ ，可以使用这里的任何一个行业代码来调用行业的股票列表：\n\n    =========================   ===================================================\n    行业代码                      行业名称\n    =========================   ===================================================\n    A01                         农业\n    A02                         林业\n    A03                         畜牧业\n    A04                         渔业\n    A05                         农、林、牧、渔服务业\n    B06                         煤炭开采和洗选业\n    B07                         石油和天然气开采业\n    B08                         黑色金属矿采选业\n    B09                         有色金属矿采选业\n    B10                         非金属矿采选业\n    B11                         开采辅助活动\n    B12                         其他采矿业\n    C13                         农副食品加工业\n    C14                         食品制造业\n    C15                         酒、饮料和精制茶制造业\n    C16                         烟草制品业\n    C17                         纺织业\n    C18                         纺织服装、服饰业\n    C19                         皮革、毛皮、羽毛及其制品和制鞋业\n    C20                         木材加工及木、竹、藤、棕、草制品业\n    C21                         家具制造业\n    C22                         造纸及纸制品业\n    C23                         印刷和记录媒介复制业\n    C24                         文教、工美、体育和娱乐用品制造业\n    C25                         石油加工、炼焦及核燃料加工业\n    C26                         化学原料及化学制品制造业\n    C27                         医药制造业\n    C28                         化学纤维制造业\n    C29                         橡胶和塑料制品业\n    C30                         非金属矿物制品业\n    C31                         黑色金属冶炼及压延加工业\n    C32                         有色金属冶炼和压延加工业\n    C33                         金属制品业\n    C34                         通用设备制造业\n    C35                         专用设备制造业\n    C36                         汽车制造业\n    C37                         铁路、船舶、航空航天和其它运输设备制造业\n    C38                         电气机械及器材制造业\n    C39                         计算机、通信和其他电子设备制造业\n    C40                         仪器仪表制造业\n    C41                         其他制造业\n    C42                         废弃资源综合利用业\n    C43                         金属制品、机械和设备修理业\n    D44                         电力、热力生产和供应业\n    D45                         燃气生产和供应业\n    D46                         水的生产和供应业\n    E47                         房屋建筑业\n    E48                         土木工程建筑业\n    E49                         建筑安装业\n    E50                         建筑装饰和其他建筑业\n    F51                         批发业\n    F52                         零售业\n    G53                         铁路运输业\n    G54                         道路运输业\n    G55                         水上运输业\n    G56                         航空运输业\n    G57                         管道运输业\n    G58                         装卸搬运和运输代理业\n    G59                         仓储业\n    G60                         邮政业\n    H61                         住宿业\n    H62                         餐饮业\n    I63                         电信、广播电视和卫星传输服务\n    I64                         互联网和相关服务\n    I65                         软件和信息技术服务业\n    J66                         货币金融服务\n    J67                         资本市场服务\n    J68                         保险业\n    J69                         其他金融业\n    K70                         房地产业\n    L71                         租赁业\n    L72                         商务服务业\n    M73                         研究和试验发展\n    M74                         专业技术服务业\n    M75                         科技推广和应用服务业\n    N76                         水利管理业\n    N77                         生态保护和环境治理业\n    N78                         公共设施管理业\n    O79                         居民服务业\n    O80                         机动车、电子产品和日用产品修理业\n    O81                         其他服务业\n    P82                         教育\n    Q83                         卫生\n    Q84                         社会工作\n    R85                         新闻和出版业\n    R86                         广播、电视、电影和影视录音制作业\n    R87                         文化艺术业\n    R88                         体育\n    R89                         娱乐业\n    S90                         综合\n    =========================   ===================================================\n\n    :example:\n\n    ..  code-block:: python3\n        :linenos:\n\n        def init(context):\n            stock_list = industry(\'A01\')\n            logger.info("农业股票列表：" + str(stock_list))\n\n        #INITINFO 农业股票列表：[\'600354.XSHG\', \'601118.XSHG\', \'002772.XSHE\', \'600371.XSHG\', \'600313.XSHG\', \'600672.XSHG\', \'600359.XSHG\', \'300143.XSHE\', \'002041.XSHE\', \'600762.XSHG\', \'600540.XSHG\', \'300189.XSHE\', \'600108.XSHG\', \'300087.XSHE\', \'600598.XSHG\', \'000998.XSHE\', \'600506.XSHG\']\n\n    '
    if isinstance(code, IndustryCodeItem):
        code = code.code
    else:
        code = to_industry_code(code)
    cs_instruments = Environment.get_instance().data_proxy.all_instruments((INSTRUMENT_TYPE.CS,))
    return [i.order_book_id for i in cs_instruments if i.industry_code == code]

@export_as_api
@ExecutionContext.enforce_phase(EXECUTION_PHASE.ON_INIT, EXECUTION_PHASE.BEFORE_TRADING, EXECUTION_PHASE.OPEN_AUCTION, EXECUTION_PHASE.ON_BAR, EXECUTION_PHASE.ON_TICK, EXECUTION_PHASE.AFTER_TRADING, EXECUTION_PHASE.SCHEDULED)
@apply_rules(verify_that('code').is_instance_of((str, SectorCodeItem)))
def sector(code):
    if False:
        for i in range(10):
            print('nop')
    '\n    获得属于某一板块的所有股票列表。\n\n    :param code: 板块名称或板块代码。例如，能源板块可填写\'Energy\'、\'能源\'或sector_code.Energy\n\n    目前支持的板块分类如下，其取值参考自MSCI发布的全球行业标准分类:\n\n    =========================   =========================   ==============================================================================\n    板块代码                      中文板块名称                  英文板块名称\n    =========================   =========================   ==============================================================================\n    Energy                      能源                         energy\n    Materials                   原材料                        materials\n    ConsumerDiscretionary       非必需消费品                   consumer discretionary\n    ConsumerStaples             必需消费品                    consumer staples\n    HealthCare                  医疗保健                      health care\n    Financials                  金融                         financials\n    InformationTechnology       信息技术                      information technology\n    TelecommunicationServices   电信服务                      telecommunication services\n    Utilities                   公共服务                      utilities\n    Industrials                 工业                         industrials\n    =========================   =========================   ==============================================================================\n\n    :example:\n\n    ..  code-block:: python3\n        :linenos:\n\n        def init(context):\n            ids1 = sector("consumer discretionary")\n            ids2 = sector("非必需消费品")\n            ids3 = sector("ConsumerDiscretionary")\n            assert ids1 == ids2 and ids1 == ids3\n            logger.info(ids1)\n        #INIT INFO\n        #[\'002045.XSHE\', \'603099.XSHG\', \'002486.XSHE\', \'002536.XSHE\', \'300100.XSHE\', \'600633.XSHG\', \'002291.XSHE\', ..., \'600233.XSHG\']\n    '
    if isinstance(code, SectorCodeItem):
        code = code.name
    else:
        code = to_sector_name(code)
    cs_instruments = Environment.get_instance().data_proxy.all_instruments((INSTRUMENT_TYPE.CS,))
    return [i.order_book_id for i in cs_instruments if i.sector_code == code]

@export_as_api
@apply_rules(verify_that('order_book_id').is_valid_instrument(), verify_that('start_date').is_valid_date(ignore_none=False))
def get_dividend(order_book_id, start_date):
    if False:
        for i in range(10):
            print('nop')
    "\n    获取某只股票到策略当前日期前一天的分红情况（包含起止日期）。\n\n    :param order_book_id: 股票代码\n    :param start_date: 开始日期，需要早于策略当前日期\n\n    =========================   ===================================================\n    fields                      字段名\n    =========================   ===================================================\n    announcement_date           分红宣布日\n    book_closure_date           股权登记日\n    dividend_cash_before_tax    税前分红\n    ex_dividend_date            除权除息日\n    payable_date                分红到帐日\n    round_lot                   分红最小单位\n    =========================   ===================================================\n\n    :example:\n\n    获取平安银行2013-01-04 到策略当前日期前一天的分红数据:\n\n    ..  code-block:: python3\n        :linenos:\n\n        get_dividend('000001.XSHE', start_date='20130104')\n        #[Out]\n        #array([(20130614, 20130619, 20130620, 20130620,  1.7 , 10),\n        #       (20140606, 20140611, 20140612, 20140612,  1.6 , 10),\n        #       (20150407, 20150410, 20150413, 20150413,  1.74, 10),\n        #       (20160608, 20160615, 20160616, 20160616,  1.53, 10)],\n        #      dtype=[('announcement_date', '<u4'), ('book_closure_date', '<u4'), ('ex_dividend_date', '<u4'), ('payable_date', '<u4'), ('dividend_cash_before_tax', '<f8'), ('round_lot', '<u4')])\n\n    "
    env = Environment.get_instance()
    dt = env.trading_dt.date() - datetime.timedelta(days=1)
    start_date = to_date(start_date)
    if start_date > dt:
        raise RQInvalidArgument(_(u'in get_dividend, start_date {} is later than the previous test day {}').format(start_date, dt))
    order_book_id = assure_order_book_id(order_book_id)
    array = env.data_proxy.get_dividend(order_book_id)
    if array is None:
        return None
    sd = start_date.year * 10000 + start_date.month * 100 + start_date.day
    ed = dt.year * 10000 + dt.month * 100 + dt.day
    return array[(array['announcement_date'] >= sd) & (array['announcement_date'] <= ed)]

def to_industry_code(s):
    if False:
        while True:
            i = 10
    for (__, v) in industry_code.__dict__.items():
        if isinstance(v, IndustryCodeItem):
            if v.name == s:
                return v.code
    return s

def to_sector_name(s):
    if False:
        for i in range(10):
            print('nop')
    for (__, v) in sector_code.__dict__.items():
        if isinstance(v, SectorCodeItem):
            if v.cn == s or v.en == s or v.name == s:
                return v.name
    return s