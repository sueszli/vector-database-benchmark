__author__ = 'chengzhi'
import copy
import json
import warnings
from tqsdk.diff import _get_obj
from tqsdk.entity import Entity
from tqsdk.utils import _query_for_init, _generate_uuid

class QuotesEntity(Entity):

    def __init__(self, api):
        if False:
            print('Hello World!')
        self._api = api
        self._not_send_init_query = True

    def __iter__(self):
        if False:
            return 10
        message = "\n            不推荐使用 api._data['quotes'] 获取全部合约，该使用方法会在 20201101 之后的版本中放弃维护。\n            需要注意：\n            * 在同步代码中，初次使用 api._data['quotes'] 获取全部合约会产生一个耗时很长的查询。\n            * 在协程中，api._data['quotes'] 这种用法不支持使用。\n            请尽快修改使用新的接口，参考链接 http://doc.shinnytech.com/tqsdk/reference/tqsdk.api.html#tqsdk.api.TqApi.query_quotes\n        "
        warnings.warn(message, DeprecationWarning, stacklevel=3)
        self._api._logger.warning('deprecation', content="Deprecation Warning in api._data['quotes']")
        if self._not_send_init_query and self._api._stock:
            self._not_send_init_query = False
            q = _query_for_init()
            self._api.query_graphql(q, {}, _generate_uuid('PYSDK_quote'))
        return super().__iter__()

class Quote(Entity):
    """ Quote 是一个行情对象 """

    def __init__(self, api):
        if False:
            i = 10
            return i + 15
        self._api = api
        self.datetime: str = ''
        self.ask_price1: float = float('nan')
        self.ask_volume1: int = 0
        self.bid_price1: float = float('nan')
        self.bid_volume1: int = 0
        self.ask_price2: float = float('nan')
        self.ask_volume2: int = 0
        self.bid_price2: float = float('nan')
        self.bid_volume2: int = 0
        self.ask_price3: float = float('nan')
        self.ask_volume3: int = 0
        self.bid_price3: float = float('nan')
        self.bid_volume3: int = 0
        self.ask_price4: float = float('nan')
        self.ask_volume4: int = 0
        self.bid_price4: float = float('nan')
        self.bid_volume4: int = 0
        self.ask_price5: float = float('nan')
        self.ask_volume5: int = 0
        self.bid_price5: float = float('nan')
        self.bid_volume5: int = 0
        self.last_price: float = float('nan')
        self.highest: float = float('nan')
        self.lowest: float = float('nan')
        self.open: float = float('nan')
        self.close: float = float('nan')
        self.average: float = float('nan')
        self.volume: int = 0
        self.amount: float = float('nan')
        self.open_interest: int = 0
        self.settlement: float = float('nan')
        self.upper_limit: float = float('nan')
        self.lower_limit: float = float('nan')
        self.pre_open_interest: int = 0
        self.pre_settlement: float = float('nan')
        self.pre_close: float = float('nan')
        self.price_tick: float = float('nan')
        self.price_decs: int = 0
        self.volume_multiple: int = 0
        self.max_limit_order_volume: int = 0
        self.max_market_order_volume: int = 0
        self.min_limit_order_volume: int = 0
        self.min_market_order_volume: int = 0
        self.underlying_symbol: str = ''
        self.strike_price: float = float('nan')
        self.ins_class: str = ''
        self.instrument_id: str = ''
        self.instrument_name: str = ''
        self.exchange_id: str = ''
        self.expired: bool = False
        self.trading_time: TradingTime = TradingTime(self._api)
        self.expire_datetime: float = float('nan')
        self.delivery_year: int = 0
        self.delivery_month: int = 0
        self.last_exercise_datetime: float = float('nan')
        self.exercise_year: int = 0
        self.exercise_month: int = 0
        self.option_class: str = ''
        self.exercise_type: str = ''
        self.product_id: str = ''
        self.iopv: float = float('nan')
        self.public_float_share_quantity: int = 0
        self.stock_dividend_ratio: list = []
        self.cash_dividend_ratio: list = []
        self.expire_rest_days: int = float('nan')

    def _instance_entity(self, path):
        if False:
            while True:
                i = 10
        super(Quote, self)._instance_entity(path)
        self.trading_time = copy.copy(self.trading_time)
        self.trading_time._instance_entity(path + ['trading_time'])

    @property
    def underlying_quote(self):
        if False:
            return 10
        '\n        标的合约 underlying_symbol 所指定的合约对象，若没有标的合约则为 None\n\n        :return: 标的指定的 :py:class:`~tqsdk.objs.Quote` 对象\n        '
        if self.underlying_symbol:
            return self._api.get_quote(self.underlying_symbol)
        return None

    def __await__(self):
        if False:
            while True:
                i = 10
        assert self._task
        return self._task.__await__()

class TradingTime(Entity):
    """ TradingTime 是一个交易时间对象
        它不是一个可单独使用的类，而是用于定义 Quote 的 trading_time 字段的类型

        (每个连续的交易时间段是一个列表，包含两个字符串元素，分别为这个时间段的起止点)"""

    def __init__(self, api):
        if False:
            return 10
        self._api = api
        self.day: list = []
        self.night: list = []

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return json.dumps({'day': self.day, 'night': self.night})

class TradingStatus(Entity):
    """ TradingStatus 是一个交易状态对象 """

    def __init__(self, api):
        if False:
            for i in range(10):
                print('nop')
        self._api = api
        self.symbol: str = ''
        self.trade_status: str = ''

    def __await__(self):
        if False:
            while True:
                i = 10
        assert self._task
        return self._task.__await__()

class Kline(Entity):
    """ Kline 是一个K线对象 """

    def __init__(self, api):
        if False:
            print('Hello World!')
        self._api = api
        self.datetime: int = 0
        self.open: float = float('nan')
        self.high: float = float('nan')
        self.low: float = float('nan')
        self.close: float = float('nan')
        self.volume: int = 0
        self.open_oi: int = 0
        self.close_oi: int = 0

class Tick(Entity):
    """ Tick 是一个tick对象 """

    def __init__(self, api):
        if False:
            i = 10
            return i + 15
        self._api = api
        self.datetime: int = 0
        self.last_price: float = float('nan')
        self.average: float = float('nan')
        self.highest: float = float('nan')
        self.lowest: float = float('nan')
        self.ask_price1: float = float('nan')
        self.ask_volume1: int = 0
        self.bid_price1: float = float('nan')
        self.bid_volume1: int = 0
        self.ask_price2: float = float('nan')
        self.ask_volume2: int = 0
        self.bid_price2: float = float('nan')
        self.bid_volume2: int = 0
        self.ask_price3: float = float('nan')
        self.ask_volume3: int = 0
        self.bid_price3: float = float('nan')
        self.bid_volume3: int = 0
        self.ask_price4: float = float('nan')
        self.ask_volume4: int = 0
        self.bid_price4: float = float('nan')
        self.bid_volume4: int = 0
        self.ask_price5: float = float('nan')
        self.ask_volume5: int = 0
        self.bid_price5: float = float('nan')
        self.bid_volume5: int = 0
        self.volume: int = 0
        self.amount: float = float('nan')
        self.open_interest: int = 0

class Account(Entity):
    """ Account 是一个账户对象 """

    def __init__(self, api):
        if False:
            print('Hello World!')
        self._api = api
        self.currency: str = ''
        self.pre_balance: float = float('nan')
        self.static_balance: float = float('nan')
        self.balance: float = float('nan')
        self.available: float = float('nan')
        self.ctp_balance: float = float('nan')
        self.ctp_available: float = float('nan')
        self.float_profit: float = float('nan')
        self.position_profit: float = float('nan')
        self.close_profit: float = float('nan')
        self.frozen_margin: float = float('nan')
        self.margin: float = float('nan')
        self.frozen_commission: float = float('nan')
        self.commission: float = float('nan')
        self.frozen_premium: float = float('nan')
        self.premium: float = float('nan')
        self.deposit: float = float('nan')
        self.withdraw: float = float('nan')
        self.risk_ratio: float = float('nan')
        self.market_value: float = float('nan')

class Position(Entity):
    """ Position 是一个持仓对象 """

    def __init__(self, api):
        if False:
            print('Hello World!')
        self._api = api
        self.exchange_id: str = ''
        self.instrument_id: str = ''
        self.pos_long_his: int = 0
        self.pos_long_today: int = 0
        self.pos_short_his: int = 0
        self.pos_short_today: int = 0
        self.volume_long_today: int = 0
        self.volume_long_his: int = 0
        self.volume_long: int = 0
        self.volume_long_frozen_today: int = 0
        self.volume_long_frozen_his: int = 0
        self.volume_long_frozen: int = 0
        self.volume_short_today: int = 0
        self.volume_short_his: int = 0
        self.volume_short: int = 0
        self.volume_short_frozen_today: int = 0
        self.volume_short_frozen_his: int = 0
        self.volume_short_frozen: int = 0
        self.open_price_long: float = float('nan')
        self.open_price_short: float = float('nan')
        self.open_cost_long: float = float('nan')
        self.open_cost_short: float = float('nan')
        self.position_price_long: float = float('nan')
        self.position_price_short: float = float('nan')
        self.position_cost_long: float = float('nan')
        self.position_cost_short: float = float('nan')
        self.float_profit_long: float = float('nan')
        self.float_profit_short: float = float('nan')
        self.float_profit: float = float('nan')
        self.position_profit_long: float = float('nan')
        self.position_profit_short: float = float('nan')
        self.position_profit: float = float('nan')
        self.margin_long: float = float('nan')
        self.margin_short: float = float('nan')
        self.margin: float = float('nan')
        self.market_value_long: float = float('nan')
        self.market_value_short: float = float('nan')
        self.market_value: float = float('nan')
        self.pos: int = 0
        self.pos_long: int = 0
        self.pos_short: int = 0

    @property
    def orders(self):
        if False:
            while True:
                i = 10
        '\n        与此持仓相关的且目前委托单状态为ALIVE的开仓/平仓挂单\n\n        :return: dict, 其中每个元素的key为委托单ID, value为 :py:class:`~tqsdk.objs.Order`\n        '
        tdict = _get_obj(self._api._data, ['trade', self._path[1], 'orders'])
        fts = {order_id: order for (order_id, order) in tdict.items() if not order_id.startswith('_') and order.instrument_id == self.instrument_id and (order.exchange_id == self.exchange_id) and (order.status == 'ALIVE')}
        return fts

class Order(Entity):
    """ Order 是一个委托单对象 """

    def __init__(self, api):
        if False:
            i = 10
            return i + 15
        self._api = api
        self.order_id: str = ''
        self.exchange_order_id: str = ''
        self.exchange_id: str = ''
        self.instrument_id: str = ''
        self.direction: str = ''
        self.offset: str = ''
        self.volume_orign: int = 0
        self.volume_left: int = 0
        self.limit_price: float = float('nan')
        self.price_type: str = ''
        self.volume_condition: str = ''
        self.time_condition: str = ''
        self.insert_date_time: int = 0
        self.last_msg: str = ''
        self.status: str = ''
        self.is_dead: bool = None
        self.is_online: bool = None
        self.is_error: bool = None
        self.trade_price: float = float('nan')
        self._this_session = False

    @property
    def trade_records(self):
        if False:
            i = 10
            return i + 15
        '\n        成交记录\n\n        :return: dict, 其中每个元素的key为成交ID, value为 :py:class:`~tqsdk.objs.Trade`\n        '
        tdict = _get_obj(self._api._data, ['trade', self._path[1], 'trades'])
        fts = {trade_id: trade for (trade_id, trade) in tdict.items() if not trade_id.startswith('_') and trade.order_id == self.order_id}
        return fts

class Trade(Entity):
    """ Trade 是一个成交对象 """

    def __init__(self, api):
        if False:
            print('Hello World!')
        self._api = api
        self.order_id: str = ''
        self.trade_id: str = ''
        self.exchange_trade_id: str = ''
        self.exchange_id: str = ''
        self.instrument_id: str = ''
        self.direction: str = ''
        self.offset: str = ''
        self.price: float = float('nan')
        self.volume: int = 0
        self.trade_date_time: int = 0

class RiskManagementRule(Entity):

    def __init__(self, api):
        if False:
            i = 10
            return i + 15
        self._api = api
        self.user_id = ''
        self.exchange_id = ''
        self.enable = False
        self.self_trade = SelfTradeRule(self._api)
        self.frequent_cancellation = FrequentCancellationRule(self._api)
        self.trade_position_ratio = TradePositionRatioRule(self._api)

    def _instance_entity(self, path):
        if False:
            i = 10
            return i + 15
        super(RiskManagementRule, self)._instance_entity(path)
        self.self_trade = copy.copy(self.self_trade)
        self.self_trade._instance_entity(path + ['self_trade'])
        self.frequent_cancellation = copy.copy(self.frequent_cancellation)
        self.frequent_cancellation._instance_entity(path + ['frequent_cancellation'])
        self.trade_position_ratio = copy.copy(self.trade_position_ratio)
        self.trade_position_ratio._instance_entity(path + ['trade_position_ratio'])

class SelfTradeRule(Entity):
    """自成交风控规则"""

    def __init__(self, api):
        if False:
            for i in range(10):
                print('nop')
        self._api = api
        self.count_limit = 0

    def __repr__(self):
        if False:
            print('Hello World!')
        return json.dumps({'count_limit': self.count_limit})

class FrequentCancellationRule(Entity):
    """频繁报撤单风控规则"""

    def __init__(self, api):
        if False:
            for i in range(10):
                print('nop')
        self._api = api
        self.insert_order_count_limit = 0
        self.cancel_order_count_limit = 0
        self.cancel_order_percent_limit = float('nan')

    def __repr__(self):
        if False:
            while True:
                i = 10
        return json.dumps({'insert_order_count_limit': self.insert_order_count_limit, 'cancel_order_count_limit': self.cancel_order_count_limit, 'cancel_order_percent_limit': self.cancel_order_percent_limit})

class TradePositionRatioRule(Entity):
    """成交持仓比风控规则"""

    def __init__(self, api):
        if False:
            i = 10
            return i + 15
        self._api = api
        self.trade_units_limit = 0
        self.trade_position_ratio_limit = float('nan')

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return json.dumps({'trade_units_limit': self.trade_units_limit, 'trade_position_ratio_limit': self.trade_position_ratio_limit})

class RiskManagementData(Entity):

    def __init__(self, api):
        if False:
            return 10
        self._api = api
        self.user_id = ''
        self.exchange_id = ''
        self.instrument_id = ''
        self.self_trade = SelfTrade(self._api)
        self.frequent_cancellation = FrequentCancellation(self._api)
        self.trade_position_ratio = TradePositionRatio(self._api)

    def _instance_entity(self, path):
        if False:
            for i in range(10):
                print('nop')
        super(RiskManagementData, self)._instance_entity(path)
        self.self_trade = copy.copy(self.self_trade)
        self.self_trade._instance_entity(path + ['self_trade'])
        self.frequent_cancellation = copy.copy(self.frequent_cancellation)
        self.frequent_cancellation._instance_entity(path + ['frequent_cancellation'])
        self.trade_position_ratio = copy.copy(self.trade_position_ratio)
        self.trade_position_ratio._instance_entity(path + ['trade_position_ratio'])

class SelfTrade(Entity):
    """自成交情况"""

    def __init__(self, api):
        if False:
            return 10
        self._api = api
        self.highest_buy_price = float('nan')
        self.lowest_sell_price = float('nan')
        self.self_trade_count = 0
        self.rejected_count = 0

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return json.dumps({'highest_buy_price': self.highest_buy_price, 'lowest_sell_price': self.lowest_sell_price, 'self_trade_count': self.self_trade_count, 'rejected_count': self.rejected_count})

class FrequentCancellation(Entity):
    """频繁报撤单情况"""

    def __init__(self, api):
        if False:
            return 10
        self._api = api
        self.insert_order_count = 0
        self.cancel_order_count = 0
        self.cancel_order_percent = float('nan')
        self.rejected_count = 0

    def __repr__(self):
        if False:
            while True:
                i = 10
        return json.dumps({'insert_order_count': self.insert_order_count, 'cancel_order_count': self.cancel_order_count, 'cancel_order_percent': self.cancel_order_percent, 'rejected_count': self.rejected_count})

class TradePositionRatio(Entity):
    """成交持仓比情况"""

    def __init__(self, api):
        if False:
            print('Hello World!')
        self._api = api
        self.trade_units = 0
        self.net_position_units = 0
        self.trade_position_ratio = float('nan')
        self.rejected_count = 0

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return json.dumps({'trade_units': self.trade_units, 'net_position_units': self.net_position_units, 'trade_position_ratio': self.trade_position_ratio, 'rejected_count': self.rejected_count})

class SecurityAccount(Entity):
    """ SecurityAccount 是一个股票账户对象"""

    def __init__(self, api):
        if False:
            print('Hello World!')
        self._api = api
        self.user_id: str = ''
        self.currency: str = 'CNY'
        self.market_value: float = float('nan')
        self.asset: float = float('nan')
        self.asset_his: float = float('nan')
        self.available: float = float('nan')
        self.available_his: float = float('nan')
        self.cost: float = float('nan')
        self.drawable: float = float('nan')
        self.deposit: float = float('nan')
        self.withdraw: float = float('nan')
        self.buy_frozen_balance: float = float('nan')
        self.buy_frozen_fee: float = float('nan')
        self.buy_balance_today: float = float('nan')
        self.buy_fee_today: float = float('nan')
        self.sell_balance_today: float = float('nan')
        self.sell_fee_today: float = float('nan')
        self.hold_profit: float = float('nan')
        self.float_profit_today: float = float('nan')
        self.real_profit_today: float = float('nan')
        self.profit_today: float = float('nan')
        self.profit_rate_today: float = float('nan')
        self.dividend_balance_today: float = float('nan')

class SecurityPosition(Entity):
    """ SecurityPosition 是一个股票账户持仓对象 """

    def __init__(self, api):
        if False:
            for i in range(10):
                print('nop')
        self._api = api
        self.user_id: str = ''
        self.exchange_id: str = ''
        self.instrument_id: str = ''
        self.create_date: str = ''
        self.cost: float = float('nan')
        self.cost_his: float = float('nan')
        self.volume: int = 0
        self.volume_his: int = 0
        self.last_price: float = float('nan')
        self.buy_volume_today: int = 0
        self.buy_balance_today: float = float('nan')
        self.buy_fee_today: float = float('nan')
        self.sell_volume_today: int = 0
        self.sell_balance_today: float = float('nan')
        self.sell_fee_today: float = float('nan')
        self.buy_volume_his: int = 0
        self.buy_balance_his: float = float('nan')
        self.buy_fee_his: float = float('nan')
        self.sell_volume_his: int = 0
        self.sell_balance_his: float = float('nan')
        self.sell_fee_his: float = float('nan')
        self.shared_volume_today: float = float('nan')
        self.devidend_balance_today: float = float('nan')
        self.market_value: float = float('nan')
        self.market_value_his: float = float('nan')
        self.float_profit_today: float = float('nan')
        self.real_profit_today: float = float('nan')
        self.real_profit_his: float = float('nan')
        self.profit_today: float = float('nan')
        self.profit_rate_today: float = float('nan')
        self.hold_profit: float = float('nan')
        self.real_profit_total: float = float('nan')
        self.profit_total: float = float('nan')
        self.profit_rate_total: float = float('nan')

    @property
    def orders(self):
        if False:
            return 10
        tdict = _get_obj(self._api._data, ['trade', self._path[1], 'orders'])
        fts = {order_id: order for (order_id, order) in tdict.items() if not order_id.startswith('_') and order.instrument_id == self.instrument_id and (order.exchange_id == self.exchange_id) and (order.status == 'ALIVE')}
        return fts

class SecurityOrder(Entity):
    """ SecurityOrder 是一个股票账户委托单对象 """

    def __init__(self, api):
        if False:
            i = 10
            return i + 15
        self._api = api
        self.user_id: str = ''
        self.order_id: str = ''
        self.exchange_order_id: str = ''
        self.exchange_id: str = ''
        self.instrument_id: str = ''
        self.direction: str = ''
        self.volume_orign: int = 0
        self.volume_left: int = 0
        self.price_type: str = ''
        self.limit_price: float = float('nan')
        self.frozen_fee: float = float('nan')
        self.insert_date_time: int = 0
        self.status: str = ''
        self.last_msg: str = ''

    @property
    def trade_records(self):
        if False:
            while True:
                i = 10
        '\n        成交记录\n\n        :return: dict, 其中每个元素的key为成交ID, value为 :py:class:`~tqsdk.objs.Trade`\n        '
        tdict = _get_obj(self._api._data, ['trade', self._path[1], 'trades'])
        fts = {trade_id: trade for (trade_id, trade) in tdict.items() if not trade_id.startswith('_') and trade.order_id == self.order_id}
        return fts

class SecurityTrade(Entity):
    """ SecurityTrade 是一个股票账户成交对象 """

    def __init__(self, api):
        if False:
            i = 10
            return i + 15
        self._api = api
        self.user_id: str = ''
        self.trade_id: str = ''
        self.exchange_id: str = ''
        self.instrument_id: str = ''
        self.order_id: str = ''
        self.exchange_order_id: str = ''
        self.direction: str = ''
        self.volume: int = 0
        self.price: float = float('nan')
        self.balance: float = float('nan')
        self.fee: float = float('nan')
        self.trade_date_time: int = 0