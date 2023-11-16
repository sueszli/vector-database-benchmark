import pickle
from datetime import datetime, date
from typing import Set
from rqalpha.portfolio import Portfolio, Account
from rqalpha.const import DEFAULT_ACCOUNT_TYPE
from rqalpha.environment import Environment
from rqalpha.utils.logger import user_system_log, system_log
from rqalpha.utils.repr import property_repr

class RunInfo(object):
    """
    策略运行信息
    """
    __repr__ = property_repr

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        self._start_date = config.base.start_date
        self._end_date = config.base.end_date
        self._frequency = config.base.frequency
        self._stock_starting_cash = config.base.accounts.get(DEFAULT_ACCOUNT_TYPE.STOCK, 0)
        self._future_starting_cash = config.base.accounts.get(DEFAULT_ACCOUNT_TYPE.FUTURE, 0)
        self._margin_multiplier = config.base.margin_multiplier
        self._run_type = config.base.run_type
        try:
            self._matching_type = config.mod.sys_simulation.matching_type
            self._slippage = config.mod.sys_simulation.slippage
            self._commission_multiplier = config.mod.sys_transaction_cost.commission_multiplier
            if config.mod.sys_transaction_cost.commission_multiplier:
                self._stock_commission_multiplier = self._commission_multiplier
                self._futures_commission_multiplier = self._commission_multiplier
            else:
                self._stock_commission_multiplier = config.mod.sys_transaction_cost.stock_commission_multiplier
                self._futures_commission_multiplier = config.mod.sys_transaction_cost.futures_commission_multiplier
        except:
            pass

    @property
    def start_date(self):
        if False:
            return 10
        '\n        策略的开始日期\n        '
        return self._start_date

    @property
    def end_date(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        策略的结束日期\n        '
        return self._end_date

    @property
    def frequency(self):
        if False:
            return 10
        "\n        '1d'或'1m'\n        "
        return self._frequency

    @property
    def stock_starting_cash(self):
        if False:
            return 10
        '\n        股票账户初始资金\n        '
        return self._stock_starting_cash

    @property
    def future_starting_cash(self):
        if False:
            while True:
                i = 10
        '\n        期货账户初始资金\n        '
        return self._future_starting_cash

    @property
    def slippage(self):
        if False:
            while True:
                i = 10
        '\n        滑点水平\n        '
        return self._slippage

    @property
    def matching_type(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        撮合方式\n        '
        return self._matching_type

    @property
    def commission_multiplier(self):
        if False:
            print('Hello World!')
        '\n        手续费倍率\n        '
        return self._commission_multiplier

    @property
    def stock_commission_multiplier(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        股票手续费倍率\n        '
        return self._stock_commission_multiplier

    @property
    def futures_commission_multiplier(self):
        if False:
            return 10
        '\n        期货手续费倍率\n        '
        return self._futures_commission_multiplier

    @property
    def margin_multiplier(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        保证金倍率\n        '
        return self._margin_multiplier

    @property
    def run_type(self):
        if False:
            return 10
        '\n        运行类型\n        '
        return self._run_type

class StrategyContext(object):
    """
    策略上下文
    """

    def __repr__(self):
        if False:
            return 10
        items = ('%s = %r' % (k, v) for (k, v) in self.__dict__.items() if not callable(v) and (not k.startswith('_')))
        return 'Context({%s})' % (', '.join(items),)

    def __init__(self):
        if False:
            while True:
                i = 10
        self._config = None

    def get_state(self):
        if False:
            print('Hello World!')
        dict_data = {}
        for (key, value) in self.__dict__.items():
            if key.startswith('_'):
                continue
            try:
                dict_data[key] = pickle.dumps(value)
            except Exception as e:
                user_system_log.warn('context.{} can not pickle', key)
        return pickle.dumps(dict_data)

    def set_state(self, state):
        if False:
            return 10
        dict_data = pickle.loads(state)
        for (key, value) in dict_data.items():
            try:
                self.__dict__[key] = pickle.loads(value)
                system_log.debug('restore context.{} {}', key, type(self.__dict__[key]))
            except Exception as e:
                user_system_log.warn('context.{} can not restore', key)

    @property
    def universe(self):
        if False:
            i = 10
            return i + 15
        '\n        在运行 :func:`update_universe`, :func:`subscribe` 或者 :func:`unsubscribe` 的时候，合约池会被更新。\n\n        需要注意，合约池内合约的交易时间（包含股票的策略默认会在股票交易时段触发）是handle_bar被触发的依据。\n        '
        return Environment.get_instance().get_universe()

    @property
    def now(self):
        if False:
            print('Hello World!')
        '\n        当前 Bar/Tick 所对应的时间\n        '
        return Environment.get_instance().calendar_dt

    @property
    def run_info(self):
        if False:
            return 10
        '\n        测略运行信息\n        '
        config = Environment.get_instance().config
        return RunInfo(config)

    @property
    def portfolio(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        策略投资组合，可通过该对象获取当前策略账户、持仓等信息\n        '
        return Environment.get_instance().portfolio

    @property
    def stock_account(self):
        if False:
            while True:
                i = 10
        '\n        股票账户\n        '
        return self.portfolio.accounts[DEFAULT_ACCOUNT_TYPE.STOCK]

    @property
    def future_account(self):
        if False:
            while True:
                i = 10
        '\n        期货账户\n        '
        return self.portfolio.accounts[DEFAULT_ACCOUNT_TYPE.FUTURE]

    @property
    def config(self):
        if False:
            i = 10
            return i + 15
        return Environment.get_instance().config