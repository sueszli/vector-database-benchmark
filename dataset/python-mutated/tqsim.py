__author__ = 'mayanqiong'
from tqsdk.tradeable.mixin import FutureMixin
from tqsdk.datetime import _timestamp_nano_to_str
from tqsdk.diff import _get_obj
from tqsdk.objs import Quote
from tqsdk.report import TqReport
from tqsdk.tradeable.sim.basesim import BaseSim
from tqsdk.tradeable.sim.trade_future import SimTrade
from tqsdk.tradeable.sim.utils import _get_future_margin, _get_commission

class TqSim(BaseSim, FutureMixin):
    """
    天勤模拟交易类

    该类实现了一个本地的模拟账户，并且在内部完成撮合交易，在回测和复盘模式下，只能使用 TqSim 账户来交易。

    限价单要求报单价格达到或超过对手盘价格才能成交, 成交价为报单价格, 如果没有对手盘(涨跌停)则无法成交

    市价单使用对手盘价格成交, 如果没有对手盘(涨跌停)则自动撤单

    模拟交易不会有部分成交的情况, 要成交就是全部成交
    """

    def __init__(self, init_balance: float=10000000.0, account_id: str=None) -> None:
        if False:
            while True:
                i = 10
        '\n        Args:\n            init_balance (float): [可选]初始资金, 默认为一千万\n\n            account_id (str): [可选]帐号, 默认为 TQSIM\n\n        Example::\n\n            # 修改TqSim模拟帐号的初始资金为100000\n            from tqsdk import TqApi, TqSim, TqAuth\n            api = TqApi(TqSim(init_balance=100000), auth=TqAuth("快期账户", "账户密码"))\n\n        '
        if float(init_balance) <= 0:
            raise Exception('初始资金(init_balance) %s 错误, 请检查 init_balance 是否填写正确' % init_balance)
        super(TqSim, self).__init__(account_id='TQSIM' if account_id is None else account_id, init_balance=float(init_balance), trade_class=SimTrade)

    @property
    def _account_info(self):
        if False:
            while True:
                i = 10
        info = super(TqSim, self)._account_info
        info.update({'account_type': self._account_type})
        return info

    def set_commission(self, symbol: str, commission: float=float('nan')):
        if False:
            print('Hello World!')
        '\n        设置指定合约模拟交易的每手手续费。\n\n        Args:\n            symbol (str): 合约代码\n\n            commission (float): 每手手续费\n\n        Returns:\n            float: 设置的每手手续费\n\n        Example::\n\n            from tqsdk import TqSim, TqApi, TqAuth\n\n            sim = TqSim()\n            api = TqApi(sim, auth=TqAuth("快期账户", "账户密码"))\n\n            sim.set_commission("SHFE.cu2112", 50)\n\n            print(sim.get_commission("SHFE.cu2112"))\n        '
        if commission != commission:
            raise Exception("合约手续费不可以设置为 float('nan')")
        quote = _get_obj(self._data, ['quotes', symbol], Quote(self._api if hasattr(self, '_api') else None))
        quote['user_commission'] = commission
        if self._quote_tasks.get(symbol):
            self._quote_tasks[symbol]['quote_chan'].send_nowait({'quotes': {symbol: {'user_commission': commission}}})
        return commission

    def set_margin(self, symbol: str, margin: float=float('nan')):
        if False:
            for i in range(10):
                print('nop')
        '\n        设置指定合约模拟交易的每手保证金。\n\n        Args:\n            symbol (str): 合约代码 (只支持期货合约)\n\n            margin (float): 每手保证金\n\n        Returns:\n            float: 设置的每手保证金\n\n        Example::\n\n            from tqsdk import TqSim, TqApi, TqAuth\n\n            sim = TqSim()\n            api = TqApi(sim, auth=TqAuth("快期账户", "账户密码"))\n\n            sim.set_margin("SHFE.cu2112", 26000)\n\n            print(sim.get_margin("SHFE.cu2112"))\n        '
        if margin != margin:
            raise Exception("合约手续费不可以设置为 float('nan')")
        quote = _get_obj(self._data, ['quotes', symbol], Quote(self._api if hasattr(self, '_api') else None))
        quote['user_margin'] = margin
        if self._quote_tasks.get(symbol):
            self._quote_tasks[symbol]['quote_chan'].send_nowait({'quotes': {symbol: {'user_margin': margin}}})
            while margin != self.get_position(symbol).get('future_margin'):
                self._api.wait_update()
        return margin

    def get_margin(self, symbol: str):
        if False:
            return 10
        '\n        获取指定合约模拟交易的每手保证金。\n\n        Args:\n            symbol (str): 合约代码\n\n        Returns:\n            float: 返回合约模拟交易的每手保证金\n\n        Example::\n\n            from tqsdk import TqSim, TqApi, TqAuth\n\n            sim = TqSim()\n            api = TqApi(sim, auth=TqAuth("快期账户", "账户密码"))\n\n            quote = api.get_quote("SHFE.cu2112")\n            print(sim.get_margin("SHFE.cu2112"))\n        '
        return _get_future_margin(self._data.get('quotes', {}).get(symbol, {}))

    def get_commission(self, symbol: str):
        if False:
            print('Hello World!')
        '\n        获取指定合约模拟交易的每手手续费\n\n        Args:\n            symbol (str): 合约代码\n\n        Returns:\n            float: 返回合约模拟交易的每手手续费\n\n        Example::\n\n            from tqsdk import TqSim, TqApi, TqAuth\n\n            sim = TqSim()\n            api = TqApi(sim, auth=TqAuth("快期账户", "账户密码"))\n\n            quote = api.get_quote("SHFE.cu2112")\n            print(sim.get_commission("SHFE.cu2112"))\n        '
        return _get_commission(self._data.get('quotes', {}).get(symbol, {}))

    def _handle_on_alive(self, msg, order):
        if False:
            while True:
                i = 10
        '\n        在 order 状态变为 ALIVE 调用，屏幕输出信息，打印日志\n        '
        symbol = f"{order['exchange_id']}.{order['instrument_id']}"
        self._api._print(f"模拟交易下单 {self._account_name}, {order['order_id']}: 时间: {_timestamp_nano_to_str(order['insert_date_time'])}, 合约: {symbol}, 开平: {order['offset']}, 方向: {order['direction']}, 手数: {order['volume_left']}, 价格: {order.get('limit_price', '市价')}")
        self._logger.debug(msg, order_id=order['order_id'], datetime=order['insert_date_time'], symbol=symbol, offset=order['offset'], direction=order['direction'], volume_left=order['volume_left'], limit_price=order.get('limit_price', '市价'))

    def _handle_on_finished(self, msg, order):
        if False:
            i = 10
            return i + 15
        '\n        在 order 状态变为 FINISHED 调用，屏幕输出信息，打印日志\n        '
        self._api._print(f"模拟交易委托单 {self._account_name}, {order['order_id']}: {order['last_msg']}")
        self._logger.debug(msg, order_id=order['order_id'], last_msg=order['last_msg'], status=order['status'], volume_orign=order['volume_orign'], volume_left=order['volume_left'])

    def _report(self):
        if False:
            while True:
                i = 10
        if not self.trade_log:
            return
        date_keys = sorted(self.trade_log.keys())
        self._api._print(f'模拟交易成交记录, 账户: {self._account_name}')
        for d in date_keys:
            for t in self.trade_log[d]['trades']:
                symbol = t['exchange_id'] + '.' + t['instrument_id']
                self._api._print(f"时间: {_timestamp_nano_to_str(t['trade_date_time'])}, 合约: {symbol}, 开平: {t['offset']}, 方向: {t['direction']}, 手数: {t['volume']}, 价格: {t['price']:.3f},手续费: {t['commission']:.2f}")
        self._api._print(f'模拟交易账户资金, 账户: {self._account_name}')
        for d in date_keys:
            account = self.trade_log[d]['account']
            self._api._print(f"日期: {d}, 账户权益: {account['balance']:.2f}, 可用资金: {account['available']:.2f}, 浮动盈亏: {account['float_profit']:.2f}, 持仓盈亏: {account['position_profit']:.2f}, 平仓盈亏: {account['close_profit']:.2f}, 市值: {account['market_value']:.2f}, 保证金: {account['margin']:.2f}, 手续费: {account['commission']:.2f}, 风险度: {account['risk_ratio'] * 100:.2f}%")
        report = TqReport(report_id=self._account_id, trade_log=self.trade_log, quotes=self._data['quotes'])
        self.tqsdk_stat = report.default_metrics
        self._api._print(f"胜率: {self.tqsdk_stat['winning_rate'] * 100:.2f}%, 盈亏额比例: {self.tqsdk_stat['profit_loss_ratio']:.2f}, 收益率: {self.tqsdk_stat['ror'] * 100:.2f}%, 年化收益率: {self.tqsdk_stat['annual_yield'] * 100:.2f}%, 最大回撤: {self.tqsdk_stat['max_drawdown'] * 100:.2f}%, 年化夏普率: {self.tqsdk_stat['sharpe_ratio']:.4f},年化索提诺比率: {self.tqsdk_stat['sortino_ratio']:.4f}")
        if self._tqsdk_backtest:
            self._api.draw_report(report.full())