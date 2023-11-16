__author__ = 'chengzhi'
import asyncio
import time
from datetime import datetime
from asyncio import gather
from inspect import isfunction
from typing import Optional, Union, Callable
from tqsdk import utils
from tqsdk.api import TqApi
from tqsdk.backtest import TqBacktest
from tqsdk.channel import TqChan
from tqsdk.datetime import _is_in_trading_time, _timestamp_nano_to_str
from tqsdk.diff import _get_obj
from tqsdk.lib.utils import _check_volume_limit, _check_direction, _check_offset, _check_volume, _check_price, _check_offset_priority
from tqsdk.tradeable import TqAccount, TqKq, TqSim

class TargetPosTaskSingleton(type):
    """
    TargetPosTask 需要保证在每个账户下每个合约只有一个 TargetPosTask 实例。

    当用户多次调用时，应该保证对于同一账户同一合约使用相同的参数构造，否则抛错。

    在修改 TargetPosTask 构造参数时，同时应该修改 TargetPosTaskSingleton.__call__ 方法的参数，要确保其个数、名称、默认值和文档描述一致，    这些参数才是实际传给 TargetPosTask.__init__ 方法的参数。

    同时应该在 TargetPosTask 实例运行结束时释放相应的资源，_instances 需要情况对应的引用。
    """
    _instances = {}

    def __call__(cls, api, symbol, price='ACTIVE', offset_priority='今昨,开', min_volume=None, max_volume=None, trade_chan=None, trade_objs_chan=None, account: Optional[Union[TqAccount, TqKq, TqSim]]=None, *args, **kwargs):
        if False:
            print('Hello World!')
        target_account = api._account._check_valid(account)
        if target_account is None:
            raise Exception(f'多账户模式下, 需要指定账户实例 account')
        key = api._account._get_account_key(target_account) + '#' + symbol
        if key not in TargetPosTaskSingleton._instances:
            TargetPosTaskSingleton._instances[key] = super(TargetPosTaskSingleton, cls).__call__(api, symbol, price, offset_priority, min_volume, max_volume, trade_chan, trade_objs_chan, target_account, *args, **kwargs)
        else:
            instance = TargetPosTaskSingleton._instances[key]
            if instance._offset_priority != offset_priority:
                raise Exception('您试图用不同的 offset_priority 参数创建两个 %s 调仓任务, offset_priority参数原为 %s, 现为 %s' % (symbol, instance._offset_priority, offset_priority))
            if instance._price != price:
                raise Exception('您试图用不同的 price 参数创建两个 %s 调仓任务, price参数原为 %s, 现为 %s' % (symbol, instance._price, price))
            if instance._min_volume != min_volume:
                raise Exception(f'您试图用不同的 min_volume 参数创建两个 {symbol} 调仓任务, min_volume 参数原为 {instance._min_volume}, 现为 {min_volume}')
            if instance._max_volume != max_volume:
                raise Exception(f'您试图用不同的 max_volume 参数创建两个 {symbol} 调仓任务, max_volume 参数原为 {instance._max_volume}, 现为 {max_volume}')
        return TargetPosTaskSingleton._instances[key]

class TargetPosTask(object, metaclass=TargetPosTaskSingleton):
    """目标持仓 task, 该 task 可以将指定合约调整到目标头寸"""

    def __init__(self, api: TqApi, symbol: str, price: Union[str, Callable[[str], Union[float, int]]]='ACTIVE', offset_priority: str='今昨,开', min_volume: Optional[int]=None, max_volume: Optional[int]=None, trade_chan: Optional[TqChan]=None, trade_objs_chan: Optional[TqChan]=None, account: Optional[Union[TqAccount, TqKq, TqSim]]=None) -> None:
        if False:
            while True:
                i = 10
        '\n        创建目标持仓task实例，负责调整归属于该task的持仓 **(默认为整个账户的该合约净持仓)**.\n\n        **注意:**\n            1. TargetPosTask 在 set_target_volume 时并不下单或撤单, 它的下单和撤单动作, 是在之后的每次 wait_update 时执行的. 因此, **需保证 set_target_volume 后还会继续调用wait_update()** 。\n\n            2. TargetPosTask 在每次下单后会根据后续的行情计算下单的最新的价格，当该价格发生变化则首先撤回由该程序发出的在场的委托，然后根据最新的价格重新下单。\n\n            3. 请勿在使用 TargetPosTask 的同时使用 insert_order() 函数, 否则将导致 TargetPosTask 报错或错误下单。\n\n            4. TargetPosTask 如果同时设置 min_volume（每笔最小下单手数），max_volume（每笔最大下单的手数）两个参数，表示采用 **大单拆分模式** 下单。\n\n                在 **大单拆分模式** 下，每次下单的手数为随机生成的正整数，值介于 min_volume、max_volume 之间。\n\n                具体说明：调用 set_target_volume 后，首先会根据目标持仓手数、开平仓顺序计算出，需要平今、平昨、开仓的目标下单手数及顺序。\n\n                + 如果在调整持仓的目标下单手数小于 max_volume，则直接以目标下单手数下单。\n\n                + 如果在调整持仓的目标下单手数大于等于 max_volume，则会以 min_volume、max_volume 之间的随机手数下一笔委托单，手数全部成交后，会接着处理剩余的手数；                继续以随机手数下一笔委托单，全部成交后，继续处理剩余的手数，直至剩余手数小于 max_volume 时，直接以剩余手数下单。\n\n                当使用大单拆分模式下单时，必须同时填写 min_volume、max_volume，且需要满足 max_volume >= min_volume > 0。\n\n        Args:\n            api (TqApi): TqApi实例，该task依托于指定api下单/撤单\n\n            symbol (str): 负责调整的合约代码\n\n            price (str / Callable): [可选]下单方式, 默认为 "ACTIVE"\n                * "ACTIVE"：对价下单，在持仓调整过程中，若下单方向为买，对价为卖一价；若下单方向为卖，对价为买一价。\n                * "PASSIVE"：排队价下单，在持仓调整过程中，若下单方向为买，对价为买一价；若下单方向为卖，对价为卖一价，该种方式可能会造成较多撤单.\n                * Callable[[str], Union[float, int]]: 函数参数为下单方向，函数返回值是下单价格。如果返回 nan，程序会抛错。\n\n            offset_priority (str): [可选]开平仓顺序，昨=平昨仓，今=平今仓，开=开仓，逗号=等待之前操作完成\n\n                                   对于下单指令区分平今/昨的交易所(如上期所)，按照今/昨仓的数量计算是否能平今/昨仓\n                                   对于下单指令不区分平今/昨的交易所(如中金所)，按照“先平当日新开仓，再平历史仓”的规则计算是否能平今/昨仓，如果这些交易所设置为"昨开"在有当日新开仓和历史仓仓的情况下，会自动跳过平昨仓进入到下一步\n\n                                   * "今昨,开" 表示先平今仓，再平昨仓，等待平仓完成后开仓，对于没有单向大边的品种避免了开仓保证金不足\n                                   * "今昨开" 表示先平今仓，再平昨仓，并开仓，所有指令同时发出，适合有单向大边的品种\n                                   * "昨开" 表示先平昨仓，再开仓，禁止平今仓，适合股指这样平今手续费较高的品种\n                                   * "开" 表示只开仓，不平仓，适合需要进行锁仓操作的品种\n\n            min_volume (int): [可选] **大单拆分模式下** 每笔最小下单的手数，默认不启用 **大单拆分模式**\n\n            max_volume (int): [可选] **大单拆分模式下** 每笔最大下单的手数，默认不启用 **大单拆分模式**\n\n            trade_chan (TqChan): [可选]成交通知channel, 当有成交发生时会将成交手数(多头为正数，空头为负数)发到该channel上\n\n            trade_objs_chan (TqChan): [可选]成交对象通知channel, 当有成交发生时会将成交对象发送到该channel上\n\n            account (TqAccount/TqKq/TqSim): [可选]指定发送下单指令的账户实例, 多账户模式下，该参数必须指定\n\n        **注意**\n\n        当 price 参数为函数类型时，该函数应该返回一个有效的价格值，应该避免返回 nan。以下为 price 参数是函数类型时的示例。\n\n        Example1::\n\n            # ... 用户代码 ...\n            quote = api.get_quote("SHFE.cu2012")\n            def get_price(direction):\n                # 在 BUY 时使用买一价加一档价格，SELL 时使用卖一价减一档价格\n                if direction == "BUY":\n                    price = quote.bid_price1 + quote.price_tick\n                else:\n                    price = quote.ask_price1 - quote.price_tick\n                # 如果 price 价格是 nan，使用最新价报单\n                if price != price:\n                    price = quote.last_price\n                return price\n\n            target_pos = TargetPosTask(api, "SHFE.cu2012", price=get_price)\n            # ... 用户代码 ...\n\n\n        Example2::\n\n            # ... 用户代码 ...\n            quote_list = api.get_quote_list(["SHFE.cu2012","SHFE.au2012"])\n\n            def get_price_by_quote(quote):\n                def get_price(direction):\n                    # 在 BUY 时使用涨停价，SELL 时使用跌停价\n                    if direction == "BUY":\n                        price = quote["upper_limit"]\n                    else:\n                        price = quote.lower_limit\n                    # 如果 price 价格是 nan，使用最新价报单\n                    if price != price:\n                        price = quote.last_price\n                    return price\n                return get_price\n\n            for quote in quote_list:\n                target_pos_active_dict[quote.instrument_id] = TargetPosTask(api, quote.instrument_id, price=get_price_by_quote(quote))\n\n            # ... 用户代码 ...\n\n        Example3::\n\n            # 大单拆分模式用法示例\n\n            from tqsdk import TqApi, TqAuth, TargetPosTask\n            api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n            position = api.get_position(\'SHFE.rb2106\')\n\n            # 同时设置 min_volume、max_volume 两个参数，表示使用大单拆分模式\n            t = TargetPosTask(api, \'SHFE.rb2106\', min_volume=2, max_volume=10)\n            t.set_target_volume(50)\n            while True:\n                api.wait_update()\n                if position.pos_long == 50:\n                    break\n            api.close()\n\n            # 说明：\n            # 以上代码使用 TqSim 交易，开始时用户没有 SHFE.cu2012 合约的任何持仓，那么在 t.set_target_volume(50) 之后应该开多仓 50 手\n            # 根据用户参数，下单使用大单拆分模式，每次下单手数在 2～10 之间，打印出的成交通知可能是这样的：\n            # 2021-03-15 11:29:48 -     INFO - 模拟交易成交记录\n            # 2021-03-15 11:29:48 -     INFO - 时间: 2021-03-15 11:29:47.516138, 合约: SHFE.rb2106, 开平: OPEN, 方向: BUY, 手数: 7, 价格: 4687.000,手续费: 32.94\n            # 2021-03-15 11:29:48 -     INFO - 时间: 2021-03-15 11:29:47.519699, 合约: SHFE.rb2106, 开平: OPEN, 方向: BUY, 手数: 8, 价格: 4687.000,手续费: 37.64\n            # 2021-03-15 11:29:48 -     INFO - 时间: 2021-03-15 11:29:47.522848, 合约: SHFE.rb2106, 开平: OPEN, 方向: BUY, 手数: 10, 价格: 4687.000,手续费: 47.05\n            # 2021-03-15 11:29:48 -     INFO - 时间: 2021-03-15 11:29:47.525617, 合约: SHFE.rb2106, 开平: OPEN, 方向: BUY, 手数: 8, 价格: 4687.000,手续费: 37.64\n            # 2021-03-15 11:29:48 -     INFO - 时间: 2021-03-15 11:29:47.528151, 合约: SHFE.rb2106, 开平: OPEN, 方向: BUY, 手数: 7, 价格: 4687.000,手续费: 32.94\n            # 2021-03-15 11:29:48 -     INFO - 时间: 2021-03-15 11:29:47.530930, 合约: SHFE.rb2106, 开平: OPEN, 方向: BUY, 手数: 7, 价格: 4687.000,手续费: 32.94\n            # 2021-03-15 11:29:48 -     INFO - 时间: 2021-03-15 11:29:47.533515, 合约: SHFE.rb2106, 开平: OPEN, 方向: BUY, 手数: 3, 价格: 4687.000,手续费: 14.12\n\n        '
        if symbol.startswith('CZCE.ZC'):
            raise Exception('动力煤期货不支持创建 targetpostask、twap、vwap 任务，交易所规定该品种最小开仓手数为大于等于 4 手，这些函数还未支持该规则!')
        if symbol.startswith('CZCE.WH'):
            raise Exception('强麦期货不支持创建 targetpostask、twap、vwap 任务，交易所规定该品种最小开仓手数为大于等于 10 手，这些函数还未支持该规则!')
        if symbol.startswith('CZCE.PM'):
            raise Exception('普麦期货不支持创建 targetpostask、twap、vwap 任务，交易所规定该品种最小开仓手数为大于等于 10 手，这些函数还未支持该规则!')
        if symbol.startswith('CZCE.RI'):
            raise Exception('早籼稻期货不支持创建 targetpostask、twap、vwap 任务，交易所规定该品种最小开仓手数为大于等于 10 手，这些函数还未支持该规则!')
        if symbol.startswith('CZCE.JR'):
            raise Exception('粳稻期货不支持创建 targetpostask、twap、vwap 任务，交易所规定该品种最小开仓手数为大于等于 10 手，这些函数还未支持该规则!')
        if symbol.startswith('CZCE.LR'):
            raise Exception('晚籼稻期货不支持创建 targetpostask、twap、vwap 任务，交易所规定该品种最小开仓手数为大于等于 10 手，这些函数还未支持该规则!')
        if symbol == 'CZCE.SA309' or symbol == 'CZCE.SA310':
            raise Exception('纯碱期货 2309 合约及 2310 合约不支持创建 targetpostask、twap、vwap 任务，交易所规定该品种最小开仓手数为大于等于 4 手，这些函数还未支持该规则!')
        super(TargetPosTask, self).__init__()
        self._api = api
        self._account = api._account._check_valid(account)
        self._symbol = symbol
        self._exchange = symbol.split('.')[0]
        self._offset_priority = _check_offset_priority(offset_priority)
        (self._min_volume, self._max_volume) = _check_volume_limit(min_volume, max_volume)
        self._price = _check_price(price)
        self._pos = self._account.get_position(self._symbol)
        self._pos_chan = TqChan(self._api, last_only=True)
        self._trade_chan = trade_chan
        self._trade_objs_chan = trade_objs_chan
        self._task = self._api.create_task(self._target_pos_task())
        self._time_update_task = self._api.create_task(self._update_time_from_md())
        self._local_time_record = time.time() - 0.005
        self._local_time_record_update_chan = TqChan(self._api, last_only=True)

    def set_target_volume(self, volume: int) -> None:
        if False:
            return 10
        '\n        设置目标持仓手数\n\n        Args:\n            volume (int): 目标持仓手数，正数表示多头，负数表示空头，0表示空仓\n\n        Example1::\n\n            # 设置 rb1810 持仓为多头5手\n            from tqsdk import TqApi, TqAuth, TargetPosTask\n\n            api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n            target_pos = TargetPosTask(api, "SHFE.rb1810")\n            target_pos.set_target_volume(5)\n            while True:\n                # 需在 set_target_volume 后调用wait_update()以发出指令\n                api.wait_update()\n\n        Example2::\n\n            # 多账户模式下使用 TargetPosTask\n            from tqsdk import TqApi, TqMultiAccount, TqAuth, TargetPosTask\n\n            account1 = TqAccount("H海通期货", "123456", "123456")\n            account2 = TqAccount("H宏源期货", "654321", "123456")\n            api = TqApi(TqMultiAccount([account1, account2]), auth=TqAuth("快期账户", "账户密码"))\n            symbol1 = "DCE.m2105"\n            symbol2 = "DCE.i2101"\n            # 多账户模式下, 调仓工具需要指定账户实例\n            target_pos1 = TargetPosTask(api, symbol1, account=account1)\n            target_pos2 = TargetPosTask(api, symbol2, account=account2)\n            target_pos1.set_target_volume(30)\n            target_pos2.set_target_volume(80)\n            while True:\n                api.wait_update()\n\n            api.close()\n\n        '
        if self._task.done():
            raise Exception('已经结束的 TargetPosTask 实例不可以再设置手数。')
        self._pos_chan.send_nowait(int(volume))

    def _get_order(self, offset, vol, pending_frozen):
        if False:
            while True:
                i = 10
        '\n        根据指定的offset和预期下单手数vol, 返回符合要求的委托单最大报单手数\n        :param offset: "昨" / "今" / "开"\n        :param vol: int, <0表示SELL, >0表示BUY\n        :return: order_offset: "CLOSE"/"CLOSETODAY"/"OPEN"; order_dir: "BUY"/"SELL"; "order_volume": >=0, 报单手数\n        '
        if vol > 0:
            order_dir = 'BUY'
            pos_all = self._pos.pos_short
        else:
            order_dir = 'SELL'
            pos_all = self._pos.pos_long
        if offset == '昨':
            order_offset = 'CLOSE'
            if self._exchange == 'SHFE' or self._exchange == 'INE':
                if vol > 0:
                    pos_all = self._pos.pos_short_his
                else:
                    pos_all = self._pos.pos_long_his
                frozen_volume = sum([order.volume_left for order in self._pos.orders.values() if not order.is_dead and order.offset == order_offset and (order.direction == order_dir)])
            else:
                frozen_volume = pending_frozen + sum([order.volume_left for order in self._pos.orders.values() if not order.is_dead and order.offset != 'OPEN' and (order.direction == order_dir)])
                if (self._pos.pos_short_today if vol > 0 else self._pos.pos_long_today) - frozen_volume > 0:
                    pos_all = frozen_volume
            order_volume = min(abs(vol), max(0, pos_all - frozen_volume))
        elif offset == '今':
            if self._exchange == 'SHFE' or self._exchange == 'INE':
                order_offset = 'CLOSETODAY'
                if vol > 0:
                    pos_all = self._pos.pos_short_today
                else:
                    pos_all = self._pos.pos_long_today
                frozen_volume = sum([order.volume_left for order in self._pos.orders.values() if not order.is_dead and order.offset == order_offset and (order.direction == order_dir)])
            else:
                order_offset = 'CLOSE'
                frozen_volume = pending_frozen + sum([order.volume_left for order in self._pos.orders.values() if not order.is_dead and order.offset != 'OPEN' and (order.direction == order_dir)])
                pos_all = self._pos.pos_short_today if vol > 0 else self._pos.pos_long_today
            order_volume = min(abs(vol), max(0, pos_all - frozen_volume))
        elif offset == '开':
            order_offset = 'OPEN'
            order_volume = abs(vol)
        else:
            order_offset = ''
            order_volume = 0
        return (order_offset, order_dir, order_volume)

    async def _update_time_from_md(self):
        """监听行情更新并记录当时本地时间的task"""
        try:
            chan = TqChan(self._api, last_only=True)
            self._quote = await self._api.get_quote(self._symbol)
            self._api.register_update_notify(self._quote, chan)
            if isinstance(self._api._backtest, TqBacktest):
                self._api.register_update_notify(_get_obj(self._api._data, ['_tqsdk_backtest']), chan)
            async for _ in chan:
                self._local_time_record = time.time() - 0.005
                self._local_time_record_update_chan.send_nowait(True)
        finally:
            await chan.close()

    async def _target_pos_task(self):
        """负责调整目标持仓的task"""
        all_tasks = []
        try:
            self._quote = await self._api.get_quote(self._symbol)
            async for target_pos in self._pos_chan:
                while True:
                    if isinstance(self._api._backtest, TqBacktest):
                        cur_timestamp = self._api._data.get('_tqsdk_backtest', {}).get('current_dt', float('nan'))
                        cur_dt = _timestamp_nano_to_str(cur_timestamp)
                        time_record = float('nan')
                    else:
                        cur_dt = self._quote['datetime']
                        time_record = self._local_time_record
                    if _is_in_trading_time(self._quote, cur_dt, time_record):
                        break
                    await self._local_time_record_update_chan.recv()
                target_pos = self._pos_chan.recv_latest(target_pos)
                delta_volume = target_pos - self._pos.pos
                pending_forzen = 0
                for each_priority in self._offset_priority + ',':
                    if each_priority == ',':
                        await gather(*[each._task for each in all_tasks])
                        pending_forzen = 0
                        all_tasks = []
                        continue
                    (order_offset, order_dir, order_volume) = self._get_order(each_priority, delta_volume, pending_forzen)
                    if order_volume == 0:
                        continue
                    elif order_offset != 'OPEN':
                        pending_forzen += order_volume
                    order_task = InsertOrderUntilAllTradedTask(self._api, self._symbol, order_dir, offset=order_offset, volume=order_volume, min_volume=self._min_volume, max_volume=self._max_volume, price=self._price, trade_chan=self._trade_chan, trade_objs_chan=self._trade_objs_chan, account=self._account)
                    all_tasks.append(order_task)
                    delta_volume -= order_volume if order_dir == 'BUY' else -order_volume
        finally:
            TargetPosTaskSingleton._instances.pop(self._account._account_key + '#' + self._symbol, None)
            await self._pos_chan.close()
            self._time_update_task.cancel()
            await asyncio.gather(*[t._task for t in all_tasks] + [self._time_update_task], return_exceptions=True)

    def cancel(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        取消当前 TargetPosTask 实例，会将该实例已经发出但还是未成交的委托单撤单，并且如果后续调用此实例的 set_target_volume 函数会报错。\n\n        任何时刻，每个账户下一个合约只能有一个 TargetPosTask 实例，并且其构造参数不能修改。\n\n        如果对于同一个合约要构造不同参数的 TargetPosTask 实例，需要调用 cancel 方法销毁，才能创建新的 TargetPosTask 实例\n\n        Example1::\n\n            from datetime import datetime, time\n            from tqsdk import TqApi, TargetPosTask\n\n            api = TqApi(auth=TqAuth("快期账户", "账户密码"))\n            quote = api.get_quote("SHFE.rb2110")\n            target_pos_passive = TargetPosTask(api, "SHFE.rb2110", price="PASSIVE")\n\n            while datetime.strptime(quote.datetime, "%Y-%m-%d %H:%M:%S.%f").time() < time(14, 50):\n                api.wait_update()\n                # ... 策略代码 ...\n\n            # 取消 TargetPosTask 实例\n            target_pos_passive.cancel()\n\n            while not target_pos_passive.is_finished():  # 此循环等待 target_pos_passive 处理 cancel 结束\n                api.wait_update()  # 调用wait_update()，会对已经发出但还是未成交的委托单撤单\n\n            # 创建新的 TargetPosTask 实例\n            target_pos_active = TargetPosTask(api, "SHFE.rb2110", price="ACTIVE")\n            target_pos_active.set_target_volume(0)  # 平所有仓位\n\n            while True:\n                api.wait_update()\n                # ... 策略代码 ...\n\n            api.close()\n\n        '
        self._task.cancel()

    def is_finished(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        返回当前 TargetPosTask 实例是否已经结束。即如果后续调用此实例的 set_target_volume 函数会报错，此实例不会再下单或者撤单。\n\n        Returns:\n            bool: 当前 TargetPosTask 实例是否已经结束\n        '
        return self._task.done()

class InsertOrderUntilAllTradedTask(object):
    """追价下单task, 该task会在行情变化后自动撤单重下，直到全部成交
     （注：此类主要在tqsdk内部使用，并非简单用法，不建议用户使用）"""

    def __init__(self, api, symbol, direction, offset, volume, min_volume: Optional[int]=None, max_volume: Optional[int]=None, price: Union[str, Callable[[str], Union[float, int]]]='ACTIVE', trade_chan=None, trade_objs_chan=None, account: Optional[Union[TqAccount, TqKq, TqSim]]=None):
        if False:
            print('Hello World!')
        '\n        创建追价下单task实例\n\n        Args:\n            api (TqApi): TqApi实例，该task依托于指定api下单/撤单\n\n            symbol (str): 拟下单的合约symbol, 格式为 交易所代码.合约代码,  例如 "SHFE.cu1801"\n\n            direction (str): "BUY" 或 "SELL"\n\n            offset (str): "OPEN", "CLOSE" 或 "CLOSETODAY"\n\n            volume (int): 需要下单的手数\n\n            min_volume (int): [可选] **大单拆分模式下** 每笔最小下单的手数，默认不启用 **大单拆分模式**\n\n            max_volume (int): [可选] **大单拆分模式下** 每笔最大下单的手数，默认不启用 **大单拆分模式**\n\n            price (str / Callable): [可选]下单方式, 默认为 "ACTIVE"。\n                * "ACTIVE"：对价下单，在持仓调整过程中，若下单方向为买，对价为卖一价；若下单方向为卖，对价为买一价。\n                * "PASSIVE"：对价下单，在持仓调整过程中，若下单方向为买，对价为买一价；若下单方向为卖，对价为卖一价。\n                * Callable[[str], Union[float, int]]: 函数参数为下单方向，函数返回值是下单价格。如果返回 nan，程序会抛错。\n\n            trade_chan (TqChan): [可选]成交通知channel, 当有成交发生时会将成交手数(多头为正数，空头为负数)发到该channel上\n\n            trade_objs_chan (TqChan): [可选]成交对象通知channel, 当有成交发生时会将成交对象发送到该channel上\n\n            account (TqAccount/TqKq/TqSim): [可选]指定发送下单指令的账户实例, 多账户模式下，该参数必须指定\n        '
        self._api = api
        self._account = account
        self._symbol = symbol
        self._direction = _check_direction(direction)
        self._offset = _check_offset(offset)
        self._volume = _check_volume(volume)
        (self._min_volume, self._max_volume) = _check_volume_limit(min_volume, max_volume)
        self._price = _check_price(price)
        self._trade_chan = trade_chan
        self._trade_objs_chan = trade_objs_chan
        self._task = self._api.create_task(self._run())

    async def _run(self):
        """负责追价下单的task"""
        self._quote = await self._api.get_quote(self._symbol)
        while self._volume != 0:
            limit_price = self._get_price(self._direction)
            if limit_price != limit_price:
                raise Exception('设置价格函数返回 nan，无法处理。请检查后重试。')
            if self._min_volume and self._max_volume and (self._volume >= self._max_volume):
                this_volume = utils.RD.randint(self._min_volume, self._max_volume)
            else:
                this_volume = self._volume
            insert_order_task = InsertOrderTask(self._api, self._symbol, self._direction, self._offset, this_volume, limit_price=limit_price, trade_chan=self._trade_chan, trade_objs_chan=self._trade_objs_chan, account=self._account)
            order = await insert_order_task._order_chan.recv()
            check_chan = TqChan(self._api, last_only=True)
            check_task = self._api.create_task(self._check_price(check_chan, limit_price, order['order_id']))
            try:
                await asyncio.shield(insert_order_task._task)
                order = insert_order_task._order_chan.recv_latest(order)
                self._volume -= this_volume - order['volume_left']
                if order['volume_left'] != 0 and (not check_task.done()):
                    raise Exception('遇到错单: %s %s %s %d手 %f %s' % (self._symbol, self._direction, self._offset, this_volume, limit_price, order['last_msg']))
            finally:
                if self._api.get_order(order['order_id'], account=self._account).status == 'ALIVE':
                    self._api.cancel_order(order['order_id'], account=self._account)
                await check_chan.close()
                await check_task
                try:
                    await asyncio.wait_for(insert_order_task._task, timeout=30)
                except asyncio.TimeoutError:
                    raise Exception(f'InsertOrderTask 执行超时，30s 内报单未执行完。此错误产生可能的原因：可能是用户调用了 api.close() 之后，已经创建的 InsertOrderTask 无法正常结束。')

    def _get_price(self, direction):
        if False:
            print('Hello World!')
        '根据最新行情和下单方式计算出最优的下单价格'
        if self._price not in ('ACTIVE', 'PASSIVE'):
            assert isfunction(self._price)
            return self._price(direction)
        price_list = [self._quote.ask_price1, self._quote.bid_price1]
        if direction == 'SELL':
            price_list.reverse()
        if self._price == 'PASSIVE':
            price_list.reverse()
        limit_price = price_list[0]
        if limit_price != limit_price:
            limit_price = price_list[1]
        if limit_price != limit_price:
            limit_price = self._quote.last_price
        if limit_price != limit_price:
            limit_price = self._quote.pre_close
        return limit_price

    async def _check_price(self, update_chan, order_price, order_id):
        """判断价格是否变化的task"""
        async with self._api.register_update_notify(chan=update_chan):
            async for _ in update_chan:
                new_price = self._get_price(self._direction)
                if self._direction == 'BUY' and new_price > order_price or (self._direction == 'SELL' and new_price < order_price):
                    self._api.cancel_order(order_id, account=self._account)
                    break

class InsertOrderTask(object):
    """下单task （注：此类主要在tqsdk内部使用，并非简单用法，不建议用户使用）"""

    def __init__(self, api, symbol, direction, offset, volume, limit_price=None, order_chan=None, trade_chan=None, trade_objs_chan=None, account: Optional[Union[TqAccount, TqKq, TqSim]]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        创建下单task实例\n\n        Args:\n            api (TqApi): TqApi实例，该task依托于指定api下单/撤单\n\n            symbol (str): 拟下单的合约symbol, 格式为 交易所代码.合约代码,  例如 "SHFE.cu1801"\n\n            direction (str): "BUY" 或 "SELL"\n\n            offset (str): "OPEN", "CLOSE" 或 "CLOSETODAY"\n\n            volume (int): 需要下单的手数\n\n            limit_price (float): [可选]下单价格, 默认市价单\n\n            order_chan (TqChan): [可选]委托单通知channel, 当委托单状态发生时会将委托单信息发到该channel上\n\n            trade_chan (TqChan): [可选]成交通知channel, 当有成交发生时会将成交手数(多头为正数，空头为负数)发到该channel上\n\n            trade_objs_chan (TqChan): [可选]成交对象通知channel, 当有成交发生时会将成交对象发送到该channel上\n\n            account (TqAccount/TqKq/TqSim): [可选]指定发送下单指令的账户实例, 多账户模式下，该参数必须指定\n        '
        self._api = api
        self._account = account
        self._symbol = symbol
        self._direction = _check_direction(direction)
        self._offset = _check_offset(offset)
        self._volume = _check_volume(volume)
        self._offset = offset
        self._volume = int(volume)
        self._limit_price = float(limit_price) if limit_price is not None else None
        self._order_chan = order_chan if order_chan is not None else TqChan(self._api)
        self._trade_chan = trade_chan
        self._trade_objs_chan = trade_objs_chan
        self._task = self._api.create_task(self._run())

    async def _run(self):
        """负责下单的task"""
        order_id = utils._generate_uuid('PYSDK_target')
        order = self._api.insert_order(self._symbol, self._direction, self._offset, self._volume, self._limit_price, order_id=order_id, account=self._account)
        last_order = order.copy()
        last_left = self._volume
        all_trades_id = set()
        async with self._api.register_update_notify() as update_chan:
            await self._order_chan.send({k: v for (k, v) in last_order.items() if not k.startswith('_')})
            while order.status != 'FINISHED' or order.volume_orign - order.volume_left != sum([trade.volume for trade in order.trade_records.values()]):
                await update_chan.recv()
                if order.volume_left != last_left:
                    vol = last_left - order.volume_left
                    last_left = order.volume_left
                    if self._trade_chan:
                        await self._trade_chan.send(vol if order.direction == 'BUY' else -vol)
                if self._trade_objs_chan:
                    rest_trades_id = set(order.trade_records) - all_trades_id
                    for trade_id in rest_trades_id:
                        await self._trade_objs_chan.send({k: v for (k, v) in order.trade_records[trade_id].items() if not k.startswith('_')})
                        all_trades_id.add(trade_id)
                if order != last_order:
                    last_order = order.copy()
                    await self._order_chan.send({k: v for (k, v) in last_order.items() if not k.startswith('_')})