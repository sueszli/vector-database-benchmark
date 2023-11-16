__author__ = 'yanqiong'
import asyncio
from typing import Optional, Union
from tqsdk.algorithm.time_table_generater import _gen_random_list
from tqsdk.api import TqApi
from tqsdk.channel import TqChan
from tqsdk.datetime import _get_trading_timestamp, _get_trade_timestamp
from tqsdk.lib import InsertOrderUntilAllTradedTask
from tqsdk.tradeable import TqAccount, TqKq, TqSim
from tqsdk.rangeset import _rangeset_slice, _rangeset_head, _rangeset_length

class Twap(object):
    """
    天勤算法 - Twap

    Twap 算法实现了在设定的交易时间段内，完成设定的下单手数。

    构造 Twap 类的实例，该算法实例就会开始运行，根据以下逻辑下单：

    1. 将用户设置的总手数，拆分为一个随机手数列表，列表的值即为每次下单的手数，列表元素之和为总下单手数，同时每次下单手数也符合用户设置的每次下单手数的上下限；
    2. 将总的交易时间段拆分为随机时间间隔列表，列表的值即为每次下单的时间间隔，这些时间间隔相加应该等于总的下单时间；
    3. 每一次下单，在两个列表中分别取出下单手数、下单预计完成的时间，先用跟盘价下单，在当前时间间隔已经过去 2/3 或者只剩下 2s 时，主动撤掉未成交单，用对手价下单剩余手数；
    4. 在当前时间段已结束并且下单手数全部成交完，会开始下一次下单，重复第 3 步。

    基于以上逻辑，用户参数应该满足：

    平均每次下单时间 = duration / 下单次数 > 3s

    其中，下单次数 = 总的下单手数 / 平均每次下单手数 = 总的下单手数 / ((单次委托单最小下单手数 + 单次委托单最大下单手数) / 2)


    **注意**：

    时间段 duration，以 s 为单位，时长可以跨非交易时间段，但是不可以跨交易日。

    比如，SHFE.cu2101 的白盘交易时间段为 ["09:00:00" ～ "10:15:00"], ["10:30:00", "11:30:00"], ["13:30:00", "15:00:00"]，duration 设置为 1200 (20分钟)。

    如果当前行情时间是 2020-09-15 09:10:00，那么下单的时间应该在 2020-09-15 09:10:00 ～ 2020-09-15 09:30:00；
    如果当前行情时间是 2020-09-15 10:10:00，那么下单的时间应该在 2020-09-15 10:10:00 ～ 2020-09-15 10:15:00，以及 2020-09-15 10:30:00 ～ 2020-09-15 10:45:00。


    本模块不支持在回测中使用。
    """

    def __init__(self, api: TqApi, symbol: str, direction: str, offset: str, volume: int, duration: float, min_volume_each_order: int, max_volume_each_order: int, account: Optional[Union[TqAccount, TqKq, TqSim]]=None):
        if False:
            while True:
                i = 10
        '\n        创建 Twap 实例\n\n        Args:\n            api (TqApi): TqApi实例，该task依托于指定api下单/撤单\n\n            symbol (str): 拟下单的合约symbol, 格式为 交易所代码.合约代码,  例如 "SHFE.cu1801"\n\n            direction (str): "BUY" 或 "SELL"\n\n            offset (str): "OPEN", "CLOSE"，"CLOSETODAY"\n\n            volume (int): 需要下单的总手数\n\n            duration (int): 算法执行的时长，以秒为单位，时长可以跨非交易时间段，但是不可以跨交易日\n            * 设置为 60*10, 可以是 10:10～10:15 + 10:30~10:35\n\n            min_volume_each_order (int):单笔最小委托单，每笔委托单数默认在最小和最大值中产生\n\n            max_volume_each_order (int):单笔最大委托单，每笔委托单数默认在最小和最大值中产生\n\n            account (TqAccount/TqKq/TqSim): [可选]指定发送下单指令的账户实例, 多账户模式下，该参数必须指定\n\n        Example1::\n\n          from tqsdk import TqApi\n          from tqsdk.algorithm import Twap\n\n          api = TqApi(auth="快期账户,用户密码")\n          # 设置twap任务参数\n          target_twap = Twap(api,"SHFE.rb2012","BUY","OPEN",500,300,10,25)\n          # 启动循环\n          while True:\n            api.wait_update()\n            if target_twap.is_finished():\n                break\n          api.close()\n\n        Example2::\n\n          from tqsdk import TqApi\n          from tqsdk.algorithm import Twap\n\n          api = TqApi(auth="快期账户,用户密码")\n          target_twap = Twap(api,"SHFE.rb2012","BUY","OPEN",500,300,10,25)\n\n          num_of_trades = 0\n\n          while True:\n            api.wait_update()\n\n            if num_of_trades < len(target_twap.trades):\n              # 最新的成交\n              for i in range(num_of_trades - len(target_twap.trades), 0):\n                print("新的成交", target_twap.trades[i])\n              print(target_twap.average_trade_price)  # 打印出当前已经成交的平均价格\n              num_of_trades = len(target_twap.trades)\n\n            if target_twap.is_finished():\n                break\n\n          print("打印出 twap 全部成交以及成交均价")\n          print(target_twap.trades)\n          print(target_twap.average_trade_price)\n          api.close()\n        '
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
        self._api = api
        self._account = api._account._check_valid(account)
        if self._account is None:
            raise Exception(f'多账户模式下, 需要指定账户实例 account')
        self._symbol = symbol
        self._direction = direction
        self._offset = offset
        self._volume = int(volume)
        self._duration = duration
        self._min_volume_each_order = int(min_volume_each_order)
        self._max_volume_each_order = int(max_volume_each_order)
        if self._max_volume_each_order <= 0 or self._min_volume_each_order <= 0:
            raise Exception('请调整参数, min_volume_each_order、max_volume_each_order 必须是大于 0 的整数。')
        if self._min_volume_each_order > self._max_volume_each_order:
            raise Exception('请调整参数, min_volume_each_order 必须小于 max_volume_each_order。')
        (volume_list, interval_list) = self._get_volume_list()
        self._task = self._api.create_task(self._run(volume_list, interval_list))
        self._order_task = None
        self.trades = []
        self._trade_sum_volume = 0
        self._trade_sum_amount = 0
        self._trade_objs_chan = TqChan(self._api)
        self._trade_recv_task = self._api.create_task(self._trade_recv())

    @property
    def average_trade_price(self):
        if False:
            print('Hello World!')
        if self._trade_sum_volume == 0:
            return float('nan')
        else:
            return self._trade_sum_amount / self._trade_sum_volume

    async def _run(self, volume_list, interval_list):
        self._quote = await self._api.get_quote(self._symbol)
        (deadline_timestamp_list, strict_deadline_timestamp_list) = self._get_deadline_timestamp(interval_list)
        for i in range(len(volume_list)):
            exit_immediately = i + 1 == len(volume_list)
            await self._insert_order(volume_list[i], deadline_timestamp_list[i], strict_deadline_timestamp_list[i], exit_immediately)

    async def _trade_recv(self):
        try:
            async for trade in self._trade_objs_chan:
                self.trades.append(trade)
                self._trade_sum_volume += trade['volume']
                self._trade_sum_amount += trade['volume'] * trade['price']
        finally:
            await self._trade_objs_chan.close()

    def _get_deadline_timestamp(self, interval_list):
        if False:
            for i in range(10):
                print('nop')
        trading_timestamp = _get_trading_timestamp(self._quote, self._quote.datetime)
        trading_timestamp_nano_range = trading_timestamp['night'] + trading_timestamp['day']
        current_timestamp_nano = _get_trade_timestamp(self._quote.datetime, float('nan'))
        if not trading_timestamp_nano_range[0][0] <= current_timestamp_nano < trading_timestamp_nano_range[-1][1]:
            raise Exception('当前时间不在指定的交易时间段内')
        deadline_timestamp_list = []
        strict_deadline_timestamp_list = []
        for interval in interval_list:
            r = _rangeset_head(_rangeset_slice(trading_timestamp_nano_range, current_timestamp_nano), int(interval * 1000000000.0))
            strict_interval = interval - min(2, interval / 3)
            strict_r = _rangeset_head(_rangeset_slice(trading_timestamp_nano_range, current_timestamp_nano), int(strict_interval * 1000000000.0))
            if _rangeset_length(r) < int(interval * 1000000000.0):
                raise Exception('指定时间段超出当前交易日')
            deadline_timestamp_list.append(r[-1][1])
            strict_deadline_timestamp_list.append(strict_r[-1][1])
            current_timestamp_nano = r[-1][1]
        return (deadline_timestamp_list, strict_deadline_timestamp_list)

    async def _insert_order(self, volume, end_time, strict_end_time, exit_immediately):
        volume_left = volume
        try:
            trade_chan = TqChan(self._api)
            self._order_task = InsertOrderUntilAllTradedTask(self._api, self._symbol, self._direction, self._offset, volume=volume, price='PASSIVE', trade_chan=trade_chan, trade_objs_chan=self._trade_objs_chan, account=self._account)
            async with self._api.register_update_notify() as update_chan:
                async for _ in update_chan:
                    if _get_trade_timestamp(self._quote.datetime, float('nan')) > strict_end_time:
                        break
                    else:
                        while not trade_chan.empty():
                            v = await trade_chan.recv()
                            volume_left = volume_left - (v if self._direction == 'BUY' else -v)
                        if exit_immediately and volume_left == 0:
                            break
        finally:
            self._order_task._task.cancel()
            await asyncio.gather(self._order_task._task, return_exceptions=True)
            while not trade_chan.empty():
                v = await trade_chan.recv()
                volume_left = volume_left - (v if self._direction == 'BUY' else -v)
            await trade_chan.close()
            if volume_left > 0:
                await self._insert_order_active(volume_left)

    async def _insert_order_active(self, volume):
        try:
            trade_chan = TqChan(self._api)
            self._order_task = InsertOrderUntilAllTradedTask(self._api, self._symbol, self._direction, self._offset, volume=volume, price='ACTIVE', trade_chan=trade_chan, trade_objs_chan=self._trade_objs_chan, account=self._account)
            async for v in trade_chan:
                volume = volume - (v if self._direction == 'BUY' else -v)
                if volume == 0:
                    break
        finally:
            await trade_chan.close()
            self._order_task._task.cancel()
            await asyncio.gather(self._order_task._task, return_exceptions=True)

    def _get_volume_list(self):
        if False:
            i = 10
            return i + 15
        if self._volume < self._max_volume_each_order:
            return ([self._volume], [self._duration])
        volume_list = _gen_random_list(sum_val=self._volume, min_val=self._min_volume_each_order, max_val=self._max_volume_each_order)
        interval = int(self._duration / len(volume_list))
        if interval < 3:
            raise Exception('请调整参数, 每次下单时间间隔不能小于3s, 将单次下单手数阈值调大或者增长下单时间。')
        min_interval = int(max(3, interval - 2))
        max_interval = int(interval * 2 - max(3, interval - 2)) + 1
        interval_list = _gen_random_list(sum_val=self._duration, min_val=min_interval, max_val=max_interval, length=len(volume_list))
        return (volume_list, interval_list)

    def cancel(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        取消当前 Twap 算法实例，会将该实例已经发出但还是未成交的委托单撤单。\n\n        Example::\n\n          from tqsdk import TqApi\n          from tqsdk.algorithm import Twap\n\n          api = TqApi(auth="快期账户,用户密码")\n          # 设置twap任务参数\n          quote = api.get_quote("SHFE.rb2012")\n          target_twap = Twap(api,"SHFE.rb2012","BUY","OPEN",500,300,10,25)\n          api.wait_update()\n          # 运行代码。。。\n          target_twap.cancel()\n          while True:\n            api.wait_update()\n            if target_twap.is_finished():\n                break\n          api.close()\n        '
        if self._task.done():
            return
        self._task.cancel()

    def is_finished(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        返回当前 Twap 算法实例是否已经结束。即此实例不会再发出下单或者撤单的任何动作。\n\n        Returns:\n            bool: 当前 Twap 算法实例是否已经结束\n        '
        return self._task.done()