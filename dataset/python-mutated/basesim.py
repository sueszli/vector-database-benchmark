__author__ = 'mayanqiong'
import asyncio
import time
from abc import abstractmethod
from typing import Type, Union
from tqsdk.channel import TqChan
from tqsdk.datetime import _get_trading_day_from_timestamp, _get_trading_day_end_time, _get_trade_timestamp, _is_in_trading_time, _timestamp_nano_to_str, _str_to_timestamp_nano
from tqsdk.diff import _get_obj, _register_update_chan, _merge_diff
from tqsdk.entity import Entity
from tqsdk.objs import Quote
from tqsdk.tradeable.tradeable import Tradeable
from tqsdk.tradeable.sim.trade_future import SimTrade
from tqsdk.tradeable.sim.trade_stock import SimTradeStock
from tqsdk.utils import _query_for_quote

class BaseSim(Tradeable):

    def __init__(self, account_id, init_balance, trade_class: Union[Type[SimTrade], Type[SimTradeStock]]) -> None:
        if False:
            i = 10
            return i + 15
        self._account_id = account_id
        super(BaseSim, self).__init__()
        self.trade_log = {}
        self.tqsdk_stat = {}
        self._init_balance = init_balance
        self._current_datetime = '1990-01-01 00:00:00.000000'
        self._trading_day_end = '1990-01-01 18:00:00.000000'
        self._local_time_record = float('nan')
        self._sim_trade = trade_class(account_key=self._account_key, account_id=self._account_id, init_balance=self._init_balance, get_trade_timestamp=self._get_trade_timestamp, is_in_trading_time=self._is_in_trading_time)
        self._data = Entity()
        self._data._instance_entity([])
        self._prototype = {'quotes': {'#': Quote(self)}}
        self._quote_tasks = {}

    @property
    def _account_name(self):
        if False:
            i = 10
            return i + 15
        return self._account_id

    @property
    def _account_info(self):
        if False:
            return 10
        info = super(BaseSim, self)._account_info
        info.update({'account_id': self._account_id})
        return info

    async def _run(self, api, api_send_chan, api_recv_chan, md_send_chan, md_recv_chan):
        """模拟交易task"""
        self._api = api
        self._tqsdk_backtest = {}
        self._logger = api._logger.getChild('TqSim')
        self._api_send_chan = api_send_chan
        self._api_recv_chan = api_recv_chan
        self._md_send_chan = md_send_chan
        self._md_recv_chan = md_recv_chan
        self._pending_subscribe_downstream = False
        self._pending_subscribe_upstream = False
        self._all_subscribe = set()
        self._has_send_init_account = False
        try:
            await super(BaseSim, self)._run(api, api_send_chan, api_recv_chan, md_send_chan, md_recv_chan)
        finally:
            self._handle_stat_report()
            for s in self._quote_tasks:
                self._quote_tasks[s]['task'].cancel()
            await asyncio.gather(*[self._quote_tasks[s]['task'] for s in self._quote_tasks], return_exceptions=True)

    async def _handle_recv_data(self, pack, chan):
        """
        处理所有上游收到的数据包，这里应该将需要发送给下游的数据 append 到 self._diffs
        pack: 收到的数据包
        chan: 收到此数据包的 channel
        """
        self._pending_subscribe_upstream = False
        if pack['aid'] == 'rtn_data':
            self._md_recv(pack)
            await asyncio.gather(*[quote_task['quote_chan'].join() for quote_task in self._quote_tasks.values()])
        if self._tqsdk_backtest != {} and self._tqsdk_backtest['current_dt'] >= self._tqsdk_backtest['end_dt']:
            self._handle_stat_report()

    async def _handle_req_data(self, pack):
        """
        处理所有下游发送的非 peek_message 数据包
        这里应该将发送的请求转发到指定的某个上游 channel
        """
        if self._is_self_trade_pack(pack):
            if pack['aid'] == 'insert_order':
                symbol = pack['exchange_id'] + '.' + pack['instrument_id']
                if symbol not in self._quote_tasks:
                    quote_chan = TqChan(self._api)
                    order_chan = TqChan(self._api)
                    self._quote_tasks[symbol] = {'quote_chan': quote_chan, 'order_chan': order_chan, 'task': self._api.create_task(self._quote_handler(symbol, quote_chan, order_chan))}
                await self._quote_tasks[symbol]['order_chan'].send(pack)
            else:
                for symbol in self._quote_tasks:
                    await self._quote_tasks[symbol]['order_chan'].send(pack)
        elif pack['aid'] == 'subscribe_quote':
            await self._subscribe_quote(set(pack['ins_list'].split(',')))
        else:
            await self._md_send_chan.send(pack)

    async def _on_send_diff(self, pending_peek):
        if pending_peek and self._pending_subscribe_downstream:
            await self._send_subscribe_quote()

    async def _subscribe_quote(self, symbols: [set, str]):
        """
        这里只会增加订阅合约，不会退订合约
        todo: 这里用到了 self._pending_peek ，父类的内部变量
        """
        symbols = symbols if isinstance(symbols, set) else {symbols}
        if symbols - self._all_subscribe:
            self._all_subscribe |= symbols
            if self._pending_peek and (not self._pending_subscribe_upstream):
                await self._send_subscribe_quote()
            else:
                self._pending_subscribe_downstream = True

    async def _send_subscribe_quote(self):
        self._pending_subscribe_upstream = True
        self._pending_subscribe_downstream = False
        await self._md_send_chan.send({'aid': 'subscribe_quote', 'ins_list': ','.join(self._all_subscribe)})

    def _handle_stat_report(self):
        if False:
            return 10
        if self.tqsdk_stat:
            return
        self._settle()
        self._report()
        self._diffs.append({'trade': {self._account_key: {'accounts': {'CNY': {'_tqsdk_stat': self.tqsdk_stat}}}}})

    async def _ensure_quote_info(self, symbol, quote_chan):
        """quote收到合约信息后返回"""
        quote = _get_obj(self._data, ['quotes', symbol], Quote(self._api))
        if quote.get('price_tick') == quote.get('price_tick'):
            return quote.copy()
        if quote.get('price_tick') != quote.get('price_tick'):
            await self._md_send_chan.send(_query_for_quote(symbol))
        async for _ in quote_chan:
            quote_chan.task_done()
            if quote.get('price_tick') == quote.get('price_tick'):
                return quote.copy()

    async def _ensure_quote(self, symbol, quote_chan):
        """quote收到行情以及合约信息后返回"""
        quote = _get_obj(self._data, ['quotes', symbol], Quote(self._api))
        _register_update_chan(quote, quote_chan)
        if quote.get('datetime', '') and quote.get('price_tick') == quote.get('price_tick'):
            return quote.copy()
        if quote.get('price_tick') != quote.get('price_tick'):
            await self._md_send_chan.send(_query_for_quote(symbol))
        async for _ in quote_chan:
            quote_chan.task_done()
            if quote.get('datetime', '') and quote.get('price_tick') == quote.get('price_tick'):
                return quote.copy()

    async def _quote_handler(self, symbol, quote_chan, order_chan):
        try:
            await self._subscribe_quote(symbol)
            quote = await self._ensure_quote(symbol, quote_chan)
            if quote['ins_class'].endswith('INDEX') and quote['exchange_id'] == 'KQ':
                if 'margin' not in quote:
                    quote_m = await self._ensure_quote_info(symbol.replace('KQ.i', 'KQ.m'), quote_chan)
                    quote_underlying = await self._ensure_quote_info(quote_m['underlying_symbol'], quote_chan)
                    self._data['quotes'][symbol]['margin'] = quote_underlying['margin']
                    self._data['quotes'][symbol]['commission'] = quote_underlying['commission']
                    quote.update(self._data['quotes'][symbol])
            underlying_quote = None
            if quote['ins_class'].endswith('OPTION'):
                underlying_symbol = quote['underlying_symbol']
                await self._subscribe_quote(underlying_symbol)
                underlying_quote = await self._ensure_quote(underlying_symbol, quote_chan)
            while not quote_chan.empty():
                quote_chan.recv_nowait()
                quote_chan.task_done()
            quote.update(self._data['quotes'][symbol])
            if underlying_quote:
                underlying_quote.update(self._data['quotes'][underlying_symbol])
            task = self._api.create_task(self._forward_chan_handler(order_chan, quote_chan))
            quotes = {symbol: quote}
            if underlying_quote:
                quotes[underlying_symbol] = underlying_quote
            self._sim_trade.update_quotes(symbol, {'quotes': quotes})
            async for pack in quote_chan:
                if 'aid' not in pack:
                    (diffs, orders_events) = self._sim_trade.update_quotes(symbol, pack)
                    self._handle_diffs(diffs, orders_events, 'match order')
                elif pack['aid'] == 'insert_order':
                    (diffs, orders_events) = self._sim_trade.insert_order(symbol, pack)
                    self._handle_diffs(diffs, orders_events, 'insert order')
                elif pack['aid'] == 'cancel_order':
                    (diffs, orders_events) = self._sim_trade.cancel_order(symbol, pack)
                    self._handle_diffs(diffs, orders_events, 'cancel order')
                quote_chan.task_done()
        finally:
            await quote_chan.close()
            await order_chan.close()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _forward_chan_handler(self, chan_from, chan_to):
        async for pack in chan_from:
            await chan_to.send(pack)

    def _md_recv(self, pack):
        if False:
            while True:
                i = 10
        for d in pack['data']:
            self._diffs.append(d)
            if not self._has_send_init_account and (not d.get('mdhis_more_data', True)):
                self._diffs.append(self._sim_trade.init_snapshot())
                self._diffs.append({'trade': {self._account_key: {'trade_more_data': False}}})
                self._has_send_init_account = True
            _tqsdk_backtest = d.get('_tqsdk_backtest', {})
            if _tqsdk_backtest:
                self._tqsdk_backtest.update(_tqsdk_backtest)
                self._current_datetime = _timestamp_nano_to_str(self._tqsdk_backtest['current_dt'])
                self._local_time_record = float('nan')
            quotes_diff = d.get('quotes', {})
            for (symbol, quote_diff) in quotes_diff.items():
                if quote_diff is None:
                    continue
                if quote_diff.get('datetime', '') > self._current_datetime:
                    self._current_datetime = quote_diff['datetime']
                    self._local_time_record = time.time() - 0.005 if not self._tqsdk_backtest else float('nan')
                if self._current_datetime > self._trading_day_end:
                    self._settle()
                    trading_day = _get_trading_day_from_timestamp(self._get_current_timestamp())
                    self._trading_day_end = _timestamp_nano_to_str(_get_trading_day_end_time(trading_day) - 999)
            if quotes_diff:
                _merge_diff(self._data, {'quotes': quotes_diff}, self._prototype, persist=False, reduce_diff=False, notify_update_diff=True)

    def _handle_diffs(self, diffs, orders_events, msg):
        if False:
            for i in range(10):
                print('nop')
        '\n        处理 sim_trade 返回的 diffs\n        orders_events 为持仓变更事件，依次屏幕输出信息，打印日志\n        '
        self._diffs += diffs
        for order in orders_events:
            if order['status'] == 'FINISHED':
                self._handle_on_finished(msg, order)
            else:
                assert order['status'] == 'ALIVE'
                self._handle_on_alive(msg, order)

    def _settle(self):
        if False:
            for i in range(10):
                print('nop')
        if self._trading_day_end[:10] == '1990-01-01':
            return
        (diffs, orders_events, trade_log) = self._sim_trade.settle()
        self._handle_diffs(diffs, orders_events, 'settle')
        self.trade_log[self._trading_day_end[:10]] = trade_log

    @abstractmethod
    def _handle_on_alive(self, msg, order):
        if False:
            i = 10
            return i + 15
        '\n        在 order 状态变为 ALIVE 调用，屏幕输出信息，打印日志\n        '
        pass

    @abstractmethod
    def _handle_on_finished(self, msg, order):
        if False:
            i = 10
            return i + 15
        '\n        在 order 状态变为 FINISHED 调用，屏幕输出信息，打印日志\n        '
        pass

    @abstractmethod
    def _report(self):
        if False:
            return 10
        pass

    def _get_current_timestamp(self):
        if False:
            i = 10
            return i + 15
        return _str_to_timestamp_nano(self._current_datetime)

    def _get_trade_timestamp(self):
        if False:
            i = 10
            return i + 15
        return _get_trade_timestamp(self._current_datetime, self._local_time_record)

    def _is_in_trading_time(self, quote):
        if False:
            for i in range(10):
                print('nop')
        return _is_in_trading_time(quote, self._current_datetime, self._local_time_record)