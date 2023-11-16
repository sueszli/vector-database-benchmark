__author__ = 'mayanqiong'
from tqsdk.tradeable.sim.trade_base import SimTradeBase
from tqsdk.tradeable.sim.utils import _get_stock_fee, _get_order_price, _get_dividend_ratio

class SimTradeStock(SimTradeBase):
    """
    天勤模拟交易账户，期货及商品期权
    """

    def _generate_account(self, init_balance):
        if False:
            for i in range(10):
                print('nop')
        return {'user_id': self._account_id, 'currency': 'CNY', 'market_value_his': 0.0, 'asset_his': init_balance, 'cost_his': 0.0, 'deposit': 0.0, 'withdraw': 0.0, 'dividend_balance_today': 0.0, 'available_his': init_balance, 'market_value': 0.0, 'asset': init_balance, 'available': init_balance, 'drawable': init_balance, 'buy_frozen_balance': 0.0, 'buy_frozen_fee': 0.0, 'buy_balance_today': 0.0, 'buy_fee_today': 0.0, 'sell_balance_today': 0.0, 'sell_fee_today': 0.0, 'cost': 0.0, 'hold_profit': 0.0, 'float_profit_today': 0.0, 'real_profit_today': 0.0, 'profit_today': 0.0, 'profit_rate_today': 0.0}

    def _generate_position(self, symbol, quote, underlying_quote) -> dict:
        if False:
            return 10
        return {'user_id': self._account_id, 'exchange_id': symbol.split('.', maxsplit=1)[0], 'instrument_id': symbol.split('.', maxsplit=1)[1], 'create_date': '', 'volume_his': 0, 'cost_his': 0.0, 'market_value_his': 0.0, 'real_profit_his': 0.0, 'shared_volume_today': 0, 'devidend_balance_today': 0.0, 'buy_volume_his': 0, 'buy_balance_his': 0.0, 'buy_fee_his': 0.0, 'sell_volume_his': 0, 'sell_balance_his': 0.0, 'sell_fee_his': 0.0, 'buy_volume_today': 0, 'buy_balance_today': 0.0, 'buy_fee_today': 0.0, 'sell_volume_today': 0, 'sell_balance_today': 0.0, 'sell_fee_today': 0.0, 'last_price': quote['last_price'], 'sell_volume_frozen': 0, 'sell_float_profit_today': 0.0, 'buy_float_profit_today': 0.0, 'cost': 0.0, 'volume': 0, 'market_value': 0.0, 'float_profit_today': 0.0, 'real_profit_today': 0.0, 'profit_today': 0.0, 'profit_rate_today': 0.0, 'hold_profit': 0.0, 'real_profit_total': 0.0, 'profit_total': 0.0, 'profit_rate_total': 0.0}

    def _generate_order(self, pack: dict) -> dict:
        if False:
            i = 10
            return i + 15
        'order 对象预处理'
        order = pack.copy()
        order['exchange_order_id'] = order['order_id']
        order['volume_orign'] = order['volume']
        order['volume_left'] = order['volume']
        order['frozen_balance'] = 0.0
        order['frozen_fee'] = 0.0
        order['last_msg'] = '报单成功'
        order['status'] = 'ALIVE'
        order['insert_date_time'] = self._get_trade_timestamp()
        del order['aid']
        del order['volume']
        self._append_to_diffs(['orders', order['order_id']], order)
        return order

    def _generate_trade(self, order, quote, price) -> dict:
        if False:
            i = 10
            return i + 15
        fee = _get_stock_fee(order['direction'], order['volume_left'], price)
        return {'user_id': order['user_id'], 'order_id': order['order_id'], 'trade_id': order['order_id'] + '|' + str(order['volume_left']), 'exchange_trade_id': order['order_id'] + '|' + str(order['volume_left']), 'exchange_id': order['exchange_id'], 'instrument_id': order['instrument_id'], 'direction': order['direction'], 'price': price, 'volume': order['volume_left'], 'trade_date_time': self._get_trade_timestamp(), 'fee': fee}

    def _on_settle(self):
        if False:
            return 10
        for symbol in self._orders:
            for order in self._orders[symbol].values():
                order['frozen_balance'] = 0.0
                order['frozen_fee'] = 0.0
                order['last_msg'] = '交易日结束，自动撤销当日有效的委托单（GFD）'
                order['status'] = 'FINISHED'
                self._append_to_diffs(['orders', order['order_id']], order)
        dividend_balance_today = 0.0
        for position in self._positions.values():
            symbol = f"{position['exchange_id']}.{position['instrument_id']}"
            (quote, _) = self._get_quotes_by_symbol(symbol)
            (stock_dividend, cash_dividend) = _get_dividend_ratio(quote)
            position['volume_his'] = position['volume']
            position['cost_his'] = position['cost']
            position['market_value_his'] = position['market_value']
            position['real_profit_his'] = position['real_profit_today']
            position['shared_volume_today'] = stock_dividend * position['volume']
            position['devidend_balance_today'] = cash_dividend * position['volume']
            if position['shared_volume_today'] > 0.0 or position['devidend_balance_today'] > 0.0:
                position['volume'] += position['shared_volume_today']
                position['market_value'] -= position['devidend_balance_today']
                position['last_price'] = position['market_value'] / position['volume']
                dividend_balance_today += position['devidend_balance_today']
            position['buy_volume_his'] = position['buy_volume_today']
            position['buy_balance_his'] = position['buy_balance_today']
            position['buy_fee_his'] = position['buy_fee_today']
            position['sell_volume_his'] = position['sell_volume_today']
            position['sell_balance_his'] = position['sell_balance_today']
            position['sell_fee_his'] = position['sell_fee_today']
            position['buy_volume_today'] = 0
            position['buy_balance_today'] = 0.0
            position['buy_fee_today'] = 0.0
            position['sell_volume_today'] = 0
            position['sell_balance_today'] = 0.0
            position['sell_fee_today'] = 0.0
            position['sell_volume_frozen'] = 0
            position['buy_avg_price'] = 0.0
            position['sell_float_profit_today'] = 0.0
            position['buy_float_profit_today'] = 0.0
            position['float_profit_today'] = 0.0
            position['real_profit_today'] = 0.0
            position['profit_today'] = 0.0
            position['profit_rate_today'] = 0.0
            position['hold_profit'] = 0.0
            self._append_to_diffs(['positions', symbol], position)
        self._account['dividend_balance_today'] = dividend_balance_today
        self._account['market_value_his'] = self._account['market_value']
        self._account['asset_his'] = self._account['asset']
        self._account['cost_his'] = self._account['cost']
        self._account['available_his'] = self._account['available'] + self._account['buy_frozen_balance'] + self._account['buy_frozen_fee']
        self._account['buy_frozen_balance'] = 0.0
        self._account['buy_frozen_fee'] = 0.0
        self._account['buy_balance_today'] = 0.0
        self._account['buy_fee_today'] = 0.0
        self._account['sell_balance_today'] = 0.0
        self._account['sell_fee_today'] = 0.0
        self._account['asset'] += self._account['dividend_balance_today']
        self._account['market_value'] -= self._account['dividend_balance_today']
        self._account['available'] = self._account['asset'] - self._account['market_value']
        self._account['drawable'] = self._account['available']
        self._account['hold_profit'] = 0.0
        self._account['float_profit_today'] = 0.0
        self._account['real_profit_today'] = 0.0
        self._account['profit_today'] = 0.0
        self._account['profit_rate_today'] = 0.0
        self._append_to_diffs(['accounts', 'CNY'], self._account)

    def _check_insert_order(self, order, symbol, position, quote, underlying_quote=None):
        if False:
            while True:
                i = 10
        if quote['ins_class'] != 'STOCK':
            order['last_msg'] = '不支持的合约类型，TqSimStock 只支持股票模拟交易'
            order['status'] = 'FINISHED'
        if order['status'] == 'ALIVE' and (not self._is_in_trading_time(quote)):
            order['last_msg'] = '下单失败, 不在可交易时间段内'
            order['status'] = 'FINISHED'
        if order['status'] == 'ALIVE' and order['direction'] == 'BUY':
            price = _get_order_price(quote, order)
            order['frozen_balance'] = price * order['volume_orign']
            order['frozen_fee'] = _get_stock_fee(order['direction'], order['volume_orign'], price)
            if order['frozen_balance'] + order['frozen_fee'] > self._account['available']:
                order['frozen_balance'] = 0.0
                order['frozen_fee'] = 0.0
                order['last_msg'] = '开仓资金不足'
                order['status'] = 'FINISHED'
        if order['status'] == 'ALIVE' and order['direction'] == 'SELL':
            if position['volume_his'] + position['shared_volume_today'] - position['sell_volume_today'] - position['sell_volume_frozen'] < order['volume_orign']:
                order['last_msg'] = '平仓手数不足'
                order['status'] = 'FINISHED'
        if order['status'] == 'FINISHED':
            self._append_to_diffs(['orders', order['order_id']], order)

    def _on_insert_order(self, order, symbol, position, quote, underlying_quote=None):
        if False:
            for i in range(10):
                print('nop')
        '记录在 orderbook'
        if order['direction'] == 'BUY':
            self._adjust_account_by_order(buy_frozen_balance=order['frozen_balance'], buy_frozen_fee=order['frozen_fee'])
            self._append_to_diffs(['accounts', 'CNY'], self._account)
        else:
            position['sell_volume_frozen'] += order['volume_orign']
            self._append_to_diffs(['positions', symbol], position)

    def _on_order_failed(self, symbol, order):
        if False:
            print('Hello World!')
        origin_frozen_balance = order['frozen_balance']
        origin_frozen_fee = order['frozen_fee']
        order['frozen_balance'] = 0.0
        order['frozen_fee'] = 0.0
        self._append_to_diffs(['orders', order['order_id']], order)
        if order['direction'] == 'BUY':
            self._adjust_account_by_order(buy_frozen_balance=-origin_frozen_balance, buy_frozen_fee=-origin_frozen_fee)
            self._append_to_diffs(['accounts', 'CNY'], self._account)
        else:
            position = self._positions[symbol]
            position['sell_volume_frozen'] -= order['volume_orign']
            self._append_to_diffs(['positions', symbol], position)

    def _on_order_traded(self, order, trade, symbol, position, quote, underlying_quote):
        if False:
            print('Hello World!')
        origin_frozen_balance = order['frozen_balance']
        origin_frozen_fee = order['frozen_fee']
        order['frozen_balance'] = 0.0
        order['frozen_fee'] = 0.0
        order['volume_left'] = 0
        self._append_to_diffs(['trades', trade['trade_id']], trade)
        self._append_to_diffs(['orders', order['order_id']], order)
        if order['direction'] == 'BUY':
            if position['volume'] == 0:
                position['create_date'] = quote['datetime'][:10]
            self._adjust_account_by_order(buy_frozen_balance=-origin_frozen_balance, buy_frozen_fee=-origin_frozen_fee)
            buy_balance = trade['volume'] * trade['price']
            position['buy_volume_today'] += trade['volume']
            position['buy_balance_today'] += buy_balance
            position['buy_fee_today'] += trade['fee']
            self._adjust_account_by_trade(buy_fee=trade['fee'], buy_balance=buy_balance)
            self._adjust_position_account(position, pre_last_price=trade['price'], last_price=position['last_price'], buy_volume=trade['volume'], buy_balance=buy_balance, buy_fee=trade['fee'])
        else:
            position['sell_volume_frozen'] -= order['volume_orign']
            sell_balance = trade['volume'] * trade['price']
            position['sell_volume_today'] += trade['volume']
            position['sell_balance_today'] += sell_balance
            position['sell_fee_today'] += trade['fee']
            self._adjust_account_by_trade(sell_fee=trade['fee'], sell_balance=sell_balance)
            self._adjust_position_account(position, last_price=quote['last_price'], sell_volume=trade['volume'], sell_balance=sell_balance, sell_fee=trade['fee'])
        self._append_to_diffs(['positions', symbol], position)
        self._append_to_diffs(['accounts', 'CNY'], self._account)

    def _on_update_quotes(self, symbol, position, quote, underlying_quote):
        if False:
            return 10
        if position['volume'] > 0:
            if position['last_price'] != quote['last_price']:
                self._adjust_position_account(position, pre_last_price=position['last_price'], last_price=quote['last_price'])
                position['last_price'] = quote['last_price']
        position['last_price'] = quote['last_price']
        self._append_to_diffs(['positions', symbol], position)
        self._append_to_diffs(['accounts', 'CNY'], self._account)

    def _adjust_position_account(self, position, pre_last_price=float('nan'), last_price=float('nan'), buy_volume=0, buy_balance=0, buy_fee=0, sell_volume=0, sell_balance=0, sell_fee=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        价格变化，使得 position 中的以下计算字段需要修改，这个函数计算出需要修改的差值部分，计算出差值部分修改 position、account\n        有两种情况下调用\n        1. 委托单 FINISHED，且全部成交，分为4种：buy_open, buy_close, sell_open, sell_close\n        2. 行情跳动\n        '
        assert [buy_volume, sell_volume].count(0) >= 1
        if buy_volume > 0:
            position['volume'] += buy_volume
            cost = buy_balance + buy_fee
            market_value = buy_volume * position['last_price']
            position['buy_avg_price'] = (position['buy_balance_today'] + position['buy_fee_today']) / position['buy_volume_today']
            buy_float_profit_today = (position['volume'] - (position['volume_his'] - position['sell_volume_today'])) * (last_price - position['buy_avg_price'])
            self._adjust_position(position, cost=cost, market_value=market_value, sell_float_profit_today=0, buy_float_profit_today=buy_float_profit_today, real_profit_today=0)
            self._adjust_account_by_position(market_value=market_value, cost=cost, float_profit_today=buy_float_profit_today, real_profit_today=0)
        elif sell_volume > 0:
            position['volume'] -= sell_volume
            cost = -sell_volume * (position['cost_his'] / position['volume_his'])
            market_value = -sell_volume * position['last_price']
            real_profit_today = sell_volume / position['volume_his'] * position['sell_float_profit_today']
            sell_float_profit_today = position['sell_float_profit_today'] - real_profit_today
            self._adjust_position(position, cost=cost, market_value=market_value, sell_float_profit_today=sell_float_profit_today, buy_float_profit_today=0, real_profit_today=real_profit_today)
            self._adjust_account_by_position(market_value=market_value, cost=cost, float_profit_today=sell_float_profit_today, real_profit_today=real_profit_today)
        else:
            market_value = position['volume'] * last_price - position['market_value']
            sell_float_profit_today = (position['volume_his'] - position['sell_volume_today']) * (last_price - pre_last_price)
            buy_float_profit_today = (position['volume'] - (position['volume_his'] - position['sell_volume_today'])) * (last_price - position['buy_avg_price'])
            self._adjust_position(position, cost=0, market_value=market_value, sell_float_profit_today=sell_float_profit_today, buy_float_profit_today=buy_float_profit_today, real_profit_today=0)
            self._adjust_account_by_position(market_value=market_value, cost=0, float_profit_today=sell_float_profit_today + buy_float_profit_today, real_profit_today=0)

    def _adjust_position(self, position, cost=0, market_value=0, sell_float_profit_today=0, buy_float_profit_today=0, real_profit_today=0):
        if False:
            for i in range(10):
                print('nop')
        position['sell_float_profit_today'] += sell_float_profit_today
        position['buy_float_profit_today'] += buy_float_profit_today
        position['cost'] += cost
        position['market_value'] += market_value
        position['float_profit_today'] += sell_float_profit_today + buy_float_profit_today
        position['real_profit_today'] += real_profit_today
        position['profit_today'] += sell_float_profit_today + buy_float_profit_today + real_profit_today
        position['hold_profit'] += market_value - cost
        position['real_profit_total'] += real_profit_today
        position['profit_total'] += real_profit_today + (market_value - cost)
        if position['cost'] > 0:
            position['profit_rate_today'] = position['profit_today'] / position['cost']
        else:
            position['profit_rate_today'] = position['profit_today'] / position['market_value_his'] if position['market_value_his'] > 0 else 0.0
        if position['cost'] > 0:
            position['profit_rate_total'] = position['profit_total'] / position['cost']
        else:
            position['profit_rate_total'] = position['profit_total'] / position['cost_his'] if position['cost_his'] > 0 else 0.0

    def _adjust_account_by_trade(self, buy_fee=0, buy_balance=0, sell_fee=0, sell_balance=0):
        if False:
            while True:
                i = 10
        '由成交引起的 account 原始字段变化，account 需要更新的计算字段'
        self._account['buy_balance_today'] += buy_balance
        self._account['buy_fee_today'] += buy_fee
        self._account['sell_balance_today'] += sell_balance
        self._account['sell_fee_today'] += sell_fee
        self._account['available'] += sell_balance - buy_fee - sell_fee - buy_balance
        self._account['asset'] += sell_balance - buy_fee - sell_fee - buy_balance
        self._account['drawable'] = max(self._account['available_his'] + min(0, self._account['sell_balance_today'] - self._account['buy_balance_today'] - self._account['buy_fee_today'] - self._account['buy_frozen_balance'] - self._account['buy_frozen_fee']), 0)

    def _adjust_account_by_position(self, market_value=0, cost=0, float_profit_today=0, real_profit_today=0):
        if False:
            return 10
        '由 position 变化，account 需要更新的计算字段'
        self._account['market_value'] += market_value
        self._account['cost'] += cost
        self._account['float_profit_today'] += float_profit_today
        self._account['real_profit_today'] += real_profit_today
        self._account['asset'] += market_value
        self._account['drawable'] = max(self._account['available_his'] + min(0, self._account['sell_balance_today'] - self._account['buy_balance_today'] - self._account['buy_fee_today'] - self._account['buy_frozen_balance'] - self._account['buy_frozen_fee']), 0)
        self._account['hold_profit'] = self._account['market_value'] - self._account['cost']
        self._account['profit_today'] = self._account['float_profit_today'] + self._account['real_profit_today']
        if self._account['cost'] > 0:
            self._account['profit_rate_today'] = self._account['profit_today'] / self._account['cost']
        else:
            self._account['profit_rate_today'] = self._account['profit_today'] / self._account['asset_his'] if self._account['asset_his'] > 0 else 0.0

    def _adjust_account_by_order(self, buy_frozen_balance=0, buy_frozen_fee=0):
        if False:
            i = 10
            return i + 15
        '由 order 变化，account 需要更新的计算字段'
        self._account['buy_frozen_balance'] += buy_frozen_balance
        self._account['buy_frozen_fee'] += buy_frozen_fee
        self._account['available'] -= buy_frozen_balance + buy_frozen_fee
        self._account['drawable'] = max(self._account['available_his'] + min(0, self._account['sell_balance_today'] - self._account['buy_balance_today'] - self._account['buy_fee_today'] - self._account['buy_frozen_balance'] - self._account['buy_frozen_fee']), 0)