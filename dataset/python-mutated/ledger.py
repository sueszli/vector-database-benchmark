from __future__ import division
from collections import namedtuple, OrderedDict
from functools import partial
from math import isnan
import logbook
import numpy as np
import pandas as pd
from six import iteritems, itervalues, PY2
from zipline.assets import Future
from zipline.finance.transaction import Transaction
import zipline.protocol as zp
from zipline.utils.sentinel import sentinel
from .position import Position
from ._finance_ext import PositionStats, calculate_position_tracker_stats, update_position_last_sale_prices
log = logbook.Logger('Performance')

class PositionTracker(object):
    """The current state of the positions held.

    Parameters
    ----------
    data_frequency : {'daily', 'minute'}
        The data frequency of the simulation.
    """

    def __init__(self, data_frequency):
        if False:
            i = 10
            return i + 15
        self.positions = OrderedDict()
        self._unpaid_dividends = {}
        self._unpaid_stock_dividends = {}
        self._positions_store = zp.Positions()
        self.data_frequency = data_frequency
        self._dirty_stats = True
        self._stats = PositionStats.new()

    def update_position(self, asset, amount=None, last_sale_price=None, last_sale_date=None, cost_basis=None):
        if False:
            i = 10
            return i + 15
        self._dirty_stats = True
        if asset not in self.positions:
            position = Position(asset)
            self.positions[asset] = position
        else:
            position = self.positions[asset]
        if amount is not None:
            position.amount = amount
        if last_sale_price is not None:
            position.last_sale_price = last_sale_price
        if last_sale_date is not None:
            position.last_sale_date = last_sale_date
        if cost_basis is not None:
            position.cost_basis = cost_basis

    def execute_transaction(self, txn):
        if False:
            while True:
                i = 10
        self._dirty_stats = True
        asset = txn.asset
        if asset not in self.positions:
            position = Position(asset)
            self.positions[asset] = position
        else:
            position = self.positions[asset]
        position.update(txn)
        if position.amount == 0:
            del self.positions[asset]
            try:
                del self._positions_store[asset]
            except KeyError:
                pass

    def handle_commission(self, asset, cost):
        if False:
            i = 10
            return i + 15
        if asset in self.positions:
            self._dirty_stats = True
            self.positions[asset].adjust_commission_cost_basis(asset, cost)

    def handle_splits(self, splits):
        if False:
            i = 10
            return i + 15
        'Processes a list of splits by modifying any positions as needed.\n\n        Parameters\n        ----------\n        splits: list\n            A list of splits.  Each split is a tuple of (asset, ratio).\n\n        Returns\n        -------\n        int: The leftover cash from fractional shares after modifying each\n            position.\n        '
        total_leftover_cash = 0
        for (asset, ratio) in splits:
            if asset in self.positions:
                self._dirty_stats = True
                position = self.positions[asset]
                leftover_cash = position.handle_split(asset, ratio)
                total_leftover_cash += leftover_cash
        return total_leftover_cash

    def earn_dividends(self, cash_dividends, stock_dividends):
        if False:
            print('Hello World!')
        "Given a list of dividends whose ex_dates are all the next trading\n        day, calculate and store the cash and/or stock payments to be paid on\n        each dividend's pay date.\n\n        Parameters\n        ----------\n        cash_dividends : iterable of (asset, amount, pay_date) namedtuples\n\n        stock_dividends: iterable of (asset, payment_asset, ratio, pay_date)\n            namedtuples.\n        "
        for cash_dividend in cash_dividends:
            self._dirty_stats = True
            div_owed = self.positions[cash_dividend.asset].earn_dividend(cash_dividend)
            try:
                self._unpaid_dividends[cash_dividend.pay_date].append(div_owed)
            except KeyError:
                self._unpaid_dividends[cash_dividend.pay_date] = [div_owed]
        for stock_dividend in stock_dividends:
            self._dirty_stats = True
            div_owed = self.positions[stock_dividend.asset].earn_stock_dividend(stock_dividend)
            try:
                self._unpaid_stock_dividends[stock_dividend.pay_date].append(div_owed)
            except KeyError:
                self._unpaid_stock_dividends[stock_dividend.pay_date] = [div_owed]

    def pay_dividends(self, next_trading_day):
        if False:
            i = 10
            return i + 15
        '\n        Returns a cash payment based on the dividends that should be paid out\n        according to the accumulated bookkeeping of earned, unpaid, and stock\n        dividends.\n        '
        net_cash_payment = 0.0
        try:
            payments = self._unpaid_dividends[next_trading_day]
            del self._unpaid_dividends[next_trading_day]
        except KeyError:
            payments = []
        for payment in payments:
            net_cash_payment += payment['amount']
        try:
            stock_payments = self._unpaid_stock_dividends[next_trading_day]
        except KeyError:
            stock_payments = []
        for stock_payment in stock_payments:
            payment_asset = stock_payment['payment_asset']
            share_count = stock_payment['share_count']
            if payment_asset in self.positions:
                position = self.positions[payment_asset]
            else:
                position = self.positions[payment_asset] = Position(payment_asset)
            position.amount += share_count
        return net_cash_payment

    def maybe_create_close_position_transaction(self, asset, dt, data_portal):
        if False:
            while True:
                i = 10
        if not self.positions.get(asset):
            return None
        amount = self.positions.get(asset).amount
        price = data_portal.get_spot_value(asset, 'price', dt, self.data_frequency)
        if isnan(price):
            price = self.positions.get(asset).last_sale_price
        return Transaction(asset=asset, amount=-amount, dt=dt, price=price, order_id=None)

    def get_positions(self):
        if False:
            for i in range(10):
                print('nop')
        positions = self._positions_store
        for (asset, pos) in iteritems(self.positions):
            positions[asset] = pos.protocol_position
        return positions

    def get_position_list(self):
        if False:
            print('Hello World!')
        return [pos.to_dict() for (asset, pos) in iteritems(self.positions) if pos.amount != 0]

    def sync_last_sale_prices(self, dt, data_portal, handle_non_market_minutes=False):
        if False:
            print('Hello World!')
        self._dirty_stats = True
        if handle_non_market_minutes:
            previous_minute = data_portal.trading_calendar.previous_minute(dt)
            get_price = partial(data_portal.get_adjusted_value, field='price', dt=previous_minute, perspective_dt=dt, data_frequency=self.data_frequency)
        else:
            get_price = partial(data_portal.get_scalar_asset_spot_value, field='price', dt=dt, data_frequency=self.data_frequency)
        update_position_last_sale_prices(self.positions, get_price, dt)

    @property
    def stats(self):
        if False:
            for i in range(10):
                print('nop')
        'The current status of the positions.\n\n        Returns\n        -------\n        stats : PositionStats\n            The current stats position stats.\n\n        Notes\n        -----\n        This is cached, repeated access will not recompute the stats until\n        the stats may have changed.\n        '
        if self._dirty_stats:
            calculate_position_tracker_stats(self.positions, self._stats)
            self._dirty_stats = False
        return self._stats
if PY2:

    def move_to_end(ordered_dict, key, last=False):
        if False:
            i = 10
            return i + 15
        if last:
            ordered_dict[key] = ordered_dict.pop(key)
        else:
            new_first_element = ordered_dict.pop(key)
            items = ordered_dict.items()
            ordered_dict.clear()
            ordered_dict[key] = new_first_element
            ordered_dict.update(items)
else:
    move_to_end = OrderedDict.move_to_end
PeriodStats = namedtuple('PeriodStats', 'net_liquidation gross_leverage net_leverage')
not_overridden = sentinel('not_overridden', 'Mark that an account field has not been overridden')

class Ledger(object):
    """The ledger tracks all orders and transactions as well as the current
    state of the portfolio and positions.

    Attributes
    ----------
    portfolio : zipline.protocol.Portfolio
        The updated portfolio being managed.
    account : zipline.protocol.Account
        The updated account being managed.
    position_tracker : PositionTracker
        The current set of positions.
    todays_returns : float
        The current day's returns. In minute emission mode, this is the partial
        day's returns. In daily emission mode, this is
        ``daily_returns[session]``.
    daily_returns_series : pd.Series
        The daily returns series. Days that have not yet finished will hold
        a value of ``np.nan``.
    daily_returns_array : np.ndarray
        The daily returns as an ndarray. Days that have not yet finished will
        hold a value of ``np.nan``.
    """

    def __init__(self, trading_sessions, capital_base, data_frequency):
        if False:
            while True:
                i = 10
        if len(trading_sessions):
            start = trading_sessions[0]
        else:
            start = None
        self.__dirty_portfolio = False
        self._immutable_portfolio = zp.Portfolio(start, capital_base)
        self._portfolio = zp.MutableView(self._immutable_portfolio)
        self.daily_returns_series = pd.Series(np.nan, index=trading_sessions)
        self.daily_returns_array = self.daily_returns_series.values
        self._previous_total_returns = 0
        self._position_stats = None
        self._dirty_account = True
        self._immutable_account = zp.Account()
        self._account = zp.MutableView(self._immutable_account)
        self._account_overrides = {}
        self.position_tracker = PositionTracker(data_frequency)
        self._processed_transactions = {}
        self._orders_by_modified = {}
        self._orders_by_id = OrderedDict()
        self._payout_last_sale_prices = {}

    @property
    def todays_returns(self):
        if False:
            i = 10
            return i + 15
        return (self.portfolio.returns + 1) / (self._previous_total_returns + 1) - 1

    @property
    def _dirty_portfolio(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__dirty_portfolio

    @_dirty_portfolio.setter
    def _dirty_portfolio(self, value):
        if False:
            return 10
        if value:
            self.__dirty_portfolio = self._dirty_account = value
        else:
            self.__dirty_portfolio = value

    def start_of_session(self, session_label):
        if False:
            return 10
        self._processed_transactions.clear()
        self._orders_by_modified.clear()
        self._orders_by_id.clear()
        self._previous_total_returns = self.portfolio.returns

    def end_of_bar(self, session_ix):
        if False:
            i = 10
            return i + 15
        self.daily_returns_array[session_ix] = self.todays_returns

    def end_of_session(self, session_ix):
        if False:
            while True:
                i = 10
        self.daily_returns_series[session_ix] = self.todays_returns

    def sync_last_sale_prices(self, dt, data_portal, handle_non_market_minutes=False):
        if False:
            return 10
        self.position_tracker.sync_last_sale_prices(dt, data_portal, handle_non_market_minutes=handle_non_market_minutes)
        self._dirty_portfolio = True

    @staticmethod
    def _calculate_payout(multiplier, amount, old_price, price):
        if False:
            for i in range(10):
                print('nop')
        return (price - old_price) * multiplier * amount

    def _cash_flow(self, amount):
        if False:
            while True:
                i = 10
        self._dirty_portfolio = True
        p = self._portfolio
        p.cash_flow += amount
        p.cash += amount

    def process_transaction(self, transaction):
        if False:
            i = 10
            return i + 15
        'Add a transaction to ledger, updating the current state as needed.\n\n        Parameters\n        ----------\n        transaction : zp.Transaction\n            The transaction to execute.\n        '
        asset = transaction.asset
        if isinstance(asset, Future):
            try:
                old_price = self._payout_last_sale_prices[asset]
            except KeyError:
                self._payout_last_sale_prices[asset] = transaction.price
            else:
                position = self.position_tracker.positions[asset]
                amount = position.amount
                price = transaction.price
                self._cash_flow(self._calculate_payout(asset.price_multiplier, amount, old_price, price))
                if amount + transaction.amount == 0:
                    del self._payout_last_sale_prices[asset]
                else:
                    self._payout_last_sale_prices[asset] = price
        else:
            self._cash_flow(-(transaction.price * transaction.amount))
        self.position_tracker.execute_transaction(transaction)
        transaction_dict = transaction.to_dict()
        try:
            self._processed_transactions[transaction.dt].append(transaction_dict)
        except KeyError:
            self._processed_transactions[transaction.dt] = [transaction_dict]

    def process_splits(self, splits):
        if False:
            print('Hello World!')
        'Processes a list of splits by modifying any positions as needed.\n\n        Parameters\n        ----------\n        splits: list[(Asset, float)]\n            A list of splits. Each split is a tuple of (asset, ratio).\n        '
        leftover_cash = self.position_tracker.handle_splits(splits)
        if leftover_cash > 0:
            self._cash_flow(leftover_cash)

    def process_order(self, order):
        if False:
            for i in range(10):
                print('nop')
        'Keep track of an order that was placed.\n\n        Parameters\n        ----------\n        order : zp.Order\n            The order to record.\n        '
        try:
            dt_orders = self._orders_by_modified[order.dt]
        except KeyError:
            self._orders_by_modified[order.dt] = OrderedDict([(order.id, order)])
            self._orders_by_id[order.id] = order
        else:
            self._orders_by_id[order.id] = dt_orders[order.id] = order
            move_to_end(dt_orders, order.id, last=True)
        move_to_end(self._orders_by_id, order.id, last=True)

    def process_commission(self, commission):
        if False:
            while True:
                i = 10
        'Process the commission.\n\n        Parameters\n        ----------\n        commission : zp.Event\n            The commission being paid.\n        '
        asset = commission['asset']
        cost = commission['cost']
        self.position_tracker.handle_commission(asset, cost)
        self._cash_flow(-cost)

    def close_position(self, asset, dt, data_portal):
        if False:
            print('Hello World!')
        txn = self.position_tracker.maybe_create_close_position_transaction(asset, dt, data_portal)
        if txn is not None:
            self.process_transaction(txn)

    def process_dividends(self, next_session, asset_finder, adjustment_reader):
        if False:
            for i in range(10):
                print('nop')
        'Process dividends for the next session.\n\n        This will earn us any dividends whose ex-date is the next session as\n        well as paying out any dividends whose pay-date is the next session\n        '
        position_tracker = self.position_tracker
        held_sids = set(position_tracker.positions)
        if held_sids:
            cash_dividends = adjustment_reader.get_dividends_with_ex_date(held_sids, next_session, asset_finder)
            stock_dividends = adjustment_reader.get_stock_dividends_with_ex_date(held_sids, next_session, asset_finder)
            position_tracker.earn_dividends(cash_dividends, stock_dividends)
        self._cash_flow(position_tracker.pay_dividends(next_session))

    def capital_change(self, change_amount):
        if False:
            return 10
        self.update_portfolio()
        portfolio = self._portfolio
        portfolio.portfolio_value += change_amount
        portfolio.cash += change_amount

    def transactions(self, dt=None):
        if False:
            return 10
        'Retrieve the dict-form of all of the transactions in a given bar or\n        for the whole simulation.\n\n        Parameters\n        ----------\n        dt : pd.Timestamp or None, optional\n            The particular datetime to look up transactions for. If not passed,\n            or None is explicitly passed, all of the transactions will be\n            returned.\n\n        Returns\n        -------\n        transactions : list[dict]\n            The transaction information.\n        '
        if dt is None:
            return [txn for by_day in itervalues(self._processed_transactions) for txn in by_day]
        return self._processed_transactions.get(dt, [])

    def orders(self, dt=None):
        if False:
            return 10
        'Retrieve the dict-form of all of the orders in a given bar or for\n        the whole simulation.\n\n        Parameters\n        ----------\n        dt : pd.Timestamp or None, optional\n            The particular datetime to look up order for. If not passed, or\n            None is explicitly passed, all of the orders will be returned.\n\n        Returns\n        -------\n        orders : list[dict]\n            The order information.\n        '
        if dt is None:
            return [o.to_dict() for o in itervalues(self._orders_by_id)]
        return [o.to_dict() for o in itervalues(self._orders_by_modified.get(dt, {}))]

    @property
    def positions(self):
        if False:
            print('Hello World!')
        return self.position_tracker.get_position_list()

    def _get_payout_total(self, positions):
        if False:
            print('Hello World!')
        calculate_payout = self._calculate_payout
        payout_last_sale_prices = self._payout_last_sale_prices
        total = 0
        for (asset, old_price) in iteritems(payout_last_sale_prices):
            position = positions[asset]
            payout_last_sale_prices[asset] = price = position.last_sale_price
            amount = position.amount
            total += calculate_payout(asset.price_multiplier, amount, old_price, price)
        return total

    def update_portfolio(self):
        if False:
            print('Hello World!')
        'Force a computation of the current portfolio state.\n        '
        if not self._dirty_portfolio:
            return
        portfolio = self._portfolio
        pt = self.position_tracker
        portfolio.positions = pt.get_positions()
        position_stats = pt.stats
        portfolio.positions_value = position_value = position_stats.net_value
        portfolio.positions_exposure = position_stats.net_exposure
        self._cash_flow(self._get_payout_total(pt.positions))
        start_value = portfolio.portfolio_value
        portfolio.portfolio_value = end_value = portfolio.cash + position_value
        pnl = end_value - start_value
        if start_value != 0:
            returns = pnl / start_value
        else:
            returns = 0.0
        portfolio.pnl += pnl
        portfolio.returns = (1 + portfolio.returns) * (1 + returns) - 1
        self._dirty_portfolio = False

    @property
    def portfolio(self):
        if False:
            return 10
        'Compute the current portfolio.\n\n        Notes\n        -----\n        This is cached, repeated access will not recompute the portfolio until\n        the portfolio may have changed.\n        '
        self.update_portfolio()
        return self._immutable_portfolio

    def calculate_period_stats(self):
        if False:
            print('Hello World!')
        position_stats = self.position_tracker.stats
        portfolio_value = self.portfolio.portfolio_value
        if portfolio_value == 0:
            gross_leverage = net_leverage = np.inf
        else:
            gross_leverage = position_stats.gross_exposure / portfolio_value
            net_leverage = position_stats.net_exposure / portfolio_value
        return (portfolio_value, gross_leverage, net_leverage)

    def override_account_fields(self, settled_cash=not_overridden, accrued_interest=not_overridden, buying_power=not_overridden, equity_with_loan=not_overridden, total_positions_value=not_overridden, total_positions_exposure=not_overridden, regt_equity=not_overridden, regt_margin=not_overridden, initial_margin_requirement=not_overridden, maintenance_margin_requirement=not_overridden, available_funds=not_overridden, excess_liquidity=not_overridden, cushion=not_overridden, day_trades_remaining=not_overridden, leverage=not_overridden, net_leverage=not_overridden, net_liquidation=not_overridden):
        if False:
            i = 10
            return i + 15
        'Override fields on ``self.account``.\n        '
        self._dirty_account = True
        self._account_overrides = kwargs = {k: v for (k, v) in locals().items() if v is not not_overridden}
        del kwargs['self']

    @property
    def account(self):
        if False:
            for i in range(10):
                print('nop')
        if self._dirty_account:
            portfolio = self.portfolio
            account = self._account
            account.settled_cash = portfolio.cash
            account.accrued_interest = 0.0
            account.buying_power = np.inf
            account.equity_with_loan = portfolio.portfolio_value
            account.total_positions_value = portfolio.portfolio_value - portfolio.cash
            account.total_positions_exposure = portfolio.positions_exposure
            account.regt_equity = portfolio.cash
            account.regt_margin = np.inf
            account.initial_margin_requirement = 0.0
            account.maintenance_margin_requirement = 0.0
            account.available_funds = portfolio.cash
            account.excess_liquidity = portfolio.cash
            account.cushion = portfolio.cash / portfolio.portfolio_value if portfolio.portfolio_value else np.nan
            account.day_trades_remaining = np.inf
            (account.net_liquidation, account.gross_leverage, account.net_leverage) = self.calculate_period_stats()
            account.leverage = account.gross_leverage
            for (k, v) in iteritems(self._account_overrides):
                setattr(account, k, v)
            self._dirty_account = False
        return self._immutable_account