from logbook import Logger
from collections import defaultdict
from copy import copy
from six import iteritems
from zipline.assets import Equity, Future, Asset
from .blotter import Blotter
from zipline.extensions import register
from zipline.finance.order import Order
from zipline.finance.slippage import DEFAULT_FUTURE_VOLUME_SLIPPAGE_BAR_LIMIT, VolatilityVolumeShare, FixedBasisPointsSlippage
from zipline.finance.commission import DEFAULT_PER_CONTRACT_COST, FUTURE_EXCHANGE_FEES_BY_SYMBOL, PerContract, PerShare
from zipline.utils.input_validation import expect_types
log = Logger('Blotter')
warning_logger = Logger('AlgoWarning')

@register(Blotter, 'default')
class SimulationBlotter(Blotter):

    def __init__(self, equity_slippage=None, future_slippage=None, equity_commission=None, future_commission=None, cancel_policy=None):
        if False:
            i = 10
            return i + 15
        super(SimulationBlotter, self).__init__(cancel_policy=cancel_policy)
        self.open_orders = defaultdict(list)
        self.orders = {}
        self.new_orders = []
        self.max_shares = int(100000000000.0)
        self.slippage_models = {Equity: equity_slippage or FixedBasisPointsSlippage(), Future: future_slippage or VolatilityVolumeShare(volume_limit=DEFAULT_FUTURE_VOLUME_SLIPPAGE_BAR_LIMIT)}
        self.commission_models = {Equity: equity_commission or PerShare(), Future: future_commission or PerContract(cost=DEFAULT_PER_CONTRACT_COST, exchange_fee=FUTURE_EXCHANGE_FEES_BY_SYMBOL)}

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '\n{class_name}(\n    slippage_models={slippage_models},\n    commission_models={commission_models},\n    open_orders={open_orders},\n    orders={orders},\n    new_orders={new_orders},\n    current_dt={current_dt})\n'.strip().format(class_name=self.__class__.__name__, slippage_models=self.slippage_models, commission_models=self.commission_models, open_orders=self.open_orders, orders=self.orders, new_orders=self.new_orders, current_dt=self.current_dt)

    @expect_types(asset=Asset)
    def order(self, asset, amount, style, order_id=None):
        if False:
            while True:
                i = 10
        'Place an order.\n\n        Parameters\n        ----------\n        asset : zipline.assets.Asset\n            The asset that this order is for.\n        amount : int\n            The amount of shares to order. If ``amount`` is positive, this is\n            the number of shares to buy or cover. If ``amount`` is negative,\n            this is the number of shares to sell or short.\n        style : zipline.finance.execution.ExecutionStyle\n            The execution style for the order.\n        order_id : str, optional\n            The unique identifier for this order.\n\n        Returns\n        -------\n        order_id : str or None\n            The unique identifier for this order, or None if no order was\n            placed.\n\n        Notes\n        -----\n        amount > 0 :: Buy/Cover\n        amount < 0 :: Sell/Short\n        Market order:    order(asset, amount)\n        Limit order:     order(asset, amount, style=LimitOrder(limit_price))\n        Stop order:      order(asset, amount, style=StopOrder(stop_price))\n        StopLimit order: order(asset, amount, style=StopLimitOrder(limit_price,\n                               stop_price))\n        '
        if amount == 0:
            return None
        elif amount > self.max_shares:
            raise OverflowError("Can't order more than %d shares" % self.max_shares)
        is_buy = amount > 0
        order = Order(dt=self.current_dt, asset=asset, amount=amount, stop=style.get_stop_price(is_buy), limit=style.get_limit_price(is_buy), id=order_id)
        self.open_orders[order.asset].append(order)
        self.orders[order.id] = order
        self.new_orders.append(order)
        return order.id

    def cancel(self, order_id, relay_status=True):
        if False:
            i = 10
            return i + 15
        if order_id not in self.orders:
            return
        cur_order = self.orders[order_id]
        if cur_order.open:
            order_list = self.open_orders[cur_order.asset]
            if cur_order in order_list:
                order_list.remove(cur_order)
            if cur_order in self.new_orders:
                self.new_orders.remove(cur_order)
            cur_order.cancel()
            cur_order.dt = self.current_dt
            if relay_status:
                self.new_orders.append(cur_order)

    def cancel_all_orders_for_asset(self, asset, warn=False, relay_status=True):
        if False:
            print('Hello World!')
        '\n        Cancel all open orders for a given asset.\n        '
        orders = self.open_orders[asset]
        for order in orders[:]:
            self.cancel(order.id, relay_status)
            if warn:
                if order.filled > 0:
                    warning_logger.warn('Your order for {order_amt} shares of {order_sym} has been partially filled. {order_filled} shares were successfully purchased. {order_failed} shares were not filled by the end of day and were canceled.'.format(order_amt=order.amount, order_sym=order.asset.symbol, order_filled=order.filled, order_failed=order.amount - order.filled))
                elif order.filled < 0:
                    warning_logger.warn('Your order for {order_amt} shares of {order_sym} has been partially filled. {order_filled} shares were successfully sold. {order_failed} shares were not filled by the end of day and were canceled.'.format(order_amt=order.amount, order_sym=order.asset.symbol, order_filled=-1 * order.filled, order_failed=-1 * (order.amount - order.filled)))
                else:
                    warning_logger.warn('Your order for {order_amt} shares of {order_sym} failed to fill by the end of day and was canceled.'.format(order_amt=order.amount, order_sym=order.asset.symbol))
        assert not orders
        del self.open_orders[asset]

    def execute_cancel_policy(self, event):
        if False:
            return 10
        if self.cancel_policy.should_cancel(event):
            warn = self.cancel_policy.warn_on_cancel
            for asset in copy(self.open_orders):
                self.cancel_all_orders_for_asset(asset, warn, relay_status=False)

    def reject(self, order_id, reason=''):
        if False:
            return 10
        "\n        Mark the given order as 'rejected', which is functionally similar to\n        cancelled. The distinction is that rejections are involuntary (and\n        usually include a message from a broker indicating why the order was\n        rejected) while cancels are typically user-driven.\n        "
        if order_id not in self.orders:
            return
        cur_order = self.orders[order_id]
        order_list = self.open_orders[cur_order.asset]
        if cur_order in order_list:
            order_list.remove(cur_order)
        if cur_order in self.new_orders:
            self.new_orders.remove(cur_order)
        cur_order.reject(reason=reason)
        cur_order.dt = self.current_dt
        self.new_orders.append(cur_order)

    def hold(self, order_id, reason=''):
        if False:
            i = 10
            return i + 15
        "\n        Mark the order with order_id as 'held'. Held is functionally similar\n        to 'open'. When a fill (full or partial) arrives, the status\n        will automatically change back to open/filled as necessary.\n        "
        if order_id not in self.orders:
            return
        cur_order = self.orders[order_id]
        if cur_order.open:
            if cur_order in self.new_orders:
                self.new_orders.remove(cur_order)
            cur_order.hold(reason=reason)
            cur_order.dt = self.current_dt
            self.new_orders.append(cur_order)

    def process_splits(self, splits):
        if False:
            print('Hello World!')
        '\n        Processes a list of splits by modifying any open orders as needed.\n\n        Parameters\n        ----------\n        splits: list\n            A list of splits.  Each split is a tuple of (asset, ratio).\n\n        Returns\n        -------\n        None\n        '
        for (asset, ratio) in splits:
            if asset not in self.open_orders:
                continue
            orders_to_modify = self.open_orders[asset]
            for order in orders_to_modify:
                order.handle_split(ratio)

    def get_transactions(self, bar_data):
        if False:
            return 10
        '\n        Creates a list of transactions based on the current open orders,\n        slippage model, and commission model.\n\n        Parameters\n        ----------\n        bar_data: zipline._protocol.BarData\n\n        Notes\n        -----\n        This method book-keeps the blotter\'s open_orders dictionary, so that\n         it is accurate by the time we\'re done processing open orders.\n\n        Returns\n        -------\n        transactions_list: List\n            transactions_list: list of transactions resulting from the current\n            open orders.  If there were no open orders, an empty list is\n            returned.\n\n        commissions_list: List\n            commissions_list: list of commissions resulting from filling the\n            open orders.  A commission is an object with "asset" and "cost"\n            parameters.\n\n        closed_orders: List\n            closed_orders: list of all the orders that have filled.\n        '
        closed_orders = []
        transactions = []
        commissions = []
        if self.open_orders:
            for (asset, asset_orders) in iteritems(self.open_orders):
                slippage = self.slippage_models[type(asset)]
                for (order, txn) in slippage.simulate(bar_data, asset, asset_orders):
                    commission = self.commission_models[type(asset)]
                    additional_commission = commission.calculate(order, txn)
                    if additional_commission > 0:
                        commissions.append({'asset': order.asset, 'order': order, 'cost': additional_commission})
                    order.filled += txn.amount
                    order.commission += additional_commission
                    order.dt = txn.dt
                    transactions.append(txn)
                    if not order.open:
                        closed_orders.append(order)
        return (transactions, commissions, closed_orders)

    def prune_orders(self, closed_orders):
        if False:
            for i in range(10):
                print('nop')
        "\n        Removes all given orders from the blotter's open_orders list.\n\n        Parameters\n        ----------\n        closed_orders: iterable of orders that are closed.\n\n        Returns\n        -------\n        None\n        "
        for order in closed_orders:
            asset = order.asset
            asset_orders = self.open_orders[asset]
            try:
                asset_orders.remove(order)
            except ValueError:
                continue
        for asset in list(self.open_orders.keys()):
            if len(self.open_orders[asset]) == 0:
                del self.open_orders[asset]