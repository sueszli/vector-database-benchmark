from abc import ABCMeta, abstractmethod
from six import with_metaclass
from zipline.extensions import extensible
from zipline.finance.cancel_policy import NeverCancel

@extensible
class Blotter(with_metaclass(ABCMeta)):

    def __init__(self, cancel_policy=None):
        if False:
            while True:
                i = 10
        self.cancel_policy = cancel_policy if cancel_policy else NeverCancel()
        self.current_dt = None

    def set_date(self, dt):
        if False:
            print('Hello World!')
        self.current_dt = dt

    @abstractmethod
    def order(self, asset, amount, style, order_id=None):
        if False:
            i = 10
            return i + 15
        'Place an order.\n\n        Parameters\n        ----------\n        asset : zipline.assets.Asset\n            The asset that this order is for.\n        amount : int\n            The amount of shares to order. If ``amount`` is positive, this is\n            the number of shares to buy or cover. If ``amount`` is negative,\n            this is the number of shares to sell or short.\n        style : zipline.finance.execution.ExecutionStyle\n            The execution style for the order.\n        order_id : str, optional\n            The unique identifier for this order.\n\n        Returns\n        -------\n        order_id : str or None\n            The unique identifier for this order, or None if no order was\n            placed.\n\n        Notes\n        -----\n        amount > 0 :: Buy/Cover\n        amount < 0 :: Sell/Short\n        Market order:    order(asset, amount)\n        Limit order:     order(asset, amount, style=LimitOrder(limit_price))\n        Stop order:      order(asset, amount, style=StopOrder(stop_price))\n        StopLimit order: order(asset, amount, style=StopLimitOrder(limit_price,\n                               stop_price))\n        '
        raise NotImplementedError('order')

    def batch_order(self, order_arg_lists):
        if False:
            i = 10
            return i + 15
        'Place a batch of orders.\n\n        Parameters\n        ----------\n        order_arg_lists : iterable[tuple]\n            Tuples of args that `order` expects.\n\n        Returns\n        -------\n        order_ids : list[str or None]\n            The unique identifier (or None) for each of the orders placed\n            (or not placed).\n\n        Notes\n        -----\n        This is required for `Blotter` subclasses to be able to place a batch\n        of orders, instead of being passed the order requests one at a time.\n        '
        return [self.order(*order_args) for order_args in order_arg_lists]

    @abstractmethod
    def cancel(self, order_id, relay_status=True):
        if False:
            print('Hello World!')
        'Cancel a single order\n\n        Parameters\n        ----------\n        order_id : int\n            The id of the order\n\n        relay_status : bool\n            Whether or not to record the status of the order\n        '
        raise NotImplementedError('cancel')

    @abstractmethod
    def cancel_all_orders_for_asset(self, asset, warn=False, relay_status=True):
        if False:
            print('Hello World!')
        '\n        Cancel all open orders for a given asset.\n        '
        raise NotImplementedError('cancel_all_orders_for_asset')

    @abstractmethod
    def execute_cancel_policy(self, event):
        if False:
            while True:
                i = 10
        raise NotImplementedError('execute_cancel_policy')

    @abstractmethod
    def reject(self, order_id, reason=''):
        if False:
            return 10
        "\n        Mark the given order as 'rejected', which is functionally similar to\n        cancelled. The distinction is that rejections are involuntary (and\n        usually include a message from a broker indicating why the order was\n        rejected) while cancels are typically user-driven.\n        "
        raise NotImplementedError('reject')

    @abstractmethod
    def hold(self, order_id, reason=''):
        if False:
            for i in range(10):
                print('nop')
        "\n        Mark the order with order_id as 'held'. Held is functionally similar\n        to 'open'. When a fill (full or partial) arrives, the status\n        will automatically change back to open/filled as necessary.\n        "
        raise NotImplementedError('hold')

    @abstractmethod
    def process_splits(self, splits):
        if False:
            while True:
                i = 10
        '\n        Processes a list of splits by modifying any open orders as needed.\n\n        Parameters\n        ----------\n        splits: list\n            A list of splits.  Each split is a tuple of (asset, ratio).\n\n        Returns\n        -------\n        None\n        '
        raise NotImplementedError('process_splits')

    @abstractmethod
    def get_transactions(self, bar_data):
        if False:
            i = 10
            return i + 15
        '\n        Creates a list of transactions based on the current open orders,\n        slippage model, and commission model.\n\n        Parameters\n        ----------\n        bar_data: zipline._protocol.BarData\n\n        Notes\n        -----\n        This method book-keeps the blotter\'s open_orders dictionary, so that\n         it is accurate by the time we\'re done processing open orders.\n\n        Returns\n        -------\n        transactions_list: List\n            transactions_list: list of transactions resulting from the current\n            open orders.  If there were no open orders, an empty list is\n            returned.\n\n        commissions_list: List\n            commissions_list: list of commissions resulting from filling the\n            open orders.  A commission is an object with "asset" and "cost"\n            parameters.\n\n        closed_orders: List\n            closed_orders: list of all the orders that have filled.\n        '
        raise NotImplementedError('get_transactions')

    @abstractmethod
    def prune_orders(self, closed_orders):
        if False:
            return 10
        "\n        Removes all given orders from the blotter's open_orders list.\n\n        Parameters\n        ----------\n        closed_orders: iterable of orders that are closed.\n\n        Returns\n        -------\n        None\n        "
        raise NotImplementedError('prune_orders')