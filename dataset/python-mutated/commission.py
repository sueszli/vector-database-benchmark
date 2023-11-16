from abc import abstractmethod
from collections import defaultdict
from six import with_metaclass
from toolz import merge
from zipline.assets import Equity, Future
from zipline.finance.constants import FUTURE_EXCHANGE_FEES_BY_SYMBOL
from zipline.finance.shared import AllowedAssetMarker, FinancialModelMeta
from zipline.utils.dummy import DummyMapping
DEFAULT_PER_SHARE_COST = 0.001
DEFAULT_PER_CONTRACT_COST = 0.85
DEFAULT_PER_DOLLAR_COST = 0.0015
DEFAULT_MINIMUM_COST_PER_EQUITY_TRADE = 0.0
DEFAULT_MINIMUM_COST_PER_FUTURE_TRADE = 0.0

class CommissionModel(with_metaclass(FinancialModelMeta)):
    """Abstract base class for commission models.

    Commission models are responsible for accepting order/transaction pairs and
    calculating how much commission should be charged to an algorithm's account
    on each transaction.

    To implement a new commission model, create a subclass of
    :class:`~zipline.finance.commission.CommissionModel` and implement
    :meth:`calculate`.
    """
    allowed_asset_types = (Equity, Future)

    @abstractmethod
    def calculate(self, order, transaction):
        if False:
            while True:
                i = 10
        "\n        Calculate the amount of commission to charge on ``order`` as a result\n        of ``transaction``.\n\n        Parameters\n        ----------\n        order : zipline.finance.order.Order\n            The order being processed.\n\n            The ``commission`` field of ``order`` is a float indicating the\n            amount of commission already charged on this order.\n\n        transaction : zipline.finance.transaction.Transaction\n            The transaction being processed. A single order may generate\n            multiple transactions if there isn't enough volume in a given bar\n            to fill the full amount requested in the order.\n\n        Returns\n        -------\n        amount_charged : float\n            The additional commission, in dollars, that we should attribute to\n            this order.\n        "
        raise NotImplementedError('calculate')

class NoCommission(CommissionModel):
    """Model commissions as free.

    Notes
    -----
    This is primarily used for testing.
    """

    @staticmethod
    def calculate(order, transaction):
        if False:
            print('Hello World!')
        return 0.0

class EquityCommissionModel(with_metaclass(AllowedAssetMarker, CommissionModel)):
    """
    Base class for commission models which only support equities.
    """
    allowed_asset_types = (Equity,)

class FutureCommissionModel(with_metaclass(AllowedAssetMarker, CommissionModel)):
    """
    Base class for commission models which only support futures.
    """
    allowed_asset_types = (Future,)

def calculate_per_unit_commission(order, transaction, cost_per_unit, initial_commission, min_trade_cost):
    if False:
        for i in range(10):
            print('nop')
    "\n    If there is a minimum commission:\n        If the order hasn't had a commission paid yet, pay the minimum\n        commission.\n\n        If the order has paid a commission, start paying additional\n        commission once the minimum commission has been reached.\n\n    If there is no minimum commission:\n        Pay commission based on number of units in the transaction.\n    "
    additional_commission = abs(transaction.amount * cost_per_unit)
    if order.commission == 0:
        return max(min_trade_cost, additional_commission + initial_commission)
    else:
        per_unit_total = abs(order.filled * cost_per_unit) + additional_commission + initial_commission
        if per_unit_total < min_trade_cost:
            return 0
        else:
            return per_unit_total - order.commission

class PerShare(EquityCommissionModel):
    """
    Calculates a commission for a transaction based on a per share cost with
    an optional minimum cost per trade.

    Parameters
    ----------
    cost : float, optional
        The amount of commissions paid per share traded. Default is one tenth
        of a cent per share.
    min_trade_cost : float, optional
        The minimum amount of commissions paid per trade. Default is no
        minimum.

    Notes
    -----
    This is zipline's default commission model for equities.
    """

    def __init__(self, cost=DEFAULT_PER_SHARE_COST, min_trade_cost=DEFAULT_MINIMUM_COST_PER_EQUITY_TRADE):
        if False:
            print('Hello World!')
        self.cost_per_share = float(cost)
        self.min_trade_cost = min_trade_cost or 0

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '{class_name}(cost_per_share={cost_per_share}, min_trade_cost={min_trade_cost})'.format(class_name=self.__class__.__name__, cost_per_share=self.cost_per_share, min_trade_cost=self.min_trade_cost)

    def calculate(self, order, transaction):
        if False:
            print('Hello World!')
        return calculate_per_unit_commission(order=order, transaction=transaction, cost_per_unit=self.cost_per_share, initial_commission=0, min_trade_cost=self.min_trade_cost)

class PerContract(FutureCommissionModel):
    """
    Calculates a commission for a transaction based on a per contract cost with
    an optional minimum cost per trade.

    Parameters
    ----------
    cost : float or dict
        The amount of commissions paid per contract traded. If given a float,
        the commission for all futures contracts is the same. If given a
        dictionary, it must map root symbols to the commission cost for
        contracts of that symbol.
    exchange_fee : float or dict
        A flat-rate fee charged by the exchange per trade. This value is a
        constant, one-time charge no matter how many contracts are being
        traded. If given a float, the fee for all contracts is the same. If
        given a dictionary, it must map root symbols to the fee for contracts
        of that symbol.
    min_trade_cost : float, optional
        The minimum amount of commissions paid per trade.
    """

    def __init__(self, cost, exchange_fee, min_trade_cost=DEFAULT_MINIMUM_COST_PER_FUTURE_TRADE):
        if False:
            while True:
                i = 10
        if isinstance(cost, (int, float)):
            self._cost_per_contract = DummyMapping(float(cost))
        else:
            self._cost_per_contract = defaultdict(lambda : DEFAULT_PER_CONTRACT_COST, **cost)
        if isinstance(exchange_fee, (int, float)):
            self._exchange_fee = DummyMapping(float(exchange_fee))
        else:
            self._exchange_fee = merge(FUTURE_EXCHANGE_FEES_BY_SYMBOL, exchange_fee)
        self.min_trade_cost = min_trade_cost or 0

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        if isinstance(self._cost_per_contract, DummyMapping):
            cost_per_contract = self._cost_per_contract['dummy key']
        else:
            cost_per_contract = '<varies>'
        if isinstance(self._exchange_fee, DummyMapping):
            exchange_fee = self._exchange_fee['dummy key']
        else:
            exchange_fee = '<varies>'
        return '{class_name}(cost_per_contract={cost_per_contract}, exchange_fee={exchange_fee}, min_trade_cost={min_trade_cost})'.format(class_name=self.__class__.__name__, cost_per_contract=cost_per_contract, exchange_fee=exchange_fee, min_trade_cost=self.min_trade_cost)

    def calculate(self, order, transaction):
        if False:
            i = 10
            return i + 15
        root_symbol = order.asset.root_symbol
        cost_per_contract = self._cost_per_contract[root_symbol]
        exchange_fee = self._exchange_fee[root_symbol]
        return calculate_per_unit_commission(order=order, transaction=transaction, cost_per_unit=cost_per_contract, initial_commission=exchange_fee, min_trade_cost=self.min_trade_cost)

class PerTrade(CommissionModel):
    """
    Calculates a commission for a transaction based on a per trade cost.

    For orders that require multiple fills, the full commission is charged to
    the first fill.

    Parameters
    ----------
    cost : float, optional
        The flat amount of commissions paid per equity trade.
    """

    def __init__(self, cost=DEFAULT_MINIMUM_COST_PER_EQUITY_TRADE):
        if False:
            print('Hello World!')
        '\n        Cost parameter is the cost of a trade, regardless of share count.\n        $5.00 per trade is fairly typical of discount brokers.\n        '
        self.cost = float(cost)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '{class_name}(cost_per_trade={cost})'.format(class_name=self.__class__.__name__, cost=self.cost)

    def calculate(self, order, transaction):
        if False:
            print('Hello World!')
        "\n        If the order hasn't had a commission paid yet, pay the fixed\n        commission.\n        "
        if order.commission == 0:
            return self.cost
        else:
            return 0.0

class PerFutureTrade(PerContract):
    """
    Calculates a commission for a transaction based on a per trade cost.

    Parameters
    ----------
    cost : float or dict
        The flat amount of commissions paid per trade, regardless of the number
        of contracts being traded. If given a float, the commission for all
        futures contracts is the same. If given a dictionary, it must map root
        symbols to the commission cost for trading contracts of that symbol.
    """

    def __init__(self, cost=DEFAULT_MINIMUM_COST_PER_FUTURE_TRADE):
        if False:
            while True:
                i = 10
        super(PerFutureTrade, self).__init__(cost=0, exchange_fee=cost, min_trade_cost=0)
        self._cost_per_trade = self._exchange_fee

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        if isinstance(self._cost_per_trade, DummyMapping):
            cost_per_trade = self._cost_per_trade['dummy key']
        else:
            cost_per_trade = '<varies>'
        return '{class_name}(cost_per_trade={cost_per_trade})'.format(class_name=self.__class__.__name__, cost_per_trade=cost_per_trade)

class PerDollar(EquityCommissionModel):
    """
    Model commissions by applying a fixed cost per dollar transacted.

    Parameters
    ----------
    cost : float, optional
        The flat amount of commissions paid per dollar of equities
        traded. Default is a commission of $0.0015 per dollar transacted.
    """

    def __init__(self, cost=DEFAULT_PER_DOLLAR_COST):
        if False:
            print('Hello World!')
        '\n        Cost parameter is the cost of a trade per-dollar. 0.0015\n        on $1 million means $1,500 commission (=1M * 0.0015)\n        '
        self.cost_per_dollar = float(cost)

    def __repr__(self):
        if False:
            return 10
        return '{class_name}(cost_per_dollar={cost})'.format(class_name=self.__class__.__name__, cost=self.cost_per_dollar)

    def calculate(self, order, transaction):
        if False:
            for i in range(10):
                print('nop')
        '\n        Pay commission based on dollar value of shares.\n        '
        cost_per_share = transaction.price * self.cost_per_dollar
        return abs(transaction.amount) * cost_per_share