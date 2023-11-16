from AlgorithmImports import *

class CustomPartialFillModelAlgorithm(QCAlgorithm):
    """Basic template algorithm that implements a fill model with partial fills"""

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2019, 1, 1)
        self.SetEndDate(2019, 3, 1)
        equity = self.AddEquity('SPY', Resolution.Hour)
        self.spy = equity.Symbol
        self.holdings = equity.Holdings
        equity.SetFillModel(CustomPartialFillModel(self))

    def OnData(self, data):
        if False:
            print('Hello World!')
        open_orders = self.Transactions.GetOpenOrders(self.spy)
        if len(open_orders) != 0:
            return
        if self.Time.day > 10 and self.holdings.Quantity <= 0:
            self.MarketOrder(self.spy, 105, True)
        elif self.Time.day > 20 and self.holdings.Quantity >= 0:
            self.MarketOrder(self.spy, -100, True)

class CustomPartialFillModel(FillModel):
    """Implements a custom fill model that inherit from FillModel. Override the MarketFill method to simulate partially fill orders"""

    def __init__(self, algorithm):
        if False:
            print('Hello World!')
        self.algorithm = algorithm
        self.absoluteRemainingByOrderId = {}

    def MarketFill(self, asset, order):
        if False:
            i = 10
            return i + 15
        absoluteRemaining = self.absoluteRemainingByOrderId.get(order.Id, order.AbsoluteQuantity)
        fill = super().MarketFill(asset, order)
        fill.FillQuantity = np.sign(order.Quantity) * 10
        if min(abs(fill.FillQuantity), absoluteRemaining) == absoluteRemaining:
            fill.FillQuantity = np.sign(order.Quantity) * absoluteRemaining
            fill.Status = OrderStatus.Filled
            self.absoluteRemainingByOrderId.pop(order.Id, None)
        else:
            fill.Status = OrderStatus.PartiallyFilled
            self.absoluteRemainingByOrderId[order.Id] = absoluteRemaining - abs(fill.FillQuantity)
            price = fill.FillPrice
        return fill