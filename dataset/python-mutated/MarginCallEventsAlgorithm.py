from AlgorithmImports import *

class MarginCallEventsAlgorithm(QCAlgorithm):
    """
    This algorithm showcases two margin related event handlers.
    OnMarginCallWarning: Fired when a portfolio's remaining margin dips below 5% of the total portfolio value
    OnMarginCall: Fired immediately before margin call orders are execued, this gives the algorithm a change to regain margin on its own through liquidation
    """

    def Initialize(self):
        if False:
            return 10
        self.SetCash(100000)
        self.SetStartDate(2013, 10, 1)
        self.SetEndDate(2013, 12, 11)
        self.AddEquity('SPY', Resolution.Second)
        self.Securities['SPY'].SetLeverage(100)

    def OnData(self, data):
        if False:
            while True:
                i = 10
        if not self.Portfolio.Invested:
            self.SetHoldings('SPY', 100)

    def OnMarginCall(self, requests):
        if False:
            return 10
        for order in requests:
            newQuantity = int(np.sign(order.Quantity) * order.Quantity * 1.1)
            requests.remove(order)
            requests.append(SubmitOrderRequest(order.OrderType, order.SecurityType, order.Symbol, newQuantity, order.StopPrice, order.LimitPrice, self.Time, 'OnMarginCall'))
        return requests

    def OnMarginCallWarning(self):
        if False:
            return 10
        spyHoldings = self.Securities['SPY'].Holdings.Quantity
        shares = int(-spyHoldings * 0.005)
        self.Error('{0} - OnMarginCallWarning(): Liquidating {1} shares of SPY to avoid margin call.'.format(self.Time, shares))
        self.MarketOrder('SPY', shares)