from AlgorithmImports import *

class ClassicRenkoConsolidatorAlgorithm(QCAlgorithm):
    """Demonstration of how to initialize and use the RenkoConsolidator"""

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetStartDate(2012, 1, 1)
        self.SetEndDate(2013, 1, 1)
        self.AddEquity('SPY', Resolution.Daily)
        renkoClose = ClassicRenkoConsolidator(2.5)
        renkoClose.DataConsolidated += self.HandleRenkoClose
        self.SubscriptionManager.AddConsolidator('SPY', renkoClose)
        renko7bar = ClassicRenkoConsolidator(2.5, lambda x: (2 * x.Open + x.High + x.Low + 3 * x.Close) / 7, lambda x: x.Volume)
        renko7bar.DataConsolidated += self.HandleRenko7Bar
        self.SubscriptionManager.AddConsolidator('SPY', renko7bar)

    def OnData(self, data):
        if False:
            for i in range(10):
                print('nop')
        pass

    def HandleRenkoClose(self, sender, data):
        if False:
            return 10
        'This function is called by our renkoClose consolidator defined in Initialize()\n        Args:\n            data: The new renko bar produced by the consolidator'
        if not self.Portfolio.Invested:
            self.SetHoldings(data.Symbol, 1)
        self.Log(f'CLOSE - {data.Time} - {data.Open} {data.Close}')

    def HandleRenko7Bar(self, sender, data):
        if False:
            i = 10
            return i + 15
        'This function is called by our renko7bar consolidator defined in Initialize()\n        Args:\n            data: The new renko bar produced by the consolidator'
        if self.Portfolio.Invested:
            self.Liquidate(data.Symbol)
        self.Log(f'7BAR - {data.Time} - {data.Open} {data.Close}')