from AlgorithmImports import *

class BasicTemplateFuturesConsolidationAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(1000000)
        futureSP500 = self.AddFuture(Futures.Indices.SP500EMini)
        futureSP500.SetFilter(0, 182)
        self.consolidators = dict()

    def OnData(self, slice):
        if False:
            i = 10
            return i + 15
        pass

    def OnDataConsolidated(self, sender, quoteBar):
        if False:
            for i in range(10):
                print('nop')
        self.Log('OnDataConsolidated called on ' + str(self.Time))
        self.Log(str(quoteBar))

    def OnSecuritiesChanged(self, changes):
        if False:
            for i in range(10):
                print('nop')
        for security in changes.AddedSecurities:
            consolidator = QuoteBarConsolidator(timedelta(minutes=5))
            consolidator.DataConsolidated += self.OnDataConsolidated
            self.SubscriptionManager.AddConsolidator(security.Symbol, consolidator)
            self.consolidators[security.Symbol] = consolidator
        for security in changes.RemovedSecurities:
            consolidator = self.consolidators.pop(security.Symbol)
            self.SubscriptionManager.RemoveConsolidator(security.Symbol, consolidator)
            consolidator.DataConsolidated -= self.OnDataConsolidated