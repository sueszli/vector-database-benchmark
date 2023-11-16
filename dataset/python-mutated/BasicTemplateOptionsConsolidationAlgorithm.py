from AlgorithmImports import *

class BasicTemplateOptionsConsolidationAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(1000000)
        option = self.AddOption('SPY')
        option.SetFilter(-2, +2, 0, 180)
        self.consolidators = dict()

    def OnData(self, slice):
        if False:
            return 10
        pass

    def OnQuoteBarConsolidated(self, sender, quoteBar):
        if False:
            for i in range(10):
                print('nop')
        self.Log('OnQuoteBarConsolidated called on ' + str(self.Time))
        self.Log(str(quoteBar))

    def OnTradeBarConsolidated(self, sender, tradeBar):
        if False:
            i = 10
            return i + 15
        self.Log('OnTradeBarConsolidated called on ' + str(self.Time))
        self.Log(str(tradeBar))

    def OnSecuritiesChanged(self, changes):
        if False:
            for i in range(10):
                print('nop')
        for security in changes.AddedSecurities:
            if security.Type == SecurityType.Equity:
                consolidator = TradeBarConsolidator(timedelta(minutes=5))
                consolidator.DataConsolidated += self.OnTradeBarConsolidated
            else:
                consolidator = QuoteBarConsolidator(timedelta(minutes=5))
                consolidator.DataConsolidated += self.OnQuoteBarConsolidated
            self.SubscriptionManager.AddConsolidator(security.Symbol, consolidator)
            self.consolidators[security.Symbol] = consolidator
        for security in changes.RemovedSecurities:
            consolidator = self.consolidators.pop(security.Symbol)
            self.SubscriptionManager.RemoveConsolidator(security.Symbol, consolidator)
            if security.Type == SecurityType.Equity:
                consolidator.DataConsolidated -= self.OnTradeBarConsolidated
            else:
                consolidator.DataConsolidated -= self.OnQuoteBarConsolidated