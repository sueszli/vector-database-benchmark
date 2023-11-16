from AlgorithmImports import *

class Collective2SignalExportDemonstrationAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        ' Initialize the date and add all equity symbols present in list _symbols '
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.AddEquity('GOOG')
        self.symbols = [Symbol.Create('SPY', SecurityType.Equity, Market.USA, None, None), Symbol.Create('EURUSD', SecurityType.Forex, Market.Oanda, None, None), Symbol.CreateFuture('ES', Market.CME, datetime(2023, 12, 15), None), Symbol.CreateOption('GOOG', Market.USA, OptionStyle.American, OptionRight.Call, 130, datetime(2023, 9, 1))]
        self.targets = []
        for item in self.symbols:
            symbol = self.AddSecurity(item).Symbol
            if symbol.SecurityType == SecurityType.Equity or symbol.SecurityType == SecurityType.Forex:
                self.targets.append(PortfolioTarget(symbol, 0.05))
            else:
                self.targets.append(PortfolioTarget(symbol, 1))
        self.fast = self.EMA('SPY', 10)
        self.slow = self.EMA('SPY', 100)
        self.emaFastIsNotSet = True
        self.emaFastWasAbove = False
        self.collective2Apikey = 'YOUR APIV4 KEY'
        self.collective2SystemId = 0
        self.SignalExport.AddSignalExportProviders(Collective2SignalExport(self.collective2Apikey, self.collective2SystemId))
        self.first_call = True
        self.SetWarmUp(100)

    def OnData(self, data):
        if False:
            return 10
        " Reduce the quantity of holdings for one security and increase the holdings to the another\n        one when the EMA's indicators crosses between themselves, then send a signal to Collective2 API "
        if self.IsWarmingUp:
            return
        if self.first_call:
            self.SetHoldings('SPY', 0.1)
            self.targets[0] = PortfolioTarget(self.Portfolio['SPY'].Symbol, 0.1)
            self.SignalExport.SetTargetPortfolio(self.targets)
            self.first_call = False
        fast = self.fast.Current.Value
        slow = self.slow.Current.Value
        if self.emaFastIsNotSet == True:
            if fast > slow * 1.001:
                self.emaFastWasAbove = True
            else:
                self.emaFastWasAbove = False
            self.emaFastIsNotSet = False
        if fast > slow * 1.001 and (not self.emaFastWasAbove):
            self.SetHoldings('SPY', 0.1)
            self.targets[0] = PortfolioTarget(self.Portfolio['SPY'].Symbol, 0.1)
            self.SignalExport.SetTargetPortfolio(self.targets)
        elif fast < slow * 0.999 and self.emaFastWasAbove:
            self.SetHoldings('SPY', 0.01)
            self.targets[0] = PortfolioTarget(self.Portfolio['SPY'].Symbol, 0.01)
            self.SignalExport.SetTargetPortfolio(self.targets)