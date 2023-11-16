from AlgorithmImports import *

class Collective2PortfolioSignalExportDemonstrationAlgorithm(QCAlgorithm):

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
        for item in self.symbols:
            self.AddSecurity(item)
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
            print('Hello World!')
        " Reduce the quantity of holdings for one security and increase the holdings to the another\n        one when the EMA's indicators crosses between themselves, then send a signal to Collective2 API "
        if self.IsWarmingUp:
            return
        if self.first_call:
            self.SetHoldings('SPY', 0.1)
            self.SignalExport.SetTargetPortfolioFromPortfolio()
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
            self.SignalExport.SetTargetPortfolioFromPortfolio()
        elif fast < slow * 0.999 and self.emaFastWasAbove:
            self.SetHoldings('SPY', 0.01)
            self.SignalExport.SetTargetPortfolioFromPortfolio()