from AlgorithmImports import *

class RawPricesUniverseRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.UniverseSettings.Resolution = Resolution.Daily
        self.UniverseSettings.DataNormalizationMode = DataNormalizationMode.Raw
        self.SetStartDate(2014, 3, 24)
        self.SetEndDate(2014, 4, 7)
        self.SetCash(50000)
        self.SetSecurityInitializer(lambda x: x.SetFeeModel(ConstantFeeModel(0)))
        self.AddUniverse('MyUniverse', Resolution.Daily, self.SelectionFunction)

    def SelectionFunction(self, dateTime):
        if False:
            while True:
                i = 10
        if dateTime.day % 2 == 0:
            return ['SPY', 'IWM', 'QQQ']
        else:
            return ['AIG', 'BAC', 'IBM']

    def OnSecuritiesChanged(self, changes):
        if False:
            for i in range(10):
                print('nop')
        for security in changes.RemovedSecurities:
            if security.Invested:
                self.Liquidate(security.Symbol)
        for security in changes.AddedSecurities:
            self.SetHoldings(security.Symbol, 0.2)