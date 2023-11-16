from AlgorithmImports import *

class RawPricesCoarseUniverseAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.UniverseSettings.Resolution = Resolution.Daily
        self.SetStartDate(2014, 1, 1)
        self.SetEndDate(2015, 1, 1)
        self.SetCash(50000)
        self.SetSecurityInitializer(self.CustomSecurityInitializer)
        self.AddUniverse(self.CoarseSelectionFunction)
        self.__numberOfSymbols = 5

    def CustomSecurityInitializer(self, security):
        if False:
            return 10
        'Initialize the security with raw prices and zero fees \n        Args:\n            security: Security which characteristics we want to change'
        security.SetDataNormalizationMode(DataNormalizationMode.Raw)
        security.SetFeeModel(ConstantFeeModel(0))

    def CoarseSelectionFunction(self, coarse):
        if False:
            while True:
                i = 10
        sortedByDollarVolume = sorted(coarse, key=lambda x: x.DollarVolume, reverse=True)
        return [x.Symbol for x in sortedByDollarVolume[:self.__numberOfSymbols]]

    def OnSecuritiesChanged(self, changes):
        if False:
            i = 10
            return i + 15
        for security in changes.RemovedSecurities:
            if security.Invested:
                self.Liquidate(security.Symbol)
        for security in changes.AddedSecurities:
            self.SetHoldings(security.Symbol, 0.2)

    def OnOrderEvent(self, orderEvent):
        if False:
            return 10
        if orderEvent.Status == OrderStatus.Filled:
            self.Log(f'OnOrderEvent({self.UtcTime}):: {orderEvent}')