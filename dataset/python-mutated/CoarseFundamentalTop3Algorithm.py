from AlgorithmImports import *

class CoarseFundamentalTop3Algorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2014, 3, 24)
        self.SetEndDate(2014, 4, 7)
        self.SetCash(50000)
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction)
        self.__numberOfSymbols = 3
        self._changes = None

    def CoarseSelectionFunction(self, coarse):
        if False:
            return 10
        sortedByDollarVolume = sorted(coarse, key=lambda x: x.DollarVolume, reverse=True)
        return [x.Symbol for x in sortedByDollarVolume[:self.__numberOfSymbols]]

    def OnData(self, data):
        if False:
            for i in range(10):
                print('nop')
        self.Log(f"OnData({self.UtcTime}): Keys: {', '.join([key.Value for key in data.Keys])}")
        if self._changes is None:
            return
        for security in self._changes.RemovedSecurities:
            if security.Invested:
                self.Liquidate(security.Symbol)
        for security in self._changes.AddedSecurities:
            self.SetHoldings(security.Symbol, 1 / self.__numberOfSymbols)
        self._changes = None

    def OnSecuritiesChanged(self, changes):
        if False:
            while True:
                i = 10
        self._changes = changes
        self.Log(f'OnSecuritiesChanged({self.UtcTime}):: {changes}')

    def OnOrderEvent(self, fill):
        if False:
            while True:
                i = 10
        self.Log(f'OnOrderEvent({self.UtcTime}):: {fill}')