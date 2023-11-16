from AlgorithmImports import *

class CoarseFineOptionUniverseChainRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2014, 6, 4)
        self.SetEndDate(2014, 6, 6)
        self.UniverseSettings.Resolution = Resolution.Minute
        self._twx = Symbol.Create('TWX', SecurityType.Equity, Market.USA)
        self._aapl = Symbol.Create('AAPL', SecurityType.Equity, Market.USA)
        self._lastEquityAdded = None
        self._changes = None
        self._optionCount = 0
        universe = self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)
        self.AddUniverseOptions(universe, self.OptionFilterFunction)

    def OptionFilterFunction(self, universe):
        if False:
            for i in range(10):
                print('nop')
        universe.IncludeWeeklys().FrontMonth()
        contracts = list()
        for symbol in universe:
            if len(contracts) == 5:
                break
            contracts.append(symbol)
        return universe.Contracts(contracts)

    def CoarseSelectionFunction(self, coarse):
        if False:
            return 10
        if self.Time <= datetime(2014, 6, 5):
            return [self._twx]
        return [self._aapl]

    def FineSelectionFunction(self, fine):
        if False:
            print('Hello World!')
        if self.Time <= datetime(2014, 6, 5):
            return [self._twx]
        return [self._aapl]

    def OnData(self, data):
        if False:
            i = 10
            return i + 15
        if self._changes == None or any((security.Price == 0 for security in self._changes.AddedSecurities)):
            return
        for security in self._changes.RemovedSecurities:
            if security.Invested:
                self.Liquidate(security.Symbol)
        for security in self._changes.AddedSecurities:
            if not security.Symbol.HasUnderlying:
                self._lastEquityAdded = security.Symbol
            else:
                if security.Symbol.Underlying != self._lastEquityAdded:
                    raise ValueError(f'Unexpected symbol added {security.Symbol}')
                self._optionCount += 1
            self.SetHoldings(security.Symbol, 0.05)
        self._changes = None

    def OnSecuritiesChanged(self, changes):
        if False:
            for i in range(10):
                print('nop')
        if self._changes == None:
            self._changes = changes
            return
        self._changes = self._changes.op_Addition(self._changes, changes)

    def OnEndOfAlgorithm(self):
        if False:
            i = 10
            return i + 15
        if self._optionCount == 0:
            raise ValueError('Option universe chain did not add any option!')