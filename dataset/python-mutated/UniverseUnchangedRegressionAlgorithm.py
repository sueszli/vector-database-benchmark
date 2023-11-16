from AlgorithmImports import *

class UniverseUnchangedRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.UniverseSettings.Resolution = Resolution.Daily
        self.SetStartDate(2014, 3, 25)
        self.SetEndDate(2014, 4, 7)
        self.SetAlpha(ConstantAlphaModel(InsightType.Price, InsightDirection.Up, timedelta(days=1), 0.025, None))
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)
        self.numberOfSymbolsFine = 2

    def CoarseSelectionFunction(self, coarse):
        if False:
            return 10
        if self.Time.date() <= date(2014, 3, 26):
            tickers = ['AAPL', 'AIG', 'IBM']
            return [Symbol.Create(x, SecurityType.Equity, Market.USA) for x in tickers]
        return Universe.Unchanged

    def FineSelectionFunction(self, fine):
        if False:
            i = 10
            return i + 15
        if self.Time.date() == date(2014, 3, 25):
            sortedByPeRatio = sorted(fine, key=lambda x: x.ValuationRatios.PERatio, reverse=True)
            return [x.Symbol for x in sortedByPeRatio[:self.numberOfSymbolsFine]]
        return Universe.Unchanged

    def OnSecuritiesChanged(self, changes):
        if False:
            i = 10
            return i + 15
        addedSymbols = [x.Symbol for x in changes.AddedSecurities]
        if len(changes.AddedSecurities) != 2 or self.Time.date() != date(2014, 3, 25) or Symbol.Create('AAPL', SecurityType.Equity, Market.USA) not in addedSymbols or (Symbol.Create('IBM', SecurityType.Equity, Market.USA) not in addedSymbols):
            raise ValueError('Unexpected security changes')
        self.Log(f'OnSecuritiesChanged({self.Time}):: {changes}')