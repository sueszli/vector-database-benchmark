from AlgorithmImports import *
from Selection.FundamentalUniverseSelectionModel import FundamentalUniverseSelectionModel
from math import ceil
from itertools import chain

class GreenblattMagicFormulaAlpha(QCAlgorithm):
    """ Alpha Streams: Benchmark Alpha: Pick stocks according to Joel Greenblatt's Magic Formula
    This alpha picks stocks according to Joel Greenblatt's Magic Formula.
    First, each stock is ranked depending on the relative value of the ratio EV/EBITDA. For example, a stock
    that has the lowest EV/EBITDA ratio in the security universe receives a score of one while a stock that has
    the tenth lowest EV/EBITDA score would be assigned 10 points.

    Then, each stock is ranked and given a score for the second valuation ratio, Return on Capital (ROC).
    Similarly, a stock that has the highest ROC value in the universe gets one score point.
    The stocks that receive the lowest combined score are chosen for insights.

    Source: Greenblatt, J. (2010) The Little Book That Beats the Market

    This alpha is part of the Benchmark Alpha Series created by QuantConnect which are open
    sourced so the community and client funds can see an example of an alpha."""

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2018, 1, 1)
        self.SetCash(100000)
        self.SetSecurityInitializer(lambda security: security.SetFeeModel(ConstantFeeModel(0)))
        self.SetUniverseSelection(GreenBlattMagicFormulaUniverseSelectionModel())
        self.SetAlpha(RateOfChangeAlphaModel())
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(NullRiskManagementModel())

class RateOfChangeAlphaModel(AlphaModel):
    """Uses Rate of Change (ROC) to create magnitude prediction for insights."""

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self.lookback = kwargs.get('lookback', 1)
        self.resolution = kwargs.get('resolution', Resolution.Daily)
        self.predictionInterval = Time.Multiply(Extensions.ToTimeSpan(self.resolution), self.lookback)
        self.symbolDataBySymbol = {}

    def Update(self, algorithm, data):
        if False:
            while True:
                i = 10
        insights = []
        for (symbol, symbolData) in self.symbolDataBySymbol.items():
            if symbolData.CanEmit:
                insights.append(Insight.Price(symbol, self.predictionInterval, InsightDirection.Up, symbolData.Return, None))
        return insights

    def OnSecuritiesChanged(self, algorithm, changes):
        if False:
            while True:
                i = 10
        for removed in changes.RemovedSecurities:
            symbolData = self.symbolDataBySymbol.pop(removed.Symbol, None)
            if symbolData is not None:
                symbolData.RemoveConsolidators(algorithm)
        symbols = [x.Symbol for x in changes.AddedSecurities if x.Symbol not in self.symbolDataBySymbol]
        history = algorithm.History(symbols, self.lookback, self.resolution)
        if history.empty:
            return
        for symbol in symbols:
            symbolData = SymbolData(algorithm, symbol, self.lookback, self.resolution)
            self.symbolDataBySymbol[symbol] = symbolData
            symbolData.WarmUpIndicators(history.loc[symbol])

class SymbolData:
    """Contains data specific to a symbol required by this model"""

    def __init__(self, algorithm, symbol, lookback, resolution):
        if False:
            return 10
        self.previous = 0
        self.symbol = symbol
        self.ROC = RateOfChange(f'{symbol}.ROC({lookback})', lookback)
        self.consolidator = algorithm.ResolveConsolidator(symbol, resolution)
        algorithm.RegisterIndicator(symbol, self.ROC, self.consolidator)

    def RemoveConsolidators(self, algorithm):
        if False:
            print('Hello World!')
        algorithm.SubscriptionManager.RemoveConsolidator(self.symbol, self.consolidator)

    def WarmUpIndicators(self, history):
        if False:
            i = 10
            return i + 15
        for tuple in history.itertuples():
            self.ROC.Update(tuple.Index, tuple.close)

    @property
    def Return(self):
        if False:
            return 10
        return self.ROC.Current.Value

    @property
    def CanEmit(self):
        if False:
            return 10
        if self.previous == self.ROC.Samples:
            return False
        self.previous = self.ROC.Samples
        return self.ROC.IsReady

    def __str__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return f'{self.ROC.Name}: {(1 + self.Return) ** 252 - 1:.2%}'

class GreenBlattMagicFormulaUniverseSelectionModel(FundamentalUniverseSelectionModel):
    """Defines a universe according to Joel Greenblatt's Magic Formula, as a universe selection model for the framework algorithm.
       From the universe QC500, stocks are ranked using the valuation ratios, Enterprise Value to EBITDA (EV/EBITDA) and Return on Assets (ROA).
    """

    def __init__(self, filterFineData=True, universeSettings=None):
        if False:
            return 10
        'Initializes a new default instance of the MagicFormulaUniverseSelectionModel'
        super().__init__(filterFineData, universeSettings)
        self.NumberOfSymbolsCoarse = 500
        self.NumberOfSymbolsFine = 20
        self.NumberOfSymbolsInPortfolio = 10
        self.lastMonth = -1
        self.dollarVolumeBySymbol = {}

    def SelectCoarse(self, algorithm, coarse):
        if False:
            return 10
        'Performs coarse selection for constituents.\n        The stocks must have fundamental data'
        month = algorithm.Time.month
        if month == self.lastMonth:
            return Universe.Unchanged
        self.lastMonth = month
        top = sorted([x for x in coarse if x.HasFundamentalData], key=lambda x: x.DollarVolume, reverse=True)[:self.NumberOfSymbolsCoarse]
        self.dollarVolumeBySymbol = {i.Symbol: i.DollarVolume for i in top}
        return list(self.dollarVolumeBySymbol.keys())

    def SelectFine(self, algorithm, fine):
        if False:
            while True:
                i = 10
        "QC500: Performs fine selection for the coarse selection constituents\n        The company's headquarter must in the U.S.\n        The stock must be traded on either the NYSE or NASDAQ\n        At least half a year since its initial public offering\n        The stock's market cap must be greater than 500 million\n\n        Magic Formula: Rank stocks by Enterprise Value to EBITDA (EV/EBITDA)\n        Rank subset of previously ranked stocks (EV/EBITDA), using the valuation ratio Return on Assets (ROA)"
        filteredFine = [x for x in fine if x.CompanyReference.CountryId == 'USA' and (x.CompanyReference.PrimaryExchangeID == 'NYS' or x.CompanyReference.PrimaryExchangeID == 'NAS') and ((algorithm.Time - x.SecurityReference.IPODate).days > 180) and (x.EarningReports.BasicAverageShares.ThreeMonths * x.EarningReports.BasicEPS.TwelveMonths * x.ValuationRatios.PERatio > 500000000.0)]
        count = len(filteredFine)
        if count == 0:
            return []
        myDict = dict()
        percent = self.NumberOfSymbolsFine / count
        for key in ['N', 'M', 'U', 'T', 'B', 'I']:
            value = [x for x in filteredFine if x.CompanyReference.IndustryTemplateCode == key]
            value = sorted(value, key=lambda x: self.dollarVolumeBySymbol[x.Symbol], reverse=True)
            myDict[key] = value[:ceil(len(value) * percent)]
        topFine = chain.from_iterable(myDict.values())
        sortedByEVToEBITDA = sorted(topFine, key=lambda x: x.ValuationRatios.EVToEBITDA, reverse=True)
        sortedByROA = sorted(sortedByEVToEBITDA[:self.NumberOfSymbolsFine], key=lambda x: x.ValuationRatios.ForwardROA, reverse=False)
        return [f.Symbol for f in sortedByROA[:self.NumberOfSymbolsInPortfolio]]