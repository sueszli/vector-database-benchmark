# QUANTCONNECT.COM - Democratizing Finance, Empowering Individuals.
# Lean Algorithmic Trading Engine v2.0. Copyright 2014 QuantConnect Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from AlgorithmImports import *
from Selection.FundamentalUniverseSelectionModel import FundamentalUniverseSelectionModel

from math import ceil
from itertools import chain

class GreenblattMagicFormulaAlpha(QCAlgorithm):
    ''' Alpha Streams: Benchmark Alpha: Pick stocks according to Joel Greenblatt's Magic Formula
    This alpha picks stocks according to Joel Greenblatt's Magic Formula.
    First, each stock is ranked depending on the relative value of the ratio EV/EBITDA. For example, a stock
    that has the lowest EV/EBITDA ratio in the security universe receives a score of one while a stock that has
    the tenth lowest EV/EBITDA score would be assigned 10 points.

    Then, each stock is ranked and given a score for the second valuation ratio, Return on Capital (ROC).
    Similarly, a stock that has the highest ROC value in the universe gets one score point.
    The stocks that receive the lowest combined score are chosen for insights.

    Source: Greenblatt, J. (2010) The Little Book That Beats the Market

    This alpha is part of the Benchmark Alpha Series created by QuantConnect which are open
    sourced so the community and client funds can see an example of an alpha.'''

    def Initialize(self):

        self.SetStartDate(2018, 1, 1)
        self.SetCash(100000)

        #Set zero transaction fees
        self.SetSecurityInitializer(lambda security: security.SetFeeModel(ConstantFeeModel(0)))

        # select stocks using MagicFormulaUniverseSelectionModel
        self.SetUniverseSelection(GreenBlattMagicFormulaUniverseSelectionModel())

        # Use MagicFormulaAlphaModel to establish insights
        self.SetAlpha(RateOfChangeAlphaModel())

        # Equally weigh securities in portfolio, based on insights
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())

        ## Set Immediate Execution Model
        self.SetExecution(ImmediateExecutionModel())

        ## Set Null Risk Management Model
        self.SetRiskManagement(NullRiskManagementModel())

class RateOfChangeAlphaModel(AlphaModel):
    '''Uses Rate of Change (ROC) to create magnitude prediction for insights.'''

    def __init__(self, *args, **kwargs):
        self.lookback = kwargs.get('lookback', 1)
        self.resolution = kwargs.get('resolution', Resolution.Daily)
        self.predictionInterval = Time.Multiply(Extensions.ToTimeSpan(self.resolution), self.lookback)
        self.symbolDataBySymbol = {}

    def Update(self, algorithm, data):
        insights = []
        for symbol, symbolData in self.symbolDataBySymbol.items():
            if symbolData.CanEmit:
                insights.append(Insight.Price(symbol, self.predictionInterval, InsightDirection.Up, symbolData.Return, None))
        return insights

    def OnSecuritiesChanged(self, algorithm, changes):

        # clean up data for removed securities
        for removed in changes.RemovedSecurities:
            symbolData = self.symbolDataBySymbol.pop(removed.Symbol, None)
            if symbolData is not None:
                symbolData.RemoveConsolidators(algorithm)

        # initialize data for added securities
        symbols = [ x.Symbol for x in changes.AddedSecurities
            if x.Symbol not in self.symbolDataBySymbol]

        history = algorithm.History(symbols, self.lookback, self.resolution)
        if history.empty: return

        for symbol in symbols:
            symbolData = SymbolData(algorithm, symbol, self.lookback, self.resolution)
            self.symbolDataBySymbol[symbol] = symbolData
            symbolData.WarmUpIndicators(history.loc[symbol])


class SymbolData:
    '''Contains data specific to a symbol required by this model'''
    def __init__(self, algorithm, symbol, lookback, resolution):
        self.previous = 0
        self.symbol = symbol
        self.ROC = RateOfChange(f'{symbol}.ROC({lookback})', lookback)
        self.consolidator = algorithm.ResolveConsolidator(symbol, resolution)
        algorithm.RegisterIndicator(symbol, self.ROC, self.consolidator)

    def RemoveConsolidators(self, algorithm):
        algorithm.SubscriptionManager.RemoveConsolidator(self.symbol, self.consolidator)

    def WarmUpIndicators(self, history):
        for tuple in history.itertuples():
            self.ROC.Update(tuple.Index, tuple.close)

    @property
    def Return(self):
        return self.ROC.Current.Value

    @property
    def CanEmit(self):
        if self.previous == self.ROC.Samples:
            return False

        self.previous = self.ROC.Samples
        return self.ROC.IsReady

    def __str__(self, **kwargs):
        return f'{self.ROC.Name}: {(1 + self.Return)**252 - 1:.2%}'


class GreenBlattMagicFormulaUniverseSelectionModel(FundamentalUniverseSelectionModel):
    '''Defines a universe according to Joel Greenblatt's Magic Formula, as a universe selection model for the framework algorithm.
       From the universe QC500, stocks are ranked using the valuation ratios, Enterprise Value to EBITDA (EV/EBITDA) and Return on Assets (ROA).
    '''

    def __init__(self,
                 filterFineData = True,
                 universeSettings = None):
        '''Initializes a new default instance of the MagicFormulaUniverseSelectionModel'''
        super().__init__(filterFineData, universeSettings)

        # Number of stocks in Coarse Universe
        self.NumberOfSymbolsCoarse = 500
        # Number of sorted stocks in the fine selection subset using the valuation ratio, EV to EBITDA (EV/EBITDA)
        self.NumberOfSymbolsFine = 20
        # Final number of stocks in security list, after sorted by the valuation ratio, Return on Assets (ROA)
        self.NumberOfSymbolsInPortfolio = 10

        self.lastMonth = -1
        self.dollarVolumeBySymbol = {}

    def SelectCoarse(self, algorithm, coarse):
        '''Performs coarse selection for constituents.
        The stocks must have fundamental data'''
        month = algorithm.Time.month
        if month == self.lastMonth:
            return Universe.Unchanged
        self.lastMonth = month

        # sort the stocks by dollar volume and take the top 1000
        top = sorted([x for x in coarse if x.HasFundamentalData],
                    key=lambda x: x.DollarVolume, reverse=True)[:self.NumberOfSymbolsCoarse]

        self.dollarVolumeBySymbol = { i.Symbol: i.DollarVolume for i in top }

        return list(self.dollarVolumeBySymbol.keys())


    def SelectFine(self, algorithm, fine):
        '''QC500: Performs fine selection for the coarse selection constituents
        The company's headquarter must in the U.S.
        The stock must be traded on either the NYSE or NASDAQ
        At least half a year since its initial public offering
        The stock's market cap must be greater than 500 million

        Magic Formula: Rank stocks by Enterprise Value to EBITDA (EV/EBITDA)
        Rank subset of previously ranked stocks (EV/EBITDA), using the valuation ratio Return on Assets (ROA)'''

        # QC500:
        ## The company's headquarter must in the U.S.
        ## The stock must be traded on either the NYSE or NASDAQ
        ## At least half a year since its initial public offering
        ## The stock's market cap must be greater than 500 million
        filteredFine = [x for x in fine if x.CompanyReference.CountryId == "USA"
                                        and (x.CompanyReference.PrimaryExchangeID == "NYS" or x.CompanyReference.PrimaryExchangeID == "NAS")
                                        and (algorithm.Time - x.SecurityReference.IPODate).days > 180
                                        and x.EarningReports.BasicAverageShares.ThreeMonths * x.EarningReports.BasicEPS.TwelveMonths * x.ValuationRatios.PERatio > 5e8]
        count = len(filteredFine)
        if count == 0: return []

        myDict = dict()
        percent = self.NumberOfSymbolsFine / count

        # select stocks with top dollar volume in every single sector
        for key in ["N", "M", "U", "T", "B", "I"]:
            value = [x for x in filteredFine if x.CompanyReference.IndustryTemplateCode == key]
            value = sorted(value, key=lambda x: self.dollarVolumeBySymbol[x.Symbol], reverse = True)
            myDict[key] = value[:ceil(len(value) * percent)]

        # stocks in QC500 universe
        topFine = chain.from_iterable(myDict.values())

        #  Magic Formula:
        ## Rank stocks by Enterprise Value to EBITDA (EV/EBITDA)
        ## Rank subset of previously ranked stocks (EV/EBITDA), using the valuation ratio Return on Assets (ROA)

        # sort stocks in the security universe of QC500 based on Enterprise Value to EBITDA valuation ratio
        sortedByEVToEBITDA = sorted(topFine, key=lambda x: x.ValuationRatios.EVToEBITDA , reverse=True)

        # sort subset of stocks that have been sorted by Enterprise Value to EBITDA, based on the valuation ratio Return on Assets (ROA)
        sortedByROA = sorted(sortedByEVToEBITDA[:self.NumberOfSymbolsFine], key=lambda x: x.ValuationRatios.ForwardROA, reverse=False)

        # retrieve list of securites in portfolio
        return [f.Symbol for f in sortedByROA[:self.NumberOfSymbolsInPortfolio]]
