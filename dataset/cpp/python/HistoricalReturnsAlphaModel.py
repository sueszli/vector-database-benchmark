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

class HistoricalReturnsAlphaModel(AlphaModel):
    '''Uses Historical returns to create insights.'''

    def __init__(self, *args, **kwargs):
        '''Initializes a new default instance of the HistoricalReturnsAlphaModel class.
        Args:
            lookback(int): Historical return lookback period
            resolution: The resolution of historical data'''
        self.lookback = kwargs['lookback'] if 'lookback' in kwargs else 1
        self.resolution = kwargs['resolution'] if 'resolution' in kwargs else Resolution.Daily
        self.predictionInterval = Time.Multiply(Extensions.ToTimeSpan(self.resolution), self.lookback)
        self.symbolDataBySymbol = {}
        self.insightCollection = InsightCollection()

    def Update(self, algorithm, data):
        '''Updates this alpha model with the latest data from the algorithm.
        This is called each time the algorithm receives data for subscribed securities
        Args:
            algorithm: The algorithm instance
            data: The new data available
        Returns:
            The new insights generated'''
        insights = []

        for symbol, symbolData in self.symbolDataBySymbol.items():
            if symbolData.CanEmit:

                direction = InsightDirection.Flat
                magnitude = symbolData.Return
                if magnitude > 0: direction = InsightDirection.Up
                if magnitude < 0: direction = InsightDirection.Down
                
                if direction == InsightDirection.Flat:
                    self.CancelInsights(algorithm, symbol)
                    continue

                insights.append(Insight.Price(symbol, self.predictionInterval, direction, magnitude, None))

        self.insightCollection.AddRange(insights)
        return insights

    def OnSecuritiesChanged(self, algorithm, changes):
        '''Event fired each time the we add/remove securities from the data feed
        Args:
            algorithm: The algorithm instance that experienced the change in securities
            changes: The security additions and removals from the algorithm'''

        # clean up data for removed securities
        for removed in changes.RemovedSecurities:
            symbolData = self.symbolDataBySymbol.pop(removed.Symbol, None)
            if symbolData is not None:
                symbolData.RemoveConsolidators(algorithm)
            self.CancelInsights(algorithm, removed.Symbol)

        # initialize data for added securities
        symbols = [ x.Symbol for x in changes.AddedSecurities ]
        history = algorithm.History(symbols, self.lookback, self.resolution)
        if history.empty: return

        tickers = history.index.levels[0]
        for ticker in tickers:
            symbol = SymbolCache.GetSymbol(ticker)

            if symbol not in self.symbolDataBySymbol:
                symbolData = SymbolData(symbol, self.lookback)
                self.symbolDataBySymbol[symbol] = symbolData
                symbolData.RegisterIndicators(algorithm, self.resolution)
                symbolData.WarmUpIndicators(history.loc[ticker])

    def CancelInsights(self, algorithm, symbol):
        if not self.insightCollection.ContainsKey(symbol):
            return
        insights = self.insightCollection[symbol]
        algorithm.Insights.Cancel(insights)
        self.insightCollection.Clear([ symbol ]);


class SymbolData:
    '''Contains data specific to a symbol required by this model'''
    def __init__(self, symbol, lookback):
        self.Symbol = symbol
        self.ROC = RateOfChange('{}.ROC({})'.format(symbol, lookback), lookback)
        self.Consolidator = None
        self.previous = 0

    def RegisterIndicators(self, algorithm, resolution):
        self.Consolidator = algorithm.ResolveConsolidator(self.Symbol, resolution)
        algorithm.RegisterIndicator(self.Symbol, self.ROC, self.Consolidator)

    def RemoveConsolidators(self, algorithm):
        if self.Consolidator is not None:
            algorithm.SubscriptionManager.RemoveConsolidator(self.Symbol, self.Consolidator)

    def WarmUpIndicators(self, history):
        for tuple in history.itertuples():
            self.ROC.Update(tuple.Index, tuple.close)

    @property
    def Return(self):
        return float(self.ROC.Current.Value)

    @property
    def CanEmit(self):
        if self.previous == self.ROC.Samples:
            return False

        self.previous = self.ROC.Samples
        return self.ROC.IsReady

    def __str__(self, **kwargs):
        return '{}: {:.2%}'.format(self.ROC.Name, (1 + self.Return)**252 - 1)
