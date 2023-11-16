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
from Portfolio.MeanVarianceOptimizationPortfolioConstructionModel import *

### <summary>
### Mean Variance Optimization algorithm
### Uses the HistoricalReturnsAlphaModel and the MeanVarianceOptimizationPortfolioConstructionModel
### to create an algorithm that rebalances the portfolio according to modern portfolio theory
### </summary>
### <meta name="tag" content="using data" />
### <meta name="tag" content="using quantconnect" />
### <meta name="tag" content="trading and orders" />
class MeanVarianceOptimizationFrameworkAlgorithm(QCAlgorithm):
    '''Mean Variance Optimization algorithm.'''

    def Initialize(self):

        # Set requested data resolution
        self.UniverseSettings.Resolution = Resolution.Minute

        self.Settings.RebalancePortfolioOnInsightChanges = False

        self.SetStartDate(2013,10,7)   #Set Start Date
        self.SetEndDate(2013,10,11)    #Set End Date
        self.SetCash(100000)           #Set Strategy Cash

        self.symbols = [ Symbol.Create(x, SecurityType.Equity, Market.USA) for x in [ 'AIG', 'BAC', 'IBM', 'SPY' ] ]

        # set algorithm framework models
        self.SetUniverseSelection(CoarseFundamentalUniverseSelectionModel(self.coarseSelector))
        self.SetAlpha(HistoricalReturnsAlphaModel(resolution = Resolution.Daily))
        self.SetPortfolioConstruction(MeanVarianceOptimizationPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(NullRiskManagementModel())

    def coarseSelector(self, coarse):
        # Drops SPY after the 8th
        last = 3 if self.Time.day > 8 else len(self.symbols)

        return self.symbols[0:last]

    def OnOrderEvent(self,  orderEvent):
        if orderEvent.Status == OrderStatus.Filled:
            self.Log(str(orderEvent))
