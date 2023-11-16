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

### <summary>
### Test algorithm using 'ConfidenceWeightedPortfolioConstructionModel' and 'ConstantAlphaModel'
### generating a constant 'Insight' with a 0.25 confidence
### </summary>
class ConfidenceWeightedFrameworkAlgorithm(QCAlgorithm):
    def Initialize(self):
        ''' Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'''

        # Set requested data resolution
        self.UniverseSettings.Resolution = Resolution.Minute

        # Order margin value has to have a minimum of 0.5% of Portfolio value, allows filtering out small trades and reduce fees.
        # Commented so regression algorithm is more sensitive
        #self.Settings.MinimumOrderMarginPortfolioPercentage = 0.005

        self.SetStartDate(2013,10,7)   #Set Start Date
        self.SetEndDate(2013,10,11)    #Set End Date
        self.SetCash(100000)           #Set Strategy Cash

        symbols = [ Symbol.Create("SPY", SecurityType.Equity, Market.USA) ]

        # set algorithm framework models
        self.SetUniverseSelection(ManualUniverseSelectionModel(symbols))
        self.SetAlpha(ConstantAlphaModel(InsightType.Price, InsightDirection.Up, timedelta(minutes = 20), 0.025, 0.25))
        self.SetPortfolioConstruction(ConfidenceWeightedPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())

    def OnEndOfAlgorithm(self):
        # holdings value should be 0.25 - to avoid price fluctuation issue we compare with 0.28 and 0.23
        if (self.Portfolio.TotalHoldingsValue > self.Portfolio.TotalPortfolioValue * 0.28
            or self.Portfolio.TotalHoldingsValue < self.Portfolio.TotalPortfolioValue * 0.23):
            raise ValueError("Unexpected Total Holdings Value: " + str(self.Portfolio.TotalHoldingsValue))
