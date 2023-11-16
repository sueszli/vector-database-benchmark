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
from Selection.ETFConstituentsUniverseSelectionModel import *

### <summary>
### Demonstration of using the ETFConstituentsUniverseSelectionModel
### </summary>
class ETFConstituentsFrameworkAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2020, 12, 1)
        self.SetEndDate(2020, 12, 7)
        self.SetCash(100000)

        self.UniverseSettings.Resolution = Resolution.Daily
        symbol = Symbol.Create("SPY", SecurityType.Equity, Market.USA)
        self.AddUniverseSelection(ETFConstituentsUniverseSelectionModel(symbol, self.UniverseSettings, self.ETFConstituentsFilter))

        self.AddAlpha(ConstantAlphaModel(InsightType.Price, InsightDirection.Up, timedelta(days=1)))

        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())


    def ETFConstituentsFilter(self, constituents: List[ETFConstituentData]) -> List[Symbol]:
        # Get the 10 securities with the largest weight in the index
        selected = sorted([c for c in constituents if c.Weight],
            key=lambda c: c.Weight, reverse=True)[:8]
        return [c.Symbol for c in selected]

