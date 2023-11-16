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

from Alphas.ConstantAlphaModel import ConstantAlphaModel
from Selection.OptionUniverseSelectionModel import OptionUniverseSelectionModel
from Execution.ImmediateExecutionModel import ImmediateExecutionModel
from Risk.NullRiskManagementModel import NullRiskManagementModel

### <summary>
### Basic template options framework algorithm uses framework components
### to define an algorithm that trades options.
### </summary>
class BasicTemplateOptionsFrameworkAlgorithm(QCAlgorithm):

    def Initialize(self):

        self.UniverseSettings.Resolution = Resolution.Minute

        self.SetStartDate(2014, 6, 5)
        self.SetEndDate(2014, 6, 9)
        self.SetCash(100000)

        # set framework models
        self.SetUniverseSelection(EarliestExpiringWeeklyAtTheMoneyPutOptionUniverseSelectionModel(self.SelectOptionChainSymbols))
        self.SetAlpha(ConstantOptionContractAlphaModel(InsightType.Price, InsightDirection.Up, timedelta(hours = 0.5)))
        self.SetPortfolioConstruction(SingleSharePortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(NullRiskManagementModel())


    def SelectOptionChainSymbols(self, utcTime):
        newYorkTime = Extensions.ConvertFromUtc(utcTime, TimeZones.NewYork)
        ticker = "TWX" if newYorkTime.date() < date(2014, 6, 6) else "AAPL"
        return [ Symbol.Create(ticker, SecurityType.Option, Market.USA, f"?{ticker}") ]

class EarliestExpiringWeeklyAtTheMoneyPutOptionUniverseSelectionModel(OptionUniverseSelectionModel):
    '''Creates option chain universes that select only the earliest expiry ATM weekly put contract
    and runs a user defined optionChainSymbolSelector every day to enable choosing different option chains'''
    def __init__(self, select_option_chain_symbols):
        super().__init__(timedelta(1), select_option_chain_symbols)

    def Filter(self, filter):
        '''Defines the option chain universe filter'''
        return (filter.Strikes(+1, +1)
                      # Expiration method accepts timedelta objects or integer for days.
                      # The following statements yield the same filtering criteria
                      .Expiration(0, 7)
                      # .Expiration(timedelta(0), timedelta(7))
                      .WeeklysOnly()
                      .PutsOnly()
                      .OnlyApplyFilterAtMarketOpen())

class ConstantOptionContractAlphaModel(ConstantAlphaModel):
    '''Implementation of a constant alpha model that only emits insights for option symbols'''
    def __init__(self, type, direction, period):
        super().__init__(type, direction, period)

    def ShouldEmitInsight(self, utcTime, symbol):
        # only emit alpha for option symbols and not underlying equity symbols
        if symbol.SecurityType != SecurityType.Option:
            return False

        return super().ShouldEmitInsight(utcTime, symbol)

class SingleSharePortfolioConstructionModel(PortfolioConstructionModel):
    '''Portfolio construction model that sets target quantities to 1 for up insights and -1 for down insights'''
    def CreateTargets(self, algorithm, insights):
        targets = []
        for insight in insights:
            targets.append(PortfolioTarget(insight.Symbol, insight.Direction))
        return targets
