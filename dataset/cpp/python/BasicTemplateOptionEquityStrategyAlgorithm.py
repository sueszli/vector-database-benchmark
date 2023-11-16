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
### This example demonstrates how to execute a Call Butterfly option equity strategy
### It adds options for a given underlying equity security, and shows how you can prefilter contracts easily based on strikes and expirations
### </summary>
### <meta name="tag" content="using data" />
### <meta name="tag" content="options" />
### <meta name="tag" content="filter selection" />
### <meta name="tag" content="trading and orders" />
class BasicTemplateOptionEquityStrategyAlgorithm(QCAlgorithm):
    UnderlyingTicker = "GOOG"

    def Initialize(self):
        self.SetStartDate(2015, 12, 24)
        self.SetEndDate(2015, 12, 24)

        equity = self.AddEquity(self.UnderlyingTicker)
        option = self.AddOption(self.UnderlyingTicker)
        self.option_symbol = option.Symbol

        # set our strike/expiry filter for this option chain
        option.SetFilter(lambda u: (u.Strikes(-2, +2)
                                     # Expiration method accepts TimeSpan objects or integer for days.
                                     # The following statements yield the same filtering criteria
                                     .Expiration(0, 180)))

    def OnData(self,slice):
        if self.Portfolio.Invested or not self.IsMarketOpen(self.option_symbol): return

        chain = slice.OptionChains.GetValue(self.option_symbol)
        if chain is None:
            return

        groupedByExpiry = dict()
        for contract in [contract for contract in chain if contract.Right == OptionRight.Call]:
            groupedByExpiry.setdefault(int(contract.Expiry.timestamp()), []).append(contract)

        firstExpiry = list(sorted(groupedByExpiry))[0]
        callContracts = sorted(groupedByExpiry[firstExpiry], key = lambda x: x.Strike)
        
        expiry = callContracts[0].Expiry
        lowerStrike = callContracts[0].Strike
        middleStrike = callContracts[1].Strike
        higherStrike = callContracts[2].Strike

        optionStrategy = OptionStrategies.CallButterfly(self.option_symbol, higherStrike, middleStrike, lowerStrike, expiry)
                    
        self.Order(optionStrategy, 10)

    def OnOrderEvent(self, orderEvent):
        self.Log(str(orderEvent))
