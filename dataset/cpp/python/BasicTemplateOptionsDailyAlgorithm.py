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
### This example demonstrates how to add options for a given underlying equity security.
### It also shows how you can prefilter contracts easily based on strikes and expirations, and how you
### can inspect the option chain to pick a specific option contract to trade.
### </summary>
### <meta name="tag" content="using data" />
### <meta name="tag" content="options" />
### <meta name="tag" content="filter selection" />
class BasicTemplateOptionsDailyAlgorithm(QCAlgorithm):
    UnderlyingTicker = "GOOG"

    def Initialize(self):
        self.SetStartDate(2015, 12, 23)
        self.SetEndDate(2016, 1, 20)
        self.SetCash(100000)
        self.optionExpired = False

        equity = self.AddEquity(self.UnderlyingTicker, Resolution.Daily)
        option = self.AddOption(self.UnderlyingTicker, Resolution.Daily)
        self.option_symbol = option.Symbol

        # set our strike/expiry filter for this option chain
        option.SetFilter(lambda u: (u.CallsOnly().Strikes(0, 1).Expiration(0, 30)))

        # use the underlying equity as the benchmark
        self.SetBenchmark(equity.Symbol)

    def OnData(self,slice):
        if self.Portfolio.Invested: return

        chain = slice.OptionChains.GetValue(self.option_symbol)
        if chain is None:
            return

        # Grab us the contract nearest expiry
        contracts = sorted(chain, key = lambda x: x.Expiry)

        # if found, trade it
        if len(contracts) == 0: return
        symbol = contracts[0].Symbol
        self.MarketOrder(symbol, 1)

    def OnOrderEvent(self, orderEvent):
        self.Log(str(orderEvent))

        # Check for our expected OTM option expiry
        if "OTM" in orderEvent.Message:

            # Assert it is at midnight 1/16 (5AM UTC)
            if orderEvent.UtcTime.month != 1 and orderEvent.UtcTime.day != 16 and orderEvent.UtcTime.hour != 5:
                raise AssertionError(f"Expiry event was not at the correct time, {orderEvent.UtcTime}")

            self.optionExpired = True

    def OnEndOfAlgorithm(self):
        # Assert we had our option expire and fill a liquidation order
        if not self.optionExpired:
            raise AssertionError("Algorithm did not process the option expiration like expected")
