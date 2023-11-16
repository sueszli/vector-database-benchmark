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
### A demonstration algorithm to check there can be placed an order of a pair not present
### in the brokerage using the conversion between stablecoins
### </summary>
class StableCoinsRegressionAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2018, 5, 1)
        self.SetEndDate(2018, 5, 2)
        self.SetCash("USDT", 200000000)
        self.SetBrokerageModel(BrokerageName.Binance, AccountType.Cash)
        self.AddCrypto("BTCUSDT", Resolution.Hour, Market.Binance)

    def OnData(self, data):
        if not self.Portfolio.Invested:
            self.SetHoldings("BTCUSDT", 1)
