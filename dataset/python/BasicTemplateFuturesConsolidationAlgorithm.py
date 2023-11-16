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
### A demonstration of consolidating futures data into larger bars for your algorithm.
### </summary>
### <meta name="tag" content="using data" />
### <meta name="tag" content="benchmarks" />
### <meta name="tag" content="consolidating data" />
### <meta name="tag" content="futures" />
class BasicTemplateFuturesConsolidationAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(1000000)

        # Subscribe and set our expiry filter for the futures chain
        futureSP500 = self.AddFuture(Futures.Indices.SP500EMini)
        # set our expiry filter for this future chain
        # SetFilter method accepts timedelta objects or integer for days.
        # The following statements yield the same filtering criteria
        futureSP500.SetFilter(0, 182)
        # future.SetFilter(timedelta(0), timedelta(182))

        self.consolidators = dict()

    def OnData(self,slice):
        pass

    def OnDataConsolidated(self, sender, quoteBar):
        self.Log("OnDataConsolidated called on " + str(self.Time))
        self.Log(str(quoteBar))
        
    def OnSecuritiesChanged(self, changes):
        for security in changes.AddedSecurities:
            consolidator = QuoteBarConsolidator(timedelta(minutes=5))
            consolidator.DataConsolidated += self.OnDataConsolidated
            self.SubscriptionManager.AddConsolidator(security.Symbol, consolidator)
            self.consolidators[security.Symbol] = consolidator
            
        for security in changes.RemovedSecurities:
            consolidator = self.consolidators.pop(security.Symbol)
            self.SubscriptionManager.RemoveConsolidator(security.Symbol, consolidator)
            consolidator.DataConsolidated -= self.OnDataConsolidated
