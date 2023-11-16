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

class ScheduledEventsBenchmark(QCAlgorithm):

    def Initialize(self):

        self.SetStartDate(2011, 1, 1)
        self.SetEndDate(2022, 1, 1)
        self.SetCash(100000)
        self.AddEquity("SPY")

        for i in range(300):
            self.Schedule.On(self.DateRules.EveryDay("SPY"), self.TimeRules.AfterMarketOpen("SPY", i), self.Rebalance)
            self.Schedule.On(self.DateRules.EveryDay("SPY"), self.TimeRules.BeforeMarketClose("SPY", i), self.Rebalance)

    def OnData(self, data):
        pass

    def Rebalance(self):
        pass
