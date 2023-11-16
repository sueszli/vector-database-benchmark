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
### Example and regression algorithm asserting the behavior of registering and unregistering an indicator from the engine
### </summary>
class UnregisterIndicatorRegressionAlgorithm(QCAlgorithm):
    def Initialize(self):
        '''Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'''
        self.SetStartDate(2013,10, 7)
        self.SetEndDate(2013,10,11)

        spy = self.AddEquity("SPY")
        ibm = self.AddEquity("IBM")

        self._symbols = [ spy.Symbol, ibm.Symbol ]
        self._trin = self.TRIN(self._symbols, Resolution.Minute)
        self._trin2 = None

    def OnData(self, data):
        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.

        Arguments:
            data: Slice object keyed by symbol containing the stock data
        '''
        if self._trin.IsReady:
            self._trin.Reset()
            self.UnregisterIndicator(self._trin)

            # let's create a new one with a differente resolution
            self._trin2 = self.TRIN(self._symbols, Resolution.Hour)

        if not self._trin2 is None and self._trin2.IsReady:
            if self._trin.IsReady:
                raise ValueError("Indicator should of stop getting updates!")

            if not self.Portfolio.Invested:
                self.SetHoldings(self._symbols[0], 0.5)
                self.SetHoldings(self._symbols[1], 0.5)
