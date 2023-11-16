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
### In this algorithm we demonstrate how to use the coarse fundamental data to define a universe as the top dollar volume and set the algorithm to use raw prices
### </summary>
### <meta name="tag" content="using data" />
### <meta name="tag" content="universes" />
### <meta name="tag" content="coarse universes" />
### <meta name="tag" content="fine universes" />
class RawPricesCoarseUniverseAlgorithm(QCAlgorithm):

    def Initialize(self):
        '''Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'''

        # what resolution should the data *added* to the universe be?
        self.UniverseSettings.Resolution = Resolution.Daily

        self.SetStartDate(2014,1,1)    #Set Start Date
        self.SetEndDate(2015,1,1)      #Set End Date
        self.SetCash(50000)            #Set Strategy Cash

        # Set the security initializer with the characteristics defined in CustomSecurityInitializer
        self.SetSecurityInitializer(self.CustomSecurityInitializer)

        # this add universe method accepts a single parameter that is a function that
        # accepts an IEnumerable<CoarseFundamental> and returns IEnumerable<Symbol>
        self.AddUniverse(self.CoarseSelectionFunction)

        self.__numberOfSymbols = 5

    def CustomSecurityInitializer(self, security):
        '''Initialize the security with raw prices and zero fees 
        Args:
            security: Security which characteristics we want to change'''
        security.SetDataNormalizationMode(DataNormalizationMode.Raw)
        security.SetFeeModel(ConstantFeeModel(0))

    # sort the data by daily dollar volume and take the top 'NumberOfSymbols'
    def CoarseSelectionFunction(self, coarse):
        # sort descending by daily dollar volume
        sortedByDollarVolume = sorted(coarse, key=lambda x: x.DollarVolume, reverse=True)

        # return the symbol objects of the top entries from our sorted collection
        return [ x.Symbol for x in sortedByDollarVolume[:self.__numberOfSymbols] ]


    # this event fires whenever we have changes to our universe
    def OnSecuritiesChanged(self, changes):
        # liquidate removed securities
        for security in changes.RemovedSecurities:
            if security.Invested:
                self.Liquidate(security.Symbol)

        # we want 20% allocation in each security in our universe
        for security in changes.AddedSecurities:
            self.SetHoldings(security.Symbol, 0.2)

    def OnOrderEvent(self, orderEvent):
        if orderEvent.Status == OrderStatus.Filled:
            self.Log(f"OnOrderEvent({self.UtcTime}):: {orderEvent}")
