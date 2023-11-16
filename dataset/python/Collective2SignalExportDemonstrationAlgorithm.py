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
### This algorithm sends a list of portfolio targets to Collective2 API every time the ema indicators crosses between themselves.
### </summary>
### <meta name="tag" content="using data" />
### <meta name="tag" content="using quantconnect" />
### <meta name="tag" content="securities and portfolio" />
class Collective2SignalExportDemonstrationAlgorithm(QCAlgorithm):

    def Initialize(self):
        ''' Initialize the date and add all equity symbols present in list _symbols '''

        self.SetStartDate(2013, 10, 7)   #Set Start Date
        self.SetEndDate(2013, 10, 11)    #Set End Date
        self.SetCash(100000)             #Set Strategy Cash

        # Symbols accepted by Collective2. Collective2 accepts stock, future, forex and US stock option symbols
        self.AddEquity("GOOG")
        self.symbols = [Symbol.Create("SPY", SecurityType.Equity, Market.USA, None, None), Symbol.Create("EURUSD", SecurityType.Forex, Market.Oanda, None, None), Symbol.CreateFuture("ES", Market.CME, datetime(2023, 12, 15), None), Symbol.CreateOption("GOOG", Market.USA, OptionStyle.American, OptionRight.Call, 130, datetime(2023, 9, 1))]
        self.targets = []

        # Create a new PortfolioTarget for each symbol, assign it an initial amount of 0.05 and save it in self.targets list
        for item in self.symbols:
            symbol = self.AddSecurity(item).Symbol
            if symbol.SecurityType == SecurityType.Equity or symbol.SecurityType == SecurityType.Forex:
                self.targets.append(PortfolioTarget(symbol, 0.05))
            else:
                self.targets.append(PortfolioTarget(symbol, 1))

        self.fast = self.EMA("SPY", 10)
        self.slow = self.EMA("SPY", 100)

        # Initialize these flags, to check when the ema indicators crosses between themselves
        self.emaFastIsNotSet = True;
        self.emaFastWasAbove = False;

        # Set Collective2 export provider
        # Collective2 APIv4 KEY: This value is provided by Collective2 in your account section (See https://collective2.com/account-info)
        # See API documentation at https://trade.collective2.com/c2-api
        self.collective2Apikey = "YOUR APIV4 KEY"

        # Collective2 System ID: This value is found beside the system's name (strategy's name) on the main system page
        self.collective2SystemId = 0

        self.SignalExport.AddSignalExportProviders(Collective2SignalExport(self.collective2Apikey, self.collective2SystemId))
        
        self.first_call = True
        
        self.SetWarmUp(100)

    def OnData(self, data):
        ''' Reduce the quantity of holdings for one security and increase the holdings to the another
        one when the EMA's indicators crosses between themselves, then send a signal to Collective2 API '''
        if self.IsWarmingUp: return
        
        # Place an order as soon as possible to send a signal.
        if self.first_call:
            self.SetHoldings("SPY", 0.1)
            self.targets[0] = PortfolioTarget(self.Portfolio["SPY"].Symbol, 0.1)
            self.SignalExport.SetTargetPortfolio(self.targets)
            self.first_call = False

        fast = self.fast.Current.Value
        slow = self.slow.Current.Value

        # Set the value of flag _emaFastWasAbove, to know when the ema indicators crosses between themselves
        if self.emaFastIsNotSet == True:
            if fast > slow *1.001:
                self.emaFastWasAbove = True
            else:
                self.emaFastWasAbove = False
            self.emaFastIsNotSet = False;

        # Check whether ema fast and ema slow crosses. If they do, set holdings to SPY
        # or reduce its holdings, change its value in self.targets list and send signals
        #  to Collective2 API from self.targets
        if fast > slow * 1.001 and (not self.emaFastWasAbove):
            self.SetHoldings("SPY", 0.1)
            self.targets[0] = PortfolioTarget(self.Portfolio["SPY"].Symbol, 0.1)
            self.SignalExport.SetTargetPortfolio(self.targets)
        elif fast < slow * 0.999 and (self.emaFastWasAbove):
            self.SetHoldings("SPY", 0.01)
            self.targets[0] = PortfolioTarget(self.Portfolio["SPY"].Symbol, 0.01)
            self.SignalExport.SetTargetPortfolio(self.targets)
