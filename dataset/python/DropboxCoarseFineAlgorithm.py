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
### In this algorithm, we fetch a list of tickers with corresponding dates from a file on Dropbox.
### We then create a fine fundamental universe which contains those symbols on their respective dates.### 
### </summary>
### <meta name="tag" content="download" />
### <meta name="tag" content="universes" />
### <meta name="tag" content="custom data" />
class DropboxCoarseFineAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2019, 9, 23) # Set Start Date
        self.SetEndDate(2019, 9, 30) # Set End Date
        self.SetCash(100000)  # Set Strategy Cash
        self.AddUniverse(self.SelectCoarse, self.SelectFine)
       
        self.universeData = None
        self.nextUpdate = datetime(1, 1, 1) # Minimum datetime 
        self.url = "https://www.dropbox.com/s/x2sb9gaiicc6hm3/tickers_with_dates.csv?dl=1" 
        
    def OnEndOfDay(self): 
        for security in self.ActiveSecurities.Values:
            self.Debug(f"{self.Time.date()} {security.Symbol.Value} with Market Cap: ${security.Fundamentals.MarketCap}")
        
    def SelectCoarse(self, coarse):
        return self.GetSymbols()
        
    def SelectFine(self, fine): 
        symbols = self.GetSymbols()
        
        # Return symbols from our list which have a market capitalization of at least 10B
        return [f.Symbol for f in fine if f.MarketCap > 1e10 and f.Symbol in symbols]
        
    def GetSymbols(self):
        
        # In live trading update every 12 hours
        if self.LiveMode:
            if self.Time < self.nextUpdate:
                # Return today's row
                return self.universeData[self.Time.date()]
            # When updating set the new reset time.
            self.nextUpdate = self.Time + timedelta(hours=12)
            self.universeData = self.Parse(self.url) 
        
        # In backtest load once if not set, then just use the dates.
        if self.universeData is None:
            self.universeData = self.Parse(self.url)

        # Check if contains the row we need
        if self.Time.date() not in self.universeData:
            return Universe.Unchanged

        return self.universeData[self.Time.date()]
        
    
    def Parse(self, url): 
        # Download file from url as string
        file = self.Download(url).split("\n")
        
        # # Remove formatting characters
        data = [x.replace("\r", "").replace(" ", "") for x in file]
        
        # # Split data by date and symbol
        split_data = [x.split(",") for x in data]
        
        # Dictionary to hold list of active symbols for each date, keyed by date
        symbolsByDate = {}
        
        # Parse data into dictionary
        for arr in split_data:
            date = datetime.strptime(arr[0], "%Y%m%d").date()
            symbols = [Symbol.Create(ticker, SecurityType.Equity, Market.USA) for ticker in arr[1:]]
            symbolsByDate[date] = symbols
        
        return symbolsByDate
        
    def OnSecuritiesChanged(self, changes):
        self.Log(f"Added Securities: {[security.Symbol.Value for security in changes.AddedSecurities]}")
