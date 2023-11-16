from AlgorithmImports import *

class DropboxCoarseFineAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2019, 9, 23)
        self.SetEndDate(2019, 9, 30)
        self.SetCash(100000)
        self.AddUniverse(self.SelectCoarse, self.SelectFine)
        self.universeData = None
        self.nextUpdate = datetime(1, 1, 1)
        self.url = 'https://www.dropbox.com/s/x2sb9gaiicc6hm3/tickers_with_dates.csv?dl=1'

    def OnEndOfDay(self):
        if False:
            while True:
                i = 10
        for security in self.ActiveSecurities.Values:
            self.Debug(f'{self.Time.date()} {security.Symbol.Value} with Market Cap: ${security.Fundamentals.MarketCap}')

    def SelectCoarse(self, coarse):
        if False:
            print('Hello World!')
        return self.GetSymbols()

    def SelectFine(self, fine):
        if False:
            for i in range(10):
                print('nop')
        symbols = self.GetSymbols()
        return [f.Symbol for f in fine if f.MarketCap > 10000000000.0 and f.Symbol in symbols]

    def GetSymbols(self):
        if False:
            return 10
        if self.LiveMode:
            if self.Time < self.nextUpdate:
                return self.universeData[self.Time.date()]
            self.nextUpdate = self.Time + timedelta(hours=12)
            self.universeData = self.Parse(self.url)
        if self.universeData is None:
            self.universeData = self.Parse(self.url)
        if self.Time.date() not in self.universeData:
            return Universe.Unchanged
        return self.universeData[self.Time.date()]

    def Parse(self, url):
        if False:
            i = 10
            return i + 15
        file = self.Download(url).split('\n')
        data = [x.replace('\r', '').replace(' ', '') for x in file]
        split_data = [x.split(',') for x in data]
        symbolsByDate = {}
        for arr in split_data:
            date = datetime.strptime(arr[0], '%Y%m%d').date()
            symbols = [Symbol.Create(ticker, SecurityType.Equity, Market.USA) for ticker in arr[1:]]
            symbolsByDate[date] = symbols
        return symbolsByDate

    def OnSecuritiesChanged(self, changes):
        if False:
            while True:
                i = 10
        self.Log(f'Added Securities: {[security.Symbol.Value for security in changes.AddedSecurities]}')