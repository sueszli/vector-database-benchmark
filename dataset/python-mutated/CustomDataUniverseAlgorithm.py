from AlgorithmImports import *

class CustomDataUniverseAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.UniverseSettings.Resolution = Resolution.Daily
        self.SetStartDate(2015, 1, 5)
        self.SetEndDate(2015, 7, 1)
        self.SetCash(100000)
        self.AddEquity('SPY', Resolution.Daily)
        self.SetBenchmark('SPY')
        self.AddUniverse(NyseTopGainers, 'universe-nyse-top-gainers', Resolution.Daily, self.nyseTopGainers)

    def nyseTopGainers(self, data):
        if False:
            print('Hello World!')
        return [x.Symbol for x in data if x['TopGainersRank'] <= 2]

    def OnData(self, slice):
        if False:
            for i in range(10):
                print('nop')
        pass

    def OnSecuritiesChanged(self, changes):
        if False:
            i = 10
            return i + 15
        self._changes = changes
        for security in changes.RemovedSecurities:
            if security.Invested:
                self.Liquidate(security.Symbol)
                self.Log('Exit {0} at {1}'.format(security.Symbol, security.Close))
        for security in changes.AddedSecurities:
            if not security.Invested and security.Close != 0:
                qty = self.CalculateOrderQuantity(security.Symbol, -0.25)
                self.MarketOnOpenOrder(security.Symbol, qty)
                self.Log('Enter {0} at {1}'.format(security.Symbol, security.Close))

class NyseTopGainers(PythonData):

    def __init__(self):
        if False:
            return 10
        self.count = 0
        self.last_date = datetime.min

    def GetSource(self, config, date, isLiveMode):
        if False:
            print('Hello World!')
        url = 'http://www.wsj.com/mdc/public/page/2_3021-gainnyse-gainer.html' if isLiveMode else 'https://www.dropbox.com/s/vrn3p38qberw3df/nyse-gainers.csv?dl=1'
        return SubscriptionDataSource(url, SubscriptionTransportMedium.RemoteFile)

    def Reader(self, config, line, date, isLiveMode):
        if False:
            while True:
                i = 10
        if not isLiveMode:
            if not (line.strip() and line[0].isdigit()):
                return None
            csv = line.split(',')
            nyse = NyseTopGainers()
            nyse.Time = datetime.strptime(csv[0], '%Y%m%d')
            nyse.EndTime = nyse.Time + timedelta(1)
            nyse.Symbol = Symbol.Create(csv[1], SecurityType.Equity, Market.USA)
            nyse['TopGainersRank'] = int(csv[2])
            return nyse
        if self.last_date != date:
            self.last_date = date
            self.count = 0
        if not line.startswith('<a href="/public/quotes/main.html?symbol='):
            return None
        last_close_paren = line.rfind(')')
        last_open_paren = line.rfind('(')
        if last_open_paren == -1 or last_close_paren == -1:
            return None
        symbol_string = line[last_open_paren + 1:last_close_paren]
        nyse = NyseTopGainers()
        nyse.Time = date
        nyse.EndTime = nyse.Time + timedelta(1)
        nyse.Symbol = Symbol.Create(symbol_string, SecurityType.Equity, Market.USA)
        nyse['TopGainersRank'] = self.count
        self.count = self.count + 1
        return nyse