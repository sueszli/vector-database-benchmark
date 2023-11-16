from AlgorithmImports import *

class CustomDataRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2011, 9, 14)
        self.SetEndDate(2015, 12, 1)
        self.SetCash(100000)
        resolution = Resolution.Second if self.LiveMode else Resolution.Daily
        self.AddData(Bitcoin, 'BTC', resolution)
        seeder = FuncSecuritySeeder(self.GetLastKnownPrices)
        self.SetSecurityInitializer(lambda x: seeder.SeedSecurity(x))
        self._warmedUpChecked = False

    def OnData(self, data):
        if False:
            return 10
        if not self.Portfolio.Invested:
            if data['BTC'].Close != 0:
                self.Order('BTC', self.Portfolio.MarginRemaining / abs(data['BTC'].Close + 1))

    def OnSecuritiesChanged(self, changes):
        if False:
            while True:
                i = 10
        changes.FilterCustomSecurities = False
        for addedSecurity in changes.AddedSecurities:
            if addedSecurity.Symbol.Value == 'BTC':
                self._warmedUpChecked = True
            if not addedSecurity.HasData:
                raise ValueError(f'Security {addedSecurity.Symbol} was not warmed up!')

    def OnEndOfAlgorithm(self):
        if False:
            while True:
                i = 10
        if not self._warmedUpChecked:
            raise ValueError('Security was not warmed up!')

class Bitcoin(PythonData):
    """Custom Data Type: Bitcoin data from Quandl - http://www.quandl.com/help/api-for-bitcoin-data"""

    def GetSource(self, config, date, isLiveMode):
        if False:
            return 10
        if isLiveMode:
            return SubscriptionDataSource('https://www.bitstamp.net/api/ticker/', SubscriptionTransportMedium.Rest)
        return SubscriptionDataSource('https://www.quantconnect.com/api/v2/proxy/quandl/api/v3/datasets/BCHARTS/BITSTAMPUSD.csv?order=asc&api_key=WyAazVXnq7ATy_fefTqm', SubscriptionTransportMedium.RemoteFile)

    def Reader(self, config, line, date, isLiveMode):
        if False:
            return 10
        coin = Bitcoin()
        coin.Symbol = config.Symbol
        if isLiveMode:
            try:
                liveBTC = json.loads(line)
                value = liveBTC['last']
                if value == 0:
                    return None
                coin.Time = datetime.now()
                coin.Value = value
                coin['Open'] = float(liveBTC['open'])
                coin['High'] = float(liveBTC['high'])
                coin['Low'] = float(liveBTC['low'])
                coin['Close'] = float(liveBTC['last'])
                coin['Ask'] = float(liveBTC['ask'])
                coin['Bid'] = float(liveBTC['bid'])
                coin['VolumeBTC'] = float(liveBTC['volume'])
                coin['WeightedPrice'] = float(liveBTC['vwap'])
                return coin
            except ValueError:
                return None
        if not (line.strip() and line[0].isdigit()):
            return None
        try:
            data = line.split(',')
            coin.Time = datetime.strptime(data[0], '%Y-%m-%d')
            coin.EndTime = coin.Time + timedelta(days=1)
            coin.Value = float(data[4])
            coin['Open'] = float(data[1])
            coin['High'] = float(data[2])
            coin['Low'] = float(data[3])
            coin['Close'] = float(data[4])
            coin['VolumeBTC'] = float(data[5])
            coin['VolumeUSD'] = float(data[6])
            coin['WeightedPrice'] = float(data[7])
            return coin
        except ValueError:
            return None