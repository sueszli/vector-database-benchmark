from AlgorithmImports import *

class CustomDataBitcoinAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetStartDate(2011, 9, 13)
        self.SetEndDate(datetime.now().date() - timedelta(1))
        self.SetCash(100000)
        self.AddData(Bitcoin, 'BTC')

    def OnData(self, data):
        if False:
            return 10
        if not data.ContainsKey('BTC'):
            return
        close = data['BTC'].Close
        if not self.Portfolio.Invested:
            self.SetHoldings('BTC', 1)
            self.Debug("Buying BTC 'Shares': BTC: {0}".format(close))
        self.Debug('Time: {0} {1}'.format(datetime.now(), close))

class Bitcoin(PythonData):
    """Custom Data Type: Bitcoin data from Quandl - http://www.quandl.com/help/api-for-bitcoin-data"""

    def GetSource(self, config, date, isLiveMode):
        if False:
            while True:
                i = 10
        if isLiveMode:
            return SubscriptionDataSource('https://www.bitstamp.net/api/ticker/', SubscriptionTransportMedium.Rest)
        return SubscriptionDataSource('https://www.quantconnect.com/api/v2/proxy/quandl/api/v3/datasets/BCHARTS/BITSTAMPUSD.csv?order=asc&api_key=WyAazVXnq7ATy_fefTqm', SubscriptionTransportMedium.RemoteFile)

    def Reader(self, config, line, date, isLiveMode):
        if False:
            i = 10
            return i + 15
        coin = Bitcoin()
        coin.Symbol = config.Symbol
        if isLiveMode:
            try:
                liveBTC = json.loads(line)
                value = liveBTC['last']
                if value == 0:
                    return None
                coin.EndTime = datetime.utcnow().astimezone(timezone(str(config.ExchangeTimeZone))).replace(tzinfo=None)
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
            value = data[4]
            if value == 0:
                return None
            coin.Time = datetime.strptime(data[0], '%Y-%m-%d')
            coin.EndTime = coin.Time + timedelta(days=1)
            coin.Value = value
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