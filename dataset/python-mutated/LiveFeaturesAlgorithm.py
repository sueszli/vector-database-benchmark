from AlgorithmImports import *
from System.Globalization import *

class LiveTradingFeaturesAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(25000)
        self.AddSecurity(SecurityType.Equity, 'IBM', Resolution.Second)
        self.AddSecurity(SecurityType.Forex, 'EURUSD', Resolution.Minute)
        self.AddData(Bitcoin, 'BTC', Resolution.Second, TimeZones.Utc)
        self.is_connected = True

    def OnData(Bitcoin, data):
        if False:
            while True:
                i = 10
        if self.LiveMode:
            self.SetRuntimeStatistic('BTC', str(data.Close))
        if not self.Portfolio.HoldStock:
            self.MarketOrder('BTC', 100)
            self.Notify.Email('myemail@gmail.com', 'Test', 'Test Body', 'test attachment')
            self.Notify.Sms('+11233456789', str(data.Time) + '>> Test message from live BTC server.')
            self.Notify.Web('http://api.quantconnect.com', str(data.Time) + '>> Test data packet posted from live BTC server.')

    def OnData(self, data):
        if False:
            print('Hello World!')
        if not self.Portfolio['IBM'].HoldStock and data.ContainsKey('IBM'):
            quantity = int(np.floor(self.Portfolio.MarginRemaining / data['IBM'].Close))
            self.MarketOrder('IBM', quantity)
            self.Debug('Purchased IBM on ' + str(self.Time.strftime('%m/%d/%Y')))
            self.Notify.Email('myemail@gmail.com', 'Test', 'Test Body', 'test attachment')

    def OnBrokerageMessage(self, messageEvent):
        if False:
            while True:
                i = 10
        self.Debug(f'Brokerage meesage received - {messageEvent.ToString()}')

    def OnBrokerageDisconnect(self):
        if False:
            return 10
        self.is_connected = False
        self.Debug(f'Brokerage disconnected!')

    def OnBrokerageReconnect(self):
        if False:
            for i in range(10):
                print('nop')
        self.is_connected = True
        self.Debug(f'Brokerage reconnected!')

class Bitcoin(PythonData):

    def GetSource(self, config, date, isLiveMode):
        if False:
            i = 10
            return i + 15
        if isLiveMode:
            return SubscriptionDataSource('https://www.bitstamp.net/api/ticker/', SubscriptionTransportMedium.Rest)
        return SubscriptionDataSource('https://www.quandl.com/api/v3/datasets/BCHARTS/BITSTAMPUSD.csv?order=asc', SubscriptionTransportMedium.RemoteFile)

    def Reader(self, config, line, date, isLiveMode):
        if False:
            for i in range(10):
                print('nop')
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