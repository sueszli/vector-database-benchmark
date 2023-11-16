from AlgorithmImports import *

class BybitCustomDataCryptoRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2022, 12, 13)
        self.SetEndDate(2022, 12, 13)
        self.SetAccountCurrency('USDT')
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.Bybit, AccountType.Cash)
        symbol = self.AddCrypto('BTCUSDT').Symbol
        self.btcUsdt = self.AddData(CustomCryptoData, symbol, Resolution.Minute).Symbol
        self.fast = self.EMA(self.btcUsdt, 30, Resolution.Minute)
        self.slow = self.EMA(self.btcUsdt, 60, Resolution.Minute)

    def OnData(self, data):
        if False:
            return 10
        if not self.slow.IsReady:
            return
        if self.fast.Current.Value > self.slow.Current.Value:
            if self.Transactions.OrdersCount == 0:
                self.Buy(self.btcUsdt, 1)
        elif self.Transactions.OrdersCount == 1:
            self.Liquidate(self.btcUsdt)

    def OnOrderEvent(self, orderEvent):
        if False:
            i = 10
            return i + 15
        self.Debug(f'{self.Time} {orderEvent}')

class CustomCryptoData(PythonData):

    def GetSource(self, config, date, isLiveMode):
        if False:
            while True:
                i = 10
        tickTypeString = Extensions.TickTypeToLower(config.TickType)
        formattedDate = date.strftime('%Y%m%d')
        source = os.path.join(Globals.DataFolder, 'crypto', 'bybit', 'minute', config.Symbol.Value.lower(), f'{formattedDate}_{tickTypeString}.zip')
        return SubscriptionDataSource(source, SubscriptionTransportMedium.LocalFile, FileFormat.Csv)

    def Reader(self, config, line, date, isLiveMode):
        if False:
            for i in range(10):
                print('nop')
        csv = line.split(',')
        data = CustomCryptoData()
        data.Symbol = config.Symbol
        data_datetime = datetime.combine(date.date(), time()) + timedelta(milliseconds=int(csv[0]))
        data.Time = Extensions.ConvertTo(data_datetime, config.DataTimeZone, config.ExchangeTimeZone)
        data.EndTime = data.Time + timedelta(minutes=1)
        data['Open'] = float(csv[1])
        data['High'] = float(csv[2])
        data['Low'] = float(csv[3])
        data['Close'] = float(csv[4])
        data['Volume'] = float(csv[5])
        data.Value = float(csv[4])
        return data