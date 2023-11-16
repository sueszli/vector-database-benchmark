from AlgorithmImports import *

class BubbleAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetCash(100000)
        self.SetStartDate(1998, 1, 1)
        self.SetEndDate(2014, 6, 1)
        self._symbols = []
        (self._macdDic, self._rsiDic) = ({}, {})
        (self._newLow, self._currCape) = (None, None)
        (self._counter, self._counter2) = (0, 0)
        (self._c, self._cCopy) = (np.empty([4]), np.empty([4]))
        self._symbols.append('SPY')
        self.AddData(Cape, 'CAPE')
        for stock in self._symbols:
            self.AddSecurity(SecurityType.Equity, stock, Resolution.Minute)
            self._macd = self.MACD(stock, 12, 26, 9, MovingAverageType.Exponential, Resolution.Daily)
            self._macdDic[stock] = self._macd
            self._rsi = self.RSI(stock, 14, MovingAverageType.Exponential, Resolution.Daily)
            self._rsiDic[stock] = self._rsi

    def OnData(self, data):
        if False:
            i = 10
            return i + 15
        if self._currCape and self._newLow is not None:
            try:
                if self._currCape > 20 and self._newLow == False:
                    for stock in self._symbols:
                        if self.Securities[stock].Holdings.Quantity == 0 and self._rsiDic[stock].Current.Value < 70 and (self.Securities[stock].Price != 0) and (self.Portfolio.Cash > self.Securities[stock].Price * 100) and (self.Time.hour == 9) and (self.Time.minute == 31):
                            self.BuyStock(stock)
                        if self._rsiDic[stock].Current.Value > 70 and self.Securities[stock].Holdings.Quantity > 0 and (self.Time.hour == 9) and (self.Time.minute == 31):
                            self.SellStock(stock)
                elif self._newLow:
                    for stock in self._symbols:
                        if self.Securities[stock].Holdings.Quantity > 0 and self._rsiDic[stock].Current.Value > 30 and (self.Time.hour == 9) and (self.Time.minute == 31):
                            self.SellStock(stock)
                        elif self.Securities[stock].Holdings.Quantity == 0 and self._rsiDic[stock].Current.Value < 30 and (Securities[stock].Price != 0) and (self.Portfolio.Cash > self.Securities[stock].Price * 100) and (self.Time.hour == 9) and (self.Time.minute == 31):
                            self.BuyStock(stock)
                elif self._currCape == 0:
                    self.Debug('Exiting due to no CAPE!')
                    self.Quit('CAPE ratio not supplied in data, exiting.')
            except:
                return None
        if not data.ContainsKey('CAPE'):
            return
        self._newLow = False
        self._currCape = data['CAPE'].Cape
        if self._counter < 4:
            self._c[self._counter] = self._currCape
            self._counter += 1
        else:
            self._cCopy = self._c
            self._cCopy = np.sort(self._cCopy)
            if self._cCopy[0] > self._currCape:
                self._newLow = True
            self._c[self._counter2] = self._currCape
            self._counter2 += 1
            if self._counter2 == 4:
                self._counter2 = 0
        self.Debug('Current Cape: ' + str(self._currCape) + ' on ' + str(self.Time))
        if self._newLow:
            self.Debug('New Low has been hit on ' + str(self.Time))

    def BuyStock(self, symbol):
        if False:
            return 10
        s = self.Securities[symbol].Holdings
        if self._macdDic[symbol].Current.Value > 0:
            self.SetHoldings(symbol, 1)
            self.Debug('Purchasing: ' + str(symbol) + '   MACD: ' + str(self._macdDic[symbol]) + '   RSI: ' + str(self._rsiDic[symbol]) + '   Price: ' + str(round(self.Securities[symbol].Price, 2)) + '   Quantity: ' + str(s.Quantity))

    def SellStock(self, symbol):
        if False:
            i = 10
            return i + 15
        s = self.Securities[symbol].Holdings
        if s.Quantity > 0 and self._macdDic[symbol].Current.Value < 0:
            self.Liquidate(symbol)
            self.Debug('Selling: ' + str(symbol) + ' at sell MACD: ' + str(self._macdDic[symbol]) + '   RSI: ' + str(self._rsiDic[symbol]) + '   Price: ' + str(round(self.Securities[symbol].Price, 2)) + '   Profit from sale: ' + str(s.LastTradeProfit))

class Cape(PythonData):

    def GetSource(self, config, date, isLiveMode):
        if False:
            for i in range(10):
                print('nop')
        return SubscriptionDataSource('https://www.dropbox.com/s/ggt6blmib54q36e/CAPE.csv?dl=1', SubscriptionTransportMedium.RemoteFile)
    ' Reader Method : using set of arguments we specify read out type. Enumerate until \n        the end of the data stream or file. E.g. Read CSV file line by line and convert into data types. '

    def Reader(self, config, line, date, isLiveMode):
        if False:
            for i in range(10):
                print('nop')
        if not (line.strip() and line[0].isdigit()):
            return None
        index = Cape()
        index.Symbol = config.Symbol
        try:
            data = line.split(',')
            index.Time = datetime.strptime(data[0], '%Y-%m')
            index['Cape'] = float(data[10])
            index.Value = data[10]
        except ValueError:
            return None
        return index