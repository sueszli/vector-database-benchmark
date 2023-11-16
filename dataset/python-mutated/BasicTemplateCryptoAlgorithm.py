from AlgorithmImports import *

class BasicTemplateCryptoAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2018, 4, 4)
        self.SetEndDate(2018, 4, 4)
        self.SetCash(10000)
        self.SetCash('EUR', 10000)
        self.SetCash('BTC', 1)
        self.SetCash('ETH', 5)
        self.SetBrokerageModel(BrokerageName.GDAX, AccountType.Cash)
        self.AddCrypto('BTCUSD', Resolution.Minute)
        self.AddCrypto('ETHUSD', Resolution.Minute)
        self.AddCrypto('BTCEUR', Resolution.Minute)
        symbol = self.AddCrypto('LTCUSD', Resolution.Minute).Symbol
        self.fast = self.EMA(symbol, 30, Resolution.Minute)
        self.slow = self.EMA(symbol, 60, Resolution.Minute)

    def OnData(self, data):
        if False:
            i = 10
            return i + 15
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n\n        Arguments:\n            data: Slice object keyed by symbol containing the stock data\n        '
        if self.Time.hour == 1 and self.Time.minute == 0:
            limitPrice = round(self.Securities['ETHUSD'].Price * 1.01, 2)
            quantity = self.Portfolio.CashBook['ETH'].Amount
            self.LimitOrder('ETHUSD', -quantity, limitPrice)
        elif self.Time.hour == 2 and self.Time.minute == 0:
            usdTotal = self.Portfolio.CashBook['USD'].Amount
            limitPrice = round(self.Securities['BTCUSD'].Price * 0.95, 2)
            quantity = usdTotal * 0.5 / limitPrice
            self.LimitOrder('BTCUSD', quantity, limitPrice)
        elif self.Time.hour == 2 and self.Time.minute == 1:
            usdTotal = self.Portfolio.CashBook['USD'].Amount
            usdReserved = sum((x.Quantity * x.LimitPrice for x in [x for x in self.Transactions.GetOpenOrders() if x.Direction == OrderDirection.Buy and x.Type == OrderType.Limit and (x.Symbol.Value == 'BTCUSD' or x.Symbol.Value == 'ETHUSD')]))
            usdAvailable = usdTotal - usdReserved
            self.Debug('usdAvailable: {}'.format(usdAvailable))
            limitPrice = round(self.Securities['ETHUSD'].Price * 1.01, 2)
            quantity = usdAvailable / limitPrice
            self.LimitOrder('ETHUSD', quantity, limitPrice)
            quantity = usdAvailable * 0.5 / limitPrice
            self.LimitOrder('ETHUSD', quantity, limitPrice)
        elif self.Time.hour == 11 and self.Time.minute == 0:
            self.SetHoldings('BTCUSD', 0)
        elif self.Time.hour == 12 and self.Time.minute == 0:
            self.Buy('BTCEUR', 1)
            limitPrice = round(self.Securities['BTCEUR'].Price * 1.1, 2)
            self.LimitOrder('BTCEUR', -1, limitPrice)
        elif self.Time.hour == 13 and self.Time.minute == 0:
            self.Transactions.CancelOpenOrders('BTCEUR')
        elif self.Time.hour > 13:
            if self.fast > self.slow:
                if self.Portfolio.CashBook['LTC'].Amount == 0:
                    self.Buy('LTCUSD', 10)
            elif self.Portfolio.CashBook['LTC'].Amount > 0:
                self.Liquidate('LTCUSD')

    def OnOrderEvent(self, orderEvent):
        if False:
            while True:
                i = 10
        self.Debug('{} {}'.format(self.Time, orderEvent.ToString()))

    def OnEndOfAlgorithm(self):
        if False:
            return 10
        self.Log('{} - TotalPortfolioValue: {}'.format(self.Time, self.Portfolio.TotalPortfolioValue))
        self.Log('{} - CashBook: {}'.format(self.Time, self.Portfolio.CashBook))