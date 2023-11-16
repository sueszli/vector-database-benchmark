from AlgorithmImports import *

class BybitCryptoRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2022, 12, 13)
        self.SetEndDate(2022, 12, 13)
        self.SetAccountCurrency('USDT')
        self.SetCash(100000)
        self.SetCash('BTC', 1)
        self.SetBrokerageModel(BrokerageName.Bybit, AccountType.Cash)
        self.btcUsdt = self.AddCrypto('BTCUSDT').Symbol
        self.fast = self.EMA(self.btcUsdt, 30, Resolution.Minute)
        self.slow = self.EMA(self.btcUsdt, 60, Resolution.Minute)
        self.liquidated = False

    def OnData(self, data):
        if False:
            print('Hello World!')
        if self.Portfolio.CashBook['USDT'].ConversionRate == 0 or self.Portfolio.CashBook['BTC'].ConversionRate == 0:
            self.Log(f"USDT conversion rate: {self.Portfolio.CashBook['USDT'].ConversionRate}")
            self.Log(f"BTC conversion rate: {self.Portfolio.CashBook['BTC'].ConversionRate}")
            raise Exception('Conversion rate is 0')
        if not self.slow.IsReady:
            return
        btcAmount = self.Portfolio.CashBook['BTC'].Amount
        if self.fast > self.slow:
            if btcAmount == 1 and (not self.liquidated):
                self.Buy(self.btcUsdt, 1)
        elif btcAmount > 1:
            self.Liquidate(self.btcUsdt)
            self.liquidated = True
        elif btcAmount > 0 and self.liquidated and (len(self.Transactions.GetOpenOrders()) == 0):
            limitPrice = round(self.Securities[self.btcUsdt].Price * 1.01, 2)
            self.LimitOrder(self.btcUsdt, -btcAmount, limitPrice)

    def OnOrderEvent(self, orderEvent):
        if False:
            return 10
        self.Debug('{} {}'.format(self.Time, orderEvent.ToString()))

    def OnEndOfAlgorithm(self):
        if False:
            return 10
        self.Log(f'{self.Time} - TotalPortfolioValue: {self.Portfolio.TotalPortfolioValue}')
        self.Log(f'{self.Time} - CashBook: {self.Portfolio.CashBook}')
        btcAmount = self.Portfolio.CashBook['BTC'].Amount
        if btcAmount > 0:
            raise Exception(f'BTC holdings should be zero at the end of the algorithm, but was {btcAmount}')