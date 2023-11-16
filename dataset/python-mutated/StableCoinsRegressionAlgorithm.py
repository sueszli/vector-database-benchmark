from AlgorithmImports import *

class StableCoinsRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2018, 5, 1)
        self.SetEndDate(2018, 5, 2)
        self.SetCash('USDT', 200000000)
        self.SetBrokerageModel(BrokerageName.Binance, AccountType.Cash)
        self.AddCrypto('BTCUSDT', Resolution.Hour, Market.Binance)

    def OnData(self, data):
        if False:
            return 10
        if not self.Portfolio.Invested:
            self.SetHoldings('BTCUSDT', 1)