from AlgorithmImports import *

class FractionalQuantityRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2015, 11, 12)
        self.SetEndDate(2016, 4, 1)
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.GDAX, AccountType.Cash)
        self.SetTimeZone(TimeZones.Utc)
        security = self.AddSecurity(SecurityType.Crypto, 'BTCUSD', Resolution.Daily, Market.GDAX, False, 1, True)
        security.SetBuyingPowerModel(SecurityMarginModel(3.3))
        con = TradeBarConsolidator(1)
        self.SubscriptionManager.AddConsolidator('BTCUSD', con)
        con.DataConsolidated += self.DataConsolidated
        self.SetBenchmark(security.Symbol)

    def DataConsolidated(self, sender, bar):
        if False:
            return 10
        quantity = math.floor((self.Portfolio.Cash + self.Portfolio.TotalFees) / abs(bar.Value + 1))
        btc_qnty = float(self.Portfolio['BTCUSD'].Quantity)
        if not self.Portfolio.Invested:
            self.Order('BTCUSD', quantity)
        elif btc_qnty == quantity:
            self.Order('BTCUSD', 0.1)
        elif btc_qnty == quantity + 0.1:
            self.Order('BTCUSD', 0.01)
        elif btc_qnty == quantity + 0.11:
            self.Order('BTCUSD', -0.02)
        elif btc_qnty == quantity + 0.09:
            self.Order('BTCUSD', 1e-05)
            self.SetHoldings('BTCUSD', -2.0)
            self.SetHoldings('BTCUSD', 2.0)
            self.Quit()