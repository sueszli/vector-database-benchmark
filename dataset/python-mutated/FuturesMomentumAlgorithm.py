from AlgorithmImports import *

class FuturesMomentumAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetStartDate(2016, 1, 1)
        self.SetEndDate(2016, 8, 18)
        self.SetCash(100000)
        fastPeriod = 20
        slowPeriod = 60
        self._tolerance = 1 + 0.001
        self.IsUpTrend = False
        self.IsDownTrend = False
        self.SetWarmUp(max(fastPeriod, slowPeriod))
        equity = self.AddEquity('SPY', Resolution.Daily)
        self._fast = self.EMA(equity.Symbol, fastPeriod, Resolution.Daily)
        self._slow = self.EMA(equity.Symbol, slowPeriod, Resolution.Daily)
        future = self.AddFuture(Futures.Indices.SP500EMini)
        future.SetFilter(timedelta(0), timedelta(182))

    def OnData(self, slice):
        if False:
            while True:
                i = 10
        if self._slow.IsReady and self._fast.IsReady:
            self.IsUpTrend = self._fast.Current.Value > self._slow.Current.Value * self._tolerance
            self.IsDownTrend = self._fast.Current.Value < self._slow.Current.Value * self._tolerance
            if not self.Portfolio.Invested and self.IsUpTrend:
                for chain in slice.FuturesChains:
                    contracts = list(filter(lambda x: x.Expiry > self.Time + timedelta(90), chain.Value))
                    if len(contracts) == 0:
                        continue
                    contract = sorted(contracts, key=lambda x: x.Expiry, reverse=True)[0]
                    self.MarketOrder(contract.Symbol, 1)
            if self.Portfolio.Invested and self.IsDownTrend:
                self.Liquidate()

    def OnEndOfDay(self, symbol):
        if False:
            for i in range(10):
                print('nop')
        if self.IsUpTrend:
            self.Plot('Indicator Signal', 'EOD', 1)
        elif self.IsDownTrend:
            self.Plot('Indicator Signal', 'EOD', -1)
        elif self._slow.IsReady and self._fast.IsReady:
            self.Plot('Indicator Signal', 'EOD', 0)

    def OnOrderEvent(self, orderEvent):
        if False:
            while True:
                i = 10
        self.Log(str(orderEvent))