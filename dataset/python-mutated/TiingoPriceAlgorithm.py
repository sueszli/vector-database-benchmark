from AlgorithmImports import *
from QuantConnect.Data.Custom.Tiingo import *

class TiingoPriceAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2017, 1, 1)
        self.SetEndDate(2017, 12, 31)
        self.SetCash(100000)
        Tiingo.SetAuthCode('my-tiingo-api-token')
        self.ticker = 'AAPL'
        self.symbol = self.AddData(TiingoPrice, self.ticker, Resolution.Daily).Symbol
        self.emaFast = self.EMA(self.symbol, 5)
        self.emaSlow = self.EMA(self.symbol, 10)

    def OnData(self, slice):
        if False:
            while True:
                i = 10
        if not slice.ContainsKey(self.ticker):
            return
        row = slice[self.ticker]
        self.Log(f'{self.Time} - {row.Symbol.Value} - {row.Close} {row.Value} {row.Price} - EmaFast:{self.emaFast} - EmaSlow:{self.emaSlow}')
        if not self.Portfolio.Invested and self.emaFast > self.emaSlow:
            self.SetHoldings(self.symbol, 1)
        elif self.Portfolio.Invested and self.emaFast < self.emaSlow:
            self.Liquidate(self.symbol)