from AlgorithmImports import *

class IndexOptionBearCallSpreadAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2021, 1, 1)
        self.SetCash(100000)
        self.spy = self.AddEquity('SPY', Resolution.Minute).Symbol
        index = self.AddIndex('VIX', Resolution.Minute).Symbol
        option = self.AddIndexOption(index, 'VIXW', Resolution.Minute)
        option.SetFilter(lambda x: x.Strikes(-5, 5).Expiration(15, 45))
        self.vixw = option.Symbol
        self.tickets = []

    def OnData(self, slice: Slice) -> None:
        if False:
            while True:
                i = 10
        if not self.Portfolio[self.spy].Invested:
            self.MarketOrder(self.spy, 100)
        if any([self.Portfolio[x.Symbol].Invested for x in self.tickets]):
            return
        chain = slice.OptionChains.get(self.vixw)
        if not chain:
            return
        expiry = min([x.Expiry for x in chain])
        calls = sorted([i for i in chain if i.Expiry == expiry and i.Right == OptionRight.Call], key=lambda x: x.Strike)
        if len(calls) < 2:
            return
        bear_call_spread = OptionStrategies.BearCallSpread(self.vixw, calls[0].Strike, calls[-1].Strike, expiry)
        self.tickets = self.Buy(bear_call_spread, 1)