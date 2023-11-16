from AlgorithmImports import *

class IndexOptionBullCallSpreadAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2021, 1, 1)
        self.SetCash(100000)
        self.spy = self.AddEquity('SPY', Resolution.Minute).Symbol
        index = self.AddIndex('SPX', Resolution.Minute).Symbol
        option = self.AddIndexOption(index, 'SPXW', Resolution.Minute)
        option.SetFilter(lambda x: x.WeeklysOnly().Strikes(-5, 5).Expiration(40, 60))
        self.spxw = option.Symbol
        self.tickets = []

    def OnData(self, slice: Slice) -> None:
        if False:
            while True:
                i = 10
        if not self.Portfolio[self.spy].Invested:
            self.MarketOrder(self.spy, 100)
        if any([self.Portfolio[x.Symbol].Invested for x in self.tickets]):
            return
        chain = slice.OptionChains.get(self.spxw)
        if not chain:
            return
        expiry = min([x.Expiry for x in chain])
        calls = sorted([i for i in chain if i.Expiry == expiry and i.Right == OptionRight.Call], key=lambda x: x.Strike)
        if len(calls) < 2:
            return
        bull_call_spread = OptionStrategies.BullCallSpread(self.spxw, calls[0].Strike, calls[-1].Strike, expiry)
        self.tickets = self.Buy(bull_call_spread, 1)