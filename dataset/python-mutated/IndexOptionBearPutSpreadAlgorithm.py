from AlgorithmImports import *

class IndexOptionBearPutSpreadAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2022, 1, 1)
        self.SetEndDate(2022, 7, 1)
        self.SetCash(100000)
        index = self.AddIndex('SPX', Resolution.Minute).Symbol
        option = self.AddIndexOption(index, 'SPXW', Resolution.Minute)
        option.SetFilter(lambda x: x.WeeklysOnly().Strikes(5, 10).Expiration(0, 0))
        self.spxw = option.Symbol
        self.tickets = []

    def OnData(self, slice: Slice) -> None:
        if False:
            return 10
        if any([self.Portfolio[x.Symbol].Invested for x in self.tickets]):
            return
        chain = slice.OptionChains.get(self.spxw)
        if not chain:
            return
        expiry = min([x.Expiry for x in chain])
        puts = sorted([i for i in chain if i.Expiry == expiry and i.Right == OptionRight.Put], key=lambda x: x.Strike)
        if len(puts) < 2:
            return
        bear_put_spread = OptionStrategies.BearPutSpread(self.spxw, puts[-1].Strike, puts[0].Strike, expiry)
        self.tickets = self.Buy(bear_put_spread, 1)