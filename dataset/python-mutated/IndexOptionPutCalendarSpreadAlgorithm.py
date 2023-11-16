from AlgorithmImports import *

class IndexOptionPutCalendarSpreadAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 1, 1)
        self.SetCash(50000)
        self.vxz = self.AddEquity('VXZ', Resolution.Minute).Symbol
        index = self.AddIndex('VIX', Resolution.Minute).Symbol
        option = self.AddIndexOption(index, 'VIXW', Resolution.Minute)
        option.SetFilter(lambda x: x.Strikes(-2, 2).Expiration(15, 45))
        self.vixw = option.Symbol
        self.tickets = []
        self.expiry = datetime.max

    def OnData(self, slice: Slice) -> None:
        if False:
            print('Hello World!')
        if not self.Portfolio[self.vxz].Invested:
            self.MarketOrder(self.vxz, 100)
        index_options_invested = [leg for leg in self.tickets if self.Portfolio[leg.Symbol].Invested]
        if self.expiry < self.Time + timedelta(2) and all([slice.ContainsKey(x.Symbol) for x in self.tickets]):
            for holding in index_options_invested:
                self.Liquidate(holding.Symbol)
        elif index_options_invested:
            return
        chain = slice.OptionChains.get(self.vixw)
        if not chain:
            return
        strike = sorted(chain, key=lambda x: abs(x.Strike - chain.Underlying.Value))[0].Strike
        puts = sorted([i for i in chain if i.Strike == strike and i.Right == OptionRight.Put], key=lambda x: x.Expiry)
        if len(puts) < 2:
            return
        self.expiry = puts[0].Expiry
        put_calendar_spread = OptionStrategies.PutCalendarSpread(self.vixw, strike, self.expiry, puts[-1].Expiry)
        self.tickets = self.Sell(put_calendar_spread, 1, asynchronous=True)