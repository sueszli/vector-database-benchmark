from AlgorithmImports import *

class IndexOptionCallCalendarSpreadAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2021, 1, 1)
        self.SetCash(50000)
        self.vxz = self.AddEquity('VXZ', Resolution.Minute).Symbol
        self.spy = self.AddEquity('SPY', Resolution.Minute).Symbol
        index = self.AddIndex('VIX', Resolution.Minute).Symbol
        option = self.AddIndexOption(index, 'VIXW', Resolution.Minute)
        option.SetFilter(lambda x: x.Strikes(-2, 2).Expiration(15, 45))
        self.vixw = option.Symbol
        self.multiplier = option.SymbolProperties.ContractMultiplier
        self.legs = []
        self.expiry = datetime.max

    def OnData(self, slice: Slice) -> None:
        if False:
            print('Hello World!')
        if self.expiry < self.Time + timedelta(2) and all([slice.ContainsKey(x.Symbol) for x in self.legs]):
            self.Liquidate()
        elif [leg for leg in self.legs if self.Portfolio[leg.Symbol].Invested]:
            return
        chain = slice.OptionChains.get(self.vixw)
        if not chain:
            return
        strike = sorted(chain, key=lambda x: abs(x.Strike - chain.Underlying.Value))[0].Strike
        calls = sorted([i for i in chain if i.Strike == strike and i.Right == OptionRight.Call], key=lambda x: x.Expiry)
        if len(calls) < 2:
            return
        self.expiry = calls[0].Expiry
        self.legs = [Leg.Create(calls[0].Symbol, -1), Leg.Create(calls[-1].Symbol, 1), Leg.Create(self.vxz, -100), Leg.Create(self.spy, -10)]
        quantity = self.Portfolio.TotalPortfolioValue // sum([abs(self.Securities[x.Symbol].Price * x.Quantity * (self.multiplier if x.Symbol.ID.SecurityType == SecurityType.IndexOption else 1)) for x in self.legs])
        self.ComboMarketOrder(self.legs, -quantity, asynchronous=True)