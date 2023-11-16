from AlgorithmImports import *

class IndexOptionCallButterflyAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2021, 1, 1)
        self.SetCash(1000000)
        self.vxz = self.AddEquity('VXZ', Resolution.Minute).Symbol
        index = self.AddIndex('SPX', Resolution.Minute).Symbol
        option = self.AddIndexOption(index, 'SPXW', Resolution.Minute)
        option.SetFilter(lambda x: x.IncludeWeeklys().Strikes(-3, 3).Expiration(15, 45))
        self.spxw = option.Symbol
        self.multiplier = option.SymbolProperties.ContractMultiplier
        self.tickets = []

    def OnData(self, slice: Slice) -> None:
        if False:
            while True:
                i = 10
        if not self.Portfolio[self.vxz].Invested:
            self.MarketOrder(self.vxz, 10000)
        if any([self.Portfolio[x.Symbol].Invested for x in self.tickets]):
            return
        chain = slice.OptionChains.get(self.spxw)
        if not chain:
            return
        expiry = min([x.Expiry for x in chain])
        calls = [x for x in chain if x.Expiry == expiry and x.Right == OptionRight.Call]
        if len(calls) < 3:
            return
        sorted_call_strikes = sorted([x.Strike for x in calls])
        atm_strike = min([abs(x - chain.Underlying.Value) for x in sorted_call_strikes])
        spread = min(atm_strike - sorted_call_strikes[0], sorted_call_strikes[-1] - atm_strike)
        itm_strike = atm_strike - spread
        otm_strike = atm_strike + spread
        if otm_strike not in sorted_call_strikes or itm_strike not in sorted_call_strikes:
            return
        call_butterfly = OptionStrategies.CallButterfly(self.spxw, otm_strike, atm_strike, itm_strike, expiry)
        price = sum([abs(self.Securities[x.Symbol].Price * x.Quantity) * self.multiplier for x in call_butterfly.UnderlyingLegs])
        if price > 0:
            quantity = self.Portfolio.TotalPortfolioValue // price
            self.tickets = self.Buy(call_butterfly, quantity, asynchronous=True)