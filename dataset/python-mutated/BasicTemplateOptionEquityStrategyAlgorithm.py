from AlgorithmImports import *

class BasicTemplateOptionEquityStrategyAlgorithm(QCAlgorithm):
    UnderlyingTicker = 'GOOG'

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2015, 12, 24)
        self.SetEndDate(2015, 12, 24)
        equity = self.AddEquity(self.UnderlyingTicker)
        option = self.AddOption(self.UnderlyingTicker)
        self.option_symbol = option.Symbol
        option.SetFilter(lambda u: u.Strikes(-2, +2).Expiration(0, 180))

    def OnData(self, slice):
        if False:
            return 10
        if self.Portfolio.Invested or not self.IsMarketOpen(self.option_symbol):
            return
        chain = slice.OptionChains.GetValue(self.option_symbol)
        if chain is None:
            return
        groupedByExpiry = dict()
        for contract in [contract for contract in chain if contract.Right == OptionRight.Call]:
            groupedByExpiry.setdefault(int(contract.Expiry.timestamp()), []).append(contract)
        firstExpiry = list(sorted(groupedByExpiry))[0]
        callContracts = sorted(groupedByExpiry[firstExpiry], key=lambda x: x.Strike)
        expiry = callContracts[0].Expiry
        lowerStrike = callContracts[0].Strike
        middleStrike = callContracts[1].Strike
        higherStrike = callContracts[2].Strike
        optionStrategy = OptionStrategies.CallButterfly(self.option_symbol, higherStrike, middleStrike, lowerStrike, expiry)
        self.Order(optionStrategy, 10)

    def OnOrderEvent(self, orderEvent):
        if False:
            while True:
                i = 10
        self.Log(str(orderEvent))