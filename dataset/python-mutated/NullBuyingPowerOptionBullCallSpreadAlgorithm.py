from AlgorithmImports import *

class NullBuyingPowerOptionBullCallSpreadAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2015, 12, 24)
        self.SetEndDate(2015, 12, 24)
        self.SetCash(200000)
        self.SetSecurityInitializer(lambda security: security.SetMarginModel(SecurityMarginModel.Null))
        self.Portfolio.SetPositions(SecurityPositionGroupModel.Null)
        equity = self.AddEquity('GOOG')
        option = self.AddOption(equity.Symbol)
        self.optionSymbol = option.Symbol
        option.SetFilter(-2, 2, 0, 180)

    def OnData(self, slice):
        if False:
            while True:
                i = 10
        if self.Portfolio.Invested or not self.IsMarketOpen(self.optionSymbol):
            return
        chain = slice.OptionChains.get(self.optionSymbol)
        if chain:
            call_contracts = [x for x in chain if x.Right == OptionRight.Call]
            expiry = min((x.Expiry for x in call_contracts))
            call_contracts = sorted([x for x in call_contracts if x.Expiry == expiry], key=lambda x: x.Strike)
            long_call = call_contracts[0]
            short_call = [x for x in call_contracts if x.Strike > long_call.Strike][0]
            quantity = 1000
            tickets = [self.MarketOrder(short_call.Symbol, -quantity), self.MarketOrder(long_call.Symbol, quantity)]
            for ticket in tickets:
                if ticket.Status != OrderStatus.Filled:
                    raise Exception(f'There should be no restriction on buying {ticket.Quantity} of {ticket.Symbol} with BuyingPowerModel.Null')

    def OnEndOfAlgorithm(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.Portfolio.TotalMarginUsed != 0:
            raise Exception('The TotalMarginUsed should be zero to avoid margin calls.')