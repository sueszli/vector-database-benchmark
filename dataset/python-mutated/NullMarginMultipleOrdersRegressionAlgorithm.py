from AlgorithmImports import *

class NullMarginMultipleOrdersRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2015, 12, 24)
        self.SetEndDate(2015, 12, 24)
        self.SetCash(10000)
        self.Portfolio.SetPositions(SecurityPositionGroupModel.Null)
        self.SetSecurityInitializer(lambda security: security.SetBuyingPowerModel(ConstantBuyingPowerModel(1)))
        equity = self.AddEquity('GOOG', leverage=4, fillForward=True)
        option = self.AddOption(equity.Symbol, fillForward=True)
        self._optionSymbol = option.Symbol
        option.SetFilter(lambda u: u.Strikes(-2, +2).Expiration(0, 180))

    def OnData(self, data: Slice):
        if False:
            while True:
                i = 10
        if not self.Portfolio.Invested:
            if self.IsMarketOpen(self._optionSymbol):
                chain = data.OptionChains.GetValue(self._optionSymbol)
                if chain is not None:
                    callContracts = [contract for contract in chain if contract.Right == OptionRight.Call]
                    callContracts.sort(key=lambda x: (x.Expiry, 1 / x.Strike), reverse=True)
                    optionContract = callContracts[0]
                    self.MarketOrder(optionContract.Symbol.Underlying, 1000)
                    self.MarketOrder(optionContract.Symbol, -10)
                    if self.Portfolio.TotalMarginUsed != 1010:
                        raise ValueError(f'Unexpected margin used {self.Portfolio.TotalMarginUsed}')