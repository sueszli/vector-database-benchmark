from AlgorithmImports import *
from QuantConnect.Securities.Positions import IPositionGroup

class OptionStrategyFactoryMethodsBaseAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2015, 12, 24)
        self.SetEndDate(2015, 12, 24)
        self.SetCash(1000000)
        option = self.AddOption('GOOG')
        self._option_symbol = option.Symbol
        option.SetFilter(-2, +2, 0, 180)
        self.SetBenchmark('GOOG')

    def OnData(self, slice):
        if False:
            print('Hello World!')
        if not self.Portfolio.Invested:
            chain = slice.OptionChains.get(self._option_symbol)
            if chain is not None:
                self.TradeStrategy(chain, self._option_symbol)
        else:
            positionGroup = list(self.Portfolio.Positions.Groups)[0]
            buyingPowerModel = positionGroup.BuyingPowerModel
            if not isinstance(buyingPowerModel, OptionStrategyPositionGroupBuyingPowerModel):
                raise Exception(f'Expected position group buying power model type: OptionStrategyPositionGroupBuyingPowerModel. Actual: {type(positionGroup.BuyingPowerModel).__name__}')
            self.AssertStrategyPositionGroup(positionGroup, self._option_symbol)
            self.LiquidateStrategy()
            self.Quit()

    def OnEndOfAlgorithm(self):
        if False:
            return 10
        if self.Portfolio.Invested:
            raise Exception('Expected no holdings at end of algorithm')
        orders_count = len(list(self.Transactions.GetOrders(lambda order: order.Status == OrderStatus.Filled)))
        if orders_count != self.ExpectedOrdersCount():
            raise Exception(f'Expected {self.ExpectedOrdersCount()} orders to have been submitted and filled, half for buying the strategy and the other half for the liquidation. Actual {orders_count}')

    def ExpectedOrdersCount(self) -> int:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('ExpectedOrdersCount method is not implemented')

    def TradeStrategy(self, chain: OptionChain, option_symbol: Symbol) -> None:
        if False:
            while True:
                i = 10
        raise NotImplementedError('TradeStrategy method is not implemented')

    def AssertStrategyPositionGroup(self, positionGroup: IPositionGroup, option_symbol: Symbol) -> None:
        if False:
            print('Hello World!')
        raise NotImplementedError('AssertStrategyPositionGroup method is not implemented')

    def LiquidateStrategy(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('LiquidateStrategy method is not implemented')