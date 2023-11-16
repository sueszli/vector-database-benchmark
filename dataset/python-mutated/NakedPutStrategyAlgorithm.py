from AlgorithmImports import *
from OptionStrategyFactoryMethodsBaseAlgorithm import *

class NakedPutStrategyAlgorithm(OptionStrategyFactoryMethodsBaseAlgorithm):

    def ExpectedOrdersCount(self) -> int:
        if False:
            return 10
        return 2

    def TradeStrategy(self, chain: OptionChain, option_symbol: Symbol):
        if False:
            for i in range(10):
                print('nop')
        contracts = sorted(sorted(chain, key=lambda x: abs(chain.Underlying.Price - x.Strike)), key=lambda x: x.Expiry, reverse=True)
        if len(contracts) == 0:
            return
        contract = contracts[0]
        if contract != None:
            self._naked_put = OptionStrategies.NakedPut(option_symbol, contract.Strike, contract.Expiry)
            self.Buy(self._naked_put, 2)

    def AssertStrategyPositionGroup(self, positionGroup: IPositionGroup, option_symbol: Symbol):
        if False:
            while True:
                i = 10
        positions = list(positionGroup.Positions)
        if len(positions) != 1:
            raise Exception(f'Expected position group to have 1 positions. Actual: {len(positions)}')
        optionPosition = [position for position in positions if position.Symbol.SecurityType == SecurityType.Option][0]
        if optionPosition.Symbol.ID.OptionRight != OptionRight.Put:
            raise Exception(f'Expected option position to be a put. Actual: {optionPosition.Symbol.ID.OptionRight}')
        expectedOptionPositionQuantity = -2
        if optionPosition.Quantity != expectedOptionPositionQuantity:
            raise Exception(f'Expected option position quantity to be {expectedOptionPositionQuantity}. Actual: {optionPosition.Quantity}')

    def LiquidateStrategy(self):
        if False:
            while True:
                i = 10
        self.Sell(self._naked_put, 2)