from AlgorithmImports import *
from OptionStrategyFactoryMethodsBaseAlgorithm import *

class CoveredAndProtectiveCallStrategiesAlgorithm(OptionStrategyFactoryMethodsBaseAlgorithm):

    def ExpectedOrdersCount(self) -> int:
        if False:
            i = 10
            return i + 15
        return 4

    def TradeStrategy(self, chain: OptionChain, option_symbol: Symbol):
        if False:
            i = 10
            return i + 15
        contracts = sorted(sorted(chain, key=lambda x: abs(chain.Underlying.Price - x.Strike)), key=lambda x: x.Expiry, reverse=True)
        if len(contracts) == 0:
            return
        contract = contracts[0]
        if contract != None:
            self._covered_call = OptionStrategies.CoveredCall(option_symbol, contract.Strike, contract.Expiry)
            self._protective_call = OptionStrategies.ProtectiveCall(option_symbol, contract.Strike, contract.Expiry)
            self.Buy(self._covered_call, 2)

    def AssertStrategyPositionGroup(self, positionGroup: IPositionGroup, option_symbol: Symbol):
        if False:
            i = 10
            return i + 15
        positions = list(positionGroup.Positions)
        if len(positions) != 2:
            raise Exception(f'Expected position group to have 2 positions. Actual: {len(positions)}')
        optionPosition = [position for position in positions if position.Symbol.SecurityType == SecurityType.Option][0]
        if optionPosition.Symbol.ID.OptionRight != OptionRight.Call:
            raise Exception(f'Expected option position to be a call. Actual: {optionPosition.Symbol.ID.OptionRight}')
        underlyingPosition = [position for position in positions if position.Symbol.SecurityType == SecurityType.Equity][0]
        expectedOptionPositionQuantity = -2
        expectedUnderlyingPositionQuantity = 2 * self.Securities[option_symbol].SymbolProperties.ContractMultiplier
        if optionPosition.Quantity != expectedOptionPositionQuantity:
            raise Exception(f'Expected option position quantity to be {expectedOptionPositionQuantity}. Actual: {optionPosition.Quantity}')
        if underlyingPosition.Quantity != expectedUnderlyingPositionQuantity:
            raise Exception(f'Expected underlying position quantity to be {expectedUnderlyingPositionQuantity}. Actual: {underlyingPosition.Quantity}')

    def LiquidateStrategy(self):
        if False:
            for i in range(10):
                print('nop')
        self.Buy(self._protective_call, 2)