import itertools
from AlgorithmImports import *
from OptionStrategyFactoryMethodsBaseAlgorithm import *

class LongAndShortStraddleStrategiesAlgorithm(OptionStrategyFactoryMethodsBaseAlgorithm):

    def ExpectedOrdersCount(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return 4

    def TradeStrategy(self, chain: OptionChain, option_symbol: Symbol):
        if False:
            for i in range(10):
                print('nop')
        contracts = sorted(sorted(chain, key=lambda x: abs(chain.Underlying.Price - x.Strike)), key=lambda x: x.Expiry, reverse=True)
        groupedContracts = [list(group) for (_, group) in itertools.groupby(contracts, lambda x: (x.Strike, x.Expiry))]
        groupedContracts = (group for group in groupedContracts if any((contract.Right == OptionRight.Call for contract in group)) and any((contract.Right == OptionRight.Put for contract in group)))
        contracts = next(groupedContracts, [])
        if len(contracts) == 0:
            return
        contract = contracts[0]
        if contract is not None:
            self._straddle = OptionStrategies.Straddle(option_symbol, contract.Strike, contract.Expiry)
            self._short_straddle = OptionStrategies.ShortStraddle(option_symbol, contract.Strike, contract.Expiry)
            self.Buy(self._straddle, 2)

    def AssertStrategyPositionGroup(self, positionGroup: IPositionGroup, option_symbol: Symbol):
        if False:
            while True:
                i = 10
        positions = list(positionGroup.Positions)
        if len(positions) != 2:
            raise Exception(f'Expected position group to have 2 positions. Actual: {len(positions)}')
        callPosition = next((position for position in positions if position.Symbol.ID.OptionRight == OptionRight.Call), None)
        if callPosition is None:
            raise Exception('Expected position group to have a call position')
        putPosition = next((position for position in positions if position.Symbol.ID.OptionRight == OptionRight.Put), None)
        if putPosition is None:
            raise Exception('Expected position group to have a put position')
        expectedCallPositionQuantity = 2
        expectedPutPositionQuantity = 2
        if callPosition.Quantity != expectedCallPositionQuantity:
            raise Exception(f'Expected call position quantity to be {expectedCallPositionQuantity}. Actual: {callPosition.Quantity}')
        if putPosition.Quantity != expectedPutPositionQuantity:
            raise Exception(f'Expected put position quantity to be {expectedPutPositionQuantity}. Actual: {putPosition.Quantity}')

    def LiquidateStrategy(self):
        if False:
            i = 10
            return i + 15
        self.Buy(self._short_straddle, 2)