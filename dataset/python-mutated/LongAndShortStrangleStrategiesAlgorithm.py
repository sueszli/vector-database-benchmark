import itertools
from AlgorithmImports import *
from OptionStrategyFactoryMethodsBaseAlgorithm import *

class LongAndShortStrangleStrategiesAlgorithm(OptionStrategyFactoryMethodsBaseAlgorithm):

    def ExpectedOrdersCount(self) -> int:
        if False:
            i = 10
            return i + 15
        return 4

    def TradeStrategy(self, chain: OptionChain, option_symbol: Symbol):
        if False:
            for i in range(10):
                print('nop')
        contracts = sorted(sorted(chain, key=lambda x: abs(chain.Underlying.Price - x.Strike)), key=lambda x: x.Expiry, reverse=True)
        groupedContracts = (list(group) for (_, group) in itertools.groupby(contracts, lambda x: x.Expiry))
        callContract = None
        putContract = None
        for group in groupedContracts:
            callContracts = sorted((contract for contract in group if contract.Right == OptionRight.Call), key=lambda x: x.Strike, reverse=True)
            putContracts = sorted((contract for contract in group if contract.Right == OptionRight.Put), key=lambda x: x.Strike)
            if len(callContracts) > 0 and len(putContracts) > 0 and (callContracts[0].Strike > putContracts[0].Strike):
                callContract = callContracts[0]
                putContract = putContracts[0]
                break
        if callContract is not None and putContract is not None:
            self._strangle = OptionStrategies.Strangle(option_symbol, callContract.Strike, putContract.Strike, callContract.Expiry)
            self._short_strangle = OptionStrategies.ShortStrangle(option_symbol, callContract.Strike, putContract.Strike, callContract.Expiry)
            self.Buy(self._strangle, 2)

    def AssertStrategyPositionGroup(self, positionGroup: IPositionGroup, option_symbol: Symbol):
        if False:
            return 10
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
            print('Hello World!')
        self.Buy(self._short_strangle, 2)