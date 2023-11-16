from AlgorithmImports import *
import itertools
from OptionStrategyFactoryMethodsBaseAlgorithm import *

class LongAndShortButterflyPutStrategiesAlgorithm(OptionStrategyFactoryMethodsBaseAlgorithm):

    def ExpectedOrdersCount(self) -> int:
        if False:
            return 10
        return 6

    def TradeStrategy(self, chain: OptionChain, option_symbol: Symbol):
        if False:
            i = 10
            return i + 15
        putContracts = (contract for contract in chain if contract.Right == OptionRight.Put)
        for (expiry, group) in itertools.groupby(putContracts, lambda x: x.Expiry):
            contracts = list(group)
            if len(contracts) < 3:
                continue
            strikes = sorted([contract.Strike for contract in contracts])
            atmStrike = min(strikes, key=lambda strike: abs(strike - chain.Underlying.Price))
            spread = min(atmStrike - strikes[0], strikes[-1] - atmStrike)
            itmStrike = atmStrike + spread
            otmStrike = atmStrike - spread
            if otmStrike in strikes and itmStrike in strikes:
                self._butterfly_put = OptionStrategies.ButterflyPut(option_symbol, itmStrike, atmStrike, otmStrike, expiry)
                self._short_butterfly_put = OptionStrategies.ShortButterflyPut(option_symbol, itmStrike, atmStrike, otmStrike, expiry)
                self.Buy(self._butterfly_put, 2)
                return

    def AssertStrategyPositionGroup(self, positionGroup: IPositionGroup, option_symbol: Symbol):
        if False:
            i = 10
            return i + 15
        positions = list(positionGroup.Positions)
        if len(positions) != 3:
            raise Exception(f'Expected position group to have 3 positions. Actual: {len(positions)}')
        higherStrike = max((leg.Strike for leg in self._butterfly_put.OptionLegs))
        higherStrikePosition = next((position for position in positions if position.Symbol.ID.OptionRight == OptionRight.Put and position.Symbol.ID.StrikePrice == higherStrike), None)
        if higherStrikePosition.Quantity != 2:
            raise Exception(f'Expected higher strike position quantity to be 2. Actual: {higherStrikePosition.Quantity}')
        lowerStrike = min((leg.Strike for leg in self._butterfly_put.OptionLegs))
        lowerStrikePosition = next((position for position in positions if position.Symbol.ID.OptionRight == OptionRight.Put and position.Symbol.ID.StrikePrice == lowerStrike), None)
        if lowerStrikePosition.Quantity != 2:
            raise Exception(f'Expected lower strike position quantity to be 2. Actual: {lowerStrikePosition.Quantity}')
        middleStrike = [leg.Strike for leg in self._butterfly_put.OptionLegs if leg.Strike < higherStrike and leg.Strike > lowerStrike][0]
        middleStrikePosition = next((position for position in positions if position.Symbol.ID.OptionRight == OptionRight.Put and position.Symbol.ID.StrikePrice == middleStrike), None)
        if middleStrikePosition.Quantity != -4:
            raise Exception(f'Expected middle strike position quantity to be -4. Actual: {middleStrikePosition.Quantity}')

    def LiquidateStrategy(self):
        if False:
            print('Hello World!')
        self.Buy(self._short_butterfly_put, 2)