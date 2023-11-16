from AlgorithmImports import *
import itertools
from OptionStrategyFactoryMethodsBaseAlgorithm import *

class LongAndShortPutCalendarSpreadStrategiesAlgorithm(OptionStrategyFactoryMethodsBaseAlgorithm):

    def ExpectedOrdersCount(self) -> int:
        if False:
            print('Hello World!')
        return 4

    def TradeStrategy(self, chain: OptionChain, option_symbol: Symbol):
        if False:
            while True:
                i = 10
        putContracts = sorted((contract for contract in chain if contract.Right == OptionRight.Put), key=lambda x: abs(x.Strike - chain.Underlying.Value))
        for (strike, group) in itertools.groupby(putContracts, lambda x: x.Strike):
            contracts = sorted(group, key=lambda x: x.Expiry)
            if len(contracts) < 2:
                continue
            self._near_expiration = contracts[0].Expiry
            self._far_expiration = contracts[1].Expiry
            self._put_calendar_spread = OptionStrategies.PutCalendarSpread(option_symbol, strike, self._near_expiration, self._far_expiration)
            self._short_put_calendar_spread = OptionStrategies.ShortPutCalendarSpread(option_symbol, strike, self._near_expiration, self._far_expiration)
            self.Buy(self._put_calendar_spread, 2)
            return

    def AssertStrategyPositionGroup(self, positionGroup: IPositionGroup, option_symbol: Symbol):
        if False:
            i = 10
            return i + 15
        positions = list(positionGroup.Positions)
        if len(positions) != 2:
            raise Exception(f'Expected position group to have 2 positions. Actual: {len(positions)}')
        nearExpirationPosition = next((position for position in positions if position.Symbol.ID.OptionRight == OptionRight.Put and position.Symbol.ID.Date == self._near_expiration), None)
        if nearExpirationPosition is None or nearExpirationPosition.Quantity != -2:
            raise Exception(f'Expected near expiration position to be -2. Actual: {nearExpirationPosition.Quantity}')
        farExpirationPosition = next((position for position in positions if position.Symbol.ID.OptionRight == OptionRight.Put and position.Symbol.ID.Date == self._far_expiration), None)
        if farExpirationPosition is None or farExpirationPosition.Quantity != 2:
            raise Exception(f'Expected far expiration position to be 2. Actual: {farExpirationPosition.Quantity}')

    def LiquidateStrategy(self):
        if False:
            return 10
        self.Buy(self._short_put_calendar_spread, 2)