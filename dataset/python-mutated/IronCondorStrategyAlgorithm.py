from AlgorithmImports import *
import itertools
from OptionStrategyFactoryMethodsBaseAlgorithm import *

class IronCondorStrategyAlgorithm(OptionStrategyFactoryMethodsBaseAlgorithm):

    def ExpectedOrdersCount(self) -> int:
        if False:
            i = 10
            return i + 15
        return 8

    def TradeStrategy(self, chain: OptionChain, option_symbol: Symbol):
        if False:
            while True:
                i = 10
        for (expiry, group) in itertools.groupby(chain, lambda x: x.Expiry):
            contracts = sorted(group, key=lambda x: x.Strike)
            if len(contracts) < 4:
                continue
            putContracts = [x for x in contracts if x.Right == OptionRight.Put]
            if len(putContracts) < 2:
                continue
            longPutStrike = putContracts[0].Strike
            shortPutStrike = putContracts[1].Strike
            callContracts = [x for x in contracts if x.Right == OptionRight.Call and x.Strike > shortPutStrike]
            if len(callContracts) < 2:
                continue
            shortCallStrike = callContracts[0].Strike
            longCallStrike = callContracts[1].Strike
            self._iron_condor = OptionStrategies.IronCondor(option_symbol, longPutStrike, shortPutStrike, shortCallStrike, longCallStrike, expiry)
            self.Buy(self._iron_condor, 2)
            return

    def AssertStrategyPositionGroup(self, positionGroup: IPositionGroup, option_symbol: Symbol):
        if False:
            while True:
                i = 10
        positions = list(positionGroup.Positions)
        if len(positions) != 4:
            raise Exception(f'Expected position group to have 4 positions. Actual: {len(positions)}')
        orderedStrikes = sorted((leg.Strike for leg in self._iron_condor.OptionLegs))
        longPutStrike = orderedStrikes[0]
        longPutPosition = next((x for x in positionGroup.Positions if x.Symbol.ID.OptionRight == OptionRight.Put and x.Symbol.ID.StrikePrice == longPutStrike), None)
        if longPutPosition is None or longPutPosition.Quantity != 2:
            raise Exception(f'Expected long put position quantity to be 2. Actual: {longPutPosition.Quantity}')
        shortPutStrike = orderedStrikes[1]
        shortPutPosition = next((x for x in positionGroup.Positions if x.Symbol.ID.OptionRight == OptionRight.Put and x.Symbol.ID.StrikePrice == shortPutStrike), None)
        if shortPutPosition is None or shortPutPosition.Quantity != -2:
            raise Exception(f'Expected short put position quantity to be -2. Actual: {shortPutPosition.Quantity}')
        shortCallStrike = orderedStrikes[2]
        shortCallPosition = next((x for x in positionGroup.Positions if x.Symbol.ID.OptionRight == OptionRight.Call and x.Symbol.ID.StrikePrice == shortCallStrike), None)
        if shortCallPosition is None or shortCallPosition.Quantity != -2:
            raise Exception(f'Expected short call position quantity to be -2. Actual: {shortCallPosition.Quantity}')
        longCallStrike = orderedStrikes[3]
        longCallPosition = next((x for x in positionGroup.Positions if x.Symbol.ID.OptionRight == OptionRight.Call and x.Symbol.ID.StrikePrice == longCallStrike), None)
        if longCallPosition is None or longCallPosition.Quantity != 2:
            raise Exception(f'Expected long call position quantity to be 2. Actual: {longCallPosition.Quantity}')

    def LiquidateStrategy(self):
        if False:
            while True:
                i = 10
        self.Sell(self._iron_condor, 2)