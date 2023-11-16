from AlgorithmImports import *
from Selection.UniverseSelectionModel import UniverseSelectionModel

class OptionUniverseSelectionModel(UniverseSelectionModel):
    """Provides an implementation of IUniverseSelectionMode that subscribes to option chains"""

    def __init__(self, refreshInterval, optionChainSymbolSelector, universeSettings=None):
        if False:
            return 10
        'Creates a new instance of OptionUniverseSelectionModel\n        Args:\n            refreshInterval: Time interval between universe refreshes</param>\n            optionChainSymbolSelector: Selects symbols from the provided option chain\n            universeSettings: Universe settings define attributes of created subscriptions, such as their resolution and the minimum time in universe before they can be removed'
        self.nextRefreshTimeUtc = datetime.min
        self.refreshInterval = refreshInterval
        self.optionChainSymbolSelector = optionChainSymbolSelector
        self.universeSettings = universeSettings

    def GetNextRefreshTimeUtc(self):
        if False:
            i = 10
            return i + 15
        'Gets the next time the framework should invoke the `CreateUniverses` method to refresh the set of universes.'
        return self.nextRefreshTimeUtc

    def CreateUniverses(self, algorithm):
        if False:
            for i in range(10):
                print('nop')
        "Creates a new fundamental universe using this class's selection functions\n        Args:\n            algorithm: The algorithm instance to create universes for\n        Returns:\n            The universe defined by this model"
        self.nextRefreshTimeUtc = (algorithm.UtcTime + self.refreshInterval).date()
        uniqueUnderlyingSymbols = set()
        for optionSymbol in self.optionChainSymbolSelector(algorithm.UtcTime):
            if not Extensions.IsOption(optionSymbol.SecurityType):
                raise ValueError('optionChainSymbolSelector must return option, index options, or futures options symbols.')
            if optionSymbol.Underlying not in uniqueUnderlyingSymbols:
                uniqueUnderlyingSymbols.add(optionSymbol.Underlying)
                yield Extensions.CreateOptionChain(algorithm, optionSymbol, self.Filter, self.universeSettings)

    def Filter(self, filter):
        if False:
            print('Hello World!')
        'Defines the option chain universe filter'
        return filter