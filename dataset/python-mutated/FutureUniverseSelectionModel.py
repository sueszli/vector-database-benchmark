from AlgorithmImports import *
from Selection.UniverseSelectionModel import UniverseSelectionModel

class FutureUniverseSelectionModel(UniverseSelectionModel):
    """Provides an implementation of IUniverseSelectionMode that subscribes to future chains"""

    def __init__(self, refreshInterval, futureChainSymbolSelector, universeSettings=None):
        if False:
            i = 10
            return i + 15
        'Creates a new instance of FutureUniverseSelectionModel\n        Args:\n            refreshInterval: Time interval between universe refreshes</param>\n            futureChainSymbolSelector: Selects symbols from the provided future chain\n            universeSettings: Universe settings define attributes of created subscriptions, such as their resolution and the minimum time in universe before they can be removed'
        self.nextRefreshTimeUtc = datetime.min
        self.refreshInterval = refreshInterval
        self.futureChainSymbolSelector = futureChainSymbolSelector
        self.universeSettings = universeSettings

    def GetNextRefreshTimeUtc(self):
        if False:
            for i in range(10):
                print('nop')
        'Gets the next time the framework should invoke the `CreateUniverses` method to refresh the set of universes.'
        return self.nextRefreshTimeUtc

    def CreateUniverses(self, algorithm):
        if False:
            print('Hello World!')
        "Creates a new fundamental universe using this class's selection functions\n        Args:\n            algorithm: The algorithm instance to create universes for\n        Returns:\n            The universe defined by this model"
        self.nextRefreshTimeUtc = algorithm.UtcTime + self.refreshInterval
        uniqueSymbols = set()
        for futureSymbol in self.futureChainSymbolSelector(algorithm.UtcTime):
            if futureSymbol.SecurityType != SecurityType.Future:
                raise ValueError('futureChainSymbolSelector must return future symbols.')
            if futureSymbol not in uniqueSymbols:
                uniqueSymbols.add(futureSymbol)
                for universe in Extensions.CreateFutureChain(algorithm, futureSymbol, self.Filter, self.universeSettings):
                    yield universe

    def Filter(self, filter):
        if False:
            i = 10
            return i + 15
        'Defines the future chain universe filter'
        return filter