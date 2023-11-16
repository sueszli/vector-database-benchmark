from AlgorithmImports import *
from Selection.UniverseSelectionModel import UniverseSelectionModel

class ETFConstituentsUniverseSelectionModel(UniverseSelectionModel):
    """Universe selection model that selects the constituents of an ETF."""

    def __init__(self, etfSymbol, universeSettings=None, universeFilterFunc=None):
        if False:
            while True:
                i = 10
        'Initializes a new instance of the ETFConstituentsUniverseSelectionModel class\n        Args:\n            etfSymbol: Symbol of the ETF to get constituents for\n            universeSettings: Universe settings\n            universeFilterFunc: Function to filter universe results'
        self.etf_symbol = etfSymbol
        self.universe_settings = universeSettings
        self.universe_filter_function = universeFilterFunc
        self.universe = None

    def CreateUniverses(self, algorithm: QCAlgorithm) -> List[Universe]:
        if False:
            for i in range(10):
                print('nop')
        "Creates a new ETF constituents universe using this class's selection function\n        Args:\n            algorithm: The algorithm instance to create universes for\n        Returns:\n            The universe defined by this model"
        if self.universe is None:
            self.universe = algorithm.Universe.ETF(self.etf_symbol, self.universe_settings, self.universe_filter_function)
        return [self.universe]