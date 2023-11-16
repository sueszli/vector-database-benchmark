from AlgorithmImports import *

class FundamentalUniverseSelectionModel:
    """Provides a base class for defining equity coarse/fine fundamental selection models"""

    def __init__(self, filterFineData=None, universeSettings=None):
        if False:
            while True:
                i = 10
        'Initializes a new instance of the FundamentalUniverseSelectionModel class\n        Args:\n            filterFineData: [Obsolete] Fine and Coarse selection are merged\n            universeSettings: The settings used when adding symbols to the algorithm, specify null to use algorithm.UniverseSettings'
        self.filterFineData = filterFineData
        if self.filterFineData == None:
            self._fundamentalData = True
        else:
            self._fundamentalData = False
        self.universeSettings = universeSettings

    def CreateUniverses(self, algorithm):
        if False:
            while True:
                i = 10
        "Creates a new fundamental universe using this class's selection functions\n        Args:\n            algorithm: The algorithm instance to create universes for\n        Returns:\n            The universe defined by this model"
        if self._fundamentalData:
            universeSettings = algorithm.UniverseSettings if self.universeSettings is None else self.universeSettings
            universe = FundamentalUniverse(universeSettings, lambda fundamental: self.Select(algorithm, fundamental))
            return [universe]
        else:
            universe = self.CreateCoarseFundamentalUniverse(algorithm)
            if self.filterFineData:
                if universe.UniverseSettings.Asynchronous:
                    raise ValueError('Asynchronous universe setting is not supported for coarse & fine selections, please use the new Fundamental single pass selection')
                universe = FineFundamentalFilteredUniverse(universe, lambda fine: self.SelectFine(algorithm, fine))
            return [universe]

    def CreateCoarseFundamentalUniverse(self, algorithm):
        if False:
            i = 10
            return i + 15
        'Creates the coarse fundamental universe object.\n        This is provided to allow more flexibility when creating coarse universe.\n        Args:\n            algorithm: The algorithm instance\n        Returns:\n            The coarse fundamental universe'
        universeSettings = algorithm.UniverseSettings if self.universeSettings is None else self.universeSettings
        return CoarseFundamentalUniverse(universeSettings, lambda coarse: self.FilteredSelectCoarse(algorithm, coarse))

    def FilteredSelectCoarse(self, algorithm, coarse):
        if False:
            return 10
        "Defines the coarse fundamental selection function.\n        If we're using fine fundamental selection than exclude symbols without fine data\n        Args:\n            algorithm: The algorithm instance\n            coarse: The coarse fundamental data used to perform filtering\n        Returns:\n            An enumerable of symbols passing the filter"
        if self.filterFineData:
            coarse = filter(lambda c: c.HasFundamentalData, coarse)
        return self.SelectCoarse(algorithm, coarse)

    def Select(self, algorithm, fundamental):
        if False:
            print('Hello World!')
        'Defines the fundamental selection function.\n        Args:\n            algorithm: The algorithm instance\n            fundamental: The fundamental data used to perform filtering\n        Returns:\n            An enumerable of symbols passing the filter'
        raise NotImplementedError("Please overrride the 'Select' fundamental function")

    def SelectCoarse(self, algorithm, coarse):
        if False:
            i = 10
            return i + 15
        'Defines the coarse fundamental selection function.\n        Args:\n            algorithm: The algorithm instance\n            coarse: The coarse fundamental data used to perform filtering\n        Returns:\n            An enumerable of symbols passing the filter'
        raise NotImplementedError("Please overrride the 'Select' fundamental function")

    def SelectFine(self, algorithm, fine):
        if False:
            print('Hello World!')
        'Defines the fine fundamental selection function.\n        Args:\n            algorithm: The algorithm instance\n            fine: The fine fundamental data used to perform filtering\n        Returns:\n            An enumerable of symbols passing the filter'
        return [f.Symbol for f in fine]