from AlgorithmImports import *
from OptionModelsConsistencyRegressionAlgorithm import OptionModelsConsistencyRegressionAlgorithm

class IndexOptionModelsConsistencyRegressionAlgorithm(OptionModelsConsistencyRegressionAlgorithm):

    def InitializeAlgorithm(self) -> Security:
        if False:
            print('Hello World!')
        self.SetStartDate(2021, 1, 4)
        self.SetEndDate(2021, 1, 5)
        index = self.AddIndex('SPX', Resolution.Minute)
        option = self.AddIndexOption(index.Symbol, 'SPX', Resolution.Minute)
        option.SetFilter(lambda u: u.Strikes(-5, +5).Expiration(0, 360))
        return option