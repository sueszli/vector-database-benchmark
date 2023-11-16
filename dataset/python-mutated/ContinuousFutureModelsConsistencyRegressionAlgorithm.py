from AlgorithmImports import *
from OptionModelsConsistencyRegressionAlgorithm import OptionModelsConsistencyRegressionAlgorithm

class ContinuousFutureModelsConsistencyRegressionAlgorithm(OptionModelsConsistencyRegressionAlgorithm):

    def InitializeAlgorithm(self) -> Security:
        if False:
            for i in range(10):
                print('nop')
        self.SetStartDate(2013, 7, 1)
        self.SetEndDate(2014, 1, 1)
        continuous_contract = self.AddFuture(Futures.Indices.SP500EMini, dataNormalizationMode=DataNormalizationMode.BackwardsPanamaCanal, dataMappingMode=DataMappingMode.OpenInterest, contractDepthOffset=1)
        return continuous_contract