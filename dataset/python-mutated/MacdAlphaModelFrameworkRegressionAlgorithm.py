from AlgorithmImports import *
from BaseFrameworkRegressionAlgorithm import BaseFrameworkRegressionAlgorithm
from Alphas.MacdAlphaModel import MacdAlphaModel

class MacdAlphaModelFrameworkRegressionAlgorithm(BaseFrameworkRegressionAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        super().Initialize()
        self.SetAlpha(MacdAlphaModel())

    def OnEndOfAlgorithm(self):
        if False:
            i = 10
            return i + 15
        expected = 4
        if self.Insights.TotalCount != expected:
            raise Exception(f'The total number of insights should be {expected}. Actual: {self.Insights.TotalCount}')