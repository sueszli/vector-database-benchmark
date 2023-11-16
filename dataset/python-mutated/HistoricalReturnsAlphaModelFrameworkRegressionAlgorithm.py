from AlgorithmImports import *
from BaseFrameworkRegressionAlgorithm import BaseFrameworkRegressionAlgorithm
from Alphas.HistoricalReturnsAlphaModel import HistoricalReturnsAlphaModel

class HistoricalReturnsAlphaModelFrameworkRegressionAlgorithm(BaseFrameworkRegressionAlgorithm):

    def Initialize(self):
        if False:
            return 10
        super().Initialize()
        self.SetAlpha(HistoricalReturnsAlphaModel())

    def OnEndOfAlgorithm(self):
        if False:
            return 10
        expected = 74
        if self.Insights.TotalCount != expected:
            raise Exception(f'The total number of insights should be {expected}. Actual: {self.Insights.TotalCount}')