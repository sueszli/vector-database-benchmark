from AlgorithmImports import *
from MeanVarianceOptimizationFrameworkAlgorithm import MeanVarianceOptimizationFrameworkAlgorithm

class CustomPortfolioOptimizerRegressionAlgorithm(MeanVarianceOptimizationFrameworkAlgorithm):

    def Initialize(self):
        if False:
            return 10
        super().Initialize()
        self.SetPortfolioConstruction(MeanVarianceOptimizationPortfolioConstructionModel(timedelta(days=1), PortfolioBias.LongShort, 1, 63, Resolution.Daily, 0.02, CustomPortfolioOptimizer()))

class CustomPortfolioOptimizer:

    def Optimize(self, historicalReturns, expectedReturns, covariance):
        if False:
            while True:
                i = 10
        return [0.5] * np.array(historicalReturns).shape[1]