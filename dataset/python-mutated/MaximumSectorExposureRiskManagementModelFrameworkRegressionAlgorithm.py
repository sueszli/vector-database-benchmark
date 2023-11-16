from AlgorithmImports import *
from BaseFrameworkRegressionAlgorithm import BaseFrameworkRegressionAlgorithm
from Risk.MaximumSectorExposureRiskManagementModel import MaximumSectorExposureRiskManagementModel

class MaximumSectorExposureRiskManagementModelFrameworkRegressionAlgorithm(BaseFrameworkRegressionAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        super().Initialize()
        self.UniverseSettings.Resolution = Resolution.Daily
        self.SetStartDate(2014, 2, 1)
        self.SetEndDate(2014, 5, 1)
        tickers = ['AAPL', 'MSFT', 'GOOG', 'AIG', 'BAC']
        self.SetUniverseSelection(FineFundamentalUniverseSelectionModel(lambda coarse: [x.Symbol for x in coarse if x.Symbol.Value in tickers], lambda fine: [x.Symbol for x in fine]))
        self.SetRiskManagement(MaximumSectorExposureRiskManagementModel(0.1))

    def OnEndOfAlgorithm(self):
        if False:
            for i in range(10):
                print('nop')
        pass