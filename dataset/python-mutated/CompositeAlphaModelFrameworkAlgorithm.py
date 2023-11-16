from AlgorithmImports import *
from Alphas.RsiAlphaModel import RsiAlphaModel
from Alphas.EmaCrossAlphaModel import EmaCrossAlphaModel
from Portfolio.EqualWeightingPortfolioConstructionModel import EqualWeightingPortfolioConstructionModel

class CompositeAlphaModelFrameworkAlgorithm(QCAlgorithm):
    """Show cases how to use the CompositeAlphaModel to define."""

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.AddEquity('SPY')
        self.AddEquity('IBM')
        self.AddEquity('BAC')
        self.AddEquity('AIG')
        self.SetUniverseSelection(ManualUniverseSelectionModel())
        self.SetAlpha(CompositeAlphaModel(RsiAlphaModel(), EmaCrossAlphaModel()))
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(NullRiskManagementModel())