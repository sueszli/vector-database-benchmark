from AlgorithmImports import *

class PortfolioTargetTagsRegressionAlgorithm(QCAlgorithm):
    """Algorithm demonstrating the portfolio target tags usage"""

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.UniverseSettings.Resolution = Resolution.Minute
        symbols = [Symbol.Create('SPY', SecurityType.Equity, Market.USA)]
        self.SetUniverseSelection(ManualUniverseSelectionModel(symbols))
        self.SetAlpha(ConstantAlphaModel(InsightType.Price, InsightDirection.Up, timedelta(minutes=20), 0.025, None))
        self.SetPortfolioConstruction(CustomPortfolioConstructionModel())
        self.SetRiskManagement(CustomRiskManagementModel())
        self.SetExecution(CustomExecutionModel(self.SetTargetTagsChecked))
        self.targetTagsChecked = False

    def SetTargetTagsChecked(self):
        if False:
            return 10
        self.targetTagsChecked = True

    def OnEndOfAlgorithm(self):
        if False:
            return 10
        if not self.targetTagsChecked:
            raise Exception('The portfolio targets tag were not checked')

class CustomPortfolioConstructionModel(EqualWeightingPortfolioConstructionModel):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__(Resolution.Daily)

    def CreateTargets(self, algorithm: QCAlgorithm, insights: List[Insight]) -> List[IPortfolioTarget]:
        if False:
            for i in range(10):
                print('nop')
        targets = super().CreateTargets(algorithm, insights)
        return CustomPortfolioConstructionModel.AddPPortfolioTargetsTags(targets)

    @staticmethod
    def GeneratePortfolioTargetTag(target: IPortfolioTarget) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'Portfolio target tag: {target.Symbol} - {target.Quantity}'

    @staticmethod
    def AddPPortfolioTargetsTags(targets: List[IPortfolioTarget]) -> List[IPortfolioTarget]:
        if False:
            return 10
        return [PortfolioTarget(target.Symbol, target.Quantity, CustomPortfolioConstructionModel.GeneratePortfolioTargetTag(target)) for target in targets]

class CustomRiskManagementModel(MaximumDrawdownPercentPerSecurity):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__(0.01)

    def ManageRisk(self, algorithm: QCAlgorithm, targets: List[IPortfolioTarget]) -> List[IPortfolioTarget]:
        if False:
            return 10
        riskManagedTargets = super().ManageRisk(algorithm, targets)
        return CustomPortfolioConstructionModel.AddPPortfolioTargetsTags(riskManagedTargets)

class CustomExecutionModel(ImmediateExecutionModel):

    def __init__(self, targetsTagCheckedCallback: Callable) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.targetsTagCheckedCallback = targetsTagCheckedCallback

    def Execute(self, algorithm: QCAlgorithm, targets: List[IPortfolioTarget]) -> None:
        if False:
            while True:
                i = 10
        if len(targets) > 0:
            self.targetsTagCheckedCallback()
        for target in targets:
            expectedTag = CustomPortfolioConstructionModel.GeneratePortfolioTargetTag(target)
            if target.Tag != expectedTag:
                raise Exception(f'Unexpected portfolio target tag: {target.Tag} - Expected: {expectedTag}')
        super().Execute(algorithm, targets)