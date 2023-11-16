from AlgorithmImports import *

class CompositeRiskManagementModel(RiskManagementModel):
    """Provides an implementation of IRiskManagementModel that combines multiple risk models
    into a single risk management model and properly sets each insights 'SourceModel' property."""

    def __init__(self, *riskManagementModels):
        if False:
            print('Hello World!')
        'Initializes a new instance of the CompositeRiskManagementModel class\n        Args:\n            riskManagementModels: The individual risk management models defining this composite model.'
        for model in riskManagementModels:
            for attributeName in ['ManageRisk', 'OnSecuritiesChanged']:
                if not hasattr(model, attributeName):
                    raise Exception(f'IRiskManagementModel.{attributeName} must be implemented. Please implement this missing method on {model.__class__.__name__}')
        self.riskManagementModels = riskManagementModels

    def ManageRisk(self, algorithm, targets):
        if False:
            i = 10
            return i + 15
        "Manages the algorithm's risk at each time step\n        Args:\n            algorithm: The algorithm instance\n            targets: The current portfolio targets to be assessed for risk"
        for model in self.riskManagementModels:
            riskAdjusted = model.ManageRisk(algorithm, targets)
            symbols = [x.Symbol for x in riskAdjusted]
            for target in targets:
                if target.Symbol not in symbols:
                    riskAdjusted.append(target)
            targets = riskAdjusted
        return targets

    def OnSecuritiesChanged(self, algorithm, changes):
        if False:
            print('Hello World!')
        'Event fired each time the we add/remove securities from the data feed.\n        This method patches this call through the each of the wrapped models.\n        Args:\n            algorithm: The algorithm instance that experienced the change in securities\n            changes: The security additions and removals from the algorithm'
        for model in self.riskManagementModels:
            model.OnSecuritiesChanged(algorithm, changes)

    def AddRiskManagement(riskManagementModel):
        if False:
            print('Hello World!')
        "Adds a new 'IRiskManagementModel' instance\n        Args:\n            riskManagementModel: The risk management model to add"
        self.riskManagementModels.Add(riskManagementModel)