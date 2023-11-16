from AlgorithmImports import *
from System import Action
from QuantConnect.Logging import *

class OptionModelsConsistencyRegressionAlgorithm(QCAlgorithm):

    def Initialize(self) -> None:
        if False:
            i = 10
            return i + 15
        security = self.InitializeAlgorithm()
        self.SetModels(security)
        self.SetSecurityInitializer(CustomSecurityInitializer(self.BrokerageModel, SecuritySeeder.Null))
        self.SetBenchmark(lambda x: 0)

    def InitializeAlgorithm(self) -> Security:
        if False:
            return 10
        self.SetStartDate(2015, 12, 24)
        self.SetEndDate(2015, 12, 24)
        equity = self.AddEquity('GOOG', leverage=4)
        option = self.AddOption(equity.Symbol)
        option.SetFilter(lambda u: u.Strikes(-2, +2).Expiration(0, 180))
        return option

    def SetModels(self, security: Security) -> None:
        if False:
            return 10
        security.SetFillModel(CustomFillModel())
        security.SetFeeModel(CustomFeeModel())
        security.SetBuyingPowerModel(CustomBuyingPowerModel())
        security.SetSlippageModel(CustomSlippageModel())
        security.SetVolatilityModel(CustomVolatilityModel())

class CustomSecurityInitializer(BrokerageModelSecurityInitializer):

    def __init__(self, brokerage_model: BrokerageModel, security_seeder: SecuritySeeder):
        if False:
            return 10
        super().__init__(brokerage_model, security_seeder)

class CustomFillModel(FillModel):
    pass

class CustomFeeModel(FeeModel):
    pass

class CustomBuyingPowerModel(BuyingPowerModel):
    pass

class CustomSlippageModel(ConstantSlippageModel):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__(0)

class CustomVolatilityModel(BaseVolatilityModel):
    pass