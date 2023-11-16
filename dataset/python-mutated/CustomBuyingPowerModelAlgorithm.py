from AlgorithmImports import *

class CustomBuyingPowerModelAlgorithm(QCAlgorithm):
    """Demonstration of using custom buying power model in backtesting.
    QuantConnect allows you to model all orders as deeply and accurately as you need."""

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2013, 10, 1)
        self.SetEndDate(2013, 10, 31)
        security = self.AddEquity('SPY', Resolution.Hour)
        self.spy = security.Symbol
        security.SetBuyingPowerModel(CustomBuyingPowerModel())

    def OnData(self, slice):
        if False:
            return 10
        if self.Portfolio.Invested:
            return
        quantity = self.CalculateOrderQuantity(self.spy, 1)
        if quantity % 100 != 0:
            raise Exception(f'CustomBuyingPowerModel only allow quantity that is multiple of 100 and {quantity} was found')
        self.MarketOrder(self.spy, quantity * 10)

class CustomBuyingPowerModel(BuyingPowerModel):

    def GetMaximumOrderQuantityForTargetBuyingPower(self, parameters):
        if False:
            i = 10
            return i + 15
        quantity = super().GetMaximumOrderQuantityForTargetBuyingPower(parameters).Quantity
        quantity = np.floor(quantity / 100) * 100
        return GetMaximumOrderQuantityResult(quantity)

    def HasSufficientBuyingPowerForOrder(self, parameters):
        if False:
            for i in range(10):
                print('nop')
        return HasSufficientBuyingPowerForOrderResult(True)

    def GetMaintenanceMargin(self, parameters):
        if False:
            return 10
        return MaintenanceMargin(0)

    def GetReservedBuyingPowerForPosition(self, parameters):
        if False:
            i = 10
            return i + 15
        return parameters.ResultInAccountCurrency(0)