from AlgorithmImports import *
from QuantConnect.Algorithm.CSharp import *

class CustomOptionExerciseModelRegressionAlgorithm(OptionAssignmentRegressionAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetSecurityInitializer(self.CustomSecurityInitializer)
        super().Initialize()

    def CustomSecurityInitializer(self, security):
        if False:
            while True:
                i = 10
        if Extensions.IsOption(security.Symbol.SecurityType):
            security.SetOptionExerciseModel(CustomExerciseModel())

    def OnData(self, data):
        if False:
            i = 10
            return i + 15
        super().OnData(data)

class CustomExerciseModel(DefaultExerciseModel):

    def OptionExercise(self, option: Option, order: OptionExerciseOrder):
        if False:
            return 10
        order_event = OrderEvent(order.Id, option.Symbol, Extensions.ConvertToUtc(option.LocalTime, option.Exchange.TimeZone), OrderStatus.Filled, Extensions.GetOrderDirection(order.Quantity), 0.0, order.Quantity, OrderFee.Zero, 'Tag')
        order_event.IsAssignment = False
        return [order_event]