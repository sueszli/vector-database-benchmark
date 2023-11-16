from AlgorithmImports import *

class ComboOrdersFillModelAlgorithm(QCAlgorithm):
    """Basic template algorithm that implements a fill model with combo orders"""

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetStartDate(2019, 1, 1)
        self.SetEndDate(2019, 1, 20)
        self.spy = self.AddEquity('SPY', Resolution.Hour)
        self.ibm = self.AddEquity('IBM', Resolution.Hour)
        self.spy.SetFillModel(CustomPartialFillModel())
        self.ibm.SetFillModel(CustomPartialFillModel())
        self.orderTypes = {}

    def OnData(self, data):
        if False:
            print('Hello World!')
        if not self.Portfolio.Invested:
            legs = [Leg.Create(self.spy.Symbol, 1), Leg.Create(self.ibm.Symbol, -1)]
            self.ComboMarketOrder(legs, 100)
            self.ComboLimitOrder(legs, 100, round(self.spy.BidPrice))
            legs = [Leg.Create(self.spy.Symbol, 1, round(self.spy.BidPrice) + 1), Leg.Create(self.ibm.Symbol, -1, round(self.ibm.BidPrice) + 1)]
            self.ComboLegLimitOrder(legs, 100)

    def OnOrderEvent(self, orderEvent):
        if False:
            print('Hello World!')
        if orderEvent.Status == OrderStatus.Filled:
            orderType = self.Transactions.GetOrderById(orderEvent.OrderId).Type
            if orderType == OrderType.ComboMarket and orderEvent.AbsoluteFillQuantity != 50:
                raise Exception(f'The absolute quantity filled for all combo market orders should be 50, but for order {orderEvent.OrderId} was {orderEvent.AbsoluteFillQuantity}')
            elif orderType == OrderType.ComboLimit and orderEvent.AbsoluteFillQuantity != 20:
                raise Exception(f'The absolute quantity filled for all combo limit orders should be 20, but for order {orderEvent.OrderId} was {orderEvent.AbsoluteFillQuantity}')
            elif orderType == OrderType.ComboLegLimit and orderEvent.AbsoluteFillQuantity != 10:
                raise Exception(f'The absolute quantity filled for all combo leg limit orders should be 10, but for order {orderEvent.OrderId} was {orderEvent.AbsoluteFillQuantity}')
            self.orderTypes[orderType] = 1

    def OnEndOfAlgorithm(self):
        if False:
            print('Hello World!')
        if len(self.orderTypes) != 3:
            raise Exception(f'Just 3 different types of order were submitted in this algorithm, but the amount of order types was {len(self.orderTypes)}')
        if OrderType.ComboMarket not in self.orderTypes.keys():
            raise Exception(f'One Combo Market Order should have been submitted but it was not')
        if OrderType.ComboLimit not in self.orderTypes.keys():
            raise Exception(f'One Combo Limit Order should have been submitted but it was not')
        if OrderType.ComboLegLimit not in self.orderTypes.keys():
            raise Exception(f'One Combo Leg Limit Order should have been submitted but it was not')

class CustomPartialFillModel(FillModel):
    """Implements a custom fill model that inherit from FillModel. Overrides ComboMarketFill, ComboLimitOrder and ComboLegLimitOrder
       methods to test FillModelPythonWrapper works as expected"""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.absoluteRemainingByOrderId = {}

    def FillOrdersPartially(self, parameters, fills, quantity):
        if False:
            return 10
        partialFills = []
        if len(fills) == 0:
            return partialFills
        for (kvp, fill) in zip(sorted(parameters.SecuritiesForOrders, key=lambda x: x.Key.Id), fills):
            order = kvp.Key
            absoluteRemaining = self.absoluteRemainingByOrderId.get(order.Id, order.AbsoluteQuantity)
            fill.FillQuantity = np.sign(order.Quantity) * quantity
            if min(abs(fill.FillQuantity), absoluteRemaining) == absoluteRemaining:
                fill.FillQuantity = np.sign(order.Quantity) * absoluteRemaining
                fill.Status = OrderStatus.Filled
                self.absoluteRemainingByOrderId.pop(order.Id, None)
            else:
                fill.Status = OrderStatus.PartiallyFilled
                self.absoluteRemainingByOrderId[order.Id] = absoluteRemaining - abs(fill.FillQuantity)
                price = fill.FillPrice
            partialFills.append(fill)
        return partialFills

    def ComboMarketFill(self, order, parameters):
        if False:
            while True:
                i = 10
        fills = super().ComboMarketFill(order, parameters)
        partialFills = self.FillOrdersPartially(parameters, fills, 50)
        return partialFills

    def ComboLimitFill(self, order, parameters):
        if False:
            while True:
                i = 10
        fills = super().ComboLimitFill(order, parameters)
        partialFills = self.FillOrdersPartially(parameters, fills, 20)
        return partialFills

    def ComboLegLimitFill(self, order, parameters):
        if False:
            i = 10
            return i + 15
        fills = super().ComboLegLimitFill(order, parameters)
        partialFills = self.FillOrdersPartially(parameters, fills, 10)
        return partialFills