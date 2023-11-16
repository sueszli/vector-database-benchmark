from AlgorithmImports import *
import random

class CustomModelsAlgorithm(QCAlgorithm):
    """Demonstration of using custom fee, slippage, fill, and buying power models for modelling transactions in backtesting.
    QuantConnect allows you to model all orders as deeply and accurately as you need."""

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetStartDate(2013, 10, 1)
        self.SetEndDate(2013, 10, 31)
        self.security = self.AddEquity('SPY', Resolution.Hour)
        self.spy = self.security.Symbol
        self.security.SetFeeModel(CustomFeeModel(self))
        self.security.SetFillModel(CustomFillModel(self))
        self.security.SetSlippageModel(CustomSlippageModel(self))
        self.security.SetBuyingPowerModel(CustomBuyingPowerModel(self))

    def OnData(self, data):
        if False:
            for i in range(10):
                print('nop')
        open_orders = self.Transactions.GetOpenOrders(self.spy)
        if len(open_orders) != 0:
            return
        if self.Time.day > 10 and self.security.Holdings.Quantity <= 0:
            quantity = self.CalculateOrderQuantity(self.spy, 0.5)
            self.Log(f'MarketOrder: {quantity}')
            self.MarketOrder(self.spy, quantity, True)
        elif self.Time.day > 20 and self.security.Holdings.Quantity >= 0:
            quantity = self.CalculateOrderQuantity(self.spy, -0.5)
            self.Log(f'MarketOrder: {quantity}')
            self.MarketOrder(self.spy, quantity, True)

class CustomFillModel(ImmediateFillModel):

    def __init__(self, algorithm):
        if False:
            while True:
                i = 10
        super().__init__()
        self.algorithm = algorithm
        self.absoluteRemainingByOrderId = {}
        self.random = Random(387510346)

    def MarketFill(self, asset, order):
        if False:
            i = 10
            return i + 15
        absoluteRemaining = order.AbsoluteQuantity
        if order.Id in self.absoluteRemainingByOrderId.keys():
            absoluteRemaining = self.absoluteRemainingByOrderId[order.Id]
        fill = super().MarketFill(asset, order)
        absoluteFillQuantity = int(min(absoluteRemaining, self.random.Next(0, 2 * int(order.AbsoluteQuantity))))
        fill.FillQuantity = np.sign(order.Quantity) * absoluteFillQuantity
        if absoluteRemaining == absoluteFillQuantity:
            fill.Status = OrderStatus.Filled
            if self.absoluteRemainingByOrderId.get(order.Id):
                self.absoluteRemainingByOrderId.pop(order.Id)
        else:
            absoluteRemaining = absoluteRemaining - absoluteFillQuantity
            self.absoluteRemainingByOrderId[order.Id] = absoluteRemaining
            fill.Status = OrderStatus.PartiallyFilled
        self.algorithm.Log(f'CustomFillModel: {fill}')
        return fill

class CustomFeeModel(FeeModel):

    def __init__(self, algorithm):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.algorithm = algorithm

    def GetOrderFee(self, parameters):
        if False:
            print('Hello World!')
        fee = max(1, parameters.Security.Price * parameters.Order.AbsoluteQuantity * 1e-05)
        self.algorithm.Log(f'CustomFeeModel: {fee}')
        return OrderFee(CashAmount(fee, 'USD'))

class CustomSlippageModel:

    def __init__(self, algorithm):
        if False:
            i = 10
            return i + 15
        self.algorithm = algorithm

    def GetSlippageApproximation(self, asset, order):
        if False:
            for i in range(10):
                print('nop')
        slippage = asset.Price * 0.0001 * np.log10(2 * float(order.AbsoluteQuantity))
        self.algorithm.Log(f'CustomSlippageModel: {slippage}')
        return slippage

class CustomBuyingPowerModel(BuyingPowerModel):

    def __init__(self, algorithm):
        if False:
            while True:
                i = 10
        super().__init__()
        self.algorithm = algorithm

    def HasSufficientBuyingPowerForOrder(self, parameters):
        if False:
            print('Hello World!')
        hasSufficientBuyingPowerForOrderResult = HasSufficientBuyingPowerForOrderResult(True)
        self.algorithm.Log(f'CustomBuyingPowerModel: {hasSufficientBuyingPowerForOrderResult.IsSufficient}')
        return hasSufficientBuyingPowerForOrderResult

class SimpleCustomFillModel(FillModel):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()

    def _create_order_event(self, asset, order):
        if False:
            print('Hello World!')
        utcTime = Extensions.ConvertToUtc(asset.LocalTime, asset.Exchange.TimeZone)
        return OrderEvent(order, utcTime, OrderFee.Zero)

    def _set_order_event_to_filled(self, fill, fill_price, fill_quantity):
        if False:
            while True:
                i = 10
        fill.Status = OrderStatus.Filled
        fill.FillQuantity = fill_quantity
        fill.FillPrice = fill_price
        return fill

    def _get_trade_bar(self, asset, orderDirection):
        if False:
            i = 10
            return i + 15
        trade_bar = asset.Cache.GetData[TradeBar]()
        if trade_bar:
            return trade_bar
        price = asset.Price
        return TradeBar(asset.LocalTime, asset.Symbol, price, price, price, price, 0)

    def MarketFill(self, asset, order):
        if False:
            return 10
        fill = self._create_order_event(asset, order)
        if order.Status == OrderStatus.Canceled:
            return fill
        return self._set_order_event_to_filled(fill, asset.Cache.AskPrice if order.Direction == OrderDirection.Buy else asset.Cache.BidPrice, order.Quantity)

    def StopMarketFill(self, asset, order):
        if False:
            i = 10
            return i + 15
        fill = self._create_order_event(asset, order)
        if order.Status == OrderStatus.Canceled:
            return fill
        stop_price = order.StopPrice
        trade_bar = self._get_trade_bar(asset, order.Direction)
        if order.Direction == OrderDirection.Sell and trade_bar.Low < stop_price:
            return self._set_order_event_to_filled(fill, stop_price, order.Quantity)
        if order.Direction == OrderDirection.Buy and trade_bar.High > stop_price:
            return self._set_order_event_to_filled(fill, stop_price, order.Quantity)
        return fill

    def LimitFill(self, asset, order):
        if False:
            while True:
                i = 10
        fill = self._create_order_event(asset, order)
        if order.Status == OrderStatus.Canceled:
            return fill
        limit_price = order.LimitPrice
        trade_bar = self._get_trade_bar(asset, order.Direction)
        if order.Direction == OrderDirection.Sell and trade_bar.High > limit_price:
            return self._set_order_event_to_filled(fill, limit_price, order.Quantity)
        if order.Direction == OrderDirection.Buy and trade_bar.Low < limit_price:
            return self._set_order_event_to_filled(fill, limit_price, order.Quantity)
        return fill