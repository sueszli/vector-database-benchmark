# QUANTCONNECT.COM - Democratizing Finance, Empowering Individuals.
# Lean Algorithmic Trading Engine v2.0. Copyright 2014 QuantConnect Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from AlgorithmImports import *
import random

### <summary>
### Demonstration of using custom fee, slippage, fill, and buying power models for modelling transactions in backtesting.
### QuantConnect allows you to model all orders as deeply and accurately as you need.
### </summary>
### <meta name="tag" content="trading and orders" />
### <meta name="tag" content="transaction fees and slippage" />
### <meta name="tag" content="custom buying power models" />
### <meta name="tag" content="custom transaction models" />
### <meta name="tag" content="custom slippage models" />
### <meta name="tag" content="custom fee models" />
class CustomModelsAlgorithm(QCAlgorithm):
    '''Demonstration of using custom fee, slippage, fill, and buying power models for modelling transactions in backtesting.
    QuantConnect allows you to model all orders as deeply and accurately as you need.'''

    def Initialize(self):
        self.SetStartDate(2013,10,1)   # Set Start Date
        self.SetEndDate(2013,10,31)    # Set End Date
        self.security = self.AddEquity("SPY", Resolution.Hour)
        self.spy = self.security.Symbol

        # set our models
        self.security.SetFeeModel(CustomFeeModel(self))
        self.security.SetFillModel(CustomFillModel(self))
        self.security.SetSlippageModel(CustomSlippageModel(self))
        self.security.SetBuyingPowerModel(CustomBuyingPowerModel(self))


    def OnData(self, data):
        open_orders = self.Transactions.GetOpenOrders(self.spy)
        if len(open_orders) != 0: return

        if self.Time.day > 10 and self.security.Holdings.Quantity <= 0:
            quantity = self.CalculateOrderQuantity(self.spy, .5)
            self.Log(f"MarketOrder: {quantity}")
            self.MarketOrder(self.spy, quantity, True)   # async needed for partial fill market orders

        elif self.Time.day > 20 and self.security.Holdings.Quantity >= 0:
            quantity = self.CalculateOrderQuantity(self.spy, -.5)
            self.Log(f"MarketOrder: {quantity}")
            self.MarketOrder(self.spy, quantity, True)   # async needed for partial fill market orders

# If we want to use methods from other models, you need to inherit from one of them
class CustomFillModel(ImmediateFillModel):
    def __init__(self, algorithm):
        super().__init__()
        self.algorithm = algorithm
        self.absoluteRemainingByOrderId = {}
        self.random = Random(387510346)

    def MarketFill(self, asset, order):
        absoluteRemaining = order.AbsoluteQuantity

        if order.Id in self.absoluteRemainingByOrderId.keys():
            absoluteRemaining = self.absoluteRemainingByOrderId[order.Id]

        fill = super().MarketFill(asset, order)
        absoluteFillQuantity = int(min(absoluteRemaining, self.random.Next(0, 2*int(order.AbsoluteQuantity))))
        fill.FillQuantity = np.sign(order.Quantity) * absoluteFillQuantity
        
        if absoluteRemaining == absoluteFillQuantity:
            fill.Status = OrderStatus.Filled
            if self.absoluteRemainingByOrderId.get(order.Id):
                self.absoluteRemainingByOrderId.pop(order.Id)
        else:
            absoluteRemaining = absoluteRemaining - absoluteFillQuantity
            self.absoluteRemainingByOrderId[order.Id] = absoluteRemaining
            fill.Status = OrderStatus.PartiallyFilled
        self.algorithm.Log(f"CustomFillModel: {fill}")
        return fill

class CustomFeeModel(FeeModel):
    def __init__(self, algorithm):
        super().__init__()
        self.algorithm = algorithm

    def GetOrderFee(self, parameters):
        # custom fee math
        fee = max(1, parameters.Security.Price
                  * parameters.Order.AbsoluteQuantity
                  * 0.00001)
        self.algorithm.Log(f"CustomFeeModel: {fee}")
        return OrderFee(CashAmount(fee, "USD"))

class CustomSlippageModel:
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def GetSlippageApproximation(self, asset, order):
        # custom slippage math
        slippage = asset.Price * 0.0001 * np.log10(2*float(order.AbsoluteQuantity))
        self.algorithm.Log(f"CustomSlippageModel: {slippage}")
        return slippage

class CustomBuyingPowerModel(BuyingPowerModel):
    def __init__(self, algorithm):
        super().__init__()
        self.algorithm = algorithm

    def HasSufficientBuyingPowerForOrder(self, parameters):
        # custom behavior: this model will assume that there is always enough buying power
        hasSufficientBuyingPowerForOrderResult = HasSufficientBuyingPowerForOrderResult(True)
        self.algorithm.Log(f"CustomBuyingPowerModel: {hasSufficientBuyingPowerForOrderResult.IsSufficient}")
        return hasSufficientBuyingPowerForOrderResult

# The simple fill model shows how to implement a simpler version of 
# the most popular order fills: Market, Stop Market and Limit
class SimpleCustomFillModel(FillModel):
    def __init__(self):
        super().__init__()

    def _create_order_event(self, asset, order):
        utcTime = Extensions.ConvertToUtc(asset.LocalTime, asset.Exchange.TimeZone)
        return OrderEvent(order, utcTime, OrderFee.Zero)

    def _set_order_event_to_filled(self, fill, fill_price, fill_quantity):
        fill.Status = OrderStatus.Filled
        fill.FillQuantity = fill_quantity
        fill.FillPrice = fill_price
        return fill

    def _get_trade_bar(self, asset, orderDirection):
        trade_bar = asset.Cache.GetData[TradeBar]()
        if trade_bar: return trade_bar

        # Tick-resolution data doesn't have TradeBar, use the asset price
        price = asset.Price
        return TradeBar(asset.LocalTime, asset.Symbol, price, price, price, price, 0)

    def MarketFill(self, asset, order):
        fill = self._create_order_event(asset, order)
        if order.Status == OrderStatus.Canceled: return fill

        return self._set_order_event_to_filled(fill, 
            asset.Cache.AskPrice \
                if order.Direction == OrderDirection.Buy else asset.Cache.BidPrice,
            order.Quantity)

    def StopMarketFill(self, asset, order):
        fill = self._create_order_event(asset, order)
        if order.Status == OrderStatus.Canceled: return fill
        
        stop_price = order.StopPrice
        trade_bar = self._get_trade_bar(asset, order.Direction)
        
        if order.Direction == OrderDirection.Sell and trade_bar.Low < stop_price:
            return self._set_order_event_to_filled(fill, stop_price, order.Quantity)

        if order.Direction == OrderDirection.Buy and trade_bar.High > stop_price:
            return self._set_order_event_to_filled(fill, stop_price, order.Quantity)

        return fill

    def LimitFill(self, asset, order):
        fill = self._create_order_event(asset, order)
        if order.Status == OrderStatus.Canceled: return fill

        limit_price = order.LimitPrice
        trade_bar = self._get_trade_bar(asset, order.Direction)

        if order.Direction == OrderDirection.Sell and trade_bar.High > limit_price:
            return self._set_order_event_to_filled(fill, limit_price, order.Quantity)

        if order.Direction == OrderDirection.Buy and trade_bar.Low < limit_price:
            return self._set_order_event_to_filled(fill, limit_price, order.Quantity)

        return fill
