from AlgorithmImports import *

class ForwardDataOnlyFillModelAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2013, 10, 1)
        self.SetEndDate(2013, 10, 31)
        self.security = self.AddEquity('SPY', Resolution.Hour)
        self.security.SetFillModel(ForwardDataOnlyFillModel())
        self.Schedule.On(self.DateRules.WeekStart(), self.TimeRules.AfterMarketOpen(self.security.Symbol), self.Trade)

    def Trade(self):
        if False:
            while True:
                i = 10
        if not self.Portfolio.Invested:
            if self.Time.hour != 9 or self.Time.minute != 30:
                raise Exception(f'Unexpected event time {self.Time}')
            ticket = self.Buy('SPY', 1)
            if ticket.Status != OrderStatus.Submitted:
                raise Exception(f'Unexpected order status {ticket.Status}')

    def OnOrderEvent(self, orderEvent: OrderEvent):
        if False:
            i = 10
            return i + 15
        self.Debug(f'OnOrderEvent:: {orderEvent}')
        if orderEvent.Status == OrderStatus.Filled and (self.Time.hour != 10 or self.Time.minute != 0):
            raise Exception(f'Unexpected fill time {self.Time}')

class ForwardDataOnlyFillModel(EquityFillModel):

    def Fill(self, parameters: FillModelParameters):
        if False:
            i = 10
            return i + 15
        orderLocalTime = Extensions.ConvertFromUtc(parameters.Order.Time, parameters.Security.Exchange.TimeZone)
        for dataType in [QuoteBar, TradeBar, Tick]:
            data = parameters.Security.Cache.GetData[dataType]()
            if not data is None and orderLocalTime <= data.EndTime:
                return super().Fill(parameters)
        return Fill([])