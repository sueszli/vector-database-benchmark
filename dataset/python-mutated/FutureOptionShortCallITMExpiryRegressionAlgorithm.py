from AlgorithmImports import *

class FutureOptionShortCallITMExpiryRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        self.SetStartDate(2020, 1, 5)
        self.SetEndDate(2020, 6, 30)
        self.es19m20 = self.AddFutureContract(Symbol.CreateFuture(Futures.Indices.SP500EMini, Market.CME, datetime(2020, 6, 19)), Resolution.Minute).Symbol
        self.esOption = self.AddFutureOptionContract(list(sorted([x for x in self.OptionChainProvider.GetOptionContractList(self.es19m20, self.Time) if x.ID.StrikePrice <= 3100.0 and x.ID.OptionRight == OptionRight.Call], key=lambda x: x.ID.StrikePrice, reverse=True))[0], Resolution.Minute).Symbol
        self.expectedContract = Symbol.CreateOption(self.es19m20, Market.CME, OptionStyle.American, OptionRight.Call, 3100.0, datetime(2020, 6, 19))
        if self.esOption != self.expectedContract:
            raise AssertionError(f'Contract {self.expectedContract} was not found in the chain')
        self.Schedule.On(self.DateRules.Tomorrow, self.TimeRules.AfterMarketOpen(self.es19m20, 1), self.ScheduledMarketOrder)

    def ScheduledMarketOrder(self):
        if False:
            while True:
                i = 10
        self.MarketOrder(self.esOption, -1)

    def OnData(self, data: Slice):
        if False:
            i = 10
            return i + 15
        for delisting in data.Delistings.Values:
            if delisting.Type == DelistingType.Warning:
                if delisting.Time != datetime(2020, 6, 19):
                    raise AssertionError(f'Delisting warning issued at unexpected date: {delisting.Time}')
            if delisting.Type == DelistingType.Delisted:
                if delisting.Time != datetime(2020, 6, 20):
                    raise AssertionError(f'Delisting happened at unexpected date: {delisting.Time}')

    def OnOrderEvent(self, orderEvent: OrderEvent):
        if False:
            while True:
                i = 10
        if orderEvent.Status != OrderStatus.Filled:
            return
        if not self.Securities.ContainsKey(orderEvent.Symbol):
            raise AssertionError(f'Order event Symbol not found in Securities collection: {orderEvent.Symbol}')
        security = self.Securities[orderEvent.Symbol]
        if security.Symbol == self.es19m20:
            self.AssertFutureOptionOrderExercise(orderEvent, security, self.Securities[self.expectedContract])
        elif security.Symbol == self.expectedContract:
            self.AssertFutureOptionContractOrder(orderEvent, security)
        else:
            raise AssertionError(f'Received order event for unknown Symbol: {orderEvent.Symbol}')
        self.Log(f'{orderEvent}')

    def AssertFutureOptionOrderExercise(self, orderEvent: OrderEvent, future: Security, optionContract: Security):
        if False:
            while True:
                i = 10
        if 'Assignment' in orderEvent.Message:
            if orderEvent.FillPrice != 3100.0:
                raise AssertionError('Option was not assigned at expected strike price (3100)')
            if orderEvent.Direction != OrderDirection.Sell or future.Holdings.Quantity != -1:
                raise AssertionError(f'Expected Qty: -1 futures holdings for assigned future {future.Symbol}, found {future.Holdings.Quantity}')
            return
        if orderEvent.Direction == OrderDirection.Buy and future.Holdings.Quantity != 0:
            raise AssertionError(f'Expected no holdings when liquidating future contract {future.Symbol}')

    def AssertFutureOptionContractOrder(self, orderEvent: OrderEvent, option: Security):
        if False:
            i = 10
            return i + 15
        if orderEvent.Direction == OrderDirection.Sell and option.Holdings.Quantity != -1:
            raise AssertionError(f'No holdings were created for option contract {option.Symbol}')
        if orderEvent.IsAssignment and option.Holdings.Quantity != 0:
            raise AssertionError(f'Holdings were found after option contract was assigned: {option.Symbol}')

    def OnEndOfAlgorithm(self):
        if False:
            print('Hello World!')
        if self.Portfolio.Invested:
            raise AssertionError(f"Expected no holdings at end of algorithm, but are invested in: {', '.join([str(i.ID) for i in self.Portfolio.Keys])}")