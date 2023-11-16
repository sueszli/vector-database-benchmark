from AlgorithmImports import *

class FutureOptionCallITMExpiryRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetStartDate(2020, 1, 5)
        self.SetEndDate(2020, 6, 30)
        self.es19m20 = self.AddFutureContract(Symbol.CreateFuture(Futures.Indices.SP500EMini, Market.CME, datetime(2020, 6, 19)), Resolution.Minute).Symbol
        self.esOption = self.AddFutureOptionContract(list(sorted([x for x in self.OptionChainProvider.GetOptionContractList(self.es19m20, self.Time) if x.ID.StrikePrice <= 3200.0 and x.ID.OptionRight == OptionRight.Call], key=lambda x: x.ID.StrikePrice, reverse=True))[0], Resolution.Minute).Symbol
        self.expectedContract = Symbol.CreateOption(self.es19m20, Market.CME, OptionStyle.American, OptionRight.Call, 3200.0, datetime(2020, 6, 19))
        if self.esOption != self.expectedContract:
            raise AssertionError(f'Contract {self.expectedContract} was not found in the chain')
        self.Schedule.On(self.DateRules.Tomorrow, self.TimeRules.AfterMarketOpen(self.es19m20, 1), self.ScheduleCallback)

    def ScheduleCallback(self):
        if False:
            return 10
        self.MarketOrder(self.esOption, 1)

    def OnData(self, data: Slice):
        if False:
            while True:
                i = 10
        for delisting in data.Delistings.Values:
            if delisting.Type == DelistingType.Warning:
                if delisting.Time != datetime(2020, 6, 19):
                    raise AssertionError(f'Delisting warning issued at unexpected date: {delisting.Time}')
            elif delisting.Type == DelistingType.Delisted:
                if delisting.Time != datetime(2020, 6, 20):
                    raise AssertionError(f'Delisting happened at unexpected date: {delisting.Time}')

    def OnOrderEvent(self, orderEvent: OrderEvent):
        if False:
            i = 10
            return i + 15
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
        self.Log(f'{self.Time} -- {orderEvent.Symbol} :: Price: {self.Securities[orderEvent.Symbol].Holdings.Price} Qty: {self.Securities[orderEvent.Symbol].Holdings.Quantity} Direction: {orderEvent.Direction} Msg: {orderEvent.Message}')

    def AssertFutureOptionOrderExercise(self, orderEvent: OrderEvent, future: Security, optionContract: Security):
        if False:
            for i in range(10):
                print('nop')
        expectedLiquidationTimeUtc = datetime(2020, 6, 20, 4, 0, 0)
        if orderEvent.Direction == OrderDirection.Sell and future.Holdings.Quantity != 0:
            raise AssertionError(f'Did not liquidate existing holdings for Symbol {future.Symbol}')
        if orderEvent.Direction == OrderDirection.Sell and orderEvent.UtcTime.replace(tzinfo=None) != expectedLiquidationTimeUtc:
            raise AssertionError(f'Liquidated future contract, but not at the expected time. Expected: {expectedLiquidationTimeUtc} - found {orderEvent.UtcTime.replace(tzinfo=None)}')
        if 'Option Exercise' in orderEvent.Message:
            if orderEvent.FillPrice != 3200.0:
                raise AssertionError('Option did not exercise at expected strike price (3200)')
            if future.Holdings.Quantity != 1:
                raise AssertionError(f'Exercised option contract, but we have no holdings for Future {future.Symbol}')
            if optionContract.Holdings.Quantity != 0:
                raise AssertionError(f'Exercised option contract, but we have holdings for Option contract {optionContract.Symbol}')

    def AssertFutureOptionContractOrder(self, orderEvent: OrderEvent, option: Security):
        if False:
            return 10
        if orderEvent.Direction == OrderDirection.Buy and option.Holdings.Quantity != 1:
            raise AssertionError(f'No holdings were created for option contract {option.Symbol}')
        if orderEvent.Direction == OrderDirection.Sell and option.Holdings.Quantity != 0:
            raise AssertionError(f'Holdings were found after a filled option exercise')
        if 'Exercise' in orderEvent.Message and option.Holdings.Quantity != 0:
            raise AssertionError(f'Holdings were found after exercising option contract {option.Symbol}')

    def OnEndOfAlgorithm(self):
        if False:
            for i in range(10):
                print('nop')
        if self.Portfolio.Invested:
            raise AssertionError(f"Expected no holdings at end of algorithm, but are invested in: {', '.join([str(i.ID) for i in self.Portfolio.Keys])}")