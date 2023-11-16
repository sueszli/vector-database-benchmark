from AlgorithmImports import *

class FutureOptionCallOTMExpiryRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetStartDate(2020, 1, 5)
        self.SetEndDate(2020, 6, 30)
        self.es19m20 = self.AddFutureContract(Symbol.CreateFuture(Futures.Indices.SP500EMini, Market.CME, datetime(2020, 6, 19)), Resolution.Minute).Symbol
        self.esOption = self.AddFutureOptionContract(list(sorted([x for x in self.OptionChainProvider.GetOptionContractList(self.es19m20, self.Time) if x.ID.StrikePrice >= 3300.0 and x.ID.OptionRight == OptionRight.Call], key=lambda x: x.ID.StrikePrice))[0], Resolution.Minute).Symbol
        self.expectedContract = Symbol.CreateOption(self.es19m20, Market.CME, OptionStyle.American, OptionRight.Call, 3300.0, datetime(2020, 6, 19))
        if self.esOption != self.expectedContract:
            raise AssertionError(f'Contract {self.expectedContract} was not found in the chain')
        self.Schedule.On(self.DateRules.Tomorrow, self.TimeRules.AfterMarketOpen(self.es19m20, 1), self.ScheduledMarketOrder)

    def ScheduledMarketOrder(self):
        if False:
            i = 10
            return i + 15
        self.MarketOrder(self.esOption, 1)

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
            return 10
        if orderEvent.Status != OrderStatus.Filled:
            return
        if not self.Securities.ContainsKey(orderEvent.Symbol):
            raise AssertionError(f'Order event Symbol not found in Securities collection: {orderEvent.Symbol}')
        security = self.Securities[orderEvent.Symbol]
        if security.Symbol == self.es19m20:
            raise AssertionError('Invalid state: did not expect a position for the underlying to be opened, since this contract expires OTM')
        if security.Symbol == self.expectedContract:
            self.AssertFutureOptionContractOrder(orderEvent, security)
        else:
            raise AssertionError(f'Received order event for unknown Symbol: {orderEvent.Symbol}')
        self.Log(f'{orderEvent}')

    def AssertFutureOptionContractOrder(self, orderEvent: OrderEvent, option: Security):
        if False:
            print('Hello World!')
        if orderEvent.Direction == OrderDirection.Buy and option.Holdings.Quantity != 1:
            raise AssertionError(f'No holdings were created for option contract {option.Symbol}')
        if orderEvent.Direction == OrderDirection.Sell and option.Holdings.Quantity != 0:
            raise AssertionError('Holdings were found after a filled option exercise')
        if orderEvent.Direction == OrderDirection.Sell and 'OTM' not in orderEvent.Message:
            raise AssertionError('Contract did not expire OTM')
        if 'Exercise' in orderEvent.Message:
            raise AssertionError('Exercised option, even though it expires OTM')

    def OnEndOfAlgorithm(self):
        if False:
            while True:
                i = 10
        if self.Portfolio.Invested:
            raise AssertionError(f"Expected no holdings at end of algorithm, but are invested in: {', '.join([str(i.ID) for i in self.Portfolio.Keys])}")