from AlgorithmImports import *

class IndexOptionShortCallITMExpiryRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetStartDate(2021, 1, 4)
        self.SetEndDate(2021, 1, 31)
        self.SetCash(1000000)
        self.Portfolio.SetMarginCallModel(MarginCallModel.Null)
        self.SetSecurityInitializer(CompositeSecurityInitializer(self.SecurityInitializer, FuncSecurityInitializer(self.CustomSecurityInitializer)))
        self.spx = self.AddIndex('SPX', Resolution.Minute).Symbol
        self.spxOption = list(self.OptionChainProvider.GetOptionContractList(self.spx, self.Time))
        self.spxOption = [i for i in self.spxOption if i.ID.StrikePrice <= 4200 and i.ID.OptionRight == OptionRight.Put and (i.ID.Date.year == 2021) and (i.ID.Date.month == 1)]
        self.spxOption = list(sorted(self.spxOption, key=lambda x: x.ID.StrikePrice, reverse=True))[0]
        self.spxOption = self.AddIndexOptionContract(self.spxOption, Resolution.Minute).Symbol
        self.expectedContract = Symbol.CreateOption(self.spx, Market.USA, OptionStyle.European, OptionRight.Put, 4200, datetime(2021, 1, 15))
        if self.spxOption != self.expectedContract:
            raise Exception(f'Contract self.expectedContract was not found in the chain')
        self.Schedule.On(self.DateRules.Tomorrow, self.TimeRules.AfterMarketOpen(self.spx, 1), lambda : self.MarketOrder(self.spxOption, -1))

    def OnData(self, data: Slice):
        if False:
            while True:
                i = 10
        for delisting in data.Delistings.Values:
            if delisting.Type == DelistingType.Warning:
                if delisting.Time != datetime(2021, 1, 15):
                    raise Exception(f'Delisting warning issued at unexpected date: {delisting.Time}')
            if delisting.Type == DelistingType.Delisted:
                if delisting.Time != datetime(2021, 1, 16):
                    raise Exception(f'Delisting happened at unexpected date: {delisting.Time}')

    def OnOrderEvent(self, orderEvent: OrderEvent):
        if False:
            for i in range(10):
                print('nop')
        if orderEvent.Status != OrderStatus.Filled:
            return
        if orderEvent.Symbol not in self.Securities:
            raise Exception(f'Order event Symbol not found in Securities collection: {orderEvent.Symbol}')
        security = self.Securities[orderEvent.Symbol]
        if security.Symbol == self.spx:
            self.AssertIndexOptionOrderExercise(orderEvent, security, self.Securities[self.expectedContract])
        elif security.Symbol == self.expectedContract:
            self.AssertIndexOptionContractOrder(orderEvent, security)
        else:
            raise Exception(f'Received order event for unknown Symbol: {orderEvent.Symbol}')

    def AssertIndexOptionOrderExercise(self, orderEvent: OrderEvent, index: Security, optionContract: Security):
        if False:
            for i in range(10):
                print('nop')
        if 'Assignment' in orderEvent.Message:
            if orderEvent.FillPrice != 4200:
                raise Exception('Option was not assigned at expected strike price (4200)')
            if orderEvent.Direction != OrderDirection.Sell or index.Holdings.Quantity != 0:
                raise Exception(f'Expected Qty: 0 index holdings for assigned index option {index.Symbol}, found {index.Holdings.Quantity}')
        elif index.Holdings.Quantity != 0:
            raise Exception(f'Expected no holdings in index: {index.Symbol}')

    def AssertIndexOptionContractOrder(self, orderEvent: OrderEvent, option: Security):
        if False:
            i = 10
            return i + 15
        if orderEvent.Direction == OrderDirection.Sell and option.Holdings.Quantity != -1:
            raise Exception(f'No holdings were created for option contract {option.Symbol}')
        if orderEvent.IsAssignment and option.Holdings.Quantity != 0:
            raise Exception(f'Holdings were found after option contract was assigned: {option.Symbol}')

    def OnEndOfAlgorithm(self):
        if False:
            for i in range(10):
                print('nop')
        if self.Portfolio.Invested:
            raise Exception(f"Expected no holdings at end of algorithm, but are invested in: {', '.join(self.Portfolio.Keys)}")

    def CustomSecurityInitializer(self, security):
        if False:
            return 10
        if Extensions.IsOption(security.Symbol.SecurityType):
            security.SetOptionAssignmentModel(NullOptionAssignmentModel())