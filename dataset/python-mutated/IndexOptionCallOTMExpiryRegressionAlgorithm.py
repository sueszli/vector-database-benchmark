from AlgorithmImports import *

class IndexOptionCallOTMExpiryRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        self.SetStartDate(2021, 1, 4)
        self.SetEndDate(2021, 1, 31)
        self.spx = self.AddIndex('SPX', Resolution.Minute).Symbol
        self.spxOption = list(self.OptionChainProvider.GetOptionContractList(self.spx, self.Time))
        self.spxOption = [i for i in self.spxOption if i.ID.StrikePrice >= 4250 and i.ID.OptionRight == OptionRight.Call and (i.ID.Date.year == 2021) and (i.ID.Date.month == 1)]
        self.spxOption = list(sorted(self.spxOption, key=lambda x: x.ID.StrikePrice))[0]
        self.spxOption = self.AddIndexOptionContract(self.spxOption, Resolution.Minute).Symbol
        self.expectedContract = Symbol.CreateOption(self.spx, Market.USA, OptionStyle.European, OptionRight.Call, 4250, datetime(2021, 1, 15))
        if self.spxOption != self.expectedContract:
            raise Exception(f'Contract {self.expectedContract} was not found in the chain')
        self.Schedule.On(self.DateRules.Tomorrow, self.TimeRules.AfterMarketOpen(self.spx, 1), lambda : self.MarketOrder(self.spxOption, 1))

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
            print('Hello World!')
        if orderEvent.Status != OrderStatus.Filled:
            return
        if orderEvent.Symbol not in self.Securities:
            raise Exception(f'Order event Symbol not found in Securities collection: {orderEvent.Symbol}')
        security = self.Securities[orderEvent.Symbol]
        if security.Symbol == self.spx:
            raise Exception('Invalid state: did not expect a position for the underlying to be opened, since this contract expires OTM')
        if security.Symbol == self.expectedContract:
            self.AssertIndexOptionContractOrder(orderEvent, security)
        else:
            raise Exception(f'Received order event for unknown Symbol: {orderEvent.Symbol}')

    def AssertIndexOptionContractOrder(self, orderEvent: OrderEvent, option: Security):
        if False:
            print('Hello World!')
        if orderEvent.Direction == OrderDirection.Buy and option.Holdings.Quantity != 1:
            raise Exception(f'No holdings were created for option contract {option.Symbol}')
        if orderEvent.Direction == OrderDirection.Sell and option.Holdings.Quantity != 0:
            raise Exception('Holdings were found after a filled option exercise')
        if orderEvent.Direction == OrderDirection.Sell and (not 'OTM' in orderEvent.Message):
            raise Exception('Contract did not expire OTM')
        if 'Exercise' in orderEvent.Message:
            raise Exception('Exercised option, even though it expires OTM')

    def OnEndOfAlgorithm(self):
        if False:
            print('Hello World!')
        if self.Portfolio.Invested:
            raise Exception(f"Expected no holdings at end of algorithm, but are invested in: {', '.join(self.Portfolio.Keys)}")