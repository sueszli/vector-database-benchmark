from AlgorithmImports import *

class DelistingEventsAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2007, 5, 16)
        self.SetEndDate(2007, 5, 25)
        self.SetCash(100000)
        self.AddEquity('AAA.1', Resolution.Daily)
        self.AddEquity('SPY', Resolution.Daily)

    def OnData(self, data):
        if False:
            while True:
                i = 10
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n\n        Arguments:\n            data: Slice object keyed by symbol containing the stock data\n        '
        if self.Transactions.OrdersCount == 0:
            self.SetHoldings('AAA.1', 1)
            self.Debug('Purchased stock')
        for kvp in data.Bars:
            symbol = kvp.Key
            value = kvp.Value
            self.Log('OnData(Slice): {0}: {1}: {2}'.format(self.Time, symbol, value.Close))
        aaa = self.Securities['AAA.1']
        if aaa.IsDelisted and aaa.IsTradable:
            raise Exception('Delisted security must NOT be tradable')
        if not aaa.IsDelisted and (not aaa.IsTradable):
            raise Exception("Securities must be marked as tradable until they're delisted or removed from the universe")
        for kvp in data.Delistings:
            symbol = kvp.Key
            value = kvp.Value
            if value.Type == DelistingType.Warning:
                self.Log('OnData(Delistings): {0}: {1} will be delisted at end of day today.'.format(self.Time, symbol))
                self.SetHoldings(symbol, 0)
            if value.Type == DelistingType.Delisted:
                self.Log('OnData(Delistings): {0}: {1} has been delisted.'.format(self.Time, symbol))
                self.SetHoldings(symbol, 1)

    def OnOrderEvent(self, orderEvent):
        if False:
            i = 10
            return i + 15
        self.Log('OnOrderEvent(OrderEvent): {0}: {1}'.format(self.Time, orderEvent))