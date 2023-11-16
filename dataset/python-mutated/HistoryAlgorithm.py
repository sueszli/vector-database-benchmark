from AlgorithmImports import *

class HistoryAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        self.SetStartDate(2013, 10, 8)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        self.AddEquity('SPY', Resolution.Daily)
        self.AddData(CustomDataEquity, 'IBM', Resolution.Daily)
        self.dailySma = SimpleMovingAverage(14)
        tradeBarHistory = self.History([self.Securities['SPY'].Symbol], timedelta(365))
        self.AssertHistoryCount('History<TradeBar>(["SPY"], timedelta(365))', tradeBarHistory, 250)
        tradeBarHistory = self.History(['SPY'], timedelta(1), Resolution.Minute)
        self.AssertHistoryCount('History(["SPY"], timedelta(1), Resolution.Minute)', tradeBarHistory, 390)
        tradeBarHistory = self.History(['SPY'], 14)
        self.AssertHistoryCount('History(["SPY"], 14)', tradeBarHistory, 14)
        tradeBarHistory = self.History(['SPY'], 14, Resolution.Minute)
        self.AssertHistoryCount('History(["SPY"], 14, Resolution.Minute)', tradeBarHistory, 14)
        intervalBarHistory = self.History(['SPY'], self.Time - timedelta(1), self.Time, Resolution.Minute, True, True)
        self.AssertHistoryCount('History(["SPY"], self.Time - timedelta(1), self.Time, Resolution.Minute, True, True)', intervalBarHistory, 960)
        intervalBarHistory = self.History(['SPY'], self.Time - timedelta(1), self.Time, Resolution.Minute, False, True)
        self.AssertHistoryCount('History(["SPY"], self.Time - timedelta(1), self.Time, Resolution.Minute, False, True)', intervalBarHistory, 828)
        intervalBarHistory = self.History(['SPY'], self.Time - timedelta(1), self.Time, Resolution.Minute, True, False)
        self.AssertHistoryCount('History(["SPY"], self.Time - timedelta(1), self.Time, Resolution.Minute, True, False)', intervalBarHistory, 390)
        intervalBarHistory = self.History(['SPY'], self.Time - timedelta(1), self.Time, Resolution.Minute, False, False)
        self.AssertHistoryCount('History(["SPY"], self.Time - timedelta(1), self.Time, Resolution.Minute, False, False)', intervalBarHistory, 390)
        for (index, tradeBar) in tradeBarHistory.loc['SPY'].iterrows():
            self.dailySma.Update(index, tradeBar['close'])
        customDataHistory = self.History(CustomDataEquity, 'IBM', timedelta(365))
        self.AssertHistoryCount('History(CustomDataEquity, "IBM", timedelta(365))', customDataHistory, 10)
        customDataHistory = self.History(CustomDataEquity, 'IBM', 14)
        self.AssertHistoryCount('History(CustomDataEquity, "IBM", 14)', customDataHistory, 10)
        self.dailySma.Reset()
        for (index, customData) in customDataHistory.loc['IBM'].iterrows():
            self.dailySma.Update(index, customData['value'])
        allCustomData = self.History(CustomDataEquity, self.Securities.Keys, 14)
        self.AssertHistoryCount('History(CustomDataEquity, self.Securities.Keys, 14)', allCustomData, 20)
        allCustomData = self.History(CustomDataEquity, self.Securities.Keys, timedelta(365))
        self.AssertHistoryCount('History(CustomDataEquity, self.Securities.Keys, timedelta(365))', allCustomData, 20)
        singleSymbolCustom = allCustomData.loc['IBM']
        self.AssertHistoryCount('allCustomData.loc["IBM"]', singleSymbolCustom, 10)
        for customData in singleSymbolCustom:
            pass
        customDataSpyValues = allCustomData.loc['IBM']['value']
        self.AssertHistoryCount('allCustomData.loc["IBM"]["value"]', customDataSpyValues, 10)
        for value in customDataSpyValues:
            pass

    def OnData(self, data):
        if False:
            while True:
                i = 10
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n\n        Arguments:\n            data: Slice object keyed by symbol containing the stock data\n        '
        if not self.Portfolio.Invested:
            self.SetHoldings('SPY', 1)

    def AssertHistoryCount(self, methodCall, tradeBarHistory, expected):
        if False:
            return 10
        count = len(tradeBarHistory.index)
        if count != expected:
            raise Exception('{} expected {}, but received {}'.format(methodCall, expected, count))

class CustomDataEquity(PythonData):

    def GetSource(self, config, date, isLive):
        if False:
            print('Hello World!')
        source = 'https://www.dl.dropboxusercontent.com/s/o6ili2svndzn556/custom_data.csv?dl=0'
        return SubscriptionDataSource(source, SubscriptionTransportMedium.RemoteFile)

    def Reader(self, config, line, date, isLive):
        if False:
            while True:
                i = 10
        if line == None:
            return None
        customData = CustomDataEquity()
        customData.Symbol = config.Symbol
        csv = line.split(',')
        customData.Time = datetime.strptime(csv[0], '%Y%m%d %H:%M')
        customData.EndTime = customData.Time + timedelta(days=1)
        customData.Value = float(csv[1])
        return customData