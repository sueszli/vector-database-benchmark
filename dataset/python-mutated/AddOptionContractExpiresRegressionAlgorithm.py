from AlgorithmImports import *

class AddOptionContractExpiresRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2014, 6, 5)
        self.SetEndDate(2014, 6, 30)
        self._expiration = datetime(2014, 6, 21)
        self._option = None
        self._traded = False
        self._twx = Symbol.Create('TWX', SecurityType.Equity, Market.USA)
        self.AddUniverse('my-daily-universe-name', self.Selector)

    def Selector(self, time):
        if False:
            print('Hello World!')
        return ['AAPL']

    def OnData(self, data):
        if False:
            while True:
                i = 10
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n\n        Arguments:\n            data: Slice object keyed by symbol containing the stock data\n        '
        if self._option == None:
            options = self.OptionChainProvider.GetOptionContractList(self._twx, self.Time)
            options = sorted(options, key=lambda x: x.ID.Symbol)
            option = next((option for option in options if option.ID.Date == self._expiration and option.ID.OptionRight == OptionRight.Call and (option.ID.OptionStyle == OptionStyle.American)), None)
            if option != None:
                self._option = self.AddOptionContract(option).Symbol
        if self._option != None and self.Securities[self._option].Price != 0 and (not self._traded):
            self._traded = True
            self.Buy(self._option, 1)
        if self.Time > self._expiration and self.Securities[self._twx].Invested:
            self.Liquidate(self._twx)