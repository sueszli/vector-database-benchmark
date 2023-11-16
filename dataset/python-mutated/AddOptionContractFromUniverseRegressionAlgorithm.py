from AlgorithmImports import *

class AddOptionContractFromUniverseRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2014, 6, 5)
        self.SetEndDate(2014, 6, 9)
        self._expiration = datetime(2014, 6, 21)
        self._securityChanges = None
        self._option = None
        self._traded = False
        self._twx = Symbol.Create('TWX', SecurityType.Equity, Market.USA)
        self._aapl = Symbol.Create('AAPL', SecurityType.Equity, Market.USA)
        self.UniverseSettings.Resolution = Resolution.Minute
        self.UniverseSettings.DataNormalizationMode = DataNormalizationMode.Raw
        self.AddUniverse(self.Selector, self.Selector)

    def Selector(self, fundamental):
        if False:
            for i in range(10):
                print('nop')
        if self.Time <= datetime(2014, 6, 5):
            return [self._twx]
        return [self._aapl]

    def OnData(self, data):
        if False:
            for i in range(10):
                print('nop')
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n\n        Arguments:\n            data: Slice object keyed by symbol containing the stock data\n        '
        if self._option != None and self.Securities[self._option].Price != 0 and (not self._traded):
            self._traded = True
            self.Buy(self._option, 1)
        if self.Time == datetime(2014, 6, 6, 14, 0, 0):
            self.RemoveOptionContract(self._option)

    def OnSecuritiesChanged(self, changes):
        if False:
            print('Hello World!')
        if self._securityChanges == None:
            self._securityChanges = changes
        else:
            self._securityChanges.op_Addition(self._securityChanges, changes)
        if any((security.Symbol.SecurityType == SecurityType.Option for security in changes.AddedSecurities)):
            return
        for addedSecurity in changes.AddedSecurities:
            options = self.OptionChainProvider.GetOptionContractList(addedSecurity.Symbol, self.Time)
            options = sorted(options, key=lambda x: x.ID.Symbol)
            option = next((option for option in options if option.ID.Date == self._expiration and option.ID.OptionRight == OptionRight.Call and (option.ID.OptionStyle == OptionStyle.American)), None)
            self.AddOptionContract(option)
            if self._option == None:
                self._option = option