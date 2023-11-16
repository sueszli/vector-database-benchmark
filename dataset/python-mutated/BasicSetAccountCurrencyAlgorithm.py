from AlgorithmImports import *

class BasicSetAccountCurrencyAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            while True:
                i = 10
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2018, 4, 4)
        self.SetEndDate(2018, 4, 4)
        self.SetBrokerageModel(BrokerageName.GDAX, AccountType.Cash)
        self.SetAccountCurrencyAndAmount()
        self._btcEur = self.AddCrypto('BTCEUR').Symbol

    def SetAccountCurrencyAndAmount(self):
        if False:
            return 10
        self.SetAccountCurrency('EUR')
        self.SetCash(100000)

    def OnData(self, data):
        if False:
            while True:
                i = 10
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n\n        Arguments:\n            data: Slice object keyed by symbol containing the stock data\n        '
        if not self.Portfolio.Invested:
            self.SetHoldings(self._btcEur, 1)