from AlgorithmImports import *

class FineFundamentalFilteredUniverseRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            print('Hello World!')
        'Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'
        self.SetStartDate(2014, 10, 7)
        self.SetEndDate(2014, 10, 11)
        self.UniverseSettings.Resolution = Resolution.Daily
        symbol = Symbol(SecurityIdentifier.GenerateConstituentIdentifier('constituents-universe-qctest', SecurityType.Equity, Market.USA), 'constituents-universe-qctest')
        self.AddUniverse(ConstituentsUniverse(symbol, self.UniverseSettings), self.FineSelectionFunction)

    def FineSelectionFunction(self, fine):
        if False:
            while True:
                i = 10
        return [x.Symbol for x in fine if x.CompanyProfile != None and x.CompanyProfile.HeadquarterCity != None and (x.CompanyProfile.HeadquarterCity.lower() == 'cupertino')]

    def OnData(self, data):
        if False:
            print('Hello World!')
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n\n        Arguments:\n            data: Slice object keyed by symbol containing the stock data\n        '
        if not self.Portfolio.Invested:
            if data.Keys[0].Value != 'AAPL':
                raise ValueError(f'Unexpected symbol was added to the universe: {data.Keys[0]}')
            self.SetHoldings(data.Keys[0], 1)