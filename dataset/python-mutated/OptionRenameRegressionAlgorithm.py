from AlgorithmImports import *

class OptionRenameRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetCash(1000000)
        self.SetStartDate(2013, 6, 28)
        self.SetEndDate(2013, 7, 2)
        option = self.AddOption('TFCFA')
        option.SetFilter(-1, 1, timedelta(0), timedelta(3650))
        self.SetBenchmark('TFCFA')

    def OnData(self, slice):
        if False:
            print('Hello World!')
        ' Event - v3.0 DATA EVENT HANDLER: (Pattern) Basic template for user to override for receiving all subscription data in a single event\n        <param name="slice">The current slice of data keyed by symbol string</param> '
        if not self.Portfolio.Invested:
            for kvp in slice.OptionChains:
                chain = kvp.Value
                if self.Time.day == 28 and self.Time.hour > 9 and (self.Time.minute > 0):
                    contracts = [i for i in sorted(chain, key=lambda x: x.Expiry) if i.Right == OptionRight.Call and i.Strike == 33 and (i.Expiry.date() == datetime(2013, 8, 17).date())]
                    if contracts:
                        contract = contracts[0]
                        self.Buy(contract.Symbol, 1)
                        underlyingSymbol = contract.Symbol.Underlying
                        self.Buy(underlyingSymbol, 100)
                        if float(contract.AskPrice) != 1.1:
                            raise ValueError('Regression test failed: current ask price was not loaded from NWSA backtest file and is not $1.1')
        elif self.Time.day == 2 and self.Time.hour > 14 and (self.Time.minute > 0):
            for kvp in slice.OptionChains:
                chain = kvp.Value
                self.Liquidate()
                contracts = [i for i in sorted(chain, key=lambda x: x.Expiry) if i.Right == OptionRight.Call and i.Strike == 33 and (i.Expiry.date() == datetime(2013, 8, 17).date())]
            if contracts:
                contract = contracts[0]
                self.Log('Bid Price' + str(contract.BidPrice))
                if float(contract.BidPrice) != 0.05:
                    raise ValueError('Regression test failed: current bid price was not loaded from FOXA file and is not $0.05')

    def OnOrderEvent(self, orderEvent):
        if False:
            for i in range(10):
                print('nop')
        self.Log(str(orderEvent))