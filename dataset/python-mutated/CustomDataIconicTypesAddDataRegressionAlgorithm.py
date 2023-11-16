from AlgorithmImports import *
from QuantConnect.Data.Custom.IconicTypes import *

class CustomDataIconicTypesAddDataRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)
        twxEquity = self.AddEquity('TWX', Resolution.Daily).Symbol
        customTwxSymbol = self.AddData(LinkedData, twxEquity, Resolution.Daily).Symbol
        self.googlEquity = self.AddEquity('GOOGL', Resolution.Daily).Symbol
        customGooglSymbol = self.AddData(LinkedData, 'GOOGL', Resolution.Daily).Symbol
        unlinkedDataSymbol = self.AddData(UnlinkedData, 'GOOGL', Resolution.Daily).Symbol
        unlinkedDataSymbolUnderlyingEquity = Symbol.Create('MSFT', SecurityType.Equity, Market.USA)
        unlinkedDataSymbolUnderlying = self.AddData(UnlinkedData, unlinkedDataSymbolUnderlyingEquity, Resolution.Daily).Symbol
        optionSymbol = self.AddOption('TWX', Resolution.Minute).Symbol
        customOptionSymbol = self.AddData(LinkedData, optionSymbol, Resolution.Daily).Symbol
        if customTwxSymbol.Underlying != twxEquity:
            raise Exception(f'Underlying symbol for {customTwxSymbol} is not equal to TWX equity. Expected {twxEquity} got {customTwxSymbol.Underlying}')
        if customGooglSymbol.Underlying != self.googlEquity:
            raise Exception(f'Underlying symbol for {customGooglSymbol} is not equal to GOOGL equity. Expected {self.googlEquity} got {customGooglSymbol.Underlying}')
        if unlinkedDataSymbol.HasUnderlying:
            raise Exception(f"Unlinked data type (no underlying) has underlying when it shouldn't. Found {unlinkedDataSymbol.Underlying}")
        if not unlinkedDataSymbolUnderlying.HasUnderlying:
            raise Exception('Unlinked data type (with underlying) has no underlying Symbol even though we added with Symbol')
        if unlinkedDataSymbolUnderlying.Underlying != unlinkedDataSymbolUnderlyingEquity:
            raise Exception(f'Unlinked data type underlying does not equal equity Symbol added. Expected {unlinkedDataSymbolUnderlyingEquity} got {unlinkedDataSymbolUnderlying.Underlying}')
        if customOptionSymbol.Underlying != optionSymbol:
            raise Exception(f'Option symbol not equal to custom underlying symbol. Expected {optionSymbol} got {customOptionSymbol.Underlying}')
        try:
            customDataNoCache = self.AddData(LinkedData, 'AAPL', Resolution.Daily)
            raise Exception('AAPL was found in the SymbolCache, though it should be missing')
        except InvalidOperationException as e:
            return

    def OnData(self, data):
        if False:
            for i in range(10):
                print('nop')
        'OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.\n\n        Arguments:\n            data: Slice object keyed by symbol containing the stock data\n        '
        if not self.Portfolio.Invested and len(self.Transactions.GetOpenOrders()) == 0:
            self.SetHoldings(self.googlEquity, 0.5)