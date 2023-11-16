# QUANTCONNECT.COM - Democratizing Finance, Empowering Individuals.
# Lean Algorithmic Trading Engine v2.0. Copyright 2014 QuantConnect Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from AlgorithmImports import *
from QuantConnect.Data.Custom.IconicTypes import *

### <summary>
### Regression algorithm checks that adding data via AddData
### works as expected
### </summary>
class CustomDataIconicTypesAddDataRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 11)
        self.SetCash(100000)

        twxEquity = self.AddEquity("TWX", Resolution.Daily).Symbol
        customTwxSymbol = self.AddData(LinkedData, twxEquity, Resolution.Daily).Symbol

        self.googlEquity = self.AddEquity("GOOGL", Resolution.Daily).Symbol
        customGooglSymbol = self.AddData(LinkedData, "GOOGL", Resolution.Daily).Symbol

        unlinkedDataSymbol = self.AddData(UnlinkedData, "GOOGL", Resolution.Daily).Symbol
        unlinkedDataSymbolUnderlyingEquity = Symbol.Create("MSFT", SecurityType.Equity, Market.USA)
        unlinkedDataSymbolUnderlying = self.AddData(UnlinkedData, unlinkedDataSymbolUnderlyingEquity, Resolution.Daily).Symbol

        optionSymbol = self.AddOption("TWX", Resolution.Minute).Symbol
        customOptionSymbol = self.AddData(LinkedData, optionSymbol, Resolution.Daily).Symbol

        if customTwxSymbol.Underlying != twxEquity:
            raise Exception(f"Underlying symbol for {customTwxSymbol} is not equal to TWX equity. Expected {twxEquity} got {customTwxSymbol.Underlying}")
        if customGooglSymbol.Underlying != self.googlEquity:
            raise Exception(f"Underlying symbol for {customGooglSymbol} is not equal to GOOGL equity. Expected {self.googlEquity} got {customGooglSymbol.Underlying}")
        if unlinkedDataSymbol.HasUnderlying:
            raise Exception(f"Unlinked data type (no underlying) has underlying when it shouldn't. Found {unlinkedDataSymbol.Underlying}")
        if not unlinkedDataSymbolUnderlying.HasUnderlying:
            raise Exception("Unlinked data type (with underlying) has no underlying Symbol even though we added with Symbol")
        if unlinkedDataSymbolUnderlying.Underlying != unlinkedDataSymbolUnderlyingEquity:
            raise Exception(f"Unlinked data type underlying does not equal equity Symbol added. Expected {unlinkedDataSymbolUnderlyingEquity} got {unlinkedDataSymbolUnderlying.Underlying}")
        if customOptionSymbol.Underlying != optionSymbol:
            raise Exception(f"Option symbol not equal to custom underlying symbol. Expected {optionSymbol} got {customOptionSymbol.Underlying}")

        try:
            customDataNoCache = self.AddData(LinkedData, "AAPL", Resolution.Daily)
            raise Exception("AAPL was found in the SymbolCache, though it should be missing")
        except InvalidOperationException as e:
            return

    def OnData(self, data):
        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.

        Arguments:
            data: Slice object keyed by symbol containing the stock data
        '''
        if not self.Portfolio.Invested and len(self.Transactions.GetOpenOrders()) == 0:
            self.SetHoldings(self.googlEquity, 0.5)
