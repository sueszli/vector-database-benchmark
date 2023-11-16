from AlgorithmImports import *

class OptionChainProviderAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetStartDate(2015, 12, 24)
        self.SetEndDate(2015, 12, 24)
        self.SetCash(100000)
        self.equity = self.AddEquity('GOOG', Resolution.Minute)
        self.equity.SetDataNormalizationMode(DataNormalizationMode.Raw)
        self.contract = str()
        self.contractsAdded = set()

    def OnData(self, data):
        if False:
            return 10
        if not self.Portfolio[self.equity.Symbol].Invested:
            self.MarketOrder(self.equity.Symbol, 100)
        if not (self.Securities.ContainsKey(self.contract) and self.Portfolio[self.contract].Invested):
            self.contract = self.OptionsFilter(data)
        if self.Securities.ContainsKey(self.contract) and (not self.Portfolio[self.contract].Invested):
            self.MarketOrder(self.contract, -1)

    def OptionsFilter(self, data):
        if False:
            for i in range(10):
                print('nop')
        ' OptionChainProvider gets a list of option contracts for an underlying symbol at requested date.\n            Then you can manually filter the contract list returned by GetOptionContractList.\n            The manual filtering will be limited to the information included in the Symbol\n            (strike, expiration, type, style) and/or prices from a History call '
        contracts = self.OptionChainProvider.GetOptionContractList(self.equity.Symbol, data.Time)
        self.underlyingPrice = self.Securities[self.equity.Symbol].Price
        otm_calls = [i for i in contracts if i.ID.OptionRight == OptionRight.Call and i.ID.StrikePrice - self.underlyingPrice > 0 and (10 < (i.ID.Date - data.Time).days < 30)]
        if len(otm_calls) > 0:
            contract = sorted(sorted(otm_calls, key=lambda x: x.ID.Date), key=lambda x: x.ID.StrikePrice - self.underlyingPrice)[0]
            if contract not in self.contractsAdded:
                self.contractsAdded.add(contract)
                self.AddOptionContract(contract, Resolution.Minute)
            return contract
        else:
            return str()