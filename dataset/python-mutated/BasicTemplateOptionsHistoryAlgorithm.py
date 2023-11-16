from AlgorithmImports import *

class BasicTemplateOptionsHistoryAlgorithm(QCAlgorithm):
    """ This example demonstrates how to get access to options history for a given underlying equity security."""

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetStartDate(2015, 12, 24)
        self.SetEndDate(2015, 12, 24)
        self.SetCash(1000000)
        option = self.AddOption('GOOG')
        option.SetFilter(-2, +2, 0, 180)
        option.PriceModel = OptionPriceModels.CrankNicolsonFD()
        self.SetWarmUp(TimeSpan.FromDays(4))
        self.SetBenchmark(lambda x: 1000000)

    def OnData(self, slice):
        if False:
            return 10
        if self.IsWarmingUp:
            return
        if not self.Portfolio.Invested:
            for chain in slice.OptionChains:
                volatility = self.Securities[chain.Key.Underlying].VolatilityModel.Volatility
                for contract in chain.Value:
                    self.Log('{0},Bid={1} Ask={2} Last={3} OI={4} sigma={5:.3f} NPV={6:.3f}                               delta={7:.3f} gamma={8:.3f} vega={9:.3f} beta={10:.2f} theta={11:.2f} IV={12:.2f}'.format(contract.Symbol.Value, contract.BidPrice, contract.AskPrice, contract.LastPrice, contract.OpenInterest, volatility, contract.TheoreticalPrice, contract.Greeks.Delta, contract.Greeks.Gamma, contract.Greeks.Vega, contract.Greeks.Rho, contract.Greeks.Theta / 365, contract.ImpliedVolatility))

    def OnSecuritiesChanged(self, changes):
        if False:
            i = 10
            return i + 15
        for change in changes.AddedSecurities:
            if change.Symbol.Value == 'GOOG':
                return
            history = self.History(change.Symbol, 10, Resolution.Minute).sort_index(level='time', ascending=False)[:3]
            for (index, row) in history.iterrows():
                self.Log('History: ' + str(index[3]) + ': ' + index[4].strftime('%m/%d/%Y %I:%M:%S %p') + ' > ' + str(row.close))