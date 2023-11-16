from AlgorithmImports import *

class NumeraiSignalExportDemonstrationAlgorithm(QCAlgorithm):
    securities = []

    def Initialize(self):
        if False:
            print('Hello World!')
        ' Initialize the date and add all equity symbols present in list _symbols '
        self.SetStartDate(2020, 10, 7)
        self.SetEndDate(2020, 10, 12)
        self.SetCash(100000)
        self.SetSecurityInitializer(BrokerageModelSecurityInitializer(self.BrokerageModel, FuncSecuritySeeder(self.GetLastKnownPrices)))
        self.etf_symbol = self.AddEquity('VTI').Symbol
        self.AddUniverse(self.Universe.ETF(self.etf_symbol))
        self.Schedule.On(self.DateRules.EveryDay(self.etf_symbol), self.TimeRules.At(13, 0, TimeZones.Utc), self.submit_signals)
        numerai_public_id = ''
        numerai_secret_id = ''
        numerai_model_id = ''
        numerai_filename = ''
        self.SignalExport.AddSignalExportProviders(NumeraiSignalExport(numerai_public_id, numerai_secret_id, numerai_model_id, numerai_filename))

    def submit_signals(self):
        if False:
            while True:
                i = 10
        symbols = sorted([security.Symbol for security in self.securities if security.HasData])
        if len(symbols) == 0:
            return
        denominator = len(symbols) * (len(symbols) + 1) / 2
        targets = [PortfolioTarget(symbol, (i + 1) / denominator) for (i, symbol) in enumerate(symbols)]
        self.SetHoldings(targets)
        success = self.SignalExport.SetTargetPortfolio(targets)
        if not success:
            self.Debug(f"Couldn't send targets at {self.Time}")

    def OnSecuritiesChanged(self, changes: SecurityChanges) -> None:
        if False:
            print('Hello World!')
        for security in changes.RemovedSecurities:
            if security in self.securities:
                self.securities.remove(security)
        self.securities.extend([security for security in changes.AddedSecurities if security.Symbol != self.etf_symbol])