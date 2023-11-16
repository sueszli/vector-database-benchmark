from AlgorithmImports import *

class CrunchDAOSignalExportDemonstrationAlgorithm(QCAlgorithm):
    crunch_universe = []

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetStartDate(2023, 5, 22)
        self.SetEndDate(2023, 5, 26)
        self.SetCash(1000000)
        api_key = ''
        model = ''
        submission_name = ''
        comment = ''
        self.SignalExport.AddSignalExportProviders(CrunchDAOSignalExport(api_key, model, submission_name, comment))
        self.SetSecurityInitializer(BrokerageModelSecurityInitializer(self.BrokerageModel, FuncSecuritySeeder(self.GetLastKnownPrices)))
        self.AddUniverse(CrunchDaoSkeleton, 'CrunchDaoSkeleton', Resolution.Daily, self.select_symbols)
        self.week = -1
        self.Schedule.On(self.DateRules.Every([DayOfWeek.Monday, DayOfWeek.Tuesday, DayOfWeek.Wednesday, DayOfWeek.Thursday, DayOfWeek.Friday]), self.TimeRules.At(13, 15, TimeZones.Utc), self.submit_signals)
        self.Settings.MinimumOrderMarginPortfolioPercentage = 0
        self.SetWarmUp(timedelta(45))

    def select_symbols(self, data: List[CrunchDaoSkeleton]) -> List[Symbol]:
        if False:
            for i in range(10):
                print('nop')
        return [x.Symbol for x in data]

    def OnSecuritiesChanged(self, changes):
        if False:
            i = 10
            return i + 15
        for security in changes.RemovedSecurities:
            if security in self.crunch_universe:
                self.crunch_universe.remove(security)
        self.crunch_universe.extend(changes.AddedSecurities)

    def submit_signals(self):
        if False:
            for i in range(10):
                print('nop')
        if self.IsWarmingUp:
            return
        week_num = self.Time.isocalendar()[1]
        if self.week == week_num:
            return
        self.week = week_num
        symbols = [security.Symbol for security in self.crunch_universe if security.Price > 0]
        weight_by_symbol = {symbol: 1 / len(symbols) for symbol in symbols}
        targets = [PortfolioTarget(symbol, weight) for (symbol, weight) in weight_by_symbol.items()]
        self.SetHoldings(targets)
        success = self.SignalExport.SetTargetPortfolio(targets)
        if not success:
            self.Debug(f"Couldn't send targets at {self.Time}")

class CrunchDaoSkeleton(PythonData):

    def GetSource(self, config, date, isLive):
        if False:
            print('Hello World!')
        return SubscriptionDataSource('https://tournament.crunchdao.com/data/skeleton.csv', SubscriptionTransportMedium.RemoteFile)

    def Reader(self, config, line, date, isLive):
        if False:
            for i in range(10):
                print('nop')
        if not line[0].isdigit():
            return None
        skeleton = CrunchDaoSkeleton()
        skeleton.Symbol = config.Symbol
        try:
            csv = line.split(',')
            skeleton.EndTime = datetime.strptime(csv[0], '%Y-%m-%d').date()
            skeleton.Symbol = Symbol(SecurityIdentifier.GenerateEquity(csv[1], Market.USA, mappingResolveDate=skeleton.Time), csv[1])
            skeleton['Ticker'] = csv[1]
        except ValueError:
            return None
        return skeleton