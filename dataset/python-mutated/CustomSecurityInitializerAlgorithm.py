from AlgorithmImports import *

class CustomSecurityInitializerAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage)
        func_security_seeder = FuncSecuritySeeder(Func[Security, BaseData](self.custom_seed_function))
        self.SetSecurityInitializer(CustomSecurityInitializer(self.BrokerageModel, func_security_seeder, DataNormalizationMode.Raw))
        self.SetStartDate(2013, 10, 1)
        self.SetEndDate(2013, 11, 1)
        self.AddEquity('SPY', Resolution.Hour)

    def OnData(self, data):
        if False:
            i = 10
            return i + 15
        if not self.Portfolio.Invested:
            self.SetHoldings('SPY', 1)

    def custom_seed_function(self, security):
        if False:
            return 10
        resolution = Resolution.Hour
        df = self.History(security.Symbol, 1, resolution)
        if df.empty:
            return None
        last_bar = df.unstack(level=0).iloc[-1]
        date_time = last_bar.name.to_pydatetime()
        open = last_bar.open.values[0]
        high = last_bar.high.values[0]
        low = last_bar.low.values[0]
        close = last_bar.close.values[0]
        volume = last_bar.volume.values[0]
        return TradeBar(date_time, security.Symbol, open, high, low, close, volume, Extensions.ToTimeSpan(resolution))

class CustomSecurityInitializer(BrokerageModelSecurityInitializer):
    """Our custom initializer that will set the data normalization mode.
    We sub-class the BrokerageModelSecurityInitializer so we can also
    take advantage of the default model/leverage setting behaviors"""

    def __init__(self, brokerageModel, securitySeeder, dataNormalizationMode):
        if False:
            return 10
        'Initializes a new instance of the CustomSecurityInitializer class with the specified normalization mode\n        brokerageModel -- The brokerage model used to get fill/fee/slippage/settlement models\n        securitySeeder -- The security seeder to be used\n        dataNormalizationMode -- The desired data normalization mode'
        self.base = BrokerageModelSecurityInitializer(brokerageModel, securitySeeder)
        self.dataNormalizationMode = dataNormalizationMode

    def Initialize(self, security):
        if False:
            print('Hello World!')
        'Initializes the specified security by setting up the models\n        security -- The security to be initialized\n        seedSecurity -- True to seed the security, false otherwise'
        self.base.Initialize(security)
        security.SetDataNormalizationMode(self.dataNormalizationMode)