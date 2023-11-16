from AlgorithmImports import *

class CustomDataBenchmarkRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetStartDate(2017, 8, 18)
        self.SetEndDate(2017, 8, 21)
        self.SetCash(100000)
        self.AddEquity('SPY', Resolution.Hour)
        self.customSymbol = self.AddData(ExampleCustomData, 'ExampleCustomData', Resolution.Hour).Symbol
        self.SetBenchmark(self.customSymbol)

    def OnData(self, data):
        if False:
            i = 10
            return i + 15
        if not self.Portfolio.Invested:
            self.SetHoldings('SPY', 1)

    def OnEndOfAlgorithm(self):
        if False:
            for i in range(10):
                print('nop')
        securityBenchmark = self.Benchmark
        if securityBenchmark.Security.Price == 0:
            raise Exception('Security benchmark price was not expected to be zero')

class ExampleCustomData(PythonData):

    def GetSource(self, config, date, isLive):
        if False:
            while True:
                i = 10
        source = 'https://www.dl.dropboxusercontent.com/s/d83xvd7mm9fzpk0/path_to_my_csv_data.csv?dl=0'
        return SubscriptionDataSource(source, SubscriptionTransportMedium.RemoteFile)

    def Reader(self, config, line, date, isLive):
        if False:
            return 10
        data = line.split(',')
        obj_data = ExampleCustomData()
        obj_data.Symbol = config.Symbol
        obj_data.Time = datetime.strptime(data[0], '%Y-%m-%d %H:%M:%S') + timedelta(hours=20)
        obj_data.Value = float(data[4])
        obj_data['Open'] = float(data[1])
        obj_data['High'] = float(data[2])
        obj_data['Low'] = float(data[3])
        obj_data['Close'] = float(data[4])
        return obj_data