from AlgorithmImports import *
from custom_data import *

class Test_CustomDataAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            return 10
        self.AddData(Nifty, 'NIFTY')
        self.AddData(CustomPythonData, 'IBM', Resolution.Daily)

class Nifty(PythonData):
    """NIFTY Custom Data Class"""

    def GetSource(self, config, date, isLiveMode):
        if False:
            for i in range(10):
                print('nop')
        return SubscriptionDataSource('https://www.dropbox.com/s/rsmg44jr6wexn2h/CNXNIFTY.csv?dl=1', SubscriptionTransportMedium.RemoteFile)

    def Reader(self, config, line, date, isLiveMode):
        if False:
            for i in range(10):
                print('nop')
        if not (line.strip() and line[0].isdigit()):
            return None
        index = Nifty()
        index.Symbol = config.Symbol
        try:
            data = line.split(',')
            index.Time = datetime.strptime(data[0], '%Y-%m-%d')
            index.Value = decimal.Decimal(data[4])
            index['Open'] = float(data[1])
            index['High'] = float(data[2])
            index['Low'] = float(data[3])
            index['Close'] = float(data[4])
        except ValueError:
            return None
        return index