from AlgorithmImports import *
import decimal

class CustomPythonData(PythonData):

    def GetSource(self, config, date, isLive):
        if False:
            print('Hello World!')
        source = Globals.DataFolder + '/equity/usa/daily/ibm.zip'
        return SubscriptionDataSource(source, SubscriptionTransportMedium.LocalFile, FileFormat.Csv)

    def Reader(self, config, line, date, isLive):
        if False:
            print('Hello World!')
        if line == None:
            return None
        customPythonData = CustomPythonData()
        customPythonData.Symbol = config.Symbol
        scaleFactor = 1 / 10000
        csv = line.split(',')
        customPythonData.Time = datetime.strptime(csv[0], '%Y%m%d %H:%M')
        customPythonData['Open'] = float(csv[1]) * scaleFactor
        customPythonData['High'] = float(csv[2]) * scaleFactor
        customPythonData['Low'] = float(csv[3]) * scaleFactor
        customPythonData['Close'] = float(csv[4]) * scaleFactor
        customPythonData['Volume'] = float(csv[5])
        return customPythonData

class Nifty(PythonData):
    """NIFTY Custom Data Class"""

    def GetSource(self, config, date, isLiveMode):
        if False:
            for i in range(10):
                print('nop')
        return SubscriptionDataSource('https://www.dropbox.com/s/rsmg44jr6wexn2h/CNXNIFTY.csv?dl=1', SubscriptionTransportMedium.RemoteFile)

    def Reader(self, config, line, date, isLiveMode):
        if False:
            i = 10
            return i + 15
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