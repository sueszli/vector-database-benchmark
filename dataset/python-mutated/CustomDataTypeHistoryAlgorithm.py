from AlgorithmImports import *

class CustomDataTypeHistoryAlgorithm(QCAlgorithm):

    def Initialize(self):
        if False:
            i = 10
            return i + 15
        self.SetStartDate(2017, 8, 20)
        self.SetEndDate(2017, 8, 20)
        self.symbol = self.AddData(CustomDataType, 'CustomDataType', Resolution.Hour).Symbol
        history = list(self.History[CustomDataType](self.symbol, 48, Resolution.Hour))
        if len(history) == 0:
            raise Exception('History request returned no data')
        self._assertHistoryData(history)
        history2 = list(self.History[CustomDataType]([self.symbol], 48, Resolution.Hour))
        if len(history2) != len(history):
            raise Exception('History requests returned different data')
        self._assertHistoryData([y.values()[0] for y in history2])

    def _assertHistoryData(self, history: List[PythonData]) -> None:
        if False:
            for i in range(10):
                print('nop')
        expectedKeys = ['open', 'close', 'high', 'low', 'some_property']
        if any((any((not x[key] for key in expectedKeys)) or x['some_property'] != 'some property value' for x in history)):
            raise Exception('History request returned data without the expected properties')

class CustomDataType(PythonData):

    def GetSource(self, config: SubscriptionDataConfig, date: datetime, isLive: bool) -> SubscriptionDataSource:
        if False:
            while True:
                i = 10
        source = 'https://www.dl.dropboxusercontent.com/s/d83xvd7mm9fzpk0/path_to_my_csv_data.csv?dl=0'
        return SubscriptionDataSource(source, SubscriptionTransportMedium.RemoteFile)

    def Reader(self, config: SubscriptionDataConfig, line: str, date: datetime, isLive: bool) -> BaseData:
        if False:
            while True:
                i = 10
        if not line.strip():
            return None
        data = line.split(',')
        obj_data = CustomDataType()
        obj_data.Symbol = config.Symbol
        try:
            obj_data.Time = datetime.strptime(data[0], '%Y-%m-%d %H:%M:%S') + timedelta(hours=20)
            obj_data['open'] = float(data[1])
            obj_data['high'] = float(data[2])
            obj_data['low'] = float(data[3])
            obj_data['close'] = float(data[4])
            obj_data.Value = obj_data['close']
            obj_data['some_property'] = 'some property value'
        except ValueError:
            return None
        return obj_data