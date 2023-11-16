from AlgorithmImports import *
from CustomDataObjectStoreRegressionAlgorithm import *

class CustomDataZippedObjectStoreRegressionAlgorithm(CustomDataObjectStoreRegressionAlgorithm):

    def GetCustomDataKey(self):
        if False:
            while True:
                i = 10
        return 'CustomData/ExampleCustomData.zip'

    def SaveDataToObjectStore(self):
        if False:
            print('Hello World!')
        self.ObjectStore.SaveBytes(self.GetCustomDataKey(), Compression.ZipBytes(bytes(self.CustomData, 'utf-8'), 'data'))