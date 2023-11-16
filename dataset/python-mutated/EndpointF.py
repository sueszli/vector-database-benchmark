from core import tarscore

class EndpointF(tarscore.struct):
    __tars_class__ = 'register.EndpointF'

    def __init__(self):
        if False:
            while True:
                i = 10
        self.host = ''
        self.port = 0
        self.timeout = 0
        self.istcp = 0
        self.grid = 0
        self.groupworkid = 0
        self.grouprealid = 0
        self.setId = ''
        self.qos = 0
        self.bakFlag = 0
        self.weight = 0
        self.weightType = 0

    @staticmethod
    def writeTo(oos, value):
        if False:
            for i in range(10):
                print('nop')
        oos.write(tarscore.string, 0, value.host)
        oos.write(tarscore.int32, 1, value.port)
        oos.write(tarscore.int32, 2, value.timeout)
        oos.write(tarscore.int32, 3, value.istcp)
        oos.write(tarscore.int32, 4, value.grid)
        oos.write(tarscore.int32, 5, value.groupworkid)
        oos.write(tarscore.int32, 6, value.grouprealid)
        oos.write(tarscore.string, 7, value.setId)
        oos.write(tarscore.int32, 8, value.qos)
        oos.write(tarscore.int32, 9, value.bakFlag)
        oos.write(tarscore.int32, 11, value.weight)
        oos.write(tarscore.int32, 12, value.weightType)

    @staticmethod
    def readFrom(ios):
        if False:
            return 10
        value = EndpointF()
        value.host = ios.read(tarscore.string, 0, True, value.host)
        value.port = ios.read(tarscore.int32, 1, True, value.port)
        value.timeout = ios.read(tarscore.int32, 2, True, value.timeout)
        value.istcp = ios.read(tarscore.int32, 3, True, value.istcp)
        value.grid = ios.read(tarscore.int32, 4, True, value.grid)
        value.groupworkid = ios.read(tarscore.int32, 5, False, value.groupworkid)
        value.grouprealid = ios.read(tarscore.int32, 6, False, value.grouprealid)
        value.setId = ios.read(tarscore.string, 7, False, value.setId)
        value.qos = ios.read(tarscore.int32, 8, False, value.qos)
        value.bakFlag = ios.read(tarscore.int32, 9, False, value.bakFlag)
        value.weight = ios.read(tarscore.int32, 11, False, value.weight)
        value.weightType = ios.read(tarscore.int32, 12, False, value.weightType)
        return value