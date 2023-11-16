from .__util import util

class RequestPacket(util.struct):
    mapcls_context = util.mapclass(util.string, util.string)
    mapcls_status = util.mapclass(util.string, util.string)

    def __init__(self):
        if False:
            while True:
                i = 10
        self.iVersion = 0
        self.cPacketType = 0
        self.iMessageType = 0
        self.iRequestId = 0
        self.sServantName = ''
        self.sFuncName = ''
        self.sBuffer = bytes()
        self.iTimeout = 0
        self.context = RequestPacket.mapcls_context()
        self.status = RequestPacket.mapcls_status()

    @staticmethod
    def writeTo(oos, value):
        if False:
            i = 10
            return i + 15
        oos.write(util.int16, 1, value.iVersion)
        oos.write(util.int8, 2, value.cPacketType)
        oos.write(util.int32, 3, value.iMessageType)
        oos.write(util.int32, 4, value.iRequestId)
        oos.write(util.string, 5, value.sServantName)
        oos.write(util.string, 6, value.sFuncName)
        oos.write(util.bytes, 7, value.sBuffer)
        oos.write(util.int32, 8, value.iTimeout)
        oos.write(RequestPacket.mapcls_context, 9, value.context)
        oos.write(RequestPacket.mapcls_status, 10, value.status)

    @staticmethod
    def readFrom(ios):
        if False:
            print('Hello World!')
        value = RequestPacket()
        value.iVersion = ios.read(util.int16, 1, True, 0)
        print('iVersion = %d' % value.iVersion)
        value.cPacketType = ios.read(util.int8, 2, True, 0)
        print('cPackerType = %d' % value.cPacketType)
        value.iMessageType = ios.read(util.int32, 3, True, 0)
        print('iMessageType = %d' % value.iMessageType)
        value.iRequestId = ios.read(util.int32, 4, True, 0)
        print('iRequestId = %d' % value.iRequestId)
        value.sServantName = ios.read(util.string, 5, True, '22222222')
        value.sFuncName = ios.read(util.string, 6, True, '')
        value.sBuffer = ios.read(util.bytes, 7, True, value.sBuffer)
        value.iTimeout = ios.read(util.int32, 8, True, 0)
        value.context = ios.read(RequestPacket.mapcls_context, 9, True, value.context)
        value.status = ios.read(RequestPacket.mapcls_status, 10, True, value.status)
        return value

class ResponsePacket(util.struct):
    __tars_class__ = 'tars.RpcMessage.ResponsePacket'
    mapcls_status = util.mapclass(util.string, util.string)

    def __init__(self):
        if False:
            return 10
        self.iVersion = 0
        self.cPacketType = 0
        self.iRequestId = 0
        self.iMessageType = 0
        self.iRet = 0
        self.sBuffer = bytes()
        self.status = RequestPacket.mapcls_status()

    @staticmethod
    def writeTo(oos, value):
        if False:
            print('Hello World!')
        oos.write(util.int16, 1, value.iVersion)
        oos.write(util.int8, 2, value.cPacketType)
        oos.write(util.int32, 3, value.iRequestId)
        oos.write(util.int32, 4, value.iMessageType)
        oos.write(util.int32, 5, value.iRet)
        oos.write(util.bytes, 6, value.sBuffer)
        oos.write(value.mapcls_status, 7, value.status)

    @staticmethod
    def readFrom(ios):
        if False:
            while True:
                i = 10
        value = ResponsePacket()
        value.iVersion = ios.read(util.int16, 1, True)
        value.cPacketType = ios.read(util.int8, 2, True)
        value.iRequestId = ios.read(util.int32, 3, True)
        value.iMessageType = ios.read(util.int32, 4, True)
        value.iRet = ios.read(util.int32, 5, True)
        value.sBuffer = ios.read(util.bytes, 6, True)
        value.status = ios.read(value.mapcls_status, 7, True)
        return value