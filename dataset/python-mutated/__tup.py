import struct
import string
from .__util import util
from .__tars import TarsOutputStream
from .__tars import TarsInputStream
from .__packet import RequestPacket

class TarsUniPacket(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.__mapa = util.mapclass(util.string, util.bytes)
        self.__mapv = util.mapclass(util.string, self.__mapa)
        self.__buffer = self.__mapv()
        self.__code = RequestPacket()

    @property
    def servant(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__code.sServantName

    @servant.setter
    def servant(self, value):
        if False:
            i = 10
            return i + 15
        self.__code.sServantName = value

    @property
    def func(self):
        if False:
            i = 10
            return i + 15
        return self.__code.sFuncName

    @func.setter
    def func(self, value):
        if False:
            return 10
        self.__code.sFuncName = value

    @property
    def requestid(self):
        if False:
            return 10
        return self.__code.iRequestId

    @requestid.setter
    def requestid(self, value):
        if False:
            while True:
                i = 10
        self.__code.iRequestId = value

    @property
    def result_code(self):
        if False:
            while True:
                i = 10
        if ('STATUS_RESULT_CODE' in self.__code.status) == False:
            return 0
        return string.atoi(self.__code.status['STATUS_RESULT_CODE'])

    @property
    def result_desc(self):
        if False:
            print('Hello World!')
        if ('STATUS_RESULT_DESC' in self.__code.status) == False:
            return ''
        return self.__code.status['STATUS_RESULT_DESC']

    def put(self, vtype, name, value):
        if False:
            return 10
        oos = TarsOutputStream()
        oos.write(vtype, 0, value)
        self.__buffer[name] = {vtype.__tars_class__: oos.getBuffer()}

    def get(self, vtype, name):
        if False:
            print('Hello World!')
        if (name in self.__buffer) == False:
            raise Exception('UniAttribute not found key:%s,type:%s' % (name, vtype.__tars_class__))
        t = self.__buffer[name]
        if (vtype.__tars_class__ in t) == False:
            raise Exception('UniAttribute not found type:' + vtype.__tars_class__)
        o = TarsInputStream(t[vtype.__tars_class__])
        return o.read(vtype, 0, True)

    def encode(self):
        if False:
            i = 10
            return i + 15
        oos = TarsOutputStream()
        oos.write(self.__mapv, 0, self.__buffer)
        self.__code.iVersion = 2
        self.__code.sBuffer = oos.getBuffer()
        sos = TarsOutputStream()
        RequestPacket.writeTo(sos, self.__code)
        return struct.pack('!i', 4 + len(sos.getBuffer())) + sos.getBuffer()

    def decode(self, buf):
        if False:
            print('Hello World!')
        ois = TarsInputStream(buf[4:])
        self.__code = RequestPacket.readFrom(ois)
        sis = TarsInputStream(self.__code.sBuffer)
        self.__buffer = sis.read(self.__mapv, 0, True)

    def clear(self):
        if False:
            while True:
                i = 10
        self.__code.__init__()

    def haskey(self, name):
        if False:
            for i in range(10):
                print('nop')
        return name in self.__buffer