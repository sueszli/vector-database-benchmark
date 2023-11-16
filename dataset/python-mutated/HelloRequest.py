import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class HelloRequest(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            return 10
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = HelloRequest()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsHelloRequest(cls, buf, offset=0):
        if False:
            print('Hello World!')
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    def Init(self, buf, pos):
        if False:
            for i in range(10):
                print('nop')
        self._tab = flatbuffers.table.Table(buf, pos)

    def Name(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

def HelloRequestStart(builder):
    if False:
        i = 10
        return i + 15
    builder.StartObject(1)

def Start(builder):
    if False:
        i = 10
        return i + 15
    HelloRequestStart(builder)

def HelloRequestAddName(builder, name):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)

def AddName(builder, name):
    if False:
        i = 10
        return i + 15
    HelloRequestAddName(builder, name)

def HelloRequestEnd(builder):
    if False:
        print('Hello World!')
    return builder.EndObject()

def End(builder):
    if False:
        for i in range(10):
            print('nop')
    return HelloRequestEnd(builder)