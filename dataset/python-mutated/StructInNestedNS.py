import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class StructInNestedNS(object):
    __slots__ = ['_tab']

    @classmethod
    def SizeOf(cls):
        if False:
            return 10
        return 8

    def Init(self, buf, pos):
        if False:
            while True:
                i = 10
        self._tab = flatbuffers.table.Table(buf, pos)

    def A(self):
        if False:
            while True:
                i = 10
        return self._tab.Get(flatbuffers.number_types.Int32Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(0))

    def B(self):
        if False:
            return 10
        return self._tab.Get(flatbuffers.number_types.Int32Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(4))

def CreateStructInNestedNS(builder, a, b):
    if False:
        for i in range(10):
            print('nop')
    builder.Prep(4, 8)
    builder.PrependInt32(b)
    builder.PrependInt32(a)
    return builder.Offset()

class StructInNestedNST(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.a = 0
        self.b = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        if False:
            while True:
                i = 10
        structInNestedNS = StructInNestedNS()
        structInNestedNS.Init(buf, pos)
        return cls.InitFromObj(structInNestedNS)

    @classmethod
    def InitFromObj(cls, structInNestedNS):
        if False:
            for i in range(10):
                print('nop')
        x = StructInNestedNST()
        x._UnPack(structInNestedNS)
        return x

    def _UnPack(self, structInNestedNS):
        if False:
            while True:
                i = 10
        if structInNestedNS is None:
            return
        self.a = structInNestedNS.A()
        self.b = structInNestedNS.B()

    def Pack(self, builder):
        if False:
            while True:
                i = 10
        return CreateStructInNestedNS(builder, self.a, self.b)