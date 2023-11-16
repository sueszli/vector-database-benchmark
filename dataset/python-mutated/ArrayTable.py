import flatbuffers
from flatbuffers.compat import import_numpy
from typing import Any
from MyGame.Example.ArrayStruct import ArrayStruct
from typing import Optional
np = import_numpy()

class ArrayTable(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset: int=0):
        if False:
            for i in range(10):
                print('nop')
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ArrayTable()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsArrayTable(cls, buf, offset=0):
        if False:
            while True:
                i = 10
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    @classmethod
    def ArrayTableBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        if False:
            return 10
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b'ARRT', size_prefixed=size_prefixed)

    def Init(self, buf: bytes, pos: int):
        if False:
            print('Hello World!')
        self._tab = flatbuffers.table.Table(buf, pos)

    def A(self) -> Optional[ArrayStruct]:
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = o + self._tab.Pos
            obj = ArrayStruct()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def ArrayTableStart(builder: flatbuffers.Builder):
    if False:
        return 10
    builder.StartObject(1)

def Start(builder: flatbuffers.Builder):
    if False:
        print('Hello World!')
    ArrayTableStart(builder)

def ArrayTableAddA(builder: flatbuffers.Builder, a: Any):
    if False:
        return 10
    builder.PrependStructSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(a), 0)

def AddA(builder: flatbuffers.Builder, a: Any):
    if False:
        i = 10
        return i + 15
    ArrayTableAddA(builder, a)

def ArrayTableEnd(builder: flatbuffers.Builder) -> int:
    if False:
        while True:
            i = 10
    return builder.EndObject()

def End(builder: flatbuffers.Builder) -> int:
    if False:
        print('Hello World!')
    return ArrayTableEnd(builder)
import MyGame.Example.ArrayStruct
try:
    from typing import Optional
except:
    pass

class ArrayTableT(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.a = None

    @classmethod
    def InitFromBuf(cls, buf, pos):
        if False:
            while True:
                i = 10
        arrayTable = ArrayTable()
        arrayTable.Init(buf, pos)
        return cls.InitFromObj(arrayTable)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        if False:
            print('Hello World!')
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, arrayTable):
        if False:
            print('Hello World!')
        x = ArrayTableT()
        x._UnPack(arrayTable)
        return x

    def _UnPack(self, arrayTable):
        if False:
            print('Hello World!')
        if arrayTable is None:
            return
        if arrayTable.A() is not None:
            self.a = MyGame.Example.ArrayStruct.ArrayStructT.InitFromObj(arrayTable.A())

    def Pack(self, builder):
        if False:
            return 10
        ArrayTableStart(builder)
        if self.a is not None:
            a = self.a.Pack(builder)
            ArrayTableAddA(builder, a)
        arrayTable = ArrayTableEnd(builder)
        return arrayTable