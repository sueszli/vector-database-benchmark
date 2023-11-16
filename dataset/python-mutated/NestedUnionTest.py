import flatbuffers
from flatbuffers.compat import import_numpy
from typing import Any
from flatbuffers.table import Table
from typing import Optional
np = import_numpy()

class NestedUnionTest(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset: int=0):
        if False:
            i = 10
            return i + 15
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = NestedUnionTest()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsNestedUnionTest(cls, buf, offset=0):
        if False:
            i = 10
            return i + 15
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    def Init(self, buf: bytes, pos: int):
        if False:
            i = 10
            return i + 15
        self._tab = flatbuffers.table.Table(buf, pos)

    def Name(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    def DataType(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    def Data(self) -> Optional[flatbuffers.table.Table]:
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            obj = Table(bytearray(), 0)
            self._tab.Union(obj, o)
            return obj
        return None

    def Id(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int16Flags, o + self._tab.Pos)
        return 0

def NestedUnionTestStart(builder: flatbuffers.Builder):
    if False:
        while True:
            i = 10
    builder.StartObject(4)

def Start(builder: flatbuffers.Builder):
    if False:
        return 10
    NestedUnionTestStart(builder)

def NestedUnionTestAddName(builder: flatbuffers.Builder, name: int):
    if False:
        print('Hello World!')
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)

def AddName(builder: flatbuffers.Builder, name: int):
    if False:
        i = 10
        return i + 15
    NestedUnionTestAddName(builder, name)

def NestedUnionTestAddDataType(builder: flatbuffers.Builder, dataType: int):
    if False:
        print('Hello World!')
    builder.PrependUint8Slot(1, dataType, 0)

def AddDataType(builder: flatbuffers.Builder, dataType: int):
    if False:
        for i in range(10):
            print('nop')
    NestedUnionTestAddDataType(builder, dataType)

def NestedUnionTestAddData(builder: flatbuffers.Builder, data: int):
    if False:
        i = 10
        return i + 15
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(data), 0)

def AddData(builder: flatbuffers.Builder, data: int):
    if False:
        while True:
            i = 10
    NestedUnionTestAddData(builder, data)

def NestedUnionTestAddId(builder: flatbuffers.Builder, id: int):
    if False:
        i = 10
        return i + 15
    builder.PrependInt16Slot(3, id, 0)

def AddId(builder: flatbuffers.Builder, id: int):
    if False:
        while True:
            i = 10
    NestedUnionTestAddId(builder, id)

def NestedUnionTestEnd(builder: flatbuffers.Builder) -> int:
    if False:
        for i in range(10):
            print('nop')
    return builder.EndObject()

def End(builder: flatbuffers.Builder) -> int:
    if False:
        while True:
            i = 10
    return NestedUnionTestEnd(builder)
import MyGame.Example.NestedUnion.Any
import MyGame.Example.NestedUnion.TestSimpleTableWithEnum
import MyGame.Example.NestedUnion.Vec3
try:
    from typing import Union
except:
    pass

class NestedUnionTestT(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.name = None
        self.dataType = 0
        self.data = None
        self.id = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        if False:
            print('Hello World!')
        nestedUnionTest = NestedUnionTest()
        nestedUnionTest.Init(buf, pos)
        return cls.InitFromObj(nestedUnionTest)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        if False:
            while True:
                i = 10
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, nestedUnionTest):
        if False:
            print('Hello World!')
        x = NestedUnionTestT()
        x._UnPack(nestedUnionTest)
        return x

    def _UnPack(self, nestedUnionTest):
        if False:
            return 10
        if nestedUnionTest is None:
            return
        self.name = nestedUnionTest.Name()
        self.dataType = nestedUnionTest.DataType()
        self.data = MyGame.Example.NestedUnion.Any.AnyCreator(self.dataType, nestedUnionTest.Data())
        self.id = nestedUnionTest.Id()

    def Pack(self, builder):
        if False:
            while True:
                i = 10
        if self.name is not None:
            name = builder.CreateString(self.name)
        if self.data is not None:
            data = self.data.Pack(builder)
        NestedUnionTestStart(builder)
        if self.name is not None:
            NestedUnionTestAddName(builder, name)
        NestedUnionTestAddDataType(builder, self.dataType)
        if self.data is not None:
            NestedUnionTestAddData(builder, data)
        NestedUnionTestAddId(builder, self.id)
        nestedUnionTest = NestedUnionTestEnd(builder)
        return nestedUnionTest