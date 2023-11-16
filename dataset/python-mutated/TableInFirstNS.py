import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class TableInFirstNS(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            print('Hello World!')
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = TableInFirstNS()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsTableInFirstNS(cls, buf, offset=0):
        if False:
            for i in range(10):
                print('nop')
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    def Init(self, buf, pos):
        if False:
            for i in range(10):
                print('nop')
        self._tab = flatbuffers.table.Table(buf, pos)

    def FooTable(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            obj = TableInNestedNS()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def FooEnum(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    def FooUnionType(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    def FooUnion(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            from flatbuffers.table import Table
            obj = Table(bytearray(), 0)
            self._tab.Union(obj, o)
            return obj
        return None

    def FooStruct(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            x = o + self._tab.Pos
            obj = StructInNestedNS()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def TableInFirstNSStart(builder):
    if False:
        for i in range(10):
            print('nop')
    builder.StartObject(5)

def Start(builder):
    if False:
        i = 10
        return i + 15
    return TableInFirstNSStart(builder)

def TableInFirstNSAddFooTable(builder, fooTable):
    if False:
        return 10
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(fooTable), 0)

def AddFooTable(builder, fooTable):
    if False:
        i = 10
        return i + 15
    return TableInFirstNSAddFooTable(builder, fooTable)

def TableInFirstNSAddFooEnum(builder, fooEnum):
    if False:
        return 10
    builder.PrependInt8Slot(1, fooEnum, 0)

def AddFooEnum(builder, fooEnum):
    if False:
        i = 10
        return i + 15
    return TableInFirstNSAddFooEnum(builder, fooEnum)

def TableInFirstNSAddFooUnionType(builder, fooUnionType):
    if False:
        while True:
            i = 10
    builder.PrependUint8Slot(2, fooUnionType, 0)

def AddFooUnionType(builder, fooUnionType):
    if False:
        for i in range(10):
            print('nop')
    return TableInFirstNSAddFooUnionType(builder, fooUnionType)

def TableInFirstNSAddFooUnion(builder, fooUnion):
    if False:
        print('Hello World!')
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(fooUnion), 0)

def AddFooUnion(builder, fooUnion):
    if False:
        while True:
            i = 10
    return TableInFirstNSAddFooUnion(builder, fooUnion)

def TableInFirstNSAddFooStruct(builder, fooStruct):
    if False:
        return 10
    builder.PrependStructSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(fooStruct), 0)

def AddFooStruct(builder, fooStruct):
    if False:
        for i in range(10):
            print('nop')
    return TableInFirstNSAddFooStruct(builder, fooStruct)

def TableInFirstNSEnd(builder):
    if False:
        while True:
            i = 10
    return builder.EndObject()

def End(builder):
    if False:
        i = 10
        return i + 15
    return TableInFirstNSEnd(builder)
try:
    from typing import Optional, Union
except:
    pass

class TableInFirstNST(object):

    def __init__(self):
        if False:
            return 10
        self.fooTable = None
        self.fooEnum = 0
        self.fooUnionType = 0
        self.fooUnion = None
        self.fooStruct = None

    @classmethod
    def InitFromBuf(cls, buf, pos):
        if False:
            while True:
                i = 10
        tableInFirstNS = TableInFirstNS()
        tableInFirstNS.Init(buf, pos)
        return cls.InitFromObj(tableInFirstNS)

    @classmethod
    def InitFromObj(cls, tableInFirstNS):
        if False:
            print('Hello World!')
        x = TableInFirstNST()
        x._UnPack(tableInFirstNS)
        return x

    def _UnPack(self, tableInFirstNS):
        if False:
            for i in range(10):
                print('nop')
        if tableInFirstNS is None:
            return
        if tableInFirstNS.FooTable() is not None:
            self.fooTable = TableInNestedNST.InitFromObj(tableInFirstNS.FooTable())
        self.fooEnum = tableInFirstNS.FooEnum()
        self.fooUnionType = tableInFirstNS.FooUnionType()
        self.fooUnion = UnionInNestedNSCreator(self.fooUnionType, tableInFirstNS.FooUnion())
        if tableInFirstNS.FooStruct() is not None:
            self.fooStruct = StructInNestedNST.InitFromObj(tableInFirstNS.FooStruct())

    def Pack(self, builder):
        if False:
            for i in range(10):
                print('nop')
        if self.fooTable is not None:
            fooTable = self.fooTable.Pack(builder)
        if self.fooUnion is not None:
            fooUnion = self.fooUnion.Pack(builder)
        TableInFirstNSStart(builder)
        if self.fooTable is not None:
            TableInFirstNSAddFooTable(builder, fooTable)
        TableInFirstNSAddFooEnum(builder, self.fooEnum)
        TableInFirstNSAddFooUnionType(builder, self.fooUnionType)
        if self.fooUnion is not None:
            TableInFirstNSAddFooUnion(builder, fooUnion)
        if self.fooStruct is not None:
            fooStruct = self.fooStruct.Pack(builder)
            TableInFirstNSAddFooStruct(builder, fooStruct)
        tableInFirstNS = TableInFirstNSEnd(builder)
        return tableInFirstNS