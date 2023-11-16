import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class TableInNestedNS(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            print('Hello World!')
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = TableInNestedNS()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsTableInNestedNS(cls, buf, offset=0):
        if False:
            print('Hello World!')
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    def Init(self, buf, pos):
        if False:
            while True:
                i = 10
        self._tab = flatbuffers.table.Table(buf, pos)

    def Foo(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

def TableInNestedNSStart(builder):
    if False:
        i = 10
        return i + 15
    builder.StartObject(1)

def Start(builder):
    if False:
        return 10
    return TableInNestedNSStart(builder)

def TableInNestedNSAddFoo(builder, foo):
    if False:
        print('Hello World!')
    builder.PrependInt32Slot(0, foo, 0)

def AddFoo(builder, foo):
    if False:
        while True:
            i = 10
    return TableInNestedNSAddFoo(builder, foo)

def TableInNestedNSEnd(builder):
    if False:
        while True:
            i = 10
    return builder.EndObject()

def End(builder):
    if False:
        print('Hello World!')
    return TableInNestedNSEnd(builder)

class TableInNestedNST(object):

    def __init__(self):
        if False:
            return 10
        self.foo = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        if False:
            print('Hello World!')
        tableInNestedNS = TableInNestedNS()
        tableInNestedNS.Init(buf, pos)
        return cls.InitFromObj(tableInNestedNS)

    @classmethod
    def InitFromObj(cls, tableInNestedNS):
        if False:
            print('Hello World!')
        x = TableInNestedNST()
        x._UnPack(tableInNestedNS)
        return x

    def _UnPack(self, tableInNestedNS):
        if False:
            return 10
        if tableInNestedNS is None:
            return
        self.foo = tableInNestedNS.Foo()

    def Pack(self, builder):
        if False:
            print('Hello World!')
        TableInNestedNSStart(builder)
        TableInNestedNSAddFoo(builder, self.foo)
        tableInNestedNS = TableInNestedNSEnd(builder)
        return tableInNestedNS