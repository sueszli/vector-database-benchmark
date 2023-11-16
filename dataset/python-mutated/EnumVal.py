import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class EnumVal(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            while True:
                i = 10
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = EnumVal()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsEnumVal(cls, buf, offset=0):
        if False:
            return 10
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    @classmethod
    def EnumValBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        if False:
            while True:
                i = 10
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b'BFBS', size_prefixed=size_prefixed)

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

    def Value(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int64Flags, o + self._tab.Pos)
        return 0

    def UnionType(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from reflection.Type import Type
            obj = Type()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def Documentation(self, j):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ''

    def DocumentationLength(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def DocumentationIsNone(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        return o == 0

    def Attributes(self, j):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from reflection.KeyValue import KeyValue
            obj = KeyValue()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def AttributesLength(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def AttributesIsNone(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        return o == 0

def EnumValStart(builder):
    if False:
        print('Hello World!')
    builder.StartObject(6)

def Start(builder):
    if False:
        return 10
    EnumValStart(builder)

def EnumValAddName(builder, name):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)

def AddName(builder, name):
    if False:
        return 10
    EnumValAddName(builder, name)

def EnumValAddValue(builder, value):
    if False:
        return 10
    builder.PrependInt64Slot(1, value, 0)

def AddValue(builder, value):
    if False:
        for i in range(10):
            print('nop')
    EnumValAddValue(builder, value)

def EnumValAddUnionType(builder, unionType):
    if False:
        print('Hello World!')
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(unionType), 0)

def AddUnionType(builder, unionType):
    if False:
        i = 10
        return i + 15
    EnumValAddUnionType(builder, unionType)

def EnumValAddDocumentation(builder, documentation):
    if False:
        i = 10
        return i + 15
    builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(documentation), 0)

def AddDocumentation(builder, documentation):
    if False:
        return 10
    EnumValAddDocumentation(builder, documentation)

def EnumValStartDocumentationVector(builder, numElems):
    if False:
        print('Hello World!')
    return builder.StartVector(4, numElems, 4)

def StartDocumentationVector(builder, numElems: int) -> int:
    if False:
        while True:
            i = 10
    return EnumValStartDocumentationVector(builder, numElems)

def EnumValAddAttributes(builder, attributes):
    if False:
        i = 10
        return i + 15
    builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(attributes), 0)

def AddAttributes(builder, attributes):
    if False:
        for i in range(10):
            print('nop')
    EnumValAddAttributes(builder, attributes)

def EnumValStartAttributesVector(builder, numElems):
    if False:
        while True:
            i = 10
    return builder.StartVector(4, numElems, 4)

def StartAttributesVector(builder, numElems: int) -> int:
    if False:
        while True:
            i = 10
    return EnumValStartAttributesVector(builder, numElems)

def EnumValEnd(builder):
    if False:
        print('Hello World!')
    return builder.EndObject()

def End(builder):
    if False:
        for i in range(10):
            print('nop')
    return EnumValEnd(builder)