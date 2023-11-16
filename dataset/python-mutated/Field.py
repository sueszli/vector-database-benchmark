import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Field(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            return 10
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Field()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsField(cls, buf, offset=0):
        if False:
            return 10
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    @classmethod
    def FieldBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        if False:
            i = 10
            return i + 15
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b'BFBS', size_prefixed=size_prefixed)

    def Init(self, buf, pos):
        if False:
            print('Hello World!')
        self._tab = flatbuffers.table.Table(buf, pos)

    def Name(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    def Type(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from reflection.Type import Type
            obj = Type()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def Id(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint16Flags, o + self._tab.Pos)
        return 0

    def Offset(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint16Flags, o + self._tab.Pos)
        return 0

    def DefaultInteger(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int64Flags, o + self._tab.Pos)
        return 0

    def DefaultReal(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float64Flags, o + self._tab.Pos)
        return 0.0

    def Deprecated(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    def Required(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    def Key(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    def Attributes(self, j):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
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
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def AttributesIsNone(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        return o == 0

    def Documentation(self, j):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ''

    def DocumentationLength(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def DocumentationIsNone(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        return o == 0

    def Optional(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    def Padding(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(28))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint16Flags, o + self._tab.Pos)
        return 0

    def Offset64(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(30))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

def FieldStart(builder):
    if False:
        i = 10
        return i + 15
    builder.StartObject(14)

def Start(builder):
    if False:
        for i in range(10):
            print('nop')
    FieldStart(builder)

def FieldAddName(builder, name):
    if False:
        i = 10
        return i + 15
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)

def AddName(builder, name):
    if False:
        for i in range(10):
            print('nop')
    FieldAddName(builder, name)

def FieldAddType(builder, type):
    if False:
        return 10
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(type), 0)

def AddType(builder, type):
    if False:
        while True:
            i = 10
    FieldAddType(builder, type)

def FieldAddId(builder, id):
    if False:
        while True:
            i = 10
    builder.PrependUint16Slot(2, id, 0)

def AddId(builder, id):
    if False:
        while True:
            i = 10
    FieldAddId(builder, id)

def FieldAddOffset(builder, offset):
    if False:
        while True:
            i = 10
    builder.PrependUint16Slot(3, offset, 0)

def AddOffset(builder, offset):
    if False:
        return 10
    FieldAddOffset(builder, offset)

def FieldAddDefaultInteger(builder, defaultInteger):
    if False:
        i = 10
        return i + 15
    builder.PrependInt64Slot(4, defaultInteger, 0)

def AddDefaultInteger(builder, defaultInteger):
    if False:
        for i in range(10):
            print('nop')
    FieldAddDefaultInteger(builder, defaultInteger)

def FieldAddDefaultReal(builder, defaultReal):
    if False:
        while True:
            i = 10
    builder.PrependFloat64Slot(5, defaultReal, 0.0)

def AddDefaultReal(builder, defaultReal):
    if False:
        i = 10
        return i + 15
    FieldAddDefaultReal(builder, defaultReal)

def FieldAddDeprecated(builder, deprecated):
    if False:
        return 10
    builder.PrependBoolSlot(6, deprecated, 0)

def AddDeprecated(builder, deprecated):
    if False:
        i = 10
        return i + 15
    FieldAddDeprecated(builder, deprecated)

def FieldAddRequired(builder, required):
    if False:
        i = 10
        return i + 15
    builder.PrependBoolSlot(7, required, 0)

def AddRequired(builder, required):
    if False:
        return 10
    FieldAddRequired(builder, required)

def FieldAddKey(builder, key):
    if False:
        return 10
    builder.PrependBoolSlot(8, key, 0)

def AddKey(builder, key):
    if False:
        for i in range(10):
            print('nop')
    FieldAddKey(builder, key)

def FieldAddAttributes(builder, attributes):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependUOffsetTRelativeSlot(9, flatbuffers.number_types.UOffsetTFlags.py_type(attributes), 0)

def AddAttributes(builder, attributes):
    if False:
        print('Hello World!')
    FieldAddAttributes(builder, attributes)

def FieldStartAttributesVector(builder, numElems):
    if False:
        return 10
    return builder.StartVector(4, numElems, 4)

def StartAttributesVector(builder, numElems: int) -> int:
    if False:
        while True:
            i = 10
    return FieldStartAttributesVector(builder, numElems)

def FieldAddDocumentation(builder, documentation):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependUOffsetTRelativeSlot(10, flatbuffers.number_types.UOffsetTFlags.py_type(documentation), 0)

def AddDocumentation(builder, documentation):
    if False:
        return 10
    FieldAddDocumentation(builder, documentation)

def FieldStartDocumentationVector(builder, numElems):
    if False:
        for i in range(10):
            print('nop')
    return builder.StartVector(4, numElems, 4)

def StartDocumentationVector(builder, numElems: int) -> int:
    if False:
        i = 10
        return i + 15
    return FieldStartDocumentationVector(builder, numElems)

def FieldAddOptional(builder, optional):
    if False:
        return 10
    builder.PrependBoolSlot(11, optional, 0)

def AddOptional(builder, optional):
    if False:
        while True:
            i = 10
    FieldAddOptional(builder, optional)

def FieldAddPadding(builder, padding):
    if False:
        return 10
    builder.PrependUint16Slot(12, padding, 0)

def AddPadding(builder, padding):
    if False:
        for i in range(10):
            print('nop')
    FieldAddPadding(builder, padding)

def FieldAddOffset64(builder, offset64):
    if False:
        while True:
            i = 10
    builder.PrependBoolSlot(13, offset64, 0)

def AddOffset64(builder, offset64):
    if False:
        return 10
    FieldAddOffset64(builder, offset64)

def FieldEnd(builder):
    if False:
        return 10
    return builder.EndObject()

def End(builder):
    if False:
        while True:
            i = 10
    return FieldEnd(builder)