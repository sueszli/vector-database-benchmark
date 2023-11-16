import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Object(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            return 10
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Object()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsObject(cls, buf, offset=0):
        if False:
            print('Hello World!')
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    @classmethod
    def ObjectBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        if False:
            i = 10
            return i + 15
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b'BFBS', size_prefixed=size_prefixed)

    def Init(self, buf, pos):
        if False:
            i = 10
            return i + 15
        self._tab = flatbuffers.table.Table(buf, pos)

    def Name(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    def Fields(self, j):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from reflection.Field import Field
            obj = Field()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def FieldsLength(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def FieldsIsNone(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

    def IsStruct(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    def Minalign(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    def Bytesize(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

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
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def AttributesIsNone(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        return o == 0

    def Documentation(self, j):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ''

    def DocumentationLength(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def DocumentationIsNone(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        return o == 0

    def DeclarationFile(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

def ObjectStart(builder):
    if False:
        i = 10
        return i + 15
    builder.StartObject(8)

def Start(builder):
    if False:
        print('Hello World!')
    ObjectStart(builder)

def ObjectAddName(builder, name):
    if False:
        i = 10
        return i + 15
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)

def AddName(builder, name):
    if False:
        i = 10
        return i + 15
    ObjectAddName(builder, name)

def ObjectAddFields(builder, fields):
    if False:
        print('Hello World!')
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(fields), 0)

def AddFields(builder, fields):
    if False:
        while True:
            i = 10
    ObjectAddFields(builder, fields)

def ObjectStartFieldsVector(builder, numElems):
    if False:
        print('Hello World!')
    return builder.StartVector(4, numElems, 4)

def StartFieldsVector(builder, numElems: int) -> int:
    if False:
        for i in range(10):
            print('nop')
    return ObjectStartFieldsVector(builder, numElems)

def ObjectAddIsStruct(builder, isStruct):
    if False:
        while True:
            i = 10
    builder.PrependBoolSlot(2, isStruct, 0)

def AddIsStruct(builder, isStruct):
    if False:
        while True:
            i = 10
    ObjectAddIsStruct(builder, isStruct)

def ObjectAddMinalign(builder, minalign):
    if False:
        print('Hello World!')
    builder.PrependInt32Slot(3, minalign, 0)

def AddMinalign(builder, minalign):
    if False:
        i = 10
        return i + 15
    ObjectAddMinalign(builder, minalign)

def ObjectAddBytesize(builder, bytesize):
    if False:
        print('Hello World!')
    builder.PrependInt32Slot(4, bytesize, 0)

def AddBytesize(builder, bytesize):
    if False:
        while True:
            i = 10
    ObjectAddBytesize(builder, bytesize)

def ObjectAddAttributes(builder, attributes):
    if False:
        print('Hello World!')
    builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(attributes), 0)

def AddAttributes(builder, attributes):
    if False:
        while True:
            i = 10
    ObjectAddAttributes(builder, attributes)

def ObjectStartAttributesVector(builder, numElems):
    if False:
        return 10
    return builder.StartVector(4, numElems, 4)

def StartAttributesVector(builder, numElems: int) -> int:
    if False:
        while True:
            i = 10
    return ObjectStartAttributesVector(builder, numElems)

def ObjectAddDocumentation(builder, documentation):
    if False:
        return 10
    builder.PrependUOffsetTRelativeSlot(6, flatbuffers.number_types.UOffsetTFlags.py_type(documentation), 0)

def AddDocumentation(builder, documentation):
    if False:
        return 10
    ObjectAddDocumentation(builder, documentation)

def ObjectStartDocumentationVector(builder, numElems):
    if False:
        return 10
    return builder.StartVector(4, numElems, 4)

def StartDocumentationVector(builder, numElems: int) -> int:
    if False:
        i = 10
        return i + 15
    return ObjectStartDocumentationVector(builder, numElems)

def ObjectAddDeclarationFile(builder, declarationFile):
    if False:
        while True:
            i = 10
    builder.PrependUOffsetTRelativeSlot(7, flatbuffers.number_types.UOffsetTFlags.py_type(declarationFile), 0)

def AddDeclarationFile(builder, declarationFile):
    if False:
        return 10
    ObjectAddDeclarationFile(builder, declarationFile)

def ObjectEnd(builder):
    if False:
        for i in range(10):
            print('nop')
    return builder.EndObject()

def End(builder):
    if False:
        return 10
    return ObjectEnd(builder)