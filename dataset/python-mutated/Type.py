import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Type(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            while True:
                i = 10
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Type()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsType(cls, buf, offset=0):
        if False:
            print('Hello World!')
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    @classmethod
    def TypeBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        if False:
            i = 10
            return i + 15
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b'BFBS', size_prefixed=size_prefixed)

    def Init(self, buf, pos):
        if False:
            print('Hello World!')
        self._tab = flatbuffers.table.Table(buf, pos)

    def BaseType(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    def Element(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    def Index(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return -1

    def FixedLength(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint16Flags, o + self._tab.Pos)
        return 0

    def BaseSize(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 4

    def ElementSize(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

def TypeStart(builder):
    if False:
        return 10
    builder.StartObject(6)

def Start(builder):
    if False:
        print('Hello World!')
    TypeStart(builder)

def TypeAddBaseType(builder, baseType):
    if False:
        while True:
            i = 10
    builder.PrependInt8Slot(0, baseType, 0)

def AddBaseType(builder, baseType):
    if False:
        i = 10
        return i + 15
    TypeAddBaseType(builder, baseType)

def TypeAddElement(builder, element):
    if False:
        return 10
    builder.PrependInt8Slot(1, element, 0)

def AddElement(builder, element):
    if False:
        for i in range(10):
            print('nop')
    TypeAddElement(builder, element)

def TypeAddIndex(builder, index):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependInt32Slot(2, index, -1)

def AddIndex(builder, index):
    if False:
        for i in range(10):
            print('nop')
    TypeAddIndex(builder, index)

def TypeAddFixedLength(builder, fixedLength):
    if False:
        i = 10
        return i + 15
    builder.PrependUint16Slot(3, fixedLength, 0)

def AddFixedLength(builder, fixedLength):
    if False:
        while True:
            i = 10
    TypeAddFixedLength(builder, fixedLength)

def TypeAddBaseSize(builder, baseSize):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependUint32Slot(4, baseSize, 4)

def AddBaseSize(builder, baseSize):
    if False:
        print('Hello World!')
    TypeAddBaseSize(builder, baseSize)

def TypeAddElementSize(builder, elementSize):
    if False:
        i = 10
        return i + 15
    builder.PrependUint32Slot(5, elementSize, 0)

def AddElementSize(builder, elementSize):
    if False:
        for i in range(10):
            print('nop')
    TypeAddElementSize(builder, elementSize)

def TypeEnd(builder):
    if False:
        while True:
            i = 10
    return builder.EndObject()

def End(builder):
    if False:
        return 10
    return TypeEnd(builder)