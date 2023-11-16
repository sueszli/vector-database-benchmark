import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class SchemaFile(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            i = 10
            return i + 15
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SchemaFile()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsSchemaFile(cls, buf, offset=0):
        if False:
            print('Hello World!')
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    @classmethod
    def SchemaFileBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        if False:
            while True:
                i = 10
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b'BFBS', size_prefixed=size_prefixed)

    def Init(self, buf, pos):
        if False:
            for i in range(10):
                print('nop')
        self._tab = flatbuffers.table.Table(buf, pos)

    def Filename(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    def IncludedFilenames(self, j):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ''

    def IncludedFilenamesLength(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def IncludedFilenamesIsNone(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

def SchemaFileStart(builder):
    if False:
        print('Hello World!')
    builder.StartObject(2)

def Start(builder):
    if False:
        print('Hello World!')
    SchemaFileStart(builder)

def SchemaFileAddFilename(builder, filename):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(filename), 0)

def AddFilename(builder, filename):
    if False:
        i = 10
        return i + 15
    SchemaFileAddFilename(builder, filename)

def SchemaFileAddIncludedFilenames(builder, includedFilenames):
    if False:
        i = 10
        return i + 15
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(includedFilenames), 0)

def AddIncludedFilenames(builder, includedFilenames):
    if False:
        i = 10
        return i + 15
    SchemaFileAddIncludedFilenames(builder, includedFilenames)

def SchemaFileStartIncludedFilenamesVector(builder, numElems):
    if False:
        return 10
    return builder.StartVector(4, numElems, 4)

def StartIncludedFilenamesVector(builder, numElems: int) -> int:
    if False:
        i = 10
        return i + 15
    return SchemaFileStartIncludedFilenamesVector(builder, numElems)

def SchemaFileEnd(builder):
    if False:
        for i in range(10):
            print('nop')
    return builder.EndObject()

def End(builder):
    if False:
        i = 10
        return i + 15
    return SchemaFileEnd(builder)