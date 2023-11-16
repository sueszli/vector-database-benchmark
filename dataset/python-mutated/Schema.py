import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Schema(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            i = 10
            return i + 15
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Schema()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsSchema(cls, buf, offset=0):
        if False:
            return 10
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    @classmethod
    def SchemaBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        if False:
            while True:
                i = 10
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b'BFBS', size_prefixed=size_prefixed)

    def Init(self, buf, pos):
        if False:
            i = 10
            return i + 15
        self._tab = flatbuffers.table.Table(buf, pos)

    def Objects(self, j):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from reflection.Object import Object
            obj = Object()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def ObjectsLength(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def ObjectsIsNone(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

    def Enums(self, j):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from reflection.Enum import Enum
            obj = Enum()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def EnumsLength(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def EnumsIsNone(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

    def FileIdent(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    def FileExt(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    def RootTable(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from reflection.Object import Object
            obj = Object()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def Services(self, j):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from reflection.Service import Service
            obj = Service()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def ServicesLength(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def ServicesIsNone(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        return o == 0

    def AdvancedFeatures(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, o + self._tab.Pos)
        return 0

    def FbsFiles(self, j):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from reflection.SchemaFile import SchemaFile
            obj = SchemaFile()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def FbsFilesLength(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def FbsFilesIsNone(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        return o == 0

def SchemaStart(builder):
    if False:
        for i in range(10):
            print('nop')
    builder.StartObject(8)

def Start(builder):
    if False:
        while True:
            i = 10
    SchemaStart(builder)

def SchemaAddObjects(builder, objects):
    if False:
        print('Hello World!')
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(objects), 0)

def AddObjects(builder, objects):
    if False:
        print('Hello World!')
    SchemaAddObjects(builder, objects)

def SchemaStartObjectsVector(builder, numElems):
    if False:
        for i in range(10):
            print('nop')
    return builder.StartVector(4, numElems, 4)

def StartObjectsVector(builder, numElems: int) -> int:
    if False:
        for i in range(10):
            print('nop')
    return SchemaStartObjectsVector(builder, numElems)

def SchemaAddEnums(builder, enums):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(enums), 0)

def AddEnums(builder, enums):
    if False:
        for i in range(10):
            print('nop')
    SchemaAddEnums(builder, enums)

def SchemaStartEnumsVector(builder, numElems):
    if False:
        for i in range(10):
            print('nop')
    return builder.StartVector(4, numElems, 4)

def StartEnumsVector(builder, numElems: int) -> int:
    if False:
        while True:
            i = 10
    return SchemaStartEnumsVector(builder, numElems)

def SchemaAddFileIdent(builder, fileIdent):
    if False:
        print('Hello World!')
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(fileIdent), 0)

def AddFileIdent(builder, fileIdent):
    if False:
        for i in range(10):
            print('nop')
    SchemaAddFileIdent(builder, fileIdent)

def SchemaAddFileExt(builder, fileExt):
    if False:
        while True:
            i = 10
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(fileExt), 0)

def AddFileExt(builder, fileExt):
    if False:
        return 10
    SchemaAddFileExt(builder, fileExt)

def SchemaAddRootTable(builder, rootTable):
    if False:
        i = 10
        return i + 15
    builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(rootTable), 0)

def AddRootTable(builder, rootTable):
    if False:
        for i in range(10):
            print('nop')
    SchemaAddRootTable(builder, rootTable)

def SchemaAddServices(builder, services):
    if False:
        while True:
            i = 10
    builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(services), 0)

def AddServices(builder, services):
    if False:
        print('Hello World!')
    SchemaAddServices(builder, services)

def SchemaStartServicesVector(builder, numElems):
    if False:
        for i in range(10):
            print('nop')
    return builder.StartVector(4, numElems, 4)

def StartServicesVector(builder, numElems: int) -> int:
    if False:
        return 10
    return SchemaStartServicesVector(builder, numElems)

def SchemaAddAdvancedFeatures(builder, advancedFeatures):
    if False:
        print('Hello World!')
    builder.PrependUint64Slot(6, advancedFeatures, 0)

def AddAdvancedFeatures(builder, advancedFeatures):
    if False:
        while True:
            i = 10
    SchemaAddAdvancedFeatures(builder, advancedFeatures)

def SchemaAddFbsFiles(builder, fbsFiles):
    if False:
        print('Hello World!')
    builder.PrependUOffsetTRelativeSlot(7, flatbuffers.number_types.UOffsetTFlags.py_type(fbsFiles), 0)

def AddFbsFiles(builder, fbsFiles):
    if False:
        print('Hello World!')
    SchemaAddFbsFiles(builder, fbsFiles)

def SchemaStartFbsFilesVector(builder, numElems):
    if False:
        for i in range(10):
            print('nop')
    return builder.StartVector(4, numElems, 4)

def StartFbsFilesVector(builder, numElems: int) -> int:
    if False:
        return 10
    return SchemaStartFbsFilesVector(builder, numElems)

def SchemaEnd(builder):
    if False:
        print('Hello World!')
    return builder.EndObject()

def End(builder):
    if False:
        for i in range(10):
            print('nop')
    return SchemaEnd(builder)