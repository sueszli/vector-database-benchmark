import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Service(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            print('Hello World!')
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Service()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsService(cls, buf, offset=0):
        if False:
            while True:
                i = 10
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    @classmethod
    def ServiceBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        if False:
            print('Hello World!')
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b'BFBS', size_prefixed=size_prefixed)

    def Init(self, buf, pos):
        if False:
            for i in range(10):
                print('nop')
        self._tab = flatbuffers.table.Table(buf, pos)

    def Name(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    def Calls(self, j):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from reflection.RPCCall import RPCCall
            obj = RPCCall()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def CallsLength(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def CallsIsNone(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

    def Attributes(self, j):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
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
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def AttributesIsNone(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

    def Documentation(self, j):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ''

    def DocumentationLength(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def DocumentationIsNone(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        return o == 0

    def DeclarationFile(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

def ServiceStart(builder):
    if False:
        print('Hello World!')
    builder.StartObject(5)

def Start(builder):
    if False:
        print('Hello World!')
    ServiceStart(builder)

def ServiceAddName(builder, name):
    if False:
        while True:
            i = 10
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)

def AddName(builder, name):
    if False:
        return 10
    ServiceAddName(builder, name)

def ServiceAddCalls(builder, calls):
    if False:
        return 10
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(calls), 0)

def AddCalls(builder, calls):
    if False:
        while True:
            i = 10
    ServiceAddCalls(builder, calls)

def ServiceStartCallsVector(builder, numElems):
    if False:
        for i in range(10):
            print('nop')
    return builder.StartVector(4, numElems, 4)

def StartCallsVector(builder, numElems: int) -> int:
    if False:
        while True:
            i = 10
    return ServiceStartCallsVector(builder, numElems)

def ServiceAddAttributes(builder, attributes):
    if False:
        i = 10
        return i + 15
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(attributes), 0)

def AddAttributes(builder, attributes):
    if False:
        i = 10
        return i + 15
    ServiceAddAttributes(builder, attributes)

def ServiceStartAttributesVector(builder, numElems):
    if False:
        for i in range(10):
            print('nop')
    return builder.StartVector(4, numElems, 4)

def StartAttributesVector(builder, numElems: int) -> int:
    if False:
        i = 10
        return i + 15
    return ServiceStartAttributesVector(builder, numElems)

def ServiceAddDocumentation(builder, documentation):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(documentation), 0)

def AddDocumentation(builder, documentation):
    if False:
        while True:
            i = 10
    ServiceAddDocumentation(builder, documentation)

def ServiceStartDocumentationVector(builder, numElems):
    if False:
        return 10
    return builder.StartVector(4, numElems, 4)

def StartDocumentationVector(builder, numElems: int) -> int:
    if False:
        return 10
    return ServiceStartDocumentationVector(builder, numElems)

def ServiceAddDeclarationFile(builder, declarationFile):
    if False:
        i = 10
        return i + 15
    builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(declarationFile), 0)

def AddDeclarationFile(builder, declarationFile):
    if False:
        for i in range(10):
            print('nop')
    ServiceAddDeclarationFile(builder, declarationFile)

def ServiceEnd(builder):
    if False:
        i = 10
        return i + 15
    return builder.EndObject()

def End(builder):
    if False:
        print('Hello World!')
    return ServiceEnd(builder)