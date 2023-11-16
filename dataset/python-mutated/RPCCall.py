import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class RPCCall(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            return 10
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = RPCCall()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsRPCCall(cls, buf, offset=0):
        if False:
            i = 10
            return i + 15
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    @classmethod
    def RPCCallBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        if False:
            for i in range(10):
                print('nop')
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b'BFBS', size_prefixed=size_prefixed)

    def Init(self, buf, pos):
        if False:
            print('Hello World!')
        self._tab = flatbuffers.table.Table(buf, pos)

    def Name(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    def Request(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from reflection.Object import Object
            obj = Object()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def Response(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from reflection.Object import Object
            obj = Object()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def Attributes(self, j):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
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
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def AttributesIsNone(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        return o == 0

    def Documentation(self, j):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ''

    def DocumentationLength(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def DocumentationIsNone(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        return o == 0

def RPCCallStart(builder):
    if False:
        for i in range(10):
            print('nop')
    builder.StartObject(5)

def Start(builder):
    if False:
        for i in range(10):
            print('nop')
    RPCCallStart(builder)

def RPCCallAddName(builder, name):
    if False:
        i = 10
        return i + 15
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)

def AddName(builder, name):
    if False:
        for i in range(10):
            print('nop')
    RPCCallAddName(builder, name)

def RPCCallAddRequest(builder, request):
    if False:
        while True:
            i = 10
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(request), 0)

def AddRequest(builder, request):
    if False:
        while True:
            i = 10
    RPCCallAddRequest(builder, request)

def RPCCallAddResponse(builder, response):
    if False:
        while True:
            i = 10
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(response), 0)

def AddResponse(builder, response):
    if False:
        while True:
            i = 10
    RPCCallAddResponse(builder, response)

def RPCCallAddAttributes(builder, attributes):
    if False:
        while True:
            i = 10
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(attributes), 0)

def AddAttributes(builder, attributes):
    if False:
        print('Hello World!')
    RPCCallAddAttributes(builder, attributes)

def RPCCallStartAttributesVector(builder, numElems):
    if False:
        while True:
            i = 10
    return builder.StartVector(4, numElems, 4)

def StartAttributesVector(builder, numElems: int) -> int:
    if False:
        print('Hello World!')
    return RPCCallStartAttributesVector(builder, numElems)

def RPCCallAddDocumentation(builder, documentation):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(documentation), 0)

def AddDocumentation(builder, documentation):
    if False:
        print('Hello World!')
    RPCCallAddDocumentation(builder, documentation)

def RPCCallStartDocumentationVector(builder, numElems):
    if False:
        return 10
    return builder.StartVector(4, numElems, 4)

def StartDocumentationVector(builder, numElems: int) -> int:
    if False:
        print('Hello World!')
    return RPCCallStartDocumentationVector(builder, numElems)

def RPCCallEnd(builder):
    if False:
        for i in range(10):
            print('nop')
    return builder.EndObject()

def End(builder):
    if False:
        for i in range(10):
            print('nop')
    return RPCCallEnd(builder)