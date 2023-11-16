import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class KeyValue(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            while True:
                i = 10
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = KeyValue()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsKeyValue(cls, buf, offset=0):
        if False:
            print('Hello World!')
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    @classmethod
    def KeyValueBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        if False:
            return 10
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b'BFBS', size_prefixed=size_prefixed)

    def Init(self, buf, pos):
        if False:
            for i in range(10):
                print('nop')
        self._tab = flatbuffers.table.Table(buf, pos)

    def Key(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    def Value(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

def KeyValueStart(builder):
    if False:
        print('Hello World!')
    builder.StartObject(2)

def Start(builder):
    if False:
        for i in range(10):
            print('nop')
    KeyValueStart(builder)

def KeyValueAddKey(builder, key):
    if False:
        return 10
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(key), 0)

def AddKey(builder, key):
    if False:
        i = 10
        return i + 15
    KeyValueAddKey(builder, key)

def KeyValueAddValue(builder, value):
    if False:
        while True:
            i = 10
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(value), 0)

def AddValue(builder, value):
    if False:
        print('Hello World!')
    KeyValueAddValue(builder, value)

def KeyValueEnd(builder):
    if False:
        return 10
    return builder.EndObject()

def End(builder):
    if False:
        print('Hello World!')
    return KeyValueEnd(builder)