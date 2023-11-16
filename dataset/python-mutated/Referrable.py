import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Referrable(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            return 10
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Referrable()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsReferrable(cls, buf, offset=0):
        if False:
            while True:
                i = 10
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    @classmethod
    def ReferrableBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        if False:
            i = 10
            return i + 15
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b'MONS', size_prefixed=size_prefixed)

    def Init(self, buf, pos):
        if False:
            print('Hello World!')
        self._tab = flatbuffers.table.Table(buf, pos)

    def Id(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, o + self._tab.Pos)
        return 0

def ReferrableStart(builder):
    if False:
        while True:
            i = 10
    builder.StartObject(1)

def Start(builder):
    if False:
        for i in range(10):
            print('nop')
    ReferrableStart(builder)

def ReferrableAddId(builder, id):
    if False:
        while True:
            i = 10
    builder.PrependUint64Slot(0, id, 0)

def AddId(builder, id):
    if False:
        for i in range(10):
            print('nop')
    ReferrableAddId(builder, id)

def ReferrableEnd(builder):
    if False:
        for i in range(10):
            print('nop')
    return builder.EndObject()

def End(builder):
    if False:
        i = 10
        return i + 15
    return ReferrableEnd(builder)

class ReferrableT(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.id = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        if False:
            print('Hello World!')
        referrable = Referrable()
        referrable.Init(buf, pos)
        return cls.InitFromObj(referrable)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        if False:
            for i in range(10):
                print('nop')
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, referrable):
        if False:
            return 10
        x = ReferrableT()
        x._UnPack(referrable)
        return x

    def _UnPack(self, referrable):
        if False:
            print('Hello World!')
        if referrable is None:
            return
        self.id = referrable.Id()

    def Pack(self, builder):
        if False:
            return 10
        ReferrableStart(builder)
        ReferrableAddId(builder, self.id)
        referrable = ReferrableEnd(builder)
        return referrable