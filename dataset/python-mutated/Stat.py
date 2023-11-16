import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Stat(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            print('Hello World!')
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Stat()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsStat(cls, buf, offset=0):
        if False:
            i = 10
            return i + 15
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    @classmethod
    def StatBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        if False:
            while True:
                i = 10
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b'MONS', size_prefixed=size_prefixed)

    def Init(self, buf, pos):
        if False:
            i = 10
            return i + 15
        self._tab = flatbuffers.table.Table(buf, pos)

    def Id(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    def Val(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int64Flags, o + self._tab.Pos)
        return 0

    def Count(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint16Flags, o + self._tab.Pos)
        return 0

def StatStart(builder):
    if False:
        while True:
            i = 10
    builder.StartObject(3)

def Start(builder):
    if False:
        print('Hello World!')
    StatStart(builder)

def StatAddId(builder, id):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(id), 0)

def AddId(builder, id):
    if False:
        while True:
            i = 10
    StatAddId(builder, id)

def StatAddVal(builder, val):
    if False:
        while True:
            i = 10
    builder.PrependInt64Slot(1, val, 0)

def AddVal(builder, val):
    if False:
        i = 10
        return i + 15
    StatAddVal(builder, val)

def StatAddCount(builder, count):
    if False:
        print('Hello World!')
    builder.PrependUint16Slot(2, count, 0)

def AddCount(builder, count):
    if False:
        for i in range(10):
            print('nop')
    StatAddCount(builder, count)

def StatEnd(builder):
    if False:
        for i in range(10):
            print('nop')
    return builder.EndObject()

def End(builder):
    if False:
        return 10
    return StatEnd(builder)

class StatT(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.id = None
        self.val = 0
        self.count = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        if False:
            while True:
                i = 10
        stat = Stat()
        stat.Init(buf, pos)
        return cls.InitFromObj(stat)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        if False:
            print('Hello World!')
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, stat):
        if False:
            while True:
                i = 10
        x = StatT()
        x._UnPack(stat)
        return x

    def _UnPack(self, stat):
        if False:
            while True:
                i = 10
        if stat is None:
            return
        self.id = stat.Id()
        self.val = stat.Val()
        self.count = stat.Count()

    def Pack(self, builder):
        if False:
            print('Hello World!')
        if self.id is not None:
            id = builder.CreateString(self.id)
        StatStart(builder)
        if self.id is not None:
            StatAddId(builder, id)
        StatAddVal(builder, self.val)
        StatAddCount(builder, self.count)
        stat = StatEnd(builder)
        return stat