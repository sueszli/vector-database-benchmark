import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class TestSimpleTableWithEnum(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            print('Hello World!')
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = TestSimpleTableWithEnum()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsTestSimpleTableWithEnum(cls, buf, offset=0):
        if False:
            return 10
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    @classmethod
    def TestSimpleTableWithEnumBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        if False:
            for i in range(10):
                print('nop')
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b'MONS', size_prefixed=size_prefixed)

    def Init(self, buf, pos):
        if False:
            return 10
        self._tab = flatbuffers.table.Table(buf, pos)

    def Color(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 2

def TestSimpleTableWithEnumStart(builder):
    if False:
        print('Hello World!')
    builder.StartObject(1)

def Start(builder):
    if False:
        return 10
    TestSimpleTableWithEnumStart(builder)

def TestSimpleTableWithEnumAddColor(builder, color):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependUint8Slot(0, color, 2)

def AddColor(builder, color):
    if False:
        for i in range(10):
            print('nop')
    TestSimpleTableWithEnumAddColor(builder, color)

def TestSimpleTableWithEnumEnd(builder):
    if False:
        while True:
            i = 10
    return builder.EndObject()

def End(builder):
    if False:
        return 10
    return TestSimpleTableWithEnumEnd(builder)

class TestSimpleTableWithEnumT(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.color = 2

    @classmethod
    def InitFromBuf(cls, buf, pos):
        if False:
            print('Hello World!')
        testSimpleTableWithEnum = TestSimpleTableWithEnum()
        testSimpleTableWithEnum.Init(buf, pos)
        return cls.InitFromObj(testSimpleTableWithEnum)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        if False:
            while True:
                i = 10
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, testSimpleTableWithEnum):
        if False:
            while True:
                i = 10
        x = TestSimpleTableWithEnumT()
        x._UnPack(testSimpleTableWithEnum)
        return x

    def _UnPack(self, testSimpleTableWithEnum):
        if False:
            print('Hello World!')
        if testSimpleTableWithEnum is None:
            return
        self.color = testSimpleTableWithEnum.Color()

    def Pack(self, builder):
        if False:
            return 10
        TestSimpleTableWithEnumStart(builder)
        TestSimpleTableWithEnumAddColor(builder, self.color)
        testSimpleTableWithEnum = TestSimpleTableWithEnumEnd(builder)
        return testSimpleTableWithEnum