import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class ScalarStuff(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            i = 10
            return i + 15
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ScalarStuff()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsScalarStuff(cls, buf, offset=0):
        if False:
            print('Hello World!')
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    @classmethod
    def ScalarStuffBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        if False:
            while True:
                i = 10
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b'NULL', size_prefixed=size_prefixed)

    def Init(self, buf, pos):
        if False:
            for i in range(10):
                print('nop')
        self._tab = flatbuffers.table.Table(buf, pos)

    def JustI8(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    def MaybeI8(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return None

    def DefaultI8(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 42

    def JustU8(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    def MaybeU8(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return None

    def DefaultU8(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 42

    def JustI16(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int16Flags, o + self._tab.Pos)
        return 0

    def MaybeI16(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int16Flags, o + self._tab.Pos)
        return None

    def DefaultI16(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int16Flags, o + self._tab.Pos)
        return 42

    def JustU16(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint16Flags, o + self._tab.Pos)
        return 0

    def MaybeU16(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint16Flags, o + self._tab.Pos)
        return None

    def DefaultU16(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint16Flags, o + self._tab.Pos)
        return 42

    def JustI32(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(28))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    def MaybeI32(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(30))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return None

    def DefaultI32(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(32))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 42

    def JustU32(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(34))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    def MaybeU32(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(36))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return None

    def DefaultU32(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(38))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 42

    def JustI64(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(40))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int64Flags, o + self._tab.Pos)
        return 0

    def MaybeI64(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(42))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int64Flags, o + self._tab.Pos)
        return None

    def DefaultI64(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(44))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int64Flags, o + self._tab.Pos)
        return 42

    def JustU64(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(46))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, o + self._tab.Pos)
        return 0

    def MaybeU64(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(48))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, o + self._tab.Pos)
        return None

    def DefaultU64(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(50))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, o + self._tab.Pos)
        return 42

    def JustF32(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(52))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

    def MaybeF32(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(54))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return None

    def DefaultF32(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(56))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 42.0

    def JustF64(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(58))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float64Flags, o + self._tab.Pos)
        return 0.0

    def MaybeF64(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(60))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float64Flags, o + self._tab.Pos)
        return None

    def DefaultF64(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(62))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float64Flags, o + self._tab.Pos)
        return 42.0

    def JustBool(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(64))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    def MaybeBool(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(66))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return None

    def DefaultBool(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(68))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return True

    def JustEnum(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(70))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    def MaybeEnum(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(72))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return None

    def DefaultEnum(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(74))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 1

def ScalarStuffStart(builder):
    if False:
        print('Hello World!')
    builder.StartObject(36)

def Start(builder):
    if False:
        return 10
    ScalarStuffStart(builder)

def ScalarStuffAddJustI8(builder, justI8):
    if False:
        print('Hello World!')
    builder.PrependInt8Slot(0, justI8, 0)

def AddJustI8(builder, justI8):
    if False:
        i = 10
        return i + 15
    ScalarStuffAddJustI8(builder, justI8)

def ScalarStuffAddMaybeI8(builder, maybeI8):
    if False:
        while True:
            i = 10
    builder.PrependInt8Slot(1, maybeI8, None)

def AddMaybeI8(builder, maybeI8):
    if False:
        i = 10
        return i + 15
    ScalarStuffAddMaybeI8(builder, maybeI8)

def ScalarStuffAddDefaultI8(builder, defaultI8):
    if False:
        return 10
    builder.PrependInt8Slot(2, defaultI8, 42)

def AddDefaultI8(builder, defaultI8):
    if False:
        return 10
    ScalarStuffAddDefaultI8(builder, defaultI8)

def ScalarStuffAddJustU8(builder, justU8):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependUint8Slot(3, justU8, 0)

def AddJustU8(builder, justU8):
    if False:
        print('Hello World!')
    ScalarStuffAddJustU8(builder, justU8)

def ScalarStuffAddMaybeU8(builder, maybeU8):
    if False:
        return 10
    builder.PrependUint8Slot(4, maybeU8, None)

def AddMaybeU8(builder, maybeU8):
    if False:
        while True:
            i = 10
    ScalarStuffAddMaybeU8(builder, maybeU8)

def ScalarStuffAddDefaultU8(builder, defaultU8):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependUint8Slot(5, defaultU8, 42)

def AddDefaultU8(builder, defaultU8):
    if False:
        print('Hello World!')
    ScalarStuffAddDefaultU8(builder, defaultU8)

def ScalarStuffAddJustI16(builder, justI16):
    if False:
        return 10
    builder.PrependInt16Slot(6, justI16, 0)

def AddJustI16(builder, justI16):
    if False:
        i = 10
        return i + 15
    ScalarStuffAddJustI16(builder, justI16)

def ScalarStuffAddMaybeI16(builder, maybeI16):
    if False:
        while True:
            i = 10
    builder.PrependInt16Slot(7, maybeI16, None)

def AddMaybeI16(builder, maybeI16):
    if False:
        print('Hello World!')
    ScalarStuffAddMaybeI16(builder, maybeI16)

def ScalarStuffAddDefaultI16(builder, defaultI16):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependInt16Slot(8, defaultI16, 42)

def AddDefaultI16(builder, defaultI16):
    if False:
        return 10
    ScalarStuffAddDefaultI16(builder, defaultI16)

def ScalarStuffAddJustU16(builder, justU16):
    if False:
        print('Hello World!')
    builder.PrependUint16Slot(9, justU16, 0)

def AddJustU16(builder, justU16):
    if False:
        for i in range(10):
            print('nop')
    ScalarStuffAddJustU16(builder, justU16)

def ScalarStuffAddMaybeU16(builder, maybeU16):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependUint16Slot(10, maybeU16, None)

def AddMaybeU16(builder, maybeU16):
    if False:
        return 10
    ScalarStuffAddMaybeU16(builder, maybeU16)

def ScalarStuffAddDefaultU16(builder, defaultU16):
    if False:
        print('Hello World!')
    builder.PrependUint16Slot(11, defaultU16, 42)

def AddDefaultU16(builder, defaultU16):
    if False:
        for i in range(10):
            print('nop')
    ScalarStuffAddDefaultU16(builder, defaultU16)

def ScalarStuffAddJustI32(builder, justI32):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependInt32Slot(12, justI32, 0)

def AddJustI32(builder, justI32):
    if False:
        for i in range(10):
            print('nop')
    ScalarStuffAddJustI32(builder, justI32)

def ScalarStuffAddMaybeI32(builder, maybeI32):
    if False:
        print('Hello World!')
    builder.PrependInt32Slot(13, maybeI32, None)

def AddMaybeI32(builder, maybeI32):
    if False:
        while True:
            i = 10
    ScalarStuffAddMaybeI32(builder, maybeI32)

def ScalarStuffAddDefaultI32(builder, defaultI32):
    if False:
        i = 10
        return i + 15
    builder.PrependInt32Slot(14, defaultI32, 42)

def AddDefaultI32(builder, defaultI32):
    if False:
        i = 10
        return i + 15
    ScalarStuffAddDefaultI32(builder, defaultI32)

def ScalarStuffAddJustU32(builder, justU32):
    if False:
        i = 10
        return i + 15
    builder.PrependUint32Slot(15, justU32, 0)

def AddJustU32(builder, justU32):
    if False:
        for i in range(10):
            print('nop')
    ScalarStuffAddJustU32(builder, justU32)

def ScalarStuffAddMaybeU32(builder, maybeU32):
    if False:
        print('Hello World!')
    builder.PrependUint32Slot(16, maybeU32, None)

def AddMaybeU32(builder, maybeU32):
    if False:
        while True:
            i = 10
    ScalarStuffAddMaybeU32(builder, maybeU32)

def ScalarStuffAddDefaultU32(builder, defaultU32):
    if False:
        i = 10
        return i + 15
    builder.PrependUint32Slot(17, defaultU32, 42)

def AddDefaultU32(builder, defaultU32):
    if False:
        print('Hello World!')
    ScalarStuffAddDefaultU32(builder, defaultU32)

def ScalarStuffAddJustI64(builder, justI64):
    if False:
        return 10
    builder.PrependInt64Slot(18, justI64, 0)

def AddJustI64(builder, justI64):
    if False:
        print('Hello World!')
    ScalarStuffAddJustI64(builder, justI64)

def ScalarStuffAddMaybeI64(builder, maybeI64):
    if False:
        return 10
    builder.PrependInt64Slot(19, maybeI64, None)

def AddMaybeI64(builder, maybeI64):
    if False:
        return 10
    ScalarStuffAddMaybeI64(builder, maybeI64)

def ScalarStuffAddDefaultI64(builder, defaultI64):
    if False:
        i = 10
        return i + 15
    builder.PrependInt64Slot(20, defaultI64, 42)

def AddDefaultI64(builder, defaultI64):
    if False:
        print('Hello World!')
    ScalarStuffAddDefaultI64(builder, defaultI64)

def ScalarStuffAddJustU64(builder, justU64):
    if False:
        i = 10
        return i + 15
    builder.PrependUint64Slot(21, justU64, 0)

def AddJustU64(builder, justU64):
    if False:
        return 10
    ScalarStuffAddJustU64(builder, justU64)

def ScalarStuffAddMaybeU64(builder, maybeU64):
    if False:
        while True:
            i = 10
    builder.PrependUint64Slot(22, maybeU64, None)

def AddMaybeU64(builder, maybeU64):
    if False:
        i = 10
        return i + 15
    ScalarStuffAddMaybeU64(builder, maybeU64)

def ScalarStuffAddDefaultU64(builder, defaultU64):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependUint64Slot(23, defaultU64, 42)

def AddDefaultU64(builder, defaultU64):
    if False:
        return 10
    ScalarStuffAddDefaultU64(builder, defaultU64)

def ScalarStuffAddJustF32(builder, justF32):
    if False:
        i = 10
        return i + 15
    builder.PrependFloat32Slot(24, justF32, 0.0)

def AddJustF32(builder, justF32):
    if False:
        i = 10
        return i + 15
    ScalarStuffAddJustF32(builder, justF32)

def ScalarStuffAddMaybeF32(builder, maybeF32):
    if False:
        while True:
            i = 10
    builder.PrependFloat32Slot(25, maybeF32, None)

def AddMaybeF32(builder, maybeF32):
    if False:
        for i in range(10):
            print('nop')
    ScalarStuffAddMaybeF32(builder, maybeF32)

def ScalarStuffAddDefaultF32(builder, defaultF32):
    if False:
        while True:
            i = 10
    builder.PrependFloat32Slot(26, defaultF32, 42.0)

def AddDefaultF32(builder, defaultF32):
    if False:
        print('Hello World!')
    ScalarStuffAddDefaultF32(builder, defaultF32)

def ScalarStuffAddJustF64(builder, justF64):
    if False:
        return 10
    builder.PrependFloat64Slot(27, justF64, 0.0)

def AddJustF64(builder, justF64):
    if False:
        print('Hello World!')
    ScalarStuffAddJustF64(builder, justF64)

def ScalarStuffAddMaybeF64(builder, maybeF64):
    if False:
        while True:
            i = 10
    builder.PrependFloat64Slot(28, maybeF64, None)

def AddMaybeF64(builder, maybeF64):
    if False:
        for i in range(10):
            print('nop')
    ScalarStuffAddMaybeF64(builder, maybeF64)

def ScalarStuffAddDefaultF64(builder, defaultF64):
    if False:
        print('Hello World!')
    builder.PrependFloat64Slot(29, defaultF64, 42.0)

def AddDefaultF64(builder, defaultF64):
    if False:
        print('Hello World!')
    ScalarStuffAddDefaultF64(builder, defaultF64)

def ScalarStuffAddJustBool(builder, justBool):
    if False:
        return 10
    builder.PrependBoolSlot(30, justBool, 0)

def AddJustBool(builder, justBool):
    if False:
        print('Hello World!')
    ScalarStuffAddJustBool(builder, justBool)

def ScalarStuffAddMaybeBool(builder, maybeBool):
    if False:
        while True:
            i = 10
    builder.PrependBoolSlot(31, maybeBool, None)

def AddMaybeBool(builder, maybeBool):
    if False:
        while True:
            i = 10
    ScalarStuffAddMaybeBool(builder, maybeBool)

def ScalarStuffAddDefaultBool(builder, defaultBool):
    if False:
        i = 10
        return i + 15
    builder.PrependBoolSlot(32, defaultBool, 1)

def AddDefaultBool(builder, defaultBool):
    if False:
        return 10
    ScalarStuffAddDefaultBool(builder, defaultBool)

def ScalarStuffAddJustEnum(builder, justEnum):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependInt8Slot(33, justEnum, 0)

def AddJustEnum(builder, justEnum):
    if False:
        for i in range(10):
            print('nop')
    ScalarStuffAddJustEnum(builder, justEnum)

def ScalarStuffAddMaybeEnum(builder, maybeEnum):
    if False:
        i = 10
        return i + 15
    builder.PrependInt8Slot(34, maybeEnum, None)

def AddMaybeEnum(builder, maybeEnum):
    if False:
        for i in range(10):
            print('nop')
    ScalarStuffAddMaybeEnum(builder, maybeEnum)

def ScalarStuffAddDefaultEnum(builder, defaultEnum):
    if False:
        return 10
    builder.PrependInt8Slot(35, defaultEnum, 1)

def AddDefaultEnum(builder, defaultEnum):
    if False:
        while True:
            i = 10
    ScalarStuffAddDefaultEnum(builder, defaultEnum)

def ScalarStuffEnd(builder):
    if False:
        i = 10
        return i + 15
    return builder.EndObject()

def End(builder):
    if False:
        return 10
    return ScalarStuffEnd(builder)

class ScalarStuffT(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.justI8 = 0
        self.maybeI8 = None
        self.defaultI8 = 42
        self.justU8 = 0
        self.maybeU8 = None
        self.defaultU8 = 42
        self.justI16 = 0
        self.maybeI16 = None
        self.defaultI16 = 42
        self.justU16 = 0
        self.maybeU16 = None
        self.defaultU16 = 42
        self.justI32 = 0
        self.maybeI32 = None
        self.defaultI32 = 42
        self.justU32 = 0
        self.maybeU32 = None
        self.defaultU32 = 42
        self.justI64 = 0
        self.maybeI64 = None
        self.defaultI64 = 42
        self.justU64 = 0
        self.maybeU64 = None
        self.defaultU64 = 42
        self.justF32 = 0.0
        self.maybeF32 = None
        self.defaultF32 = 42.0
        self.justF64 = 0.0
        self.maybeF64 = None
        self.defaultF64 = 42.0
        self.justBool = False
        self.maybeBool = None
        self.defaultBool = True
        self.justEnum = 0
        self.maybeEnum = None
        self.defaultEnum = 1

    @classmethod
    def InitFromBuf(cls, buf, pos):
        if False:
            for i in range(10):
                print('nop')
        scalarStuff = ScalarStuff()
        scalarStuff.Init(buf, pos)
        return cls.InitFromObj(scalarStuff)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        if False:
            return 10
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, scalarStuff):
        if False:
            for i in range(10):
                print('nop')
        x = ScalarStuffT()
        x._UnPack(scalarStuff)
        return x

    def _UnPack(self, scalarStuff):
        if False:
            print('Hello World!')
        if scalarStuff is None:
            return
        self.justI8 = scalarStuff.JustI8()
        self.maybeI8 = scalarStuff.MaybeI8()
        self.defaultI8 = scalarStuff.DefaultI8()
        self.justU8 = scalarStuff.JustU8()
        self.maybeU8 = scalarStuff.MaybeU8()
        self.defaultU8 = scalarStuff.DefaultU8()
        self.justI16 = scalarStuff.JustI16()
        self.maybeI16 = scalarStuff.MaybeI16()
        self.defaultI16 = scalarStuff.DefaultI16()
        self.justU16 = scalarStuff.JustU16()
        self.maybeU16 = scalarStuff.MaybeU16()
        self.defaultU16 = scalarStuff.DefaultU16()
        self.justI32 = scalarStuff.JustI32()
        self.maybeI32 = scalarStuff.MaybeI32()
        self.defaultI32 = scalarStuff.DefaultI32()
        self.justU32 = scalarStuff.JustU32()
        self.maybeU32 = scalarStuff.MaybeU32()
        self.defaultU32 = scalarStuff.DefaultU32()
        self.justI64 = scalarStuff.JustI64()
        self.maybeI64 = scalarStuff.MaybeI64()
        self.defaultI64 = scalarStuff.DefaultI64()
        self.justU64 = scalarStuff.JustU64()
        self.maybeU64 = scalarStuff.MaybeU64()
        self.defaultU64 = scalarStuff.DefaultU64()
        self.justF32 = scalarStuff.JustF32()
        self.maybeF32 = scalarStuff.MaybeF32()
        self.defaultF32 = scalarStuff.DefaultF32()
        self.justF64 = scalarStuff.JustF64()
        self.maybeF64 = scalarStuff.MaybeF64()
        self.defaultF64 = scalarStuff.DefaultF64()
        self.justBool = scalarStuff.JustBool()
        self.maybeBool = scalarStuff.MaybeBool()
        self.defaultBool = scalarStuff.DefaultBool()
        self.justEnum = scalarStuff.JustEnum()
        self.maybeEnum = scalarStuff.MaybeEnum()
        self.defaultEnum = scalarStuff.DefaultEnum()

    def Pack(self, builder):
        if False:
            i = 10
            return i + 15
        ScalarStuffStart(builder)
        ScalarStuffAddJustI8(builder, self.justI8)
        ScalarStuffAddMaybeI8(builder, self.maybeI8)
        ScalarStuffAddDefaultI8(builder, self.defaultI8)
        ScalarStuffAddJustU8(builder, self.justU8)
        ScalarStuffAddMaybeU8(builder, self.maybeU8)
        ScalarStuffAddDefaultU8(builder, self.defaultU8)
        ScalarStuffAddJustI16(builder, self.justI16)
        ScalarStuffAddMaybeI16(builder, self.maybeI16)
        ScalarStuffAddDefaultI16(builder, self.defaultI16)
        ScalarStuffAddJustU16(builder, self.justU16)
        ScalarStuffAddMaybeU16(builder, self.maybeU16)
        ScalarStuffAddDefaultU16(builder, self.defaultU16)
        ScalarStuffAddJustI32(builder, self.justI32)
        ScalarStuffAddMaybeI32(builder, self.maybeI32)
        ScalarStuffAddDefaultI32(builder, self.defaultI32)
        ScalarStuffAddJustU32(builder, self.justU32)
        ScalarStuffAddMaybeU32(builder, self.maybeU32)
        ScalarStuffAddDefaultU32(builder, self.defaultU32)
        ScalarStuffAddJustI64(builder, self.justI64)
        ScalarStuffAddMaybeI64(builder, self.maybeI64)
        ScalarStuffAddDefaultI64(builder, self.defaultI64)
        ScalarStuffAddJustU64(builder, self.justU64)
        ScalarStuffAddMaybeU64(builder, self.maybeU64)
        ScalarStuffAddDefaultU64(builder, self.defaultU64)
        ScalarStuffAddJustF32(builder, self.justF32)
        ScalarStuffAddMaybeF32(builder, self.maybeF32)
        ScalarStuffAddDefaultF32(builder, self.defaultF32)
        ScalarStuffAddJustF64(builder, self.justF64)
        ScalarStuffAddMaybeF64(builder, self.maybeF64)
        ScalarStuffAddDefaultF64(builder, self.defaultF64)
        ScalarStuffAddJustBool(builder, self.justBool)
        ScalarStuffAddMaybeBool(builder, self.maybeBool)
        ScalarStuffAddDefaultBool(builder, self.defaultBool)
        ScalarStuffAddJustEnum(builder, self.justEnum)
        ScalarStuffAddMaybeEnum(builder, self.maybeEnum)
        ScalarStuffAddDefaultEnum(builder, self.defaultEnum)
        scalarStuff = ScalarStuffEnd(builder)
        return scalarStuff