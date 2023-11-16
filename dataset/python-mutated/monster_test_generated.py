import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Color(object):
    Red = 1
    Green = 2
    Blue = 8

class Race(object):
    None_ = -1
    Human = 0
    Dwarf = 1
    Elf = 2

class LongEnum(object):
    LongOne = 2
    LongTwo = 4
    LongBig = 1099511627776

class Any(object):
    NONE = 0
    Monster = 1
    TestSimpleTableWithEnum = 2
    MyGame_Example2_Monster = 3

def AnyCreator(unionType, table):
    if False:
        while True:
            i = 10
    from flatbuffers.table import Table
    if not isinstance(table, Table):
        return None
    if unionType == Any().Monster:
        return MonsterT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == Any().TestSimpleTableWithEnum:
        return TestSimpleTableWithEnumT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == Any().MyGame_Example2_Monster:
        return MonsterT.InitFromBuf(table.Bytes, table.Pos)
    return None

class AnyUniqueAliases(object):
    NONE = 0
    M = 1
    TS = 2
    M2 = 3

def AnyUniqueAliasesCreator(unionType, table):
    if False:
        i = 10
        return i + 15
    from flatbuffers.table import Table
    if not isinstance(table, Table):
        return None
    if unionType == AnyUniqueAliases().M:
        return MonsterT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == AnyUniqueAliases().TS:
        return TestSimpleTableWithEnumT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == AnyUniqueAliases().M2:
        return MonsterT.InitFromBuf(table.Bytes, table.Pos)
    return None

class AnyAmbiguousAliases(object):
    NONE = 0
    M1 = 1
    M2 = 2
    M3 = 3

def AnyAmbiguousAliasesCreator(unionType, table):
    if False:
        return 10
    from flatbuffers.table import Table
    if not isinstance(table, Table):
        return None
    if unionType == AnyAmbiguousAliases().M1:
        return MonsterT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == AnyAmbiguousAliases().M2:
        return MonsterT.InitFromBuf(table.Bytes, table.Pos)
    if unionType == AnyAmbiguousAliases().M3:
        return MonsterT.InitFromBuf(table.Bytes, table.Pos)
    return None

class InParentNamespace(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            print('Hello World!')
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = InParentNamespace()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsInParentNamespace(cls, buf, offset=0):
        if False:
            for i in range(10):
                print('nop')
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    @classmethod
    def InParentNamespaceBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        if False:
            return 10
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b'MONS', size_prefixed=size_prefixed)

    def Init(self, buf, pos):
        if False:
            print('Hello World!')
        self._tab = flatbuffers.table.Table(buf, pos)

def InParentNamespaceStart(builder):
    if False:
        while True:
            i = 10
    builder.StartObject(0)

def InParentNamespaceEnd(builder):
    if False:
        return 10
    return builder.EndObject()

class InParentNamespaceT(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        if False:
            while True:
                i = 10
        inParentNamespace = InParentNamespace()
        inParentNamespace.Init(buf, pos)
        return cls.InitFromObj(inParentNamespace)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        if False:
            print('Hello World!')
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, inParentNamespace):
        if False:
            i = 10
            return i + 15
        x = InParentNamespaceT()
        x._UnPack(inParentNamespace)
        return x

    def _UnPack(self, inParentNamespace):
        if False:
            while True:
                i = 10
        if inParentNamespace is None:
            return

    def Pack(self, builder):
        if False:
            i = 10
            return i + 15
        InParentNamespaceStart(builder)
        inParentNamespace = InParentNamespaceEnd(builder)
        return inParentNamespace

class Monster(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            i = 10
            return i + 15
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Monster()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsMonster(cls, buf, offset=0):
        if False:
            i = 10
            return i + 15
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    @classmethod
    def MonsterBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        if False:
            while True:
                i = 10
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b'MONS', size_prefixed=size_prefixed)

    def Init(self, buf, pos):
        if False:
            return 10
        self._tab = flatbuffers.table.Table(buf, pos)

def MonsterStart(builder):
    if False:
        print('Hello World!')
    builder.StartObject(0)

def MonsterEnd(builder):
    if False:
        for i in range(10):
            print('nop')
    return builder.EndObject()

class MonsterT(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        pass

    @classmethod
    def InitFromBuf(cls, buf, pos):
        if False:
            print('Hello World!')
        monster = Monster()
        monster.Init(buf, pos)
        return cls.InitFromObj(monster)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        if False:
            for i in range(10):
                print('nop')
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, monster):
        if False:
            while True:
                i = 10
        x = MonsterT()
        x._UnPack(monster)
        return x

    def _UnPack(self, monster):
        if False:
            return 10
        if monster is None:
            return

    def Pack(self, builder):
        if False:
            return 10
        MonsterStart(builder)
        monster = MonsterEnd(builder)
        return monster

class Test(object):
    __slots__ = ['_tab']

    @classmethod
    def SizeOf(cls):
        if False:
            print('Hello World!')
        return 4

    def Init(self, buf, pos):
        if False:
            while True:
                i = 10
        self._tab = flatbuffers.table.Table(buf, pos)

    def A(self):
        if False:
            for i in range(10):
                print('nop')
        return self._tab.Get(flatbuffers.number_types.Int16Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(0))

    def B(self):
        if False:
            return 10
        return self._tab.Get(flatbuffers.number_types.Int8Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(2))

def CreateTest(builder, a, b):
    if False:
        for i in range(10):
            print('nop')
    builder.Prep(2, 4)
    builder.Pad(1)
    builder.PrependInt8(b)
    builder.PrependInt16(a)
    return builder.Offset()

class TestT(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.a = 0
        self.b = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        if False:
            while True:
                i = 10
        test = Test()
        test.Init(buf, pos)
        return cls.InitFromObj(test)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        if False:
            return 10
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, test):
        if False:
            print('Hello World!')
        x = TestT()
        x._UnPack(test)
        return x

    def _UnPack(self, test):
        if False:
            return 10
        if test is None:
            return
        self.a = test.A()
        self.b = test.B()

    def Pack(self, builder):
        if False:
            for i in range(10):
                print('nop')
        return CreateTest(builder, self.a, self.b)

class TestSimpleTableWithEnum(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            for i in range(10):
                print('nop')
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = TestSimpleTableWithEnum()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsTestSimpleTableWithEnum(cls, buf, offset=0):
        if False:
            print('Hello World!')
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    @classmethod
    def TestSimpleTableWithEnumBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        if False:
            return 10
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b'MONS', size_prefixed=size_prefixed)

    def Init(self, buf, pos):
        if False:
            return 10
        self._tab = flatbuffers.table.Table(buf, pos)

    def Color(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 2

def TestSimpleTableWithEnumStart(builder):
    if False:
        i = 10
        return i + 15
    builder.StartObject(1)

def TestSimpleTableWithEnumAddColor(builder, color):
    if False:
        while True:
            i = 10
    builder.PrependUint8Slot(0, color, 2)

def TestSimpleTableWithEnumEnd(builder):
    if False:
        print('Hello World!')
    return builder.EndObject()

class TestSimpleTableWithEnumT(object):

    def __init__(self):
        if False:
            return 10
        self.color = 2

    @classmethod
    def InitFromBuf(cls, buf, pos):
        if False:
            while True:
                i = 10
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
            i = 10
            return i + 15
        x = TestSimpleTableWithEnumT()
        x._UnPack(testSimpleTableWithEnum)
        return x

    def _UnPack(self, testSimpleTableWithEnum):
        if False:
            while True:
                i = 10
        if testSimpleTableWithEnum is None:
            return
        self.color = testSimpleTableWithEnum.Color()

    def Pack(self, builder):
        if False:
            print('Hello World!')
        TestSimpleTableWithEnumStart(builder)
        TestSimpleTableWithEnumAddColor(builder, self.color)
        testSimpleTableWithEnum = TestSimpleTableWithEnumEnd(builder)
        return testSimpleTableWithEnum

class Vec3(object):
    __slots__ = ['_tab']

    @classmethod
    def SizeOf(cls):
        if False:
            i = 10
            return i + 15
        return 32

    def Init(self, buf, pos):
        if False:
            i = 10
            return i + 15
        self._tab = flatbuffers.table.Table(buf, pos)

    def X(self):
        if False:
            i = 10
            return i + 15
        return self._tab.Get(flatbuffers.number_types.Float32Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(0))

    def Y(self):
        if False:
            i = 10
            return i + 15
        return self._tab.Get(flatbuffers.number_types.Float32Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(4))

    def Z(self):
        if False:
            i = 10
            return i + 15
        return self._tab.Get(flatbuffers.number_types.Float32Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(8))

    def Test1(self):
        if False:
            return 10
        return self._tab.Get(flatbuffers.number_types.Float64Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(16))

    def Test2(self):
        if False:
            return 10
        return self._tab.Get(flatbuffers.number_types.Uint8Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(24))

    def Test3(self, obj):
        if False:
            return 10
        obj.Init(self._tab.Bytes, self._tab.Pos + 26)
        return obj

def CreateVec3(builder, x, y, z, test1, test2, test3_a, test3_b):
    if False:
        for i in range(10):
            print('nop')
    builder.Prep(8, 32)
    builder.Pad(2)
    builder.Prep(2, 4)
    builder.Pad(1)
    builder.PrependInt8(test3_b)
    builder.PrependInt16(test3_a)
    builder.Pad(1)
    builder.PrependUint8(test2)
    builder.PrependFloat64(test1)
    builder.Pad(4)
    builder.PrependFloat32(z)
    builder.PrependFloat32(y)
    builder.PrependFloat32(x)
    return builder.Offset()
try:
    from typing import Optional
except:
    pass

class Vec3T(object):

    def __init__(self):
        if False:
            return 10
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.test1 = 0.0
        self.test2 = 0
        self.test3 = None

    @classmethod
    def InitFromBuf(cls, buf, pos):
        if False:
            print('Hello World!')
        vec3 = Vec3()
        vec3.Init(buf, pos)
        return cls.InitFromObj(vec3)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        if False:
            return 10
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, vec3):
        if False:
            while True:
                i = 10
        x = Vec3T()
        x._UnPack(vec3)
        return x

    def _UnPack(self, vec3):
        if False:
            return 10
        if vec3 is None:
            return
        self.x = vec3.X()
        self.y = vec3.Y()
        self.z = vec3.Z()
        self.test1 = vec3.Test1()
        self.test2 = vec3.Test2()
        if vec3.Test3(Test()) is not None:
            self.test3 = TestT.InitFromObj(vec3.Test3(Test()))

    def Pack(self, builder):
        if False:
            i = 10
            return i + 15
        return CreateVec3(builder, self.x, self.y, self.z, self.test1, self.test2, self.test3.a, self.test3.b)

class Ability(object):
    __slots__ = ['_tab']

    @classmethod
    def SizeOf(cls):
        if False:
            i = 10
            return i + 15
        return 8

    def Init(self, buf, pos):
        if False:
            i = 10
            return i + 15
        self._tab = flatbuffers.table.Table(buf, pos)

    def Id(self):
        if False:
            for i in range(10):
                print('nop')
        return self._tab.Get(flatbuffers.number_types.Uint32Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(0))

    def Distance(self):
        if False:
            while True:
                i = 10
        return self._tab.Get(flatbuffers.number_types.Uint32Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(4))

def CreateAbility(builder, id, distance):
    if False:
        i = 10
        return i + 15
    builder.Prep(4, 8)
    builder.PrependUint32(distance)
    builder.PrependUint32(id)
    return builder.Offset()

class AbilityT(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.id = 0
        self.distance = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        if False:
            for i in range(10):
                print('nop')
        ability = Ability()
        ability.Init(buf, pos)
        return cls.InitFromObj(ability)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        if False:
            i = 10
            return i + 15
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, ability):
        if False:
            i = 10
            return i + 15
        x = AbilityT()
        x._UnPack(ability)
        return x

    def _UnPack(self, ability):
        if False:
            return 10
        if ability is None:
            return
        self.id = ability.Id()
        self.distance = ability.Distance()

    def Pack(self, builder):
        if False:
            i = 10
            return i + 15
        return CreateAbility(builder, self.id, self.distance)

class StructOfStructs(object):
    __slots__ = ['_tab']

    @classmethod
    def SizeOf(cls):
        if False:
            i = 10
            return i + 15
        return 20

    def Init(self, buf, pos):
        if False:
            return 10
        self._tab = flatbuffers.table.Table(buf, pos)

    def A(self, obj):
        if False:
            while True:
                i = 10
        obj.Init(self._tab.Bytes, self._tab.Pos + 0)
        return obj

    def B(self, obj):
        if False:
            print('Hello World!')
        obj.Init(self._tab.Bytes, self._tab.Pos + 8)
        return obj

    def C(self, obj):
        if False:
            return 10
        obj.Init(self._tab.Bytes, self._tab.Pos + 12)
        return obj

def CreateStructOfStructs(builder, a_id, a_distance, b_a, b_b, c_id, c_distance):
    if False:
        print('Hello World!')
    builder.Prep(4, 20)
    builder.Prep(4, 8)
    builder.PrependUint32(c_distance)
    builder.PrependUint32(c_id)
    builder.Prep(2, 4)
    builder.Pad(1)
    builder.PrependInt8(b_b)
    builder.PrependInt16(b_a)
    builder.Prep(4, 8)
    builder.PrependUint32(a_distance)
    builder.PrependUint32(a_id)
    return builder.Offset()
try:
    from typing import Optional
except:
    pass

class StructOfStructsT(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.a = None
        self.b = None
        self.c = None

    @classmethod
    def InitFromBuf(cls, buf, pos):
        if False:
            i = 10
            return i + 15
        structOfStructs = StructOfStructs()
        structOfStructs.Init(buf, pos)
        return cls.InitFromObj(structOfStructs)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        if False:
            print('Hello World!')
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, structOfStructs):
        if False:
            return 10
        x = StructOfStructsT()
        x._UnPack(structOfStructs)
        return x

    def _UnPack(self, structOfStructs):
        if False:
            while True:
                i = 10
        if structOfStructs is None:
            return
        if structOfStructs.A(Ability()) is not None:
            self.a = AbilityT.InitFromObj(structOfStructs.A(Ability()))
        if structOfStructs.B(Test()) is not None:
            self.b = TestT.InitFromObj(structOfStructs.B(Test()))
        if structOfStructs.C(Ability()) is not None:
            self.c = AbilityT.InitFromObj(structOfStructs.C(Ability()))

    def Pack(self, builder):
        if False:
            return 10
        return CreateStructOfStructs(builder, self.a.id, self.a.distance, self.b.a, self.b.b, self.c.id, self.c.distance)

class StructOfStructsOfStructs(object):
    __slots__ = ['_tab']

    @classmethod
    def SizeOf(cls):
        if False:
            for i in range(10):
                print('nop')
        return 20

    def Init(self, buf, pos):
        if False:
            return 10
        self._tab = flatbuffers.table.Table(buf, pos)

    def A(self, obj):
        if False:
            while True:
                i = 10
        obj.Init(self._tab.Bytes, self._tab.Pos + 0)
        return obj

def CreateStructOfStructsOfStructs(builder, a_a_id, a_a_distance, a_b_a, a_b_b, a_c_id, a_c_distance):
    if False:
        for i in range(10):
            print('nop')
    builder.Prep(4, 20)
    builder.Prep(4, 20)
    builder.Prep(4, 8)
    builder.PrependUint32(a_c_distance)
    builder.PrependUint32(a_c_id)
    builder.Prep(2, 4)
    builder.Pad(1)
    builder.PrependInt8(a_b_b)
    builder.PrependInt16(a_b_a)
    builder.Prep(4, 8)
    builder.PrependUint32(a_a_distance)
    builder.PrependUint32(a_a_id)
    return builder.Offset()
try:
    from typing import Optional
except:
    pass

class StructOfStructsOfStructsT(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.a = None

    @classmethod
    def InitFromBuf(cls, buf, pos):
        if False:
            while True:
                i = 10
        structOfStructsOfStructs = StructOfStructsOfStructs()
        structOfStructsOfStructs.Init(buf, pos)
        return cls.InitFromObj(structOfStructsOfStructs)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        if False:
            while True:
                i = 10
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, structOfStructsOfStructs):
        if False:
            print('Hello World!')
        x = StructOfStructsOfStructsT()
        x._UnPack(structOfStructsOfStructs)
        return x

    def _UnPack(self, structOfStructsOfStructs):
        if False:
            return 10
        if structOfStructsOfStructs is None:
            return
        if structOfStructsOfStructs.A(StructOfStructs()) is not None:
            self.a = StructOfStructsT.InitFromObj(structOfStructsOfStructs.A(StructOfStructs()))

    def Pack(self, builder):
        if False:
            for i in range(10):
                print('nop')
        return CreateStructOfStructsOfStructs(builder, self.a.a.id, self.a.a.distance, self.a.b.a, self.a.b.b, self.a.c.id, self.a.c.distance)

class Stat(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            for i in range(10):
                print('nop')
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Stat()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsStat(cls, buf, offset=0):
        if False:
            for i in range(10):
                print('nop')
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    @classmethod
    def StatBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
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
            return self._tab.String(o + self._tab.Pos)
        return None

    def Val(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int64Flags, o + self._tab.Pos)
        return 0

    def Count(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint16Flags, o + self._tab.Pos)
        return 0

def StatStart(builder):
    if False:
        print('Hello World!')
    builder.StartObject(3)

def StatAddId(builder, id):
    if False:
        i = 10
        return i + 15
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(id), 0)

def StatAddVal(builder, val):
    if False:
        return 10
    builder.PrependInt64Slot(1, val, 0)

def StatAddCount(builder, count):
    if False:
        return 10
    builder.PrependUint16Slot(2, count, 0)

def StatEnd(builder):
    if False:
        i = 10
        return i + 15
    return builder.EndObject()

class StatT(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.id = None
        self.val = 0
        self.count = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        if False:
            print('Hello World!')
        stat = Stat()
        stat.Init(buf, pos)
        return cls.InitFromObj(stat)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        if False:
            while True:
                i = 10
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, stat):
        if False:
            print('Hello World!')
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

class Referrable(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            i = 10
            return i + 15
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
            for i in range(10):
                print('nop')
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
        return 10
    builder.StartObject(1)

def ReferrableAddId(builder, id):
    if False:
        print('Hello World!')
    builder.PrependUint64Slot(0, id, 0)

def ReferrableEnd(builder):
    if False:
        print('Hello World!')
    return builder.EndObject()

class ReferrableT(object):

    def __init__(self):
        if False:
            return 10
        self.id = 0

    @classmethod
    def InitFromBuf(cls, buf, pos):
        if False:
            i = 10
            return i + 15
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
            for i in range(10):
                print('nop')
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
            while True:
                i = 10
        ReferrableStart(builder)
        ReferrableAddId(builder, self.id)
        referrable = ReferrableEnd(builder)
        return referrable

class Monster(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            i = 10
            return i + 15
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Monster()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsMonster(cls, buf, offset=0):
        if False:
            for i in range(10):
                print('nop')
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    @classmethod
    def MonsterBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        if False:
            while True:
                i = 10
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b'MONS', size_prefixed=size_prefixed)

    def Init(self, buf, pos):
        if False:
            for i in range(10):
                print('nop')
        self._tab = flatbuffers.table.Table(buf, pos)

    def Pos(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = o + self._tab.Pos
            obj = Vec3()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def Mana(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int16Flags, o + self._tab.Pos)
        return 150

    def Hp(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int16Flags, o + self._tab.Pos)
        return 100

    def Name(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    def Inventory(self, j):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    def InventoryAsNumpy(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    def InventoryLength(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def InventoryIsNone(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        return o == 0

    def Color(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 8

    def TestType(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    def Test(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            from flatbuffers.table import Table
            obj = Table(bytearray(), 0)
            self._tab.Union(obj, o)
            return obj
        return None

    def Test4(self, j):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            obj = Test()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def Test4Length(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def Test4IsNone(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        return o == 0

    def Testarrayofstring(self, j):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ''

    def TestarrayofstringLength(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def TestarrayofstringIsNone(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        return o == 0

    def Testarrayoftables(self, j):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            obj = Monster()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def TestarrayoftablesLength(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def TestarrayoftablesIsNone(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        return o == 0

    def Enemy(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(28))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            obj = Monster()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def Testnestedflatbuffer(self, j):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(30))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    def TestnestedflatbufferAsNumpy(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(30))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    def TestnestedflatbufferNestedRoot(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(30))
        if o != 0:
            from MyGame.Example.Monster import Monster
            return Monster.GetRootAs(self._tab.Bytes, self._tab.Vector(o))
        return 0

    def TestnestedflatbufferLength(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(30))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def TestnestedflatbufferIsNone(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(30))
        return o == 0

    def Testempty(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(32))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            obj = Stat()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def Testbool(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(34))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    def Testhashs32Fnv1(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(36))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    def Testhashu32Fnv1(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(38))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    def Testhashs64Fnv1(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(40))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int64Flags, o + self._tab.Pos)
        return 0

    def Testhashu64Fnv1(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(42))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, o + self._tab.Pos)
        return 0

    def Testhashs32Fnv1a(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(44))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    def Testhashu32Fnv1a(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(46))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    def Testhashs64Fnv1a(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(48))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int64Flags, o + self._tab.Pos)
        return 0

    def Testhashu64Fnv1a(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(50))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, o + self._tab.Pos)
        return 0

    def Testarrayofbools(self, j):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(52))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.BoolFlags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    def TestarrayofboolsAsNumpy(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(52))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.BoolFlags, o)
        return 0

    def TestarrayofboolsLength(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(52))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def TestarrayofboolsIsNone(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(52))
        return o == 0

    def Testf(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(54))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 3.14159

    def Testf2(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(56))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 3.0

    def Testf3(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(58))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

    def Testarrayofstring2(self, j):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(60))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ''

    def Testarrayofstring2Length(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(60))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def Testarrayofstring2IsNone(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(60))
        return o == 0

    def Testarrayofsortedstruct(self, j):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(62))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 8
            obj = Ability()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def TestarrayofsortedstructLength(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(62))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def TestarrayofsortedstructIsNone(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(62))
        return o == 0

    def Flex(self, j):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(64))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    def FlexAsNumpy(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(64))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    def FlexLength(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(64))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def FlexIsNone(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(64))
        return o == 0

    def Test5(self, j):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(66))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            obj = Test()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def Test5Length(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(66))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def Test5IsNone(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(66))
        return o == 0

    def VectorOfLongs(self, j):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(68))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int64Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 8))
        return 0

    def VectorOfLongsAsNumpy(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(68))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int64Flags, o)
        return 0

    def VectorOfLongsLength(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(68))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def VectorOfLongsIsNone(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(68))
        return o == 0

    def VectorOfDoubles(self, j):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(70))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Float64Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 8))
        return 0

    def VectorOfDoublesAsNumpy(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(70))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Float64Flags, o)
        return 0

    def VectorOfDoublesLength(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(70))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def VectorOfDoublesIsNone(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(70))
        return o == 0

    def ParentNamespaceTest(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(72))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            obj = InParentNamespace()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def VectorOfReferrables(self, j):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(74))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            obj = Referrable()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def VectorOfReferrablesLength(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(74))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def VectorOfReferrablesIsNone(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(74))
        return o == 0

    def SingleWeakReference(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(76))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, o + self._tab.Pos)
        return 0

    def VectorOfWeakReferences(self, j):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(78))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 8))
        return 0

    def VectorOfWeakReferencesAsNumpy(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(78))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint64Flags, o)
        return 0

    def VectorOfWeakReferencesLength(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(78))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def VectorOfWeakReferencesIsNone(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(78))
        return o == 0

    def VectorOfStrongReferrables(self, j):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(80))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            obj = Referrable()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def VectorOfStrongReferrablesLength(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(80))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def VectorOfStrongReferrablesIsNone(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(80))
        return o == 0

    def CoOwningReference(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(82))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, o + self._tab.Pos)
        return 0

    def VectorOfCoOwningReferences(self, j):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(84))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 8))
        return 0

    def VectorOfCoOwningReferencesAsNumpy(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(84))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint64Flags, o)
        return 0

    def VectorOfCoOwningReferencesLength(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(84))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def VectorOfCoOwningReferencesIsNone(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(84))
        return o == 0

    def NonOwningReference(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(86))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, o + self._tab.Pos)
        return 0

    def VectorOfNonOwningReferences(self, j):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(88))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 8))
        return 0

    def VectorOfNonOwningReferencesAsNumpy(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(88))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint64Flags, o)
        return 0

    def VectorOfNonOwningReferencesLength(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(88))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def VectorOfNonOwningReferencesIsNone(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(88))
        return o == 0

    def AnyUniqueType(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(90))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    def AnyUnique(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(92))
        if o != 0:
            from flatbuffers.table import Table
            obj = Table(bytearray(), 0)
            self._tab.Union(obj, o)
            return obj
        return None

    def AnyAmbiguousType(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(94))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    def AnyAmbiguous(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(96))
        if o != 0:
            from flatbuffers.table import Table
            obj = Table(bytearray(), 0)
            self._tab.Union(obj, o)
            return obj
        return None

    def VectorOfEnums(self, j):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(98))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    def VectorOfEnumsAsNumpy(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(98))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    def VectorOfEnumsLength(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(98))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def VectorOfEnumsIsNone(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(98))
        return o == 0

    def SignedEnum(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(100))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return -1

    def Testrequirednestedflatbuffer(self, j):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(102))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    def TestrequirednestedflatbufferAsNumpy(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(102))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    def TestrequirednestedflatbufferNestedRoot(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(102))
        if o != 0:
            from MyGame.Example.Monster import Monster
            return Monster.GetRootAs(self._tab.Bytes, self._tab.Vector(o))
        return 0

    def TestrequirednestedflatbufferLength(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(102))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def TestrequirednestedflatbufferIsNone(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(102))
        return o == 0

    def ScalarKeySortedTables(self, j):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(104))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            obj = Stat()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def ScalarKeySortedTablesLength(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(104))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def ScalarKeySortedTablesIsNone(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(104))
        return o == 0

    def NativeInline(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(106))
        if o != 0:
            x = o + self._tab.Pos
            obj = Test()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def LongEnumNonEnumDefault(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(108))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, o + self._tab.Pos)
        return 0

    def LongEnumNormalDefault(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(110))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, o + self._tab.Pos)
        return 2

    def NanDefault(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(112))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return float('nan')

    def InfDefault(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(114))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return float('inf')

    def PositiveInfDefault(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(116))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return float('inf')

    def InfinityDefault(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(118))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return float('inf')

    def PositiveInfinityDefault(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(120))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return float('inf')

    def NegativeInfDefault(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(122))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return float('-inf')

    def NegativeInfinityDefault(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(124))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return float('-inf')

    def DoubleInfDefault(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(126))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float64Flags, o + self._tab.Pos)
        return float('inf')

def MonsterStart(builder):
    if False:
        while True:
            i = 10
    builder.StartObject(62)

def MonsterAddPos(builder, pos):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependStructSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(pos), 0)

def MonsterAddMana(builder, mana):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependInt16Slot(1, mana, 150)

def MonsterAddHp(builder, hp):
    if False:
        print('Hello World!')
    builder.PrependInt16Slot(2, hp, 100)

def MonsterAddName(builder, name):
    if False:
        return 10
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)

def MonsterAddInventory(builder, inventory):
    if False:
        i = 10
        return i + 15
    builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(inventory), 0)

def MonsterStartInventoryVector(builder, numElems):
    if False:
        i = 10
        return i + 15
    return builder.StartVector(1, numElems, 1)

def MonsterAddColor(builder, color):
    if False:
        while True:
            i = 10
    builder.PrependUint8Slot(6, color, 8)

def MonsterAddTestType(builder, testType):
    if False:
        i = 10
        return i + 15
    builder.PrependUint8Slot(7, testType, 0)

def MonsterAddTest(builder, test):
    if False:
        while True:
            i = 10
    builder.PrependUOffsetTRelativeSlot(8, flatbuffers.number_types.UOffsetTFlags.py_type(test), 0)

def MonsterAddTest4(builder, test4):
    if False:
        i = 10
        return i + 15
    builder.PrependUOffsetTRelativeSlot(9, flatbuffers.number_types.UOffsetTFlags.py_type(test4), 0)

def MonsterStartTest4Vector(builder, numElems):
    if False:
        while True:
            i = 10
    return builder.StartVector(4, numElems, 2)

def MonsterAddTestarrayofstring(builder, testarrayofstring):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependUOffsetTRelativeSlot(10, flatbuffers.number_types.UOffsetTFlags.py_type(testarrayofstring), 0)

def MonsterStartTestarrayofstringVector(builder, numElems):
    if False:
        return 10
    return builder.StartVector(4, numElems, 4)

def MonsterAddTestarrayoftables(builder, testarrayoftables):
    if False:
        i = 10
        return i + 15
    builder.PrependUOffsetTRelativeSlot(11, flatbuffers.number_types.UOffsetTFlags.py_type(testarrayoftables), 0)

def MonsterStartTestarrayoftablesVector(builder, numElems):
    if False:
        for i in range(10):
            print('nop')
    return builder.StartVector(4, numElems, 4)

def MonsterAddEnemy(builder, enemy):
    if False:
        i = 10
        return i + 15
    builder.PrependUOffsetTRelativeSlot(12, flatbuffers.number_types.UOffsetTFlags.py_type(enemy), 0)

def MonsterAddTestnestedflatbuffer(builder, testnestedflatbuffer):
    if False:
        return 10
    builder.PrependUOffsetTRelativeSlot(13, flatbuffers.number_types.UOffsetTFlags.py_type(testnestedflatbuffer), 0)

def MonsterStartTestnestedflatbufferVector(builder, numElems):
    if False:
        print('Hello World!')
    return builder.StartVector(1, numElems, 1)

def MonsterMakeTestnestedflatbufferVectorFromBytes(builder, bytes):
    if False:
        i = 10
        return i + 15
    builder.StartVector(1, len(bytes), 1)
    builder.head = builder.head - len(bytes)
    builder.Bytes[builder.head:builder.head + len(bytes)] = bytes
    return builder.EndVector()

def MonsterAddTestempty(builder, testempty):
    if False:
        i = 10
        return i + 15
    builder.PrependUOffsetTRelativeSlot(14, flatbuffers.number_types.UOffsetTFlags.py_type(testempty), 0)

def MonsterAddTestbool(builder, testbool):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependBoolSlot(15, testbool, 0)

def MonsterAddTesthashs32Fnv1(builder, testhashs32Fnv1):
    if False:
        while True:
            i = 10
    builder.PrependInt32Slot(16, testhashs32Fnv1, 0)

def MonsterAddTesthashu32Fnv1(builder, testhashu32Fnv1):
    if False:
        while True:
            i = 10
    builder.PrependUint32Slot(17, testhashu32Fnv1, 0)

def MonsterAddTesthashs64Fnv1(builder, testhashs64Fnv1):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependInt64Slot(18, testhashs64Fnv1, 0)

def MonsterAddTesthashu64Fnv1(builder, testhashu64Fnv1):
    if False:
        return 10
    builder.PrependUint64Slot(19, testhashu64Fnv1, 0)

def MonsterAddTesthashs32Fnv1a(builder, testhashs32Fnv1a):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependInt32Slot(20, testhashs32Fnv1a, 0)

def MonsterAddTesthashu32Fnv1a(builder, testhashu32Fnv1a):
    if False:
        i = 10
        return i + 15
    builder.PrependUint32Slot(21, testhashu32Fnv1a, 0)

def MonsterAddTesthashs64Fnv1a(builder, testhashs64Fnv1a):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependInt64Slot(22, testhashs64Fnv1a, 0)

def MonsterAddTesthashu64Fnv1a(builder, testhashu64Fnv1a):
    if False:
        while True:
            i = 10
    builder.PrependUint64Slot(23, testhashu64Fnv1a, 0)

def MonsterAddTestarrayofbools(builder, testarrayofbools):
    if False:
        i = 10
        return i + 15
    builder.PrependUOffsetTRelativeSlot(24, flatbuffers.number_types.UOffsetTFlags.py_type(testarrayofbools), 0)

def MonsterStartTestarrayofboolsVector(builder, numElems):
    if False:
        for i in range(10):
            print('nop')
    return builder.StartVector(1, numElems, 1)

def MonsterAddTestf(builder, testf):
    if False:
        i = 10
        return i + 15
    builder.PrependFloat32Slot(25, testf, 3.14159)

def MonsterAddTestf2(builder, testf2):
    if False:
        while True:
            i = 10
    builder.PrependFloat32Slot(26, testf2, 3.0)

def MonsterAddTestf3(builder, testf3):
    if False:
        i = 10
        return i + 15
    builder.PrependFloat32Slot(27, testf3, 0.0)

def MonsterAddTestarrayofstring2(builder, testarrayofstring2):
    if False:
        i = 10
        return i + 15
    builder.PrependUOffsetTRelativeSlot(28, flatbuffers.number_types.UOffsetTFlags.py_type(testarrayofstring2), 0)

def MonsterStartTestarrayofstring2Vector(builder, numElems):
    if False:
        i = 10
        return i + 15
    return builder.StartVector(4, numElems, 4)

def MonsterAddTestarrayofsortedstruct(builder, testarrayofsortedstruct):
    if False:
        while True:
            i = 10
    builder.PrependUOffsetTRelativeSlot(29, flatbuffers.number_types.UOffsetTFlags.py_type(testarrayofsortedstruct), 0)

def MonsterStartTestarrayofsortedstructVector(builder, numElems):
    if False:
        return 10
    return builder.StartVector(8, numElems, 4)

def MonsterAddFlex(builder, flex):
    if False:
        while True:
            i = 10
    builder.PrependUOffsetTRelativeSlot(30, flatbuffers.number_types.UOffsetTFlags.py_type(flex), 0)

def MonsterStartFlexVector(builder, numElems):
    if False:
        print('Hello World!')
    return builder.StartVector(1, numElems, 1)

def MonsterAddTest5(builder, test5):
    if False:
        while True:
            i = 10
    builder.PrependUOffsetTRelativeSlot(31, flatbuffers.number_types.UOffsetTFlags.py_type(test5), 0)

def MonsterStartTest5Vector(builder, numElems):
    if False:
        i = 10
        return i + 15
    return builder.StartVector(4, numElems, 2)

def MonsterAddVectorOfLongs(builder, vectorOfLongs):
    if False:
        return 10
    builder.PrependUOffsetTRelativeSlot(32, flatbuffers.number_types.UOffsetTFlags.py_type(vectorOfLongs), 0)

def MonsterStartVectorOfLongsVector(builder, numElems):
    if False:
        return 10
    return builder.StartVector(8, numElems, 8)

def MonsterAddVectorOfDoubles(builder, vectorOfDoubles):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependUOffsetTRelativeSlot(33, flatbuffers.number_types.UOffsetTFlags.py_type(vectorOfDoubles), 0)

def MonsterStartVectorOfDoublesVector(builder, numElems):
    if False:
        while True:
            i = 10
    return builder.StartVector(8, numElems, 8)

def MonsterAddParentNamespaceTest(builder, parentNamespaceTest):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependUOffsetTRelativeSlot(34, flatbuffers.number_types.UOffsetTFlags.py_type(parentNamespaceTest), 0)

def MonsterAddVectorOfReferrables(builder, vectorOfReferrables):
    if False:
        print('Hello World!')
    builder.PrependUOffsetTRelativeSlot(35, flatbuffers.number_types.UOffsetTFlags.py_type(vectorOfReferrables), 0)

def MonsterStartVectorOfReferrablesVector(builder, numElems):
    if False:
        return 10
    return builder.StartVector(4, numElems, 4)

def MonsterAddSingleWeakReference(builder, singleWeakReference):
    if False:
        return 10
    builder.PrependUint64Slot(36, singleWeakReference, 0)

def MonsterAddVectorOfWeakReferences(builder, vectorOfWeakReferences):
    if False:
        i = 10
        return i + 15
    builder.PrependUOffsetTRelativeSlot(37, flatbuffers.number_types.UOffsetTFlags.py_type(vectorOfWeakReferences), 0)

def MonsterStartVectorOfWeakReferencesVector(builder, numElems):
    if False:
        i = 10
        return i + 15
    return builder.StartVector(8, numElems, 8)

def MonsterAddVectorOfStrongReferrables(builder, vectorOfStrongReferrables):
    if False:
        return 10
    builder.PrependUOffsetTRelativeSlot(38, flatbuffers.number_types.UOffsetTFlags.py_type(vectorOfStrongReferrables), 0)

def MonsterStartVectorOfStrongReferrablesVector(builder, numElems):
    if False:
        for i in range(10):
            print('nop')
    return builder.StartVector(4, numElems, 4)

def MonsterAddCoOwningReference(builder, coOwningReference):
    if False:
        while True:
            i = 10
    builder.PrependUint64Slot(39, coOwningReference, 0)

def MonsterAddVectorOfCoOwningReferences(builder, vectorOfCoOwningReferences):
    if False:
        print('Hello World!')
    builder.PrependUOffsetTRelativeSlot(40, flatbuffers.number_types.UOffsetTFlags.py_type(vectorOfCoOwningReferences), 0)

def MonsterStartVectorOfCoOwningReferencesVector(builder, numElems):
    if False:
        for i in range(10):
            print('nop')
    return builder.StartVector(8, numElems, 8)

def MonsterAddNonOwningReference(builder, nonOwningReference):
    if False:
        print('Hello World!')
    builder.PrependUint64Slot(41, nonOwningReference, 0)

def MonsterAddVectorOfNonOwningReferences(builder, vectorOfNonOwningReferences):
    if False:
        i = 10
        return i + 15
    builder.PrependUOffsetTRelativeSlot(42, flatbuffers.number_types.UOffsetTFlags.py_type(vectorOfNonOwningReferences), 0)

def MonsterStartVectorOfNonOwningReferencesVector(builder, numElems):
    if False:
        return 10
    return builder.StartVector(8, numElems, 8)

def MonsterAddAnyUniqueType(builder, anyUniqueType):
    if False:
        i = 10
        return i + 15
    builder.PrependUint8Slot(43, anyUniqueType, 0)

def MonsterAddAnyUnique(builder, anyUnique):
    if False:
        while True:
            i = 10
    builder.PrependUOffsetTRelativeSlot(44, flatbuffers.number_types.UOffsetTFlags.py_type(anyUnique), 0)

def MonsterAddAnyAmbiguousType(builder, anyAmbiguousType):
    if False:
        print('Hello World!')
    builder.PrependUint8Slot(45, anyAmbiguousType, 0)

def MonsterAddAnyAmbiguous(builder, anyAmbiguous):
    if False:
        return 10
    builder.PrependUOffsetTRelativeSlot(46, flatbuffers.number_types.UOffsetTFlags.py_type(anyAmbiguous), 0)

def MonsterAddVectorOfEnums(builder, vectorOfEnums):
    if False:
        while True:
            i = 10
    builder.PrependUOffsetTRelativeSlot(47, flatbuffers.number_types.UOffsetTFlags.py_type(vectorOfEnums), 0)

def MonsterStartVectorOfEnumsVector(builder, numElems):
    if False:
        while True:
            i = 10
    return builder.StartVector(1, numElems, 1)

def MonsterAddSignedEnum(builder, signedEnum):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependInt8Slot(48, signedEnum, -1)

def MonsterAddTestrequirednestedflatbuffer(builder, testrequirednestedflatbuffer):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependUOffsetTRelativeSlot(49, flatbuffers.number_types.UOffsetTFlags.py_type(testrequirednestedflatbuffer), 0)

def MonsterStartTestrequirednestedflatbufferVector(builder, numElems):
    if False:
        print('Hello World!')
    return builder.StartVector(1, numElems, 1)

def MonsterMakeTestrequirednestedflatbufferVectorFromBytes(builder, bytes):
    if False:
        return 10
    builder.StartVector(1, len(bytes), 1)
    builder.head = builder.head - len(bytes)
    builder.Bytes[builder.head:builder.head + len(bytes)] = bytes
    return builder.EndVector()

def MonsterAddScalarKeySortedTables(builder, scalarKeySortedTables):
    if False:
        i = 10
        return i + 15
    builder.PrependUOffsetTRelativeSlot(50, flatbuffers.number_types.UOffsetTFlags.py_type(scalarKeySortedTables), 0)

def MonsterStartScalarKeySortedTablesVector(builder, numElems):
    if False:
        print('Hello World!')
    return builder.StartVector(4, numElems, 4)

def MonsterAddNativeInline(builder, nativeInline):
    if False:
        while True:
            i = 10
    builder.PrependStructSlot(51, flatbuffers.number_types.UOffsetTFlags.py_type(nativeInline), 0)

def MonsterAddLongEnumNonEnumDefault(builder, longEnumNonEnumDefault):
    if False:
        i = 10
        return i + 15
    builder.PrependUint64Slot(52, longEnumNonEnumDefault, 0)

def MonsterAddLongEnumNormalDefault(builder, longEnumNormalDefault):
    if False:
        return 10
    builder.PrependUint64Slot(53, longEnumNormalDefault, 2)

def MonsterAddNanDefault(builder, nanDefault):
    if False:
        return 10
    builder.PrependFloat32Slot(54, nanDefault, float('nan'))

def MonsterAddInfDefault(builder, infDefault):
    if False:
        i = 10
        return i + 15
    builder.PrependFloat32Slot(55, infDefault, float('inf'))

def MonsterAddPositiveInfDefault(builder, positiveInfDefault):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependFloat32Slot(56, positiveInfDefault, float('inf'))

def MonsterAddInfinityDefault(builder, infinityDefault):
    if False:
        return 10
    builder.PrependFloat32Slot(57, infinityDefault, float('inf'))

def MonsterAddPositiveInfinityDefault(builder, positiveInfinityDefault):
    if False:
        while True:
            i = 10
    builder.PrependFloat32Slot(58, positiveInfinityDefault, float('inf'))

def MonsterAddNegativeInfDefault(builder, negativeInfDefault):
    if False:
        while True:
            i = 10
    builder.PrependFloat32Slot(59, negativeInfDefault, float('-inf'))

def MonsterAddNegativeInfinityDefault(builder, negativeInfinityDefault):
    if False:
        print('Hello World!')
    builder.PrependFloat32Slot(60, negativeInfinityDefault, float('-inf'))

def MonsterAddDoubleInfDefault(builder, doubleInfDefault):
    if False:
        print('Hello World!')
    builder.PrependFloat64Slot(61, doubleInfDefault, float('inf'))

def MonsterEnd(builder):
    if False:
        i = 10
        return i + 15
    return builder.EndObject()
try:
    from typing import List, Optional, Union
except:
    pass

class MonsterT(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.pos = None
        self.mana = 150
        self.hp = 100
        self.name = None
        self.inventory = None
        self.color = 8
        self.testType = 0
        self.test = None
        self.test4 = None
        self.testarrayofstring = None
        self.testarrayoftables = None
        self.enemy = None
        self.testnestedflatbuffer = None
        self.testempty = None
        self.testbool = False
        self.testhashs32Fnv1 = 0
        self.testhashu32Fnv1 = 0
        self.testhashs64Fnv1 = 0
        self.testhashu64Fnv1 = 0
        self.testhashs32Fnv1a = 0
        self.testhashu32Fnv1a = 0
        self.testhashs64Fnv1a = 0
        self.testhashu64Fnv1a = 0
        self.testarrayofbools = None
        self.testf = 3.14159
        self.testf2 = 3.0
        self.testf3 = 0.0
        self.testarrayofstring2 = None
        self.testarrayofsortedstruct = None
        self.flex = None
        self.test5 = None
        self.vectorOfLongs = None
        self.vectorOfDoubles = None
        self.parentNamespaceTest = None
        self.vectorOfReferrables = None
        self.singleWeakReference = 0
        self.vectorOfWeakReferences = None
        self.vectorOfStrongReferrables = None
        self.coOwningReference = 0
        self.vectorOfCoOwningReferences = None
        self.nonOwningReference = 0
        self.vectorOfNonOwningReferences = None
        self.anyUniqueType = 0
        self.anyUnique = None
        self.anyAmbiguousType = 0
        self.anyAmbiguous = None
        self.vectorOfEnums = None
        self.signedEnum = -1
        self.testrequirednestedflatbuffer = None
        self.scalarKeySortedTables = None
        self.nativeInline = None
        self.longEnumNonEnumDefault = 0
        self.longEnumNormalDefault = 2
        self.nanDefault = float('nan')
        self.infDefault = float('inf')
        self.positiveInfDefault = float('inf')
        self.infinityDefault = float('inf')
        self.positiveInfinityDefault = float('inf')
        self.negativeInfDefault = float('-inf')
        self.negativeInfinityDefault = float('-inf')
        self.doubleInfDefault = float('inf')

    @classmethod
    def InitFromBuf(cls, buf, pos):
        if False:
            i = 10
            return i + 15
        monster = Monster()
        monster.Init(buf, pos)
        return cls.InitFromObj(monster)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        if False:
            i = 10
            return i + 15
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, monster):
        if False:
            for i in range(10):
                print('nop')
        x = MonsterT()
        x._UnPack(monster)
        return x

    def _UnPack(self, monster):
        if False:
            for i in range(10):
                print('nop')
        if monster is None:
            return
        if monster.Pos() is not None:
            self.pos = Vec3T.InitFromObj(monster.Pos())
        self.mana = monster.Mana()
        self.hp = monster.Hp()
        self.name = monster.Name()
        if not monster.InventoryIsNone():
            if np is None:
                self.inventory = []
                for i in range(monster.InventoryLength()):
                    self.inventory.append(monster.Inventory(i))
            else:
                self.inventory = monster.InventoryAsNumpy()
        self.color = monster.Color()
        self.testType = monster.TestType()
        self.test = AnyCreator(self.testType, monster.Test())
        if not monster.Test4IsNone():
            self.test4 = []
            for i in range(monster.Test4Length()):
                if monster.Test4(i) is None:
                    self.test4.append(None)
                else:
                    test_ = TestT.InitFromObj(monster.Test4(i))
                    self.test4.append(test_)
        if not monster.TestarrayofstringIsNone():
            self.testarrayofstring = []
            for i in range(monster.TestarrayofstringLength()):
                self.testarrayofstring.append(monster.Testarrayofstring(i))
        if not monster.TestarrayoftablesIsNone():
            self.testarrayoftables = []
            for i in range(monster.TestarrayoftablesLength()):
                if monster.Testarrayoftables(i) is None:
                    self.testarrayoftables.append(None)
                else:
                    monster_ = MonsterT.InitFromObj(monster.Testarrayoftables(i))
                    self.testarrayoftables.append(monster_)
        if monster.Enemy() is not None:
            self.enemy = MonsterT.InitFromObj(monster.Enemy())
        if not monster.TestnestedflatbufferIsNone():
            if np is None:
                self.testnestedflatbuffer = []
                for i in range(monster.TestnestedflatbufferLength()):
                    self.testnestedflatbuffer.append(monster.Testnestedflatbuffer(i))
            else:
                self.testnestedflatbuffer = monster.TestnestedflatbufferAsNumpy()
        if monster.Testempty() is not None:
            self.testempty = StatT.InitFromObj(monster.Testempty())
        self.testbool = monster.Testbool()
        self.testhashs32Fnv1 = monster.Testhashs32Fnv1()
        self.testhashu32Fnv1 = monster.Testhashu32Fnv1()
        self.testhashs64Fnv1 = monster.Testhashs64Fnv1()
        self.testhashu64Fnv1 = monster.Testhashu64Fnv1()
        self.testhashs32Fnv1a = monster.Testhashs32Fnv1a()
        self.testhashu32Fnv1a = monster.Testhashu32Fnv1a()
        self.testhashs64Fnv1a = monster.Testhashs64Fnv1a()
        self.testhashu64Fnv1a = monster.Testhashu64Fnv1a()
        if not monster.TestarrayofboolsIsNone():
            if np is None:
                self.testarrayofbools = []
                for i in range(monster.TestarrayofboolsLength()):
                    self.testarrayofbools.append(monster.Testarrayofbools(i))
            else:
                self.testarrayofbools = monster.TestarrayofboolsAsNumpy()
        self.testf = monster.Testf()
        self.testf2 = monster.Testf2()
        self.testf3 = monster.Testf3()
        if not monster.Testarrayofstring2IsNone():
            self.testarrayofstring2 = []
            for i in range(monster.Testarrayofstring2Length()):
                self.testarrayofstring2.append(monster.Testarrayofstring2(i))
        if not monster.TestarrayofsortedstructIsNone():
            self.testarrayofsortedstruct = []
            for i in range(monster.TestarrayofsortedstructLength()):
                if monster.Testarrayofsortedstruct(i) is None:
                    self.testarrayofsortedstruct.append(None)
                else:
                    ability_ = AbilityT.InitFromObj(monster.Testarrayofsortedstruct(i))
                    self.testarrayofsortedstruct.append(ability_)
        if not monster.FlexIsNone():
            if np is None:
                self.flex = []
                for i in range(monster.FlexLength()):
                    self.flex.append(monster.Flex(i))
            else:
                self.flex = monster.FlexAsNumpy()
        if not monster.Test5IsNone():
            self.test5 = []
            for i in range(monster.Test5Length()):
                if monster.Test5(i) is None:
                    self.test5.append(None)
                else:
                    test_ = TestT.InitFromObj(monster.Test5(i))
                    self.test5.append(test_)
        if not monster.VectorOfLongsIsNone():
            if np is None:
                self.vectorOfLongs = []
                for i in range(monster.VectorOfLongsLength()):
                    self.vectorOfLongs.append(monster.VectorOfLongs(i))
            else:
                self.vectorOfLongs = monster.VectorOfLongsAsNumpy()
        if not monster.VectorOfDoublesIsNone():
            if np is None:
                self.vectorOfDoubles = []
                for i in range(monster.VectorOfDoublesLength()):
                    self.vectorOfDoubles.append(monster.VectorOfDoubles(i))
            else:
                self.vectorOfDoubles = monster.VectorOfDoublesAsNumpy()
        if monster.ParentNamespaceTest() is not None:
            self.parentNamespaceTest = InParentNamespaceT.InitFromObj(monster.ParentNamespaceTest())
        if not monster.VectorOfReferrablesIsNone():
            self.vectorOfReferrables = []
            for i in range(monster.VectorOfReferrablesLength()):
                if monster.VectorOfReferrables(i) is None:
                    self.vectorOfReferrables.append(None)
                else:
                    referrable_ = ReferrableT.InitFromObj(monster.VectorOfReferrables(i))
                    self.vectorOfReferrables.append(referrable_)
        self.singleWeakReference = monster.SingleWeakReference()
        if not monster.VectorOfWeakReferencesIsNone():
            if np is None:
                self.vectorOfWeakReferences = []
                for i in range(monster.VectorOfWeakReferencesLength()):
                    self.vectorOfWeakReferences.append(monster.VectorOfWeakReferences(i))
            else:
                self.vectorOfWeakReferences = monster.VectorOfWeakReferencesAsNumpy()
        if not monster.VectorOfStrongReferrablesIsNone():
            self.vectorOfStrongReferrables = []
            for i in range(monster.VectorOfStrongReferrablesLength()):
                if monster.VectorOfStrongReferrables(i) is None:
                    self.vectorOfStrongReferrables.append(None)
                else:
                    referrable_ = ReferrableT.InitFromObj(monster.VectorOfStrongReferrables(i))
                    self.vectorOfStrongReferrables.append(referrable_)
        self.coOwningReference = monster.CoOwningReference()
        if not monster.VectorOfCoOwningReferencesIsNone():
            if np is None:
                self.vectorOfCoOwningReferences = []
                for i in range(monster.VectorOfCoOwningReferencesLength()):
                    self.vectorOfCoOwningReferences.append(monster.VectorOfCoOwningReferences(i))
            else:
                self.vectorOfCoOwningReferences = monster.VectorOfCoOwningReferencesAsNumpy()
        self.nonOwningReference = monster.NonOwningReference()
        if not monster.VectorOfNonOwningReferencesIsNone():
            if np is None:
                self.vectorOfNonOwningReferences = []
                for i in range(monster.VectorOfNonOwningReferencesLength()):
                    self.vectorOfNonOwningReferences.append(monster.VectorOfNonOwningReferences(i))
            else:
                self.vectorOfNonOwningReferences = monster.VectorOfNonOwningReferencesAsNumpy()
        self.anyUniqueType = monster.AnyUniqueType()
        self.anyUnique = AnyUniqueAliasesCreator(self.anyUniqueType, monster.AnyUnique())
        self.anyAmbiguousType = monster.AnyAmbiguousType()
        self.anyAmbiguous = AnyAmbiguousAliasesCreator(self.anyAmbiguousType, monster.AnyAmbiguous())
        if not monster.VectorOfEnumsIsNone():
            if np is None:
                self.vectorOfEnums = []
                for i in range(monster.VectorOfEnumsLength()):
                    self.vectorOfEnums.append(monster.VectorOfEnums(i))
            else:
                self.vectorOfEnums = monster.VectorOfEnumsAsNumpy()
        self.signedEnum = monster.SignedEnum()
        if not monster.TestrequirednestedflatbufferIsNone():
            if np is None:
                self.testrequirednestedflatbuffer = []
                for i in range(monster.TestrequirednestedflatbufferLength()):
                    self.testrequirednestedflatbuffer.append(monster.Testrequirednestedflatbuffer(i))
            else:
                self.testrequirednestedflatbuffer = monster.TestrequirednestedflatbufferAsNumpy()
        if not monster.ScalarKeySortedTablesIsNone():
            self.scalarKeySortedTables = []
            for i in range(monster.ScalarKeySortedTablesLength()):
                if monster.ScalarKeySortedTables(i) is None:
                    self.scalarKeySortedTables.append(None)
                else:
                    stat_ = StatT.InitFromObj(monster.ScalarKeySortedTables(i))
                    self.scalarKeySortedTables.append(stat_)
        if monster.NativeInline() is not None:
            self.nativeInline = TestT.InitFromObj(monster.NativeInline())
        self.longEnumNonEnumDefault = monster.LongEnumNonEnumDefault()
        self.longEnumNormalDefault = monster.LongEnumNormalDefault()
        self.nanDefault = monster.NanDefault()
        self.infDefault = monster.InfDefault()
        self.positiveInfDefault = monster.PositiveInfDefault()
        self.infinityDefault = monster.InfinityDefault()
        self.positiveInfinityDefault = monster.PositiveInfinityDefault()
        self.negativeInfDefault = monster.NegativeInfDefault()
        self.negativeInfinityDefault = monster.NegativeInfinityDefault()
        self.doubleInfDefault = monster.DoubleInfDefault()

    def Pack(self, builder):
        if False:
            return 10
        if self.name is not None:
            name = builder.CreateString(self.name)
        if self.inventory is not None:
            if np is not None and type(self.inventory) is np.ndarray:
                inventory = builder.CreateNumpyVector(self.inventory)
            else:
                MonsterStartInventoryVector(builder, len(self.inventory))
                for i in reversed(range(len(self.inventory))):
                    builder.PrependUint8(self.inventory[i])
                inventory = builder.EndVector()
        if self.test is not None:
            test = self.test.Pack(builder)
        if self.test4 is not None:
            MonsterStartTest4Vector(builder, len(self.test4))
            for i in reversed(range(len(self.test4))):
                self.test4[i].Pack(builder)
            test4 = builder.EndVector()
        if self.testarrayofstring is not None:
            testarrayofstringlist = []
            for i in range(len(self.testarrayofstring)):
                testarrayofstringlist.append(builder.CreateString(self.testarrayofstring[i]))
            MonsterStartTestarrayofstringVector(builder, len(self.testarrayofstring))
            for i in reversed(range(len(self.testarrayofstring))):
                builder.PrependUOffsetTRelative(testarrayofstringlist[i])
            testarrayofstring = builder.EndVector()
        if self.testarrayoftables is not None:
            testarrayoftableslist = []
            for i in range(len(self.testarrayoftables)):
                testarrayoftableslist.append(self.testarrayoftables[i].Pack(builder))
            MonsterStartTestarrayoftablesVector(builder, len(self.testarrayoftables))
            for i in reversed(range(len(self.testarrayoftables))):
                builder.PrependUOffsetTRelative(testarrayoftableslist[i])
            testarrayoftables = builder.EndVector()
        if self.enemy is not None:
            enemy = self.enemy.Pack(builder)
        if self.testnestedflatbuffer is not None:
            if np is not None and type(self.testnestedflatbuffer) is np.ndarray:
                testnestedflatbuffer = builder.CreateNumpyVector(self.testnestedflatbuffer)
            else:
                MonsterStartTestnestedflatbufferVector(builder, len(self.testnestedflatbuffer))
                for i in reversed(range(len(self.testnestedflatbuffer))):
                    builder.PrependUint8(self.testnestedflatbuffer[i])
                testnestedflatbuffer = builder.EndVector()
        if self.testempty is not None:
            testempty = self.testempty.Pack(builder)
        if self.testarrayofbools is not None:
            if np is not None and type(self.testarrayofbools) is np.ndarray:
                testarrayofbools = builder.CreateNumpyVector(self.testarrayofbools)
            else:
                MonsterStartTestarrayofboolsVector(builder, len(self.testarrayofbools))
                for i in reversed(range(len(self.testarrayofbools))):
                    builder.PrependBool(self.testarrayofbools[i])
                testarrayofbools = builder.EndVector()
        if self.testarrayofstring2 is not None:
            testarrayofstring2list = []
            for i in range(len(self.testarrayofstring2)):
                testarrayofstring2list.append(builder.CreateString(self.testarrayofstring2[i]))
            MonsterStartTestarrayofstring2Vector(builder, len(self.testarrayofstring2))
            for i in reversed(range(len(self.testarrayofstring2))):
                builder.PrependUOffsetTRelative(testarrayofstring2list[i])
            testarrayofstring2 = builder.EndVector()
        if self.testarrayofsortedstruct is not None:
            MonsterStartTestarrayofsortedstructVector(builder, len(self.testarrayofsortedstruct))
            for i in reversed(range(len(self.testarrayofsortedstruct))):
                self.testarrayofsortedstruct[i].Pack(builder)
            testarrayofsortedstruct = builder.EndVector()
        if self.flex is not None:
            if np is not None and type(self.flex) is np.ndarray:
                flex = builder.CreateNumpyVector(self.flex)
            else:
                MonsterStartFlexVector(builder, len(self.flex))
                for i in reversed(range(len(self.flex))):
                    builder.PrependUint8(self.flex[i])
                flex = builder.EndVector()
        if self.test5 is not None:
            MonsterStartTest5Vector(builder, len(self.test5))
            for i in reversed(range(len(self.test5))):
                self.test5[i].Pack(builder)
            test5 = builder.EndVector()
        if self.vectorOfLongs is not None:
            if np is not None and type(self.vectorOfLongs) is np.ndarray:
                vectorOfLongs = builder.CreateNumpyVector(self.vectorOfLongs)
            else:
                MonsterStartVectorOfLongsVector(builder, len(self.vectorOfLongs))
                for i in reversed(range(len(self.vectorOfLongs))):
                    builder.PrependInt64(self.vectorOfLongs[i])
                vectorOfLongs = builder.EndVector()
        if self.vectorOfDoubles is not None:
            if np is not None and type(self.vectorOfDoubles) is np.ndarray:
                vectorOfDoubles = builder.CreateNumpyVector(self.vectorOfDoubles)
            else:
                MonsterStartVectorOfDoublesVector(builder, len(self.vectorOfDoubles))
                for i in reversed(range(len(self.vectorOfDoubles))):
                    builder.PrependFloat64(self.vectorOfDoubles[i])
                vectorOfDoubles = builder.EndVector()
        if self.parentNamespaceTest is not None:
            parentNamespaceTest = self.parentNamespaceTest.Pack(builder)
        if self.vectorOfReferrables is not None:
            vectorOfReferrableslist = []
            for i in range(len(self.vectorOfReferrables)):
                vectorOfReferrableslist.append(self.vectorOfReferrables[i].Pack(builder))
            MonsterStartVectorOfReferrablesVector(builder, len(self.vectorOfReferrables))
            for i in reversed(range(len(self.vectorOfReferrables))):
                builder.PrependUOffsetTRelative(vectorOfReferrableslist[i])
            vectorOfReferrables = builder.EndVector()
        if self.vectorOfWeakReferences is not None:
            if np is not None and type(self.vectorOfWeakReferences) is np.ndarray:
                vectorOfWeakReferences = builder.CreateNumpyVector(self.vectorOfWeakReferences)
            else:
                MonsterStartVectorOfWeakReferencesVector(builder, len(self.vectorOfWeakReferences))
                for i in reversed(range(len(self.vectorOfWeakReferences))):
                    builder.PrependUint64(self.vectorOfWeakReferences[i])
                vectorOfWeakReferences = builder.EndVector()
        if self.vectorOfStrongReferrables is not None:
            vectorOfStrongReferrableslist = []
            for i in range(len(self.vectorOfStrongReferrables)):
                vectorOfStrongReferrableslist.append(self.vectorOfStrongReferrables[i].Pack(builder))
            MonsterStartVectorOfStrongReferrablesVector(builder, len(self.vectorOfStrongReferrables))
            for i in reversed(range(len(self.vectorOfStrongReferrables))):
                builder.PrependUOffsetTRelative(vectorOfStrongReferrableslist[i])
            vectorOfStrongReferrables = builder.EndVector()
        if self.vectorOfCoOwningReferences is not None:
            if np is not None and type(self.vectorOfCoOwningReferences) is np.ndarray:
                vectorOfCoOwningReferences = builder.CreateNumpyVector(self.vectorOfCoOwningReferences)
            else:
                MonsterStartVectorOfCoOwningReferencesVector(builder, len(self.vectorOfCoOwningReferences))
                for i in reversed(range(len(self.vectorOfCoOwningReferences))):
                    builder.PrependUint64(self.vectorOfCoOwningReferences[i])
                vectorOfCoOwningReferences = builder.EndVector()
        if self.vectorOfNonOwningReferences is not None:
            if np is not None and type(self.vectorOfNonOwningReferences) is np.ndarray:
                vectorOfNonOwningReferences = builder.CreateNumpyVector(self.vectorOfNonOwningReferences)
            else:
                MonsterStartVectorOfNonOwningReferencesVector(builder, len(self.vectorOfNonOwningReferences))
                for i in reversed(range(len(self.vectorOfNonOwningReferences))):
                    builder.PrependUint64(self.vectorOfNonOwningReferences[i])
                vectorOfNonOwningReferences = builder.EndVector()
        if self.anyUnique is not None:
            anyUnique = self.anyUnique.Pack(builder)
        if self.anyAmbiguous is not None:
            anyAmbiguous = self.anyAmbiguous.Pack(builder)
        if self.vectorOfEnums is not None:
            if np is not None and type(self.vectorOfEnums) is np.ndarray:
                vectorOfEnums = builder.CreateNumpyVector(self.vectorOfEnums)
            else:
                MonsterStartVectorOfEnumsVector(builder, len(self.vectorOfEnums))
                for i in reversed(range(len(self.vectorOfEnums))):
                    builder.PrependUint8(self.vectorOfEnums[i])
                vectorOfEnums = builder.EndVector()
        if self.testrequirednestedflatbuffer is not None:
            if np is not None and type(self.testrequirednestedflatbuffer) is np.ndarray:
                testrequirednestedflatbuffer = builder.CreateNumpyVector(self.testrequirednestedflatbuffer)
            else:
                MonsterStartTestrequirednestedflatbufferVector(builder, len(self.testrequirednestedflatbuffer))
                for i in reversed(range(len(self.testrequirednestedflatbuffer))):
                    builder.PrependUint8(self.testrequirednestedflatbuffer[i])
                testrequirednestedflatbuffer = builder.EndVector()
        if self.scalarKeySortedTables is not None:
            scalarKeySortedTableslist = []
            for i in range(len(self.scalarKeySortedTables)):
                scalarKeySortedTableslist.append(self.scalarKeySortedTables[i].Pack(builder))
            MonsterStartScalarKeySortedTablesVector(builder, len(self.scalarKeySortedTables))
            for i in reversed(range(len(self.scalarKeySortedTables))):
                builder.PrependUOffsetTRelative(scalarKeySortedTableslist[i])
            scalarKeySortedTables = builder.EndVector()
        MonsterStart(builder)
        if self.pos is not None:
            pos = self.pos.Pack(builder)
            MonsterAddPos(builder, pos)
        MonsterAddMana(builder, self.mana)
        MonsterAddHp(builder, self.hp)
        if self.name is not None:
            MonsterAddName(builder, name)
        if self.inventory is not None:
            MonsterAddInventory(builder, inventory)
        MonsterAddColor(builder, self.color)
        MonsterAddTestType(builder, self.testType)
        if self.test is not None:
            MonsterAddTest(builder, test)
        if self.test4 is not None:
            MonsterAddTest4(builder, test4)
        if self.testarrayofstring is not None:
            MonsterAddTestarrayofstring(builder, testarrayofstring)
        if self.testarrayoftables is not None:
            MonsterAddTestarrayoftables(builder, testarrayoftables)
        if self.enemy is not None:
            MonsterAddEnemy(builder, enemy)
        if self.testnestedflatbuffer is not None:
            MonsterAddTestnestedflatbuffer(builder, testnestedflatbuffer)
        if self.testempty is not None:
            MonsterAddTestempty(builder, testempty)
        MonsterAddTestbool(builder, self.testbool)
        MonsterAddTesthashs32Fnv1(builder, self.testhashs32Fnv1)
        MonsterAddTesthashu32Fnv1(builder, self.testhashu32Fnv1)
        MonsterAddTesthashs64Fnv1(builder, self.testhashs64Fnv1)
        MonsterAddTesthashu64Fnv1(builder, self.testhashu64Fnv1)
        MonsterAddTesthashs32Fnv1a(builder, self.testhashs32Fnv1a)
        MonsterAddTesthashu32Fnv1a(builder, self.testhashu32Fnv1a)
        MonsterAddTesthashs64Fnv1a(builder, self.testhashs64Fnv1a)
        MonsterAddTesthashu64Fnv1a(builder, self.testhashu64Fnv1a)
        if self.testarrayofbools is not None:
            MonsterAddTestarrayofbools(builder, testarrayofbools)
        MonsterAddTestf(builder, self.testf)
        MonsterAddTestf2(builder, self.testf2)
        MonsterAddTestf3(builder, self.testf3)
        if self.testarrayofstring2 is not None:
            MonsterAddTestarrayofstring2(builder, testarrayofstring2)
        if self.testarrayofsortedstruct is not None:
            MonsterAddTestarrayofsortedstruct(builder, testarrayofsortedstruct)
        if self.flex is not None:
            MonsterAddFlex(builder, flex)
        if self.test5 is not None:
            MonsterAddTest5(builder, test5)
        if self.vectorOfLongs is not None:
            MonsterAddVectorOfLongs(builder, vectorOfLongs)
        if self.vectorOfDoubles is not None:
            MonsterAddVectorOfDoubles(builder, vectorOfDoubles)
        if self.parentNamespaceTest is not None:
            MonsterAddParentNamespaceTest(builder, parentNamespaceTest)
        if self.vectorOfReferrables is not None:
            MonsterAddVectorOfReferrables(builder, vectorOfReferrables)
        MonsterAddSingleWeakReference(builder, self.singleWeakReference)
        if self.vectorOfWeakReferences is not None:
            MonsterAddVectorOfWeakReferences(builder, vectorOfWeakReferences)
        if self.vectorOfStrongReferrables is not None:
            MonsterAddVectorOfStrongReferrables(builder, vectorOfStrongReferrables)
        MonsterAddCoOwningReference(builder, self.coOwningReference)
        if self.vectorOfCoOwningReferences is not None:
            MonsterAddVectorOfCoOwningReferences(builder, vectorOfCoOwningReferences)
        MonsterAddNonOwningReference(builder, self.nonOwningReference)
        if self.vectorOfNonOwningReferences is not None:
            MonsterAddVectorOfNonOwningReferences(builder, vectorOfNonOwningReferences)
        MonsterAddAnyUniqueType(builder, self.anyUniqueType)
        if self.anyUnique is not None:
            MonsterAddAnyUnique(builder, anyUnique)
        MonsterAddAnyAmbiguousType(builder, self.anyAmbiguousType)
        if self.anyAmbiguous is not None:
            MonsterAddAnyAmbiguous(builder, anyAmbiguous)
        if self.vectorOfEnums is not None:
            MonsterAddVectorOfEnums(builder, vectorOfEnums)
        MonsterAddSignedEnum(builder, self.signedEnum)
        if self.testrequirednestedflatbuffer is not None:
            MonsterAddTestrequirednestedflatbuffer(builder, testrequirednestedflatbuffer)
        if self.scalarKeySortedTables is not None:
            MonsterAddScalarKeySortedTables(builder, scalarKeySortedTables)
        if self.nativeInline is not None:
            nativeInline = self.nativeInline.Pack(builder)
            MonsterAddNativeInline(builder, nativeInline)
        MonsterAddLongEnumNonEnumDefault(builder, self.longEnumNonEnumDefault)
        MonsterAddLongEnumNormalDefault(builder, self.longEnumNormalDefault)
        MonsterAddNanDefault(builder, self.nanDefault)
        MonsterAddInfDefault(builder, self.infDefault)
        MonsterAddPositiveInfDefault(builder, self.positiveInfDefault)
        MonsterAddInfinityDefault(builder, self.infinityDefault)
        MonsterAddPositiveInfinityDefault(builder, self.positiveInfinityDefault)
        MonsterAddNegativeInfDefault(builder, self.negativeInfDefault)
        MonsterAddNegativeInfinityDefault(builder, self.negativeInfinityDefault)
        MonsterAddDoubleInfDefault(builder, self.doubleInfDefault)
        monster = MonsterEnd(builder)
        return monster

class TypeAliases(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            return 10
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = TypeAliases()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsTypeAliases(cls, buf, offset=0):
        if False:
            for i in range(10):
                print('nop')
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    @classmethod
    def TypeAliasesBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        if False:
            while True:
                i = 10
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b'MONS', size_prefixed=size_prefixed)

    def Init(self, buf, pos):
        if False:
            for i in range(10):
                print('nop')
        self._tab = flatbuffers.table.Table(buf, pos)

    def I8(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    def U8(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    def I16(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int16Flags, o + self._tab.Pos)
        return 0

    def U16(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint16Flags, o + self._tab.Pos)
        return 0

    def I32(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    def U32(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    def I64(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int64Flags, o + self._tab.Pos)
        return 0

    def U64(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, o + self._tab.Pos)
        return 0

    def F32(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

    def F64(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float64Flags, o + self._tab.Pos)
        return 0.0

    def V8(self, j):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    def V8AsNumpy(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int8Flags, o)
        return 0

    def V8Length(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def V8IsNone(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        return o == 0

    def Vf64(self, j):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Float64Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 8))
        return 0

    def Vf64AsNumpy(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Float64Flags, o)
        return 0

    def Vf64Length(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def Vf64IsNone(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        return o == 0

def TypeAliasesStart(builder):
    if False:
        print('Hello World!')
    builder.StartObject(12)

def TypeAliasesAddI8(builder, i8):
    if False:
        i = 10
        return i + 15
    builder.PrependInt8Slot(0, i8, 0)

def TypeAliasesAddU8(builder, u8):
    if False:
        return 10
    builder.PrependUint8Slot(1, u8, 0)

def TypeAliasesAddI16(builder, i16):
    if False:
        i = 10
        return i + 15
    builder.PrependInt16Slot(2, i16, 0)

def TypeAliasesAddU16(builder, u16):
    if False:
        while True:
            i = 10
    builder.PrependUint16Slot(3, u16, 0)

def TypeAliasesAddI32(builder, i32):
    if False:
        print('Hello World!')
    builder.PrependInt32Slot(4, i32, 0)

def TypeAliasesAddU32(builder, u32):
    if False:
        print('Hello World!')
    builder.PrependUint32Slot(5, u32, 0)

def TypeAliasesAddI64(builder, i64):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependInt64Slot(6, i64, 0)

def TypeAliasesAddU64(builder, u64):
    if False:
        i = 10
        return i + 15
    builder.PrependUint64Slot(7, u64, 0)

def TypeAliasesAddF32(builder, f32):
    if False:
        i = 10
        return i + 15
    builder.PrependFloat32Slot(8, f32, 0.0)

def TypeAliasesAddF64(builder, f64):
    if False:
        i = 10
        return i + 15
    builder.PrependFloat64Slot(9, f64, 0.0)

def TypeAliasesAddV8(builder, v8):
    if False:
        return 10
    builder.PrependUOffsetTRelativeSlot(10, flatbuffers.number_types.UOffsetTFlags.py_type(v8), 0)

def TypeAliasesStartV8Vector(builder, numElems):
    if False:
        for i in range(10):
            print('nop')
    return builder.StartVector(1, numElems, 1)

def TypeAliasesAddVf64(builder, vf64):
    if False:
        while True:
            i = 10
    builder.PrependUOffsetTRelativeSlot(11, flatbuffers.number_types.UOffsetTFlags.py_type(vf64), 0)

def TypeAliasesStartVf64Vector(builder, numElems):
    if False:
        while True:
            i = 10
    return builder.StartVector(8, numElems, 8)

def TypeAliasesEnd(builder):
    if False:
        while True:
            i = 10
    return builder.EndObject()
try:
    from typing import List
except:
    pass

class TypeAliasesT(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.i8 = 0
        self.u8 = 0
        self.i16 = 0
        self.u16 = 0
        self.i32 = 0
        self.u32 = 0
        self.i64 = 0
        self.u64 = 0
        self.f32 = 0.0
        self.f64 = 0.0
        self.v8 = None
        self.vf64 = None

    @classmethod
    def InitFromBuf(cls, buf, pos):
        if False:
            i = 10
            return i + 15
        typeAliases = TypeAliases()
        typeAliases.Init(buf, pos)
        return cls.InitFromObj(typeAliases)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        if False:
            for i in range(10):
                print('nop')
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, typeAliases):
        if False:
            while True:
                i = 10
        x = TypeAliasesT()
        x._UnPack(typeAliases)
        return x

    def _UnPack(self, typeAliases):
        if False:
            return 10
        if typeAliases is None:
            return
        self.i8 = typeAliases.I8()
        self.u8 = typeAliases.U8()
        self.i16 = typeAliases.I16()
        self.u16 = typeAliases.U16()
        self.i32 = typeAliases.I32()
        self.u32 = typeAliases.U32()
        self.i64 = typeAliases.I64()
        self.u64 = typeAliases.U64()
        self.f32 = typeAliases.F32()
        self.f64 = typeAliases.F64()
        if not typeAliases.V8IsNone():
            if np is None:
                self.v8 = []
                for i in range(typeAliases.V8Length()):
                    self.v8.append(typeAliases.V8(i))
            else:
                self.v8 = typeAliases.V8AsNumpy()
        if not typeAliases.Vf64IsNone():
            if np is None:
                self.vf64 = []
                for i in range(typeAliases.Vf64Length()):
                    self.vf64.append(typeAliases.Vf64(i))
            else:
                self.vf64 = typeAliases.Vf64AsNumpy()

    def Pack(self, builder):
        if False:
            for i in range(10):
                print('nop')
        if self.v8 is not None:
            if np is not None and type(self.v8) is np.ndarray:
                v8 = builder.CreateNumpyVector(self.v8)
            else:
                TypeAliasesStartV8Vector(builder, len(self.v8))
                for i in reversed(range(len(self.v8))):
                    builder.PrependByte(self.v8[i])
                v8 = builder.EndVector()
        if self.vf64 is not None:
            if np is not None and type(self.vf64) is np.ndarray:
                vf64 = builder.CreateNumpyVector(self.vf64)
            else:
                TypeAliasesStartVf64Vector(builder, len(self.vf64))
                for i in reversed(range(len(self.vf64))):
                    builder.PrependFloat64(self.vf64[i])
                vf64 = builder.EndVector()
        TypeAliasesStart(builder)
        TypeAliasesAddI8(builder, self.i8)
        TypeAliasesAddU8(builder, self.u8)
        TypeAliasesAddI16(builder, self.i16)
        TypeAliasesAddU16(builder, self.u16)
        TypeAliasesAddI32(builder, self.i32)
        TypeAliasesAddU32(builder, self.u32)
        TypeAliasesAddI64(builder, self.i64)
        TypeAliasesAddU64(builder, self.u64)
        TypeAliasesAddF32(builder, self.f32)
        TypeAliasesAddF64(builder, self.f64)
        if self.v8 is not None:
            TypeAliasesAddV8(builder, v8)
        if self.vf64 is not None:
            TypeAliasesAddVf64(builder, vf64)
        typeAliases = TypeAliasesEnd(builder)
        return typeAliases