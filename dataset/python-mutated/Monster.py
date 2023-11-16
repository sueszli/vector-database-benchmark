import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Monster(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            return 10
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Monster()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsMonster(cls, buf, offset=0):
        if False:
            return 10
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
            print('Hello World!')
        self._tab = flatbuffers.table.Table(buf, pos)

    def Pos(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = o + self._tab.Pos
            from MyGame.Example.Vec3 import Vec3
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
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int16Flags, o + self._tab.Pos)
        return 100

    def Name(self):
        if False:
            return 10
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
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        return o == 0

    def Color(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 8

    def TestType(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    def Test(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            from flatbuffers.table import Table
            obj = Table(bytearray(), 0)
            self._tab.Union(obj, o)
            return obj
        return None

    def Test4(self, j):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            from MyGame.Example.Test import Test
            obj = Test()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def Test4Length(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def Test4IsNone(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        return o == 0

    def Testarrayofstring(self, j):
        if False:
            i = 10
            return i + 15
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
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        return o == 0

    def Testarrayoftables(self, j):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from MyGame.Example.Monster import Monster
            obj = Monster()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def TestarrayoftablesLength(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def TestarrayoftablesIsNone(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        return o == 0

    def Enemy(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(28))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from MyGame.Example.Monster import Monster
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
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(30))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    def TestnestedflatbufferNestedRoot(self):
        if False:
            for i in range(10):
                print('nop')
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
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(30))
        return o == 0

    def Testempty(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(32))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from MyGame.Example.Stat import Stat
            obj = Stat()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def Testbool(self):
        if False:
            while True:
                i = 10
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
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(42))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, o + self._tab.Pos)
        return 0

    def Testhashs32Fnv1a(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(44))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    def Testhashu32Fnv1a(self):
        if False:
            i = 10
            return i + 15
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
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(50))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, o + self._tab.Pos)
        return 0

    def Testarrayofbools(self, j):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(52))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.BoolFlags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    def TestarrayofboolsAsNumpy(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(52))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.BoolFlags, o)
        return 0

    def TestarrayofboolsLength(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(52))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def TestarrayofboolsIsNone(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(52))
        return o == 0

    def Testf(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(54))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 3.14159

    def Testf2(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(56))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 3.0

    def Testf3(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(58))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

    def Testarrayofstring2(self, j):
        if False:
            while True:
                i = 10
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
            return 10
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
            from MyGame.Example.Ability import Ability
            obj = Ability()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def TestarrayofsortedstructLength(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(62))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def TestarrayofsortedstructIsNone(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(62))
        return o == 0

    def Flex(self, j):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(64))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    def FlexAsNumpy(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(64))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    def FlexLength(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(64))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def FlexIsNone(self):
        if False:
            i = 10
            return i + 15
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
            from MyGame.Example.Test import Test
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
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(68))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int64Flags, o)
        return 0

    def VectorOfLongsLength(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(68))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def VectorOfLongsIsNone(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(68))
        return o == 0

    def VectorOfDoubles(self, j):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(70))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Float64Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 8))
        return 0

    def VectorOfDoublesAsNumpy(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(70))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Float64Flags, o)
        return 0

    def VectorOfDoublesLength(self):
        if False:
            return 10
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
            from MyGame.InParentNamespace import InParentNamespace
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
            from MyGame.Example.Referrable import Referrable
            obj = Referrable()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def VectorOfReferrablesLength(self):
        if False:
            for i in range(10):
                print('nop')
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
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(78))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 8))
        return 0

    def VectorOfWeakReferencesAsNumpy(self):
        if False:
            return 10
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
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(78))
        return o == 0

    def VectorOfStrongReferrables(self, j):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(80))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from MyGame.Example.Referrable import Referrable
            obj = Referrable()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def VectorOfStrongReferrablesLength(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(80))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def VectorOfStrongReferrablesIsNone(self):
        if False:
            return 10
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
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(84))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 8))
        return 0

    def VectorOfCoOwningReferencesAsNumpy(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(84))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint64Flags, o)
        return 0

    def VectorOfCoOwningReferencesLength(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(84))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def VectorOfCoOwningReferencesIsNone(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(84))
        return o == 0

    def NonOwningReference(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(86))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, o + self._tab.Pos)
        return 0

    def VectorOfNonOwningReferences(self, j):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(88))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 8))
        return 0

    def VectorOfNonOwningReferencesAsNumpy(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(88))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint64Flags, o)
        return 0

    def VectorOfNonOwningReferencesLength(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(88))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def VectorOfNonOwningReferencesIsNone(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(88))
        return o == 0

    def AnyUniqueType(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(90))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    def AnyUnique(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(92))
        if o != 0:
            from flatbuffers.table import Table
            obj = Table(bytearray(), 0)
            self._tab.Union(obj, o)
            return obj
        return None

    def AnyAmbiguousType(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(94))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    def AnyAmbiguous(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(96))
        if o != 0:
            from flatbuffers.table import Table
            obj = Table(bytearray(), 0)
            self._tab.Union(obj, o)
            return obj
        return None

    def VectorOfEnums(self, j):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(98))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    def VectorOfEnumsAsNumpy(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(98))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    def VectorOfEnumsLength(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(98))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def VectorOfEnumsIsNone(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(98))
        return o == 0

    def SignedEnum(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(100))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return -1

    def Testrequirednestedflatbuffer(self, j):
        if False:
            for i in range(10):
                print('nop')
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
            for i in range(10):
                print('nop')
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
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(102))
        return o == 0

    def ScalarKeySortedTables(self, j):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(104))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from MyGame.Example.Stat import Stat
            obj = Stat()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def ScalarKeySortedTablesLength(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(104))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def ScalarKeySortedTablesIsNone(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(104))
        return o == 0

    def NativeInline(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(106))
        if o != 0:
            x = o + self._tab.Pos
            from MyGame.Example.Test import Test
            obj = Test()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def LongEnumNonEnumDefault(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(108))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, o + self._tab.Pos)
        return 0

    def LongEnumNormalDefault(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(110))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, o + self._tab.Pos)
        return 2

    def NanDefault(self):
        if False:
            print('Hello World!')
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
            return 10
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
            i = 10
            return i + 15
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
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(126))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float64Flags, o + self._tab.Pos)
        return float('inf')

def MonsterStart(builder):
    if False:
        return 10
    builder.StartObject(62)

def Start(builder):
    if False:
        for i in range(10):
            print('nop')
    MonsterStart(builder)

def MonsterAddPos(builder, pos):
    if False:
        i = 10
        return i + 15
    builder.PrependStructSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(pos), 0)

def AddPos(builder, pos):
    if False:
        while True:
            i = 10
    MonsterAddPos(builder, pos)

def MonsterAddMana(builder, mana):
    if False:
        while True:
            i = 10
    builder.PrependInt16Slot(1, mana, 150)

def AddMana(builder, mana):
    if False:
        i = 10
        return i + 15
    MonsterAddMana(builder, mana)

def MonsterAddHp(builder, hp):
    if False:
        while True:
            i = 10
    builder.PrependInt16Slot(2, hp, 100)

def AddHp(builder, hp):
    if False:
        i = 10
        return i + 15
    MonsterAddHp(builder, hp)

def MonsterAddName(builder, name):
    if False:
        i = 10
        return i + 15
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)

def AddName(builder, name):
    if False:
        for i in range(10):
            print('nop')
    MonsterAddName(builder, name)

def MonsterAddInventory(builder, inventory):
    if False:
        while True:
            i = 10
    builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(inventory), 0)

def AddInventory(builder, inventory):
    if False:
        for i in range(10):
            print('nop')
    MonsterAddInventory(builder, inventory)

def MonsterStartInventoryVector(builder, numElems):
    if False:
        i = 10
        return i + 15
    return builder.StartVector(1, numElems, 1)

def StartInventoryVector(builder, numElems: int) -> int:
    if False:
        print('Hello World!')
    return MonsterStartInventoryVector(builder, numElems)

def MonsterAddColor(builder, color):
    if False:
        i = 10
        return i + 15
    builder.PrependUint8Slot(6, color, 8)

def AddColor(builder, color):
    if False:
        i = 10
        return i + 15
    MonsterAddColor(builder, color)

def MonsterAddTestType(builder, testType):
    if False:
        while True:
            i = 10
    builder.PrependUint8Slot(7, testType, 0)

def AddTestType(builder, testType):
    if False:
        print('Hello World!')
    MonsterAddTestType(builder, testType)

def MonsterAddTest(builder, test):
    if False:
        print('Hello World!')
    builder.PrependUOffsetTRelativeSlot(8, flatbuffers.number_types.UOffsetTFlags.py_type(test), 0)

def AddTest(builder, test):
    if False:
        for i in range(10):
            print('nop')
    MonsterAddTest(builder, test)

def MonsterAddTest4(builder, test4):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependUOffsetTRelativeSlot(9, flatbuffers.number_types.UOffsetTFlags.py_type(test4), 0)

def AddTest4(builder, test4):
    if False:
        print('Hello World!')
    MonsterAddTest4(builder, test4)

def MonsterStartTest4Vector(builder, numElems):
    if False:
        while True:
            i = 10
    return builder.StartVector(4, numElems, 2)

def StartTest4Vector(builder, numElems: int) -> int:
    if False:
        print('Hello World!')
    return MonsterStartTest4Vector(builder, numElems)

def MonsterAddTestarrayofstring(builder, testarrayofstring):
    if False:
        return 10
    builder.PrependUOffsetTRelativeSlot(10, flatbuffers.number_types.UOffsetTFlags.py_type(testarrayofstring), 0)

def AddTestarrayofstring(builder, testarrayofstring):
    if False:
        return 10
    MonsterAddTestarrayofstring(builder, testarrayofstring)

def MonsterStartTestarrayofstringVector(builder, numElems):
    if False:
        print('Hello World!')
    return builder.StartVector(4, numElems, 4)

def StartTestarrayofstringVector(builder, numElems: int) -> int:
    if False:
        i = 10
        return i + 15
    return MonsterStartTestarrayofstringVector(builder, numElems)

def MonsterAddTestarrayoftables(builder, testarrayoftables):
    if False:
        print('Hello World!')
    builder.PrependUOffsetTRelativeSlot(11, flatbuffers.number_types.UOffsetTFlags.py_type(testarrayoftables), 0)

def AddTestarrayoftables(builder, testarrayoftables):
    if False:
        i = 10
        return i + 15
    MonsterAddTestarrayoftables(builder, testarrayoftables)

def MonsterStartTestarrayoftablesVector(builder, numElems):
    if False:
        i = 10
        return i + 15
    return builder.StartVector(4, numElems, 4)

def StartTestarrayoftablesVector(builder, numElems: int) -> int:
    if False:
        return 10
    return MonsterStartTestarrayoftablesVector(builder, numElems)

def MonsterAddEnemy(builder, enemy):
    if False:
        while True:
            i = 10
    builder.PrependUOffsetTRelativeSlot(12, flatbuffers.number_types.UOffsetTFlags.py_type(enemy), 0)

def AddEnemy(builder, enemy):
    if False:
        print('Hello World!')
    MonsterAddEnemy(builder, enemy)

def MonsterAddTestnestedflatbuffer(builder, testnestedflatbuffer):
    if False:
        while True:
            i = 10
    builder.PrependUOffsetTRelativeSlot(13, flatbuffers.number_types.UOffsetTFlags.py_type(testnestedflatbuffer), 0)

def AddTestnestedflatbuffer(builder, testnestedflatbuffer):
    if False:
        i = 10
        return i + 15
    MonsterAddTestnestedflatbuffer(builder, testnestedflatbuffer)

def MonsterStartTestnestedflatbufferVector(builder, numElems):
    if False:
        return 10
    return builder.StartVector(1, numElems, 1)

def StartTestnestedflatbufferVector(builder, numElems: int) -> int:
    if False:
        i = 10
        return i + 15
    return MonsterStartTestnestedflatbufferVector(builder, numElems)

def MonsterMakeTestnestedflatbufferVectorFromBytes(builder, bytes):
    if False:
        return 10
    builder.StartVector(1, len(bytes), 1)
    builder.head = builder.head - len(bytes)
    builder.Bytes[builder.head:builder.head + len(bytes)] = bytes
    return builder.EndVector()

def MakeTestnestedflatbufferVectorFromBytes(builder, bytes):
    if False:
        i = 10
        return i + 15
    return MonsterMakeTestnestedflatbufferVectorFromBytes(builder, bytes)

def MonsterAddTestempty(builder, testempty):
    if False:
        while True:
            i = 10
    builder.PrependUOffsetTRelativeSlot(14, flatbuffers.number_types.UOffsetTFlags.py_type(testempty), 0)

def AddTestempty(builder, testempty):
    if False:
        i = 10
        return i + 15
    MonsterAddTestempty(builder, testempty)

def MonsterAddTestbool(builder, testbool):
    if False:
        return 10
    builder.PrependBoolSlot(15, testbool, 0)

def AddTestbool(builder, testbool):
    if False:
        i = 10
        return i + 15
    MonsterAddTestbool(builder, testbool)

def MonsterAddTesthashs32Fnv1(builder, testhashs32Fnv1):
    if False:
        while True:
            i = 10
    builder.PrependInt32Slot(16, testhashs32Fnv1, 0)

def AddTesthashs32Fnv1(builder, testhashs32Fnv1):
    if False:
        return 10
    MonsterAddTesthashs32Fnv1(builder, testhashs32Fnv1)

def MonsterAddTesthashu32Fnv1(builder, testhashu32Fnv1):
    if False:
        print('Hello World!')
    builder.PrependUint32Slot(17, testhashu32Fnv1, 0)

def AddTesthashu32Fnv1(builder, testhashu32Fnv1):
    if False:
        i = 10
        return i + 15
    MonsterAddTesthashu32Fnv1(builder, testhashu32Fnv1)

def MonsterAddTesthashs64Fnv1(builder, testhashs64Fnv1):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependInt64Slot(18, testhashs64Fnv1, 0)

def AddTesthashs64Fnv1(builder, testhashs64Fnv1):
    if False:
        print('Hello World!')
    MonsterAddTesthashs64Fnv1(builder, testhashs64Fnv1)

def MonsterAddTesthashu64Fnv1(builder, testhashu64Fnv1):
    if False:
        return 10
    builder.PrependUint64Slot(19, testhashu64Fnv1, 0)

def AddTesthashu64Fnv1(builder, testhashu64Fnv1):
    if False:
        for i in range(10):
            print('nop')
    MonsterAddTesthashu64Fnv1(builder, testhashu64Fnv1)

def MonsterAddTesthashs32Fnv1a(builder, testhashs32Fnv1a):
    if False:
        while True:
            i = 10
    builder.PrependInt32Slot(20, testhashs32Fnv1a, 0)

def AddTesthashs32Fnv1a(builder, testhashs32Fnv1a):
    if False:
        for i in range(10):
            print('nop')
    MonsterAddTesthashs32Fnv1a(builder, testhashs32Fnv1a)

def MonsterAddTesthashu32Fnv1a(builder, testhashu32Fnv1a):
    if False:
        i = 10
        return i + 15
    builder.PrependUint32Slot(21, testhashu32Fnv1a, 0)

def AddTesthashu32Fnv1a(builder, testhashu32Fnv1a):
    if False:
        print('Hello World!')
    MonsterAddTesthashu32Fnv1a(builder, testhashu32Fnv1a)

def MonsterAddTesthashs64Fnv1a(builder, testhashs64Fnv1a):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependInt64Slot(22, testhashs64Fnv1a, 0)

def AddTesthashs64Fnv1a(builder, testhashs64Fnv1a):
    if False:
        while True:
            i = 10
    MonsterAddTesthashs64Fnv1a(builder, testhashs64Fnv1a)

def MonsterAddTesthashu64Fnv1a(builder, testhashu64Fnv1a):
    if False:
        return 10
    builder.PrependUint64Slot(23, testhashu64Fnv1a, 0)

def AddTesthashu64Fnv1a(builder, testhashu64Fnv1a):
    if False:
        print('Hello World!')
    MonsterAddTesthashu64Fnv1a(builder, testhashu64Fnv1a)

def MonsterAddTestarrayofbools(builder, testarrayofbools):
    if False:
        i = 10
        return i + 15
    builder.PrependUOffsetTRelativeSlot(24, flatbuffers.number_types.UOffsetTFlags.py_type(testarrayofbools), 0)

def AddTestarrayofbools(builder, testarrayofbools):
    if False:
        for i in range(10):
            print('nop')
    MonsterAddTestarrayofbools(builder, testarrayofbools)

def MonsterStartTestarrayofboolsVector(builder, numElems):
    if False:
        return 10
    return builder.StartVector(1, numElems, 1)

def StartTestarrayofboolsVector(builder, numElems: int) -> int:
    if False:
        while True:
            i = 10
    return MonsterStartTestarrayofboolsVector(builder, numElems)

def MonsterAddTestf(builder, testf):
    if False:
        while True:
            i = 10
    builder.PrependFloat32Slot(25, testf, 3.14159)

def AddTestf(builder, testf):
    if False:
        return 10
    MonsterAddTestf(builder, testf)

def MonsterAddTestf2(builder, testf2):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependFloat32Slot(26, testf2, 3.0)

def AddTestf2(builder, testf2):
    if False:
        i = 10
        return i + 15
    MonsterAddTestf2(builder, testf2)

def MonsterAddTestf3(builder, testf3):
    if False:
        i = 10
        return i + 15
    builder.PrependFloat32Slot(27, testf3, 0.0)

def AddTestf3(builder, testf3):
    if False:
        for i in range(10):
            print('nop')
    MonsterAddTestf3(builder, testf3)

def MonsterAddTestarrayofstring2(builder, testarrayofstring2):
    if False:
        print('Hello World!')
    builder.PrependUOffsetTRelativeSlot(28, flatbuffers.number_types.UOffsetTFlags.py_type(testarrayofstring2), 0)

def AddTestarrayofstring2(builder, testarrayofstring2):
    if False:
        i = 10
        return i + 15
    MonsterAddTestarrayofstring2(builder, testarrayofstring2)

def MonsterStartTestarrayofstring2Vector(builder, numElems):
    if False:
        i = 10
        return i + 15
    return builder.StartVector(4, numElems, 4)

def StartTestarrayofstring2Vector(builder, numElems: int) -> int:
    if False:
        for i in range(10):
            print('nop')
    return MonsterStartTestarrayofstring2Vector(builder, numElems)

def MonsterAddTestarrayofsortedstruct(builder, testarrayofsortedstruct):
    if False:
        print('Hello World!')
    builder.PrependUOffsetTRelativeSlot(29, flatbuffers.number_types.UOffsetTFlags.py_type(testarrayofsortedstruct), 0)

def AddTestarrayofsortedstruct(builder, testarrayofsortedstruct):
    if False:
        return 10
    MonsterAddTestarrayofsortedstruct(builder, testarrayofsortedstruct)

def MonsterStartTestarrayofsortedstructVector(builder, numElems):
    if False:
        print('Hello World!')
    return builder.StartVector(8, numElems, 4)

def StartTestarrayofsortedstructVector(builder, numElems: int) -> int:
    if False:
        return 10
    return MonsterStartTestarrayofsortedstructVector(builder, numElems)

def MonsterAddFlex(builder, flex):
    if False:
        i = 10
        return i + 15
    builder.PrependUOffsetTRelativeSlot(30, flatbuffers.number_types.UOffsetTFlags.py_type(flex), 0)

def AddFlex(builder, flex):
    if False:
        print('Hello World!')
    MonsterAddFlex(builder, flex)

def MonsterStartFlexVector(builder, numElems):
    if False:
        print('Hello World!')
    return builder.StartVector(1, numElems, 1)

def StartFlexVector(builder, numElems: int) -> int:
    if False:
        return 10
    return MonsterStartFlexVector(builder, numElems)

def MonsterAddTest5(builder, test5):
    if False:
        while True:
            i = 10
    builder.PrependUOffsetTRelativeSlot(31, flatbuffers.number_types.UOffsetTFlags.py_type(test5), 0)

def AddTest5(builder, test5):
    if False:
        print('Hello World!')
    MonsterAddTest5(builder, test5)

def MonsterStartTest5Vector(builder, numElems):
    if False:
        print('Hello World!')
    return builder.StartVector(4, numElems, 2)

def StartTest5Vector(builder, numElems: int) -> int:
    if False:
        i = 10
        return i + 15
    return MonsterStartTest5Vector(builder, numElems)

def MonsterAddVectorOfLongs(builder, vectorOfLongs):
    if False:
        while True:
            i = 10
    builder.PrependUOffsetTRelativeSlot(32, flatbuffers.number_types.UOffsetTFlags.py_type(vectorOfLongs), 0)

def AddVectorOfLongs(builder, vectorOfLongs):
    if False:
        while True:
            i = 10
    MonsterAddVectorOfLongs(builder, vectorOfLongs)

def MonsterStartVectorOfLongsVector(builder, numElems):
    if False:
        i = 10
        return i + 15
    return builder.StartVector(8, numElems, 8)

def StartVectorOfLongsVector(builder, numElems: int) -> int:
    if False:
        return 10
    return MonsterStartVectorOfLongsVector(builder, numElems)

def MonsterAddVectorOfDoubles(builder, vectorOfDoubles):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependUOffsetTRelativeSlot(33, flatbuffers.number_types.UOffsetTFlags.py_type(vectorOfDoubles), 0)

def AddVectorOfDoubles(builder, vectorOfDoubles):
    if False:
        for i in range(10):
            print('nop')
    MonsterAddVectorOfDoubles(builder, vectorOfDoubles)

def MonsterStartVectorOfDoublesVector(builder, numElems):
    if False:
        for i in range(10):
            print('nop')
    return builder.StartVector(8, numElems, 8)

def StartVectorOfDoublesVector(builder, numElems: int) -> int:
    if False:
        i = 10
        return i + 15
    return MonsterStartVectorOfDoublesVector(builder, numElems)

def MonsterAddParentNamespaceTest(builder, parentNamespaceTest):
    if False:
        i = 10
        return i + 15
    builder.PrependUOffsetTRelativeSlot(34, flatbuffers.number_types.UOffsetTFlags.py_type(parentNamespaceTest), 0)

def AddParentNamespaceTest(builder, parentNamespaceTest):
    if False:
        i = 10
        return i + 15
    MonsterAddParentNamespaceTest(builder, parentNamespaceTest)

def MonsterAddVectorOfReferrables(builder, vectorOfReferrables):
    if False:
        while True:
            i = 10
    builder.PrependUOffsetTRelativeSlot(35, flatbuffers.number_types.UOffsetTFlags.py_type(vectorOfReferrables), 0)

def AddVectorOfReferrables(builder, vectorOfReferrables):
    if False:
        print('Hello World!')
    MonsterAddVectorOfReferrables(builder, vectorOfReferrables)

def MonsterStartVectorOfReferrablesVector(builder, numElems):
    if False:
        print('Hello World!')
    return builder.StartVector(4, numElems, 4)

def StartVectorOfReferrablesVector(builder, numElems: int) -> int:
    if False:
        while True:
            i = 10
    return MonsterStartVectorOfReferrablesVector(builder, numElems)

def MonsterAddSingleWeakReference(builder, singleWeakReference):
    if False:
        return 10
    builder.PrependUint64Slot(36, singleWeakReference, 0)

def AddSingleWeakReference(builder, singleWeakReference):
    if False:
        return 10
    MonsterAddSingleWeakReference(builder, singleWeakReference)

def MonsterAddVectorOfWeakReferences(builder, vectorOfWeakReferences):
    if False:
        i = 10
        return i + 15
    builder.PrependUOffsetTRelativeSlot(37, flatbuffers.number_types.UOffsetTFlags.py_type(vectorOfWeakReferences), 0)

def AddVectorOfWeakReferences(builder, vectorOfWeakReferences):
    if False:
        return 10
    MonsterAddVectorOfWeakReferences(builder, vectorOfWeakReferences)

def MonsterStartVectorOfWeakReferencesVector(builder, numElems):
    if False:
        while True:
            i = 10
    return builder.StartVector(8, numElems, 8)

def StartVectorOfWeakReferencesVector(builder, numElems: int) -> int:
    if False:
        print('Hello World!')
    return MonsterStartVectorOfWeakReferencesVector(builder, numElems)

def MonsterAddVectorOfStrongReferrables(builder, vectorOfStrongReferrables):
    if False:
        i = 10
        return i + 15
    builder.PrependUOffsetTRelativeSlot(38, flatbuffers.number_types.UOffsetTFlags.py_type(vectorOfStrongReferrables), 0)

def AddVectorOfStrongReferrables(builder, vectorOfStrongReferrables):
    if False:
        for i in range(10):
            print('nop')
    MonsterAddVectorOfStrongReferrables(builder, vectorOfStrongReferrables)

def MonsterStartVectorOfStrongReferrablesVector(builder, numElems):
    if False:
        i = 10
        return i + 15
    return builder.StartVector(4, numElems, 4)

def StartVectorOfStrongReferrablesVector(builder, numElems: int) -> int:
    if False:
        while True:
            i = 10
    return MonsterStartVectorOfStrongReferrablesVector(builder, numElems)

def MonsterAddCoOwningReference(builder, coOwningReference):
    if False:
        return 10
    builder.PrependUint64Slot(39, coOwningReference, 0)

def AddCoOwningReference(builder, coOwningReference):
    if False:
        print('Hello World!')
    MonsterAddCoOwningReference(builder, coOwningReference)

def MonsterAddVectorOfCoOwningReferences(builder, vectorOfCoOwningReferences):
    if False:
        return 10
    builder.PrependUOffsetTRelativeSlot(40, flatbuffers.number_types.UOffsetTFlags.py_type(vectorOfCoOwningReferences), 0)

def AddVectorOfCoOwningReferences(builder, vectorOfCoOwningReferences):
    if False:
        i = 10
        return i + 15
    MonsterAddVectorOfCoOwningReferences(builder, vectorOfCoOwningReferences)

def MonsterStartVectorOfCoOwningReferencesVector(builder, numElems):
    if False:
        i = 10
        return i + 15
    return builder.StartVector(8, numElems, 8)

def StartVectorOfCoOwningReferencesVector(builder, numElems: int) -> int:
    if False:
        i = 10
        return i + 15
    return MonsterStartVectorOfCoOwningReferencesVector(builder, numElems)

def MonsterAddNonOwningReference(builder, nonOwningReference):
    if False:
        i = 10
        return i + 15
    builder.PrependUint64Slot(41, nonOwningReference, 0)

def AddNonOwningReference(builder, nonOwningReference):
    if False:
        while True:
            i = 10
    MonsterAddNonOwningReference(builder, nonOwningReference)

def MonsterAddVectorOfNonOwningReferences(builder, vectorOfNonOwningReferences):
    if False:
        return 10
    builder.PrependUOffsetTRelativeSlot(42, flatbuffers.number_types.UOffsetTFlags.py_type(vectorOfNonOwningReferences), 0)

def AddVectorOfNonOwningReferences(builder, vectorOfNonOwningReferences):
    if False:
        for i in range(10):
            print('nop')
    MonsterAddVectorOfNonOwningReferences(builder, vectorOfNonOwningReferences)

def MonsterStartVectorOfNonOwningReferencesVector(builder, numElems):
    if False:
        while True:
            i = 10
    return builder.StartVector(8, numElems, 8)

def StartVectorOfNonOwningReferencesVector(builder, numElems: int) -> int:
    if False:
        print('Hello World!')
    return MonsterStartVectorOfNonOwningReferencesVector(builder, numElems)

def MonsterAddAnyUniqueType(builder, anyUniqueType):
    if False:
        print('Hello World!')
    builder.PrependUint8Slot(43, anyUniqueType, 0)

def AddAnyUniqueType(builder, anyUniqueType):
    if False:
        print('Hello World!')
    MonsterAddAnyUniqueType(builder, anyUniqueType)

def MonsterAddAnyUnique(builder, anyUnique):
    if False:
        while True:
            i = 10
    builder.PrependUOffsetTRelativeSlot(44, flatbuffers.number_types.UOffsetTFlags.py_type(anyUnique), 0)

def AddAnyUnique(builder, anyUnique):
    if False:
        while True:
            i = 10
    MonsterAddAnyUnique(builder, anyUnique)

def MonsterAddAnyAmbiguousType(builder, anyAmbiguousType):
    if False:
        i = 10
        return i + 15
    builder.PrependUint8Slot(45, anyAmbiguousType, 0)

def AddAnyAmbiguousType(builder, anyAmbiguousType):
    if False:
        i = 10
        return i + 15
    MonsterAddAnyAmbiguousType(builder, anyAmbiguousType)

def MonsterAddAnyAmbiguous(builder, anyAmbiguous):
    if False:
        i = 10
        return i + 15
    builder.PrependUOffsetTRelativeSlot(46, flatbuffers.number_types.UOffsetTFlags.py_type(anyAmbiguous), 0)

def AddAnyAmbiguous(builder, anyAmbiguous):
    if False:
        print('Hello World!')
    MonsterAddAnyAmbiguous(builder, anyAmbiguous)

def MonsterAddVectorOfEnums(builder, vectorOfEnums):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependUOffsetTRelativeSlot(47, flatbuffers.number_types.UOffsetTFlags.py_type(vectorOfEnums), 0)

def AddVectorOfEnums(builder, vectorOfEnums):
    if False:
        for i in range(10):
            print('nop')
    MonsterAddVectorOfEnums(builder, vectorOfEnums)

def MonsterStartVectorOfEnumsVector(builder, numElems):
    if False:
        while True:
            i = 10
    return builder.StartVector(1, numElems, 1)

def StartVectorOfEnumsVector(builder, numElems: int) -> int:
    if False:
        i = 10
        return i + 15
    return MonsterStartVectorOfEnumsVector(builder, numElems)

def MonsterAddSignedEnum(builder, signedEnum):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependInt8Slot(48, signedEnum, -1)

def AddSignedEnum(builder, signedEnum):
    if False:
        print('Hello World!')
    MonsterAddSignedEnum(builder, signedEnum)

def MonsterAddTestrequirednestedflatbuffer(builder, testrequirednestedflatbuffer):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependUOffsetTRelativeSlot(49, flatbuffers.number_types.UOffsetTFlags.py_type(testrequirednestedflatbuffer), 0)

def AddTestrequirednestedflatbuffer(builder, testrequirednestedflatbuffer):
    if False:
        print('Hello World!')
    MonsterAddTestrequirednestedflatbuffer(builder, testrequirednestedflatbuffer)

def MonsterStartTestrequirednestedflatbufferVector(builder, numElems):
    if False:
        while True:
            i = 10
    return builder.StartVector(1, numElems, 1)

def StartTestrequirednestedflatbufferVector(builder, numElems: int) -> int:
    if False:
        return 10
    return MonsterStartTestrequirednestedflatbufferVector(builder, numElems)

def MonsterMakeTestrequirednestedflatbufferVectorFromBytes(builder, bytes):
    if False:
        return 10
    builder.StartVector(1, len(bytes), 1)
    builder.head = builder.head - len(bytes)
    builder.Bytes[builder.head:builder.head + len(bytes)] = bytes
    return builder.EndVector()

def MakeTestrequirednestedflatbufferVectorFromBytes(builder, bytes):
    if False:
        print('Hello World!')
    return MonsterMakeTestrequirednestedflatbufferVectorFromBytes(builder, bytes)

def MonsterAddScalarKeySortedTables(builder, scalarKeySortedTables):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependUOffsetTRelativeSlot(50, flatbuffers.number_types.UOffsetTFlags.py_type(scalarKeySortedTables), 0)

def AddScalarKeySortedTables(builder, scalarKeySortedTables):
    if False:
        for i in range(10):
            print('nop')
    MonsterAddScalarKeySortedTables(builder, scalarKeySortedTables)

def MonsterStartScalarKeySortedTablesVector(builder, numElems):
    if False:
        for i in range(10):
            print('nop')
    return builder.StartVector(4, numElems, 4)

def StartScalarKeySortedTablesVector(builder, numElems: int) -> int:
    if False:
        print('Hello World!')
    return MonsterStartScalarKeySortedTablesVector(builder, numElems)

def MonsterAddNativeInline(builder, nativeInline):
    if False:
        print('Hello World!')
    builder.PrependStructSlot(51, flatbuffers.number_types.UOffsetTFlags.py_type(nativeInline), 0)

def AddNativeInline(builder, nativeInline):
    if False:
        while True:
            i = 10
    MonsterAddNativeInline(builder, nativeInline)

def MonsterAddLongEnumNonEnumDefault(builder, longEnumNonEnumDefault):
    if False:
        i = 10
        return i + 15
    builder.PrependUint64Slot(52, longEnumNonEnumDefault, 0)

def AddLongEnumNonEnumDefault(builder, longEnumNonEnumDefault):
    if False:
        i = 10
        return i + 15
    MonsterAddLongEnumNonEnumDefault(builder, longEnumNonEnumDefault)

def MonsterAddLongEnumNormalDefault(builder, longEnumNormalDefault):
    if False:
        print('Hello World!')
    builder.PrependUint64Slot(53, longEnumNormalDefault, 2)

def AddLongEnumNormalDefault(builder, longEnumNormalDefault):
    if False:
        while True:
            i = 10
    MonsterAddLongEnumNormalDefault(builder, longEnumNormalDefault)

def MonsterAddNanDefault(builder, nanDefault):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependFloat32Slot(54, nanDefault, float('nan'))

def AddNanDefault(builder, nanDefault):
    if False:
        while True:
            i = 10
    MonsterAddNanDefault(builder, nanDefault)

def MonsterAddInfDefault(builder, infDefault):
    if False:
        while True:
            i = 10
    builder.PrependFloat32Slot(55, infDefault, float('inf'))

def AddInfDefault(builder, infDefault):
    if False:
        return 10
    MonsterAddInfDefault(builder, infDefault)

def MonsterAddPositiveInfDefault(builder, positiveInfDefault):
    if False:
        while True:
            i = 10
    builder.PrependFloat32Slot(56, positiveInfDefault, float('inf'))

def AddPositiveInfDefault(builder, positiveInfDefault):
    if False:
        for i in range(10):
            print('nop')
    MonsterAddPositiveInfDefault(builder, positiveInfDefault)

def MonsterAddInfinityDefault(builder, infinityDefault):
    if False:
        while True:
            i = 10
    builder.PrependFloat32Slot(57, infinityDefault, float('inf'))

def AddInfinityDefault(builder, infinityDefault):
    if False:
        return 10
    MonsterAddInfinityDefault(builder, infinityDefault)

def MonsterAddPositiveInfinityDefault(builder, positiveInfinityDefault):
    if False:
        while True:
            i = 10
    builder.PrependFloat32Slot(58, positiveInfinityDefault, float('inf'))

def AddPositiveInfinityDefault(builder, positiveInfinityDefault):
    if False:
        while True:
            i = 10
    MonsterAddPositiveInfinityDefault(builder, positiveInfinityDefault)

def MonsterAddNegativeInfDefault(builder, negativeInfDefault):
    if False:
        i = 10
        return i + 15
    builder.PrependFloat32Slot(59, negativeInfDefault, float('-inf'))

def AddNegativeInfDefault(builder, negativeInfDefault):
    if False:
        while True:
            i = 10
    MonsterAddNegativeInfDefault(builder, negativeInfDefault)

def MonsterAddNegativeInfinityDefault(builder, negativeInfinityDefault):
    if False:
        while True:
            i = 10
    builder.PrependFloat32Slot(60, negativeInfinityDefault, float('-inf'))

def AddNegativeInfinityDefault(builder, negativeInfinityDefault):
    if False:
        print('Hello World!')
    MonsterAddNegativeInfinityDefault(builder, negativeInfinityDefault)

def MonsterAddDoubleInfDefault(builder, doubleInfDefault):
    if False:
        i = 10
        return i + 15
    builder.PrependFloat64Slot(61, doubleInfDefault, float('inf'))

def AddDoubleInfDefault(builder, doubleInfDefault):
    if False:
        i = 10
        return i + 15
    MonsterAddDoubleInfDefault(builder, doubleInfDefault)

def MonsterEnd(builder):
    if False:
        return 10
    return builder.EndObject()

def End(builder):
    if False:
        print('Hello World!')
    return MonsterEnd(builder)
import MyGame.Example.Ability
import MyGame.Example.Any
import MyGame.Example.AnyAmbiguousAliases
import MyGame.Example.AnyUniqueAliases
import MyGame.Example.Referrable
import MyGame.Example.Stat
import MyGame.Example.Test
import MyGame.Example.TestSimpleTableWithEnum
import MyGame.Example.Vec3
import MyGame.Example2.Monster
import MyGame.InParentNamespace
try:
    from typing import List, Optional, Union
except:
    pass

class MonsterT(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
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
            return 10
        monster = Monster()
        monster.Init(buf, pos)
        return cls.InitFromObj(monster)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        if False:
            while True:
                i = 10
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
            for i in range(10):
                print('nop')
        if monster is None:
            return
        if monster.Pos() is not None:
            self.pos = MyGame.Example.Vec3.Vec3T.InitFromObj(monster.Pos())
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
        self.test = MyGame.Example.Any.AnyCreator(self.testType, monster.Test())
        if not monster.Test4IsNone():
            self.test4 = []
            for i in range(monster.Test4Length()):
                if monster.Test4(i) is None:
                    self.test4.append(None)
                else:
                    test_ = MyGame.Example.Test.TestT.InitFromObj(monster.Test4(i))
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
                    monster_ = MyGame.Example.Monster.MonsterT.InitFromObj(monster.Testarrayoftables(i))
                    self.testarrayoftables.append(monster_)
        if monster.Enemy() is not None:
            self.enemy = MyGame.Example.Monster.MonsterT.InitFromObj(monster.Enemy())
        if not monster.TestnestedflatbufferIsNone():
            if np is None:
                self.testnestedflatbuffer = []
                for i in range(monster.TestnestedflatbufferLength()):
                    self.testnestedflatbuffer.append(monster.Testnestedflatbuffer(i))
            else:
                self.testnestedflatbuffer = monster.TestnestedflatbufferAsNumpy()
        if monster.Testempty() is not None:
            self.testempty = MyGame.Example.Stat.StatT.InitFromObj(monster.Testempty())
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
                    ability_ = MyGame.Example.Ability.AbilityT.InitFromObj(monster.Testarrayofsortedstruct(i))
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
                    test_ = MyGame.Example.Test.TestT.InitFromObj(monster.Test5(i))
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
            self.parentNamespaceTest = MyGame.InParentNamespace.InParentNamespaceT.InitFromObj(monster.ParentNamespaceTest())
        if not monster.VectorOfReferrablesIsNone():
            self.vectorOfReferrables = []
            for i in range(monster.VectorOfReferrablesLength()):
                if monster.VectorOfReferrables(i) is None:
                    self.vectorOfReferrables.append(None)
                else:
                    referrable_ = MyGame.Example.Referrable.ReferrableT.InitFromObj(monster.VectorOfReferrables(i))
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
                    referrable_ = MyGame.Example.Referrable.ReferrableT.InitFromObj(monster.VectorOfStrongReferrables(i))
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
        self.anyUnique = MyGame.Example.AnyUniqueAliases.AnyUniqueAliasesCreator(self.anyUniqueType, monster.AnyUnique())
        self.anyAmbiguousType = monster.AnyAmbiguousType()
        self.anyAmbiguous = MyGame.Example.AnyAmbiguousAliases.AnyAmbiguousAliasesCreator(self.anyAmbiguousType, monster.AnyAmbiguous())
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
                    stat_ = MyGame.Example.Stat.StatT.InitFromObj(monster.ScalarKeySortedTables(i))
                    self.scalarKeySortedTables.append(stat_)
        if monster.NativeInline() is not None:
            self.nativeInline = MyGame.Example.Test.TestT.InitFromObj(monster.NativeInline())
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