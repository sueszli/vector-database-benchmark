import os.path
import sys
PY_VERSION = sys.version_info[:2]
import ctypes
from collections import defaultdict
import math
import random
import timeit
import unittest
from flatbuffers import compat
from flatbuffers import util
from flatbuffers.compat import range_func as compat_range
from flatbuffers.compat import NumpyRequiredForThisFeature
import flatbuffers
from flatbuffers import number_types as N
import MyGame
import MyGame.Example
import MyGame.Example.Any
import MyGame.Example.Color
import MyGame.Example.Monster
import MyGame.Example.Test
import MyGame.Example.Stat
import MyGame.Example.Vec3
import MyGame.MonsterExtra
import MyGame.InParentNamespace
import MyGame.Example.ArrayTable
import MyGame.Example.ArrayStruct
import MyGame.Example.NestedStruct
import MyGame.Example.TestEnum
import MyGame.Example.NestedUnion.NestedUnionTest
import MyGame.Example.NestedUnion.Vec3
import MyGame.Example.NestedUnion.Any
import MyGame.Example.NestedUnion.Test
import MyGame.Example.NestedUnion.Color
import monster_test_generated
import optional_scalars
import optional_scalars.ScalarStuff

def create_namespace_shortcut(is_onefile):
    if False:
        i = 10
        return i + 15
    global _ANY
    global _COLOR
    global _MONSTER
    global _TEST
    global _STAT
    global _VEC3
    global _IN_PARENT_NAMESPACE
    if is_onefile:
        print('Testing with the one-file generated code')
        _ANY = monster_test_generated
        _COLOR = monster_test_generated
        _MONSTER = monster_test_generated
        _TEST = monster_test_generated
        _STAT = monster_test_generated
        _VEC3 = monster_test_generated
        _IN_PARENT_NAMESPACE = monster_test_generated
    else:
        print('Testing with multi-file generated code')
        _ANY = MyGame.Example.Any
        _COLOR = MyGame.Example.Color
        _MONSTER = MyGame.Example.Monster
        _TEST = MyGame.Example.Test
        _STAT = MyGame.Example.Stat
        _VEC3 = MyGame.Example.Vec3
        _IN_PARENT_NAMESPACE = MyGame.InParentNamespace

def assertRaises(test_case, fn, exception_class):
    if False:
        print('Hello World!')
    ' Backwards-compatible assertion for exceptions raised. '
    exc = None
    try:
        fn()
    except Exception as e:
        exc = e
    test_case.assertTrue(exc is not None)
    test_case.assertTrue(isinstance(exc, exception_class))

class TestWireFormat(unittest.TestCase):

    def test_wire_format(self):
        if False:
            i = 10
            return i + 15
        for sizePrefix in [True, False]:
            for file_identifier in [None, b'MONS']:
                (gen_buf, gen_off) = make_monster_from_generated_code(sizePrefix=sizePrefix, file_identifier=file_identifier)
                CheckReadBuffer(gen_buf, gen_off, sizePrefix=sizePrefix, file_identifier=file_identifier)
        f = open('monsterdata_test.mon', 'rb')
        canonicalWireData = f.read()
        f.close()
        CheckReadBuffer(bytearray(canonicalWireData), 0, file_identifier=b'MONS')
        f = open('monsterdata_python_wire.mon', 'wb')
        f.write(gen_buf[gen_off:])
        f.close()

class TestObjectBasedAPI(unittest.TestCase):
    """ Tests the generated object based API."""

    def test_consistency_with_repeated_pack_and_unpack(self):
        if False:
            i = 10
            return i + 15
        ' Checks the serialization and deserialization between a buffer and\n\n        its python object. It tests in the same way as the C++ object API test,\n        ObjectFlatBuffersTest in test.cpp.\n    '
        (buf, off) = make_monster_from_generated_code()
        monster1 = _MONSTER.Monster.GetRootAs(buf, off)
        monsterT1 = _MONSTER.MonsterT.InitFromObj(monster1)
        for sizePrefix in [True, False]:
            b1 = flatbuffers.Builder(0)
            if sizePrefix:
                b1.FinishSizePrefixed(monsterT1.Pack(b1))
            else:
                b1.Finish(monsterT1.Pack(b1))
            CheckReadBuffer(b1.Bytes, b1.Head(), sizePrefix)
        monster2 = _MONSTER.Monster.GetRootAs(b1.Bytes, b1.Head())
        monsterT2 = _MONSTER.MonsterT.InitFromObj(monster2)
        for sizePrefix in [True, False]:
            b2 = flatbuffers.Builder(0)
            if sizePrefix:
                b2.FinishSizePrefixed(monsterT2.Pack(b2))
            else:
                b2.Finish(monsterT2.Pack(b2))
            CheckReadBuffer(b2.Bytes, b2.Head(), sizePrefix)

    def test_default_values_with_pack_and_unpack(self):
        if False:
            print('Hello World!')
        ' Serializes and deserializes between a buffer with default values (no\n\n        specific values are filled when the buffer is created) and its python\n        object.\n    '
        b1 = flatbuffers.Builder(0)
        _MONSTER.MonsterStart(b1)
        gen_mon = _MONSTER.MonsterEnd(b1)
        b1.Finish(gen_mon)
        monster1 = _MONSTER.Monster.GetRootAs(b1.Bytes, b1.Head())
        monsterT1 = _MONSTER.MonsterT.InitFromObj(monster1)
        b2 = flatbuffers.Builder(0)
        b2.Finish(monsterT1.Pack(b2))
        monster2 = _MONSTER.Monster.GetRootAs(b2.Bytes, b2.Head())
        self.assertTrue(monster2.Pos() is None)
        self.assertEqual(monster2.Mana(), 150)
        self.assertEqual(monster2.Hp(), 100)
        self.assertTrue(monster2.Name() is None)
        self.assertEqual(monster2.Inventory(0), 0)
        self.assertEqual(monster2.InventoryAsNumpy(), 0)
        self.assertEqual(monster2.InventoryLength(), 0)
        self.assertTrue(monster2.InventoryIsNone())
        self.assertEqual(monster2.Color(), 8)
        self.assertEqual(monster2.TestType(), 0)
        self.assertTrue(monster2.Test() is None)
        self.assertTrue(monster2.Test4(0) is None)
        self.assertEqual(monster2.Test4Length(), 0)
        self.assertTrue(monster2.Test4IsNone())
        self.assertEqual(monster2.Testarrayofstring(0), '')
        self.assertEqual(monster2.TestarrayofstringLength(), 0)
        self.assertTrue(monster2.TestarrayofstringIsNone())
        self.assertTrue(monster2.Testarrayoftables(0) is None)
        self.assertEqual(monster2.TestarrayoftablesLength(), 0)
        self.assertTrue(monster2.TestarrayoftablesIsNone())
        self.assertTrue(monster2.Enemy() is None)
        self.assertEqual(monster2.Testnestedflatbuffer(0), 0)
        self.assertEqual(monster2.TestnestedflatbufferAsNumpy(), 0)
        self.assertEqual(monster2.TestnestedflatbufferLength(), 0)
        self.assertTrue(monster2.TestnestedflatbufferIsNone())
        self.assertTrue(monster2.Testempty() is None)
        self.assertFalse(monster2.Testbool())
        self.assertEqual(monster2.Testhashs32Fnv1(), 0)
        self.assertEqual(monster2.Testhashu32Fnv1(), 0)
        self.assertEqual(monster2.Testhashs64Fnv1(), 0)
        self.assertEqual(monster2.Testhashu64Fnv1(), 0)
        self.assertEqual(monster2.Testhashs32Fnv1a(), 0)
        self.assertEqual(monster2.Testhashu32Fnv1a(), 0)
        self.assertEqual(monster2.Testhashs64Fnv1a(), 0)
        self.assertEqual(monster2.Testhashu64Fnv1a(), 0)
        self.assertEqual(monster2.Testarrayofbools(0), 0)
        self.assertEqual(monster2.TestarrayofboolsAsNumpy(), 0)
        self.assertEqual(monster2.TestarrayofboolsLength(), 0)
        self.assertTrue(monster2.TestarrayofboolsIsNone())
        self.assertEqual(monster2.Testf(), 3.14159)
        self.assertEqual(monster2.Testf2(), 3.0)
        self.assertEqual(monster2.Testf3(), 0.0)
        self.assertEqual(monster2.Testarrayofstring2(0), '')
        self.assertEqual(monster2.Testarrayofstring2Length(), 0)
        self.assertTrue(monster2.Testarrayofstring2IsNone())
        self.assertTrue(monster2.Testarrayofsortedstruct(0) is None)
        self.assertEqual(monster2.TestarrayofsortedstructLength(), 0)
        self.assertTrue(monster2.TestarrayofsortedstructIsNone())
        self.assertEqual(monster2.Flex(0), 0)
        self.assertEqual(monster2.FlexAsNumpy(), 0)
        self.assertEqual(monster2.FlexLength(), 0)
        self.assertTrue(monster2.FlexIsNone())
        self.assertTrue(monster2.Test5(0) is None)
        self.assertEqual(monster2.Test5Length(), 0)
        self.assertTrue(monster2.Test5IsNone())
        self.assertEqual(monster2.VectorOfLongs(0), 0)
        self.assertEqual(monster2.VectorOfLongsAsNumpy(), 0)
        self.assertEqual(monster2.VectorOfLongsLength(), 0)
        self.assertTrue(monster2.VectorOfLongsIsNone())
        self.assertEqual(monster2.VectorOfDoubles(0), 0)
        self.assertEqual(monster2.VectorOfDoublesAsNumpy(), 0)
        self.assertEqual(monster2.VectorOfDoublesLength(), 0)
        self.assertTrue(monster2.VectorOfDoublesIsNone())
        self.assertTrue(monster2.ParentNamespaceTest() is None)
        self.assertTrue(monster2.VectorOfReferrables(0) is None)
        self.assertEqual(monster2.VectorOfReferrablesLength(), 0)
        self.assertTrue(monster2.VectorOfReferrablesIsNone())
        self.assertEqual(monster2.SingleWeakReference(), 0)
        self.assertEqual(monster2.VectorOfWeakReferences(0), 0)
        self.assertEqual(monster2.VectorOfWeakReferencesAsNumpy(), 0)
        self.assertEqual(monster2.VectorOfWeakReferencesLength(), 0)
        self.assertTrue(monster2.VectorOfWeakReferencesIsNone())
        self.assertTrue(monster2.VectorOfStrongReferrables(0) is None)
        self.assertEqual(monster2.VectorOfStrongReferrablesLength(), 0)
        self.assertTrue(monster2.VectorOfStrongReferrablesIsNone())
        self.assertEqual(monster2.CoOwningReference(), 0)
        self.assertEqual(monster2.VectorOfCoOwningReferences(0), 0)
        self.assertEqual(monster2.VectorOfCoOwningReferencesAsNumpy(), 0)
        self.assertEqual(monster2.VectorOfCoOwningReferencesLength(), 0)
        self.assertTrue(monster2.VectorOfCoOwningReferencesIsNone())
        self.assertEqual(monster2.NonOwningReference(), 0)
        self.assertEqual(monster2.VectorOfNonOwningReferences(0), 0)
        self.assertEqual(monster2.VectorOfNonOwningReferencesAsNumpy(), 0)
        self.assertEqual(monster2.VectorOfNonOwningReferencesLength(), 0)
        self.assertTrue(monster2.VectorOfNonOwningReferencesIsNone())
        self.assertEqual(monster2.AnyUniqueType(), 0)
        self.assertTrue(monster2.AnyUnique() is None)
        self.assertEqual(monster2.AnyAmbiguousType(), 0)
        self.assertTrue(monster2.AnyAmbiguous() is None)
        self.assertEqual(monster2.VectorOfEnums(0), 0)
        self.assertEqual(monster2.VectorOfEnumsAsNumpy(), 0)
        self.assertEqual(monster2.VectorOfEnumsLength(), 0)
        self.assertTrue(monster2.VectorOfEnumsIsNone())

    def test_optional_scalars_with_pack_and_unpack(self):
        if False:
            i = 10
            return i + 15
        ' Serializes and deserializes between a buffer with optional values (no\n        specific values are filled when the buffer is created) and its python\n        object.\n    '
        b1 = flatbuffers.Builder(0)
        optional_scalars.ScalarStuff.ScalarStuffStart(b1)
        gen_opt = optional_scalars.ScalarStuff.ScalarStuffEnd(b1)
        b1.Finish(gen_opt)
        opts1 = optional_scalars.ScalarStuff.ScalarStuff.GetRootAs(b1.Bytes, b1.Head())
        optsT1 = optional_scalars.ScalarStuff.ScalarStuffT.InitFromObj(opts1)
        b2 = flatbuffers.Builder(0)
        b2.Finish(optsT1.Pack(b2))
        opts2 = optional_scalars.ScalarStuff.ScalarStuff.GetRootAs(b2.Bytes, b2.Head())
        optsT2 = optional_scalars.ScalarStuff.ScalarStuffT.InitFromObj(opts2)
        self.assertTrue(opts2.JustI8() == 0)
        self.assertTrue(opts2.MaybeF32() is None)
        self.assertTrue(opts2.DefaultBool() is True)
        self.assertTrue(optsT2.justU16 == 0)
        self.assertTrue(optsT2.maybeEnum is None)
        self.assertTrue(optsT2.defaultU64 == 42)

class TestAllMutableCodePathsOfExampleSchema(unittest.TestCase):
    """ Tests the object API generated for monster_test.fbs for mutation

        purposes. In each test, the default values will be changed through the
        object API. We'll then pack the object class into the buf class and read
        the updated values out from it to validate if the values are mutated as
        expected.
  """

    def setUp(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(TestAllMutableCodePathsOfExampleSchema, self).setUp(*args, **kwargs)
        b = flatbuffers.Builder(0)
        _MONSTER.MonsterStart(b)
        self.monsterT = self._create_and_load_object_class(b)

    def _pack_and_load_buf_class(self, monsterT):
        if False:
            i = 10
            return i + 15
        ' Packs the object class into a flatbuffer and loads it into a buf\n\n        class.\n    '
        b = flatbuffers.Builder(0)
        b.Finish(monsterT.Pack(b))
        monster = _MONSTER.Monster.GetRootAs(b.Bytes, b.Head())
        return monster

    def _create_and_load_object_class(self, b):
        if False:
            i = 10
            return i + 15
        ' Finishs the creation of a monster flatbuffer and loads it into an\n\n        object class.\n    '
        gen_mon = _MONSTER.MonsterEnd(b)
        b.Finish(gen_mon)
        monster = _MONSTER.Monster.GetRootAs(b.Bytes, b.Head())
        monsterT = _MONSTER.MonsterT()
        monsterT.InitFromObj(monster)
        return monsterT

    def test_mutate_pos(self):
        if False:
            for i in range(10):
                print('nop')
        posT = _VEC3.Vec3T()
        posT.x = 4.0
        posT.y = 5.0
        posT.z = 6.0
        posT.test1 = 6.0
        posT.test2 = 7
        test3T = _TEST.TestT()
        test3T.a = 8
        test3T.b = 9
        posT.test3 = test3T
        self.monsterT.pos = posT
        monster = self._pack_and_load_buf_class(self.monsterT)
        pos = monster.Pos()
        self.assertEqual(pos.X(), 4.0)
        self.assertEqual(pos.Y(), 5.0)
        self.assertEqual(pos.Z(), 6.0)
        self.assertEqual(pos.Test1(), 6.0)
        self.assertEqual(pos.Test2(), 7)
        t3 = _TEST.Test()
        t3 = pos.Test3(t3)
        self.assertEqual(t3.A(), 8)
        self.assertEqual(t3.B(), 9)

    def test_mutate_mana(self):
        if False:
            print('Hello World!')
        self.monsterT.mana = 200
        monster = self._pack_and_load_buf_class(self.monsterT)
        self.assertEqual(monster.Mana(), 200)

    def test_mutate_hp(self):
        if False:
            i = 10
            return i + 15
        self.monsterT.hp = 200
        monster = self._pack_and_load_buf_class(self.monsterT)
        self.assertEqual(monster.Hp(), 200)

    def test_mutate_name(self):
        if False:
            return 10
        self.monsterT.name = 'MyMonster'
        monster = self._pack_and_load_buf_class(self.monsterT)
        self.assertEqual(monster.Name(), b'MyMonster')

    def test_mutate_inventory(self):
        if False:
            for i in range(10):
                print('nop')
        self.monsterT.inventory = [1, 7, 8]
        monster = self._pack_and_load_buf_class(self.monsterT)
        self.assertEqual(monster.Inventory(0), 1)
        self.assertEqual(monster.Inventory(1), 7)
        self.assertEqual(monster.Inventory(2), 8)

    def test_empty_inventory(self):
        if False:
            while True:
                i = 10
        self.monsterT.inventory = []
        monster = self._pack_and_load_buf_class(self.monsterT)
        self.assertFalse(monster.InventoryIsNone())

    def test_mutate_color(self):
        if False:
            return 10
        self.monsterT.color = _COLOR.Color.Red
        monster = self._pack_and_load_buf_class(self.monsterT)
        self.assertEqual(monster.Color(), _COLOR.Color.Red)

    def test_mutate_testtype(self):
        if False:
            for i in range(10):
                print('nop')
        self.monsterT.testType = _ANY.Any.Monster
        monster = self._pack_and_load_buf_class(self.monsterT)
        self.assertEqual(monster.TestType(), _ANY.Any.Monster)

    def test_mutate_test(self):
        if False:
            i = 10
            return i + 15
        testT = _MONSTER.MonsterT()
        testT.hp = 200
        self.monsterT.test = testT
        monster = self._pack_and_load_buf_class(self.monsterT)
        table = monster.Test()
        test_monster = _MONSTER.Monster()
        test_monster.Init(table.Bytes, table.Pos)
        self.assertEqual(test_monster.Hp(), 200)

    def test_mutate_test4(self):
        if False:
            while True:
                i = 10
        test0T = _TEST.TestT()
        test0T.a = 10
        test0T.b = 20
        test1T = _TEST.TestT()
        test1T.a = 30
        test1T.b = 40
        self.monsterT.test4 = [test0T, test1T]
        monster = self._pack_and_load_buf_class(self.monsterT)
        test0 = monster.Test4(0)
        self.assertEqual(test0.A(), 10)
        self.assertEqual(test0.B(), 20)
        test1 = monster.Test4(1)
        self.assertEqual(test1.A(), 30)
        self.assertEqual(test1.B(), 40)

    def test_empty_test4(self):
        if False:
            return 10
        self.monsterT.test4 = []
        monster = self._pack_and_load_buf_class(self.monsterT)
        self.assertFalse(monster.Test4IsNone())

    def test_mutate_testarrayofstring(self):
        if False:
            i = 10
            return i + 15
        self.monsterT.testarrayofstring = []
        self.monsterT.testarrayofstring.append('test1')
        self.monsterT.testarrayofstring.append('test2')
        monster = self._pack_and_load_buf_class(self.monsterT)
        self.assertEqual(monster.Testarrayofstring(0), b'test1')
        self.assertEqual(monster.Testarrayofstring(1), b'test2')

    def test_empty_testarrayofstring(self):
        if False:
            return 10
        self.monsterT.testarrayofstring = []
        monster = self._pack_and_load_buf_class(self.monsterT)
        self.assertFalse(monster.TestarrayofstringIsNone())

    def test_mutate_testarrayoftables(self):
        if False:
            print('Hello World!')
        monsterT0 = _MONSTER.MonsterT()
        monsterT0.hp = 200
        monsterT1 = _MONSTER.MonsterT()
        monsterT1.hp = 400
        self.monsterT.testarrayoftables = []
        self.monsterT.testarrayoftables.append(monsterT0)
        self.monsterT.testarrayoftables.append(monsterT1)
        monster = self._pack_and_load_buf_class(self.monsterT)
        self.assertEqual(monster.Testarrayoftables(0).Hp(), 200)
        self.assertEqual(monster.Testarrayoftables(1).Hp(), 400)

    def test_empty_testarrayoftables(self):
        if False:
            print('Hello World!')
        self.monsterT.testarrayoftables = []
        monster = self._pack_and_load_buf_class(self.monsterT)
        self.assertFalse(monster.TestarrayoftablesIsNone())

    def test_mutate_enemy(self):
        if False:
            while True:
                i = 10
        monsterT = _MONSTER.MonsterT()
        monsterT.hp = 200
        self.monsterT.enemy = monsterT
        monster = self._pack_and_load_buf_class(self.monsterT)
        self.assertEqual(monster.Enemy().Hp(), 200)

    def test_mutate_testnestedflatbuffer(self):
        if False:
            for i in range(10):
                print('nop')
        self.monsterT.testnestedflatbuffer = [8, 2, 4]
        monster = self._pack_and_load_buf_class(self.monsterT)
        self.assertEqual(monster.Testnestedflatbuffer(0), 8)
        self.assertEqual(monster.Testnestedflatbuffer(1), 2)
        self.assertEqual(monster.Testnestedflatbuffer(2), 4)

    def test_empty_testnestedflatbuffer(self):
        if False:
            return 10
        self.monsterT.testnestedflatbuffer = []
        monster = self._pack_and_load_buf_class(self.monsterT)
        self.assertFalse(monster.TestnestedflatbufferIsNone())

    def test_mutate_testbool(self):
        if False:
            print('Hello World!')
        self.monsterT.testbool = True
        monster = self._pack_and_load_buf_class(self.monsterT)
        self.assertTrue(monster.Testbool())

    def test_mutate_testhashes(self):
        if False:
            print('Hello World!')
        self.monsterT.testhashs32Fnv1 = 1
        self.monsterT.testhashu32Fnv1 = 2
        self.monsterT.testhashs64Fnv1 = 3
        self.monsterT.testhashu64Fnv1 = 4
        self.monsterT.testhashs32Fnv1a = 5
        self.monsterT.testhashu32Fnv1a = 6
        self.monsterT.testhashs64Fnv1a = 7
        self.monsterT.testhashu64Fnv1a = 8
        monster = self._pack_and_load_buf_class(self.monsterT)
        self.assertEqual(monster.Testhashs32Fnv1(), 1)
        self.assertEqual(monster.Testhashu32Fnv1(), 2)
        self.assertEqual(monster.Testhashs64Fnv1(), 3)
        self.assertEqual(monster.Testhashu64Fnv1(), 4)
        self.assertEqual(monster.Testhashs32Fnv1a(), 5)
        self.assertEqual(monster.Testhashu32Fnv1a(), 6)
        self.assertEqual(monster.Testhashs64Fnv1a(), 7)
        self.assertEqual(monster.Testhashu64Fnv1a(), 8)

    def test_mutate_testarrayofbools(self):
        if False:
            return 10
        self.monsterT.testarrayofbools = []
        self.monsterT.testarrayofbools.append(True)
        self.monsterT.testarrayofbools.append(True)
        self.monsterT.testarrayofbools.append(False)
        monster = self._pack_and_load_buf_class(self.monsterT)
        self.assertEqual(monster.Testarrayofbools(0), True)
        self.assertEqual(monster.Testarrayofbools(1), True)
        self.assertEqual(monster.Testarrayofbools(2), False)

    def test_empty_testarrayofbools(self):
        if False:
            for i in range(10):
                print('nop')
        self.monsterT.testarrayofbools = []
        monster = self._pack_and_load_buf_class(self.monsterT)
        self.assertFalse(monster.TestarrayofboolsIsNone())

    def test_mutate_testf(self):
        if False:
            print('Hello World!')
        self.monsterT.testf = 2.0
        monster = self._pack_and_load_buf_class(self.monsterT)
        self.assertEqual(monster.Testf(), 2.0)

    def test_mutate_vectoroflongs(self):
        if False:
            for i in range(10):
                print('nop')
        self.monsterT.vectorOfLongs = []
        self.monsterT.vectorOfLongs.append(1)
        self.monsterT.vectorOfLongs.append(100)
        self.monsterT.vectorOfLongs.append(10000)
        self.monsterT.vectorOfLongs.append(1000000)
        self.monsterT.vectorOfLongs.append(100000000)
        monster = self._pack_and_load_buf_class(self.monsterT)
        self.assertEqual(monster.VectorOfLongs(0), 1)
        self.assertEqual(monster.VectorOfLongs(1), 100)
        self.assertEqual(monster.VectorOfLongs(2), 10000)
        self.assertEqual(monster.VectorOfLongs(3), 1000000)
        self.assertEqual(monster.VectorOfLongs(4), 100000000)

    def test_empty_vectoroflongs(self):
        if False:
            i = 10
            return i + 15
        self.monsterT.vectorOfLongs = []
        monster = self._pack_and_load_buf_class(self.monsterT)
        self.assertFalse(monster.VectorOfLongsIsNone())

    def test_mutate_vectorofdoubles(self):
        if False:
            return 10
        self.monsterT.vectorOfDoubles = []
        self.monsterT.vectorOfDoubles.append(-1.7976931348623157e+308)
        self.monsterT.vectorOfDoubles.append(0)
        self.monsterT.vectorOfDoubles.append(1.7976931348623157e+308)
        monster = self._pack_and_load_buf_class(self.monsterT)
        self.assertEqual(monster.VectorOfDoubles(0), -1.7976931348623157e+308)
        self.assertEqual(monster.VectorOfDoubles(1), 0)
        self.assertEqual(monster.VectorOfDoubles(2), 1.7976931348623157e+308)

    def test_empty_vectorofdoubles(self):
        if False:
            i = 10
            return i + 15
        self.monsterT.vectorOfDoubles = []
        monster = self._pack_and_load_buf_class(self.monsterT)
        self.assertFalse(monster.VectorOfDoublesIsNone())

    def test_mutate_parentnamespacetest(self):
        if False:
            for i in range(10):
                print('nop')
        self.monsterT.parentNamespaceTest = _IN_PARENT_NAMESPACE.InParentNamespaceT()
        monster = self._pack_and_load_buf_class(self.monsterT)
        self.assertTrue(isinstance(monster.ParentNamespaceTest(), _IN_PARENT_NAMESPACE.InParentNamespace))

    def test_mutate_vectorofEnums(self):
        if False:
            while True:
                i = 10
        self.monsterT.vectorOfEnums = []
        self.monsterT.vectorOfEnums.append(_COLOR.Color.Red)
        self.monsterT.vectorOfEnums.append(_COLOR.Color.Blue)
        self.monsterT.vectorOfEnums.append(_COLOR.Color.Red)
        monster = self._pack_and_load_buf_class(self.monsterT)
        self.assertEqual(monster.VectorOfEnums(0), _COLOR.Color.Red)
        self.assertEqual(monster.VectorOfEnums(1), _COLOR.Color.Blue)
        self.assertEqual(monster.VectorOfEnums(2), _COLOR.Color.Red)

    def test_empty_vectorofEnums(self):
        if False:
            i = 10
            return i + 15
        self.monsterT.vectorOfEnums = []
        monster = self._pack_and_load_buf_class(self.monsterT)
        self.assertFalse(monster.VectorOfEnumsIsNone())

def CheckReadBuffer(buf, offset, sizePrefix=False, file_identifier=None):
    if False:
        for i in range(10):
            print('nop')
    ' CheckReadBuffer checks that the given buffer is evaluated correctly\n\n        as the example Monster.\n  '

    def asserter(stmt):
        if False:
            print('Hello World!')
        ' An assertion helper that is separated from TestCase classes. '
        if not stmt:
            raise AssertionError('CheckReadBuffer case failed')
    if file_identifier:
        asserter(util.GetBufferIdentifier(buf, offset, size_prefixed=sizePrefix) == file_identifier)
        asserter(util.BufferHasIdentifier(buf, offset, file_identifier=file_identifier, size_prefixed=sizePrefix))
        asserter(_MONSTER.Monster.MonsterBufferHasIdentifier(buf, offset, size_prefixed=sizePrefix))
    if sizePrefix:
        size = util.GetSizePrefix(buf, offset)
        asserter(size == len(buf[offset:]) - 4)
        (buf, offset) = util.RemoveSizePrefix(buf, offset)
    if file_identifier:
        asserter(_MONSTER.Monster.MonsterBufferHasIdentifier(buf, offset))
    else:
        asserter(not _MONSTER.Monster.MonsterBufferHasIdentifier(buf, offset))
    monster = _MONSTER.Monster.GetRootAs(buf, offset)
    asserter(monster.Hp() == 80)
    asserter(monster.Mana() == 150)
    asserter(monster.Name() == b'MyMonster')
    vec = monster.Pos()
    asserter(vec is not None)
    asserter(vec.X() == 1.0)
    asserter(vec.Y() == 2.0)
    asserter(vec.Z() == 3.0)
    asserter(vec.Test1() == 3.0)
    asserter(vec.Test2() == 2)
    t = _TEST.Test()
    t = vec.Test3(t)
    asserter(t is not None)
    asserter(t.A() == 5)
    asserter(t.B() == 6)
    union_type = _ANY.Any
    asserter(monster.TestType() == union_type.Monster)
    table2 = monster.Test()
    asserter(type(table2) is flatbuffers.table.Table)
    monster2 = _MONSTER.Monster()
    monster2.Init(table2.Bytes, table2.Pos)
    asserter(monster2.Name() == b'Fred')
    asserter(monster.InventoryLength() == 5)
    asserter(not monster.InventoryIsNone())
    invsum = 0
    for i in compat_range(monster.InventoryLength()):
        v = monster.Inventory(i)
        invsum += int(v)
    asserter(invsum == 10)
    for i in range(5):
        asserter(monster.VectorOfLongs(i) == 10 ** (i * 2))
    asserter(not monster.VectorOfDoublesIsNone())
    asserter([-1.7976931348623157e+308, 0, 1.7976931348623157e+308] == [monster.VectorOfDoubles(i) for i in range(monster.VectorOfDoublesLength())])
    try:
        import numpy as np
        asserter(monster.InventoryAsNumpy().sum() == 10)
        asserter(monster.InventoryAsNumpy().dtype == np.dtype('<u1'))
        VectorOfLongs = monster.VectorOfLongsAsNumpy()
        asserter(VectorOfLongs.dtype == np.dtype('<i8'))
        for i in range(5):
            asserter(VectorOfLongs[i] == 10 ** (i * 2))
        VectorOfDoubles = monster.VectorOfDoublesAsNumpy()
        asserter(VectorOfDoubles.dtype == np.dtype('<f8'))
        asserter(VectorOfDoubles[0] == np.finfo('<f8').min)
        asserter(VectorOfDoubles[1] == 0.0)
        asserter(VectorOfDoubles[2] == np.finfo('<f8').max)
    except ImportError:
        pass
    asserter(monster.Test4Length() == 2)
    asserter(not monster.Test4IsNone())
    test0 = monster.Test4(0)
    asserter(type(test0) is _TEST.Test)
    test1 = monster.Test4(1)
    asserter(type(test1) is _TEST.Test)
    v0 = test0.A()
    v1 = test0.B()
    v2 = test1.A()
    v3 = test1.B()
    sumtest12 = int(v0) + int(v1) + int(v2) + int(v3)
    asserter(sumtest12 == 100)
    asserter(not monster.TestarrayofstringIsNone())
    asserter(monster.TestarrayofstringLength() == 2)
    asserter(monster.Testarrayofstring(0) == b'test1')
    asserter(monster.Testarrayofstring(1) == b'test2')
    asserter(monster.TestarrayoftablesIsNone())
    asserter(monster.TestarrayoftablesLength() == 0)
    asserter(monster.TestnestedflatbufferIsNone())
    asserter(monster.TestnestedflatbufferLength() == 0)
    asserter(monster.Testempty() is None)

class TestFuzz(unittest.TestCase):
    """ Low level stress/fuzz test: serialize/deserialize a variety of

        different kinds of data in different combinations
  """
    binary_type = compat.binary_types[0]
    ofInt32Bytes = binary_type([131, 51, 51, 51])
    ofInt64Bytes = binary_type([132, 68, 68, 68, 68, 68, 68, 68])
    overflowingInt32Val = flatbuffers.encode.Get(flatbuffers.packer.int32, ofInt32Bytes, 0)
    overflowingInt64Val = flatbuffers.encode.Get(flatbuffers.packer.int64, ofInt64Bytes, 0)
    boolVal = True
    int8Val = N.Int8Flags.py_type(-127)
    uint8Val = N.Uint8Flags.py_type(255)
    int16Val = N.Int16Flags.py_type(-32222)
    uint16Val = N.Uint16Flags.py_type(65262)
    int32Val = N.Int32Flags.py_type(overflowingInt32Val)
    uint32Val = N.Uint32Flags.py_type(4259175901)
    int64Val = N.Int64Flags.py_type(overflowingInt64Val)
    uint64Val = N.Uint64Flags.py_type(18216159772788182220)
    float32Val = N.Float32Flags.py_type(ctypes.c_float(3.14159).value)
    float64Val = N.Float64Flags.py_type(3.14159265359)

    def test_fuzz(self):
        if False:
            return 10
        return self.check_once(11, 100)

    def check_once(self, fuzzFields, fuzzObjects):
        if False:
            for i in range(10):
                print('nop')
        testValuesMax = 11
        builder = flatbuffers.Builder(0)
        l = LCG()
        objects = [0 for _ in compat_range(fuzzObjects)]
        for i in compat_range(fuzzObjects):
            builder.StartObject(fuzzFields)
            for j in compat_range(fuzzFields):
                choice = int(l.Next()) % testValuesMax
                if choice == 0:
                    builder.PrependBoolSlot(int(j), self.boolVal, False)
                elif choice == 1:
                    builder.PrependInt8Slot(int(j), self.int8Val, 0)
                elif choice == 2:
                    builder.PrependUint8Slot(int(j), self.uint8Val, 0)
                elif choice == 3:
                    builder.PrependInt16Slot(int(j), self.int16Val, 0)
                elif choice == 4:
                    builder.PrependUint16Slot(int(j), self.uint16Val, 0)
                elif choice == 5:
                    builder.PrependInt32Slot(int(j), self.int32Val, 0)
                elif choice == 6:
                    builder.PrependUint32Slot(int(j), self.uint32Val, 0)
                elif choice == 7:
                    builder.PrependInt64Slot(int(j), self.int64Val, 0)
                elif choice == 8:
                    builder.PrependUint64Slot(int(j), self.uint64Val, 0)
                elif choice == 9:
                    builder.PrependFloat32Slot(int(j), self.float32Val, 0)
                elif choice == 10:
                    builder.PrependFloat64Slot(int(j), self.float64Val, 0)
                else:
                    raise RuntimeError('unreachable')
            off = builder.EndObject()
            objects[i] = off
        stats = defaultdict(int)

        def check(table, desc, want, got):
            if False:
                for i in range(10):
                    print('nop')
            stats[desc] += 1
            self.assertEqual(want, got, '%s != %s, %s' % (want, got, desc))
        l = LCG()
        for i in compat_range(fuzzObjects):
            table = flatbuffers.table.Table(builder.Bytes, len(builder.Bytes) - objects[i])
            for j in compat_range(fuzzFields):
                field_count = flatbuffers.builder.VtableMetadataFields + j
                f = N.VOffsetTFlags.py_type(field_count * N.VOffsetTFlags.bytewidth)
                choice = int(l.Next()) % testValuesMax
                if choice == 0:
                    check(table, 'bool', self.boolVal, table.GetSlot(f, False, N.BoolFlags))
                elif choice == 1:
                    check(table, '<i1', self.int8Val, table.GetSlot(f, 0, N.Int8Flags))
                elif choice == 2:
                    check(table, '<u1', self.uint8Val, table.GetSlot(f, 0, N.Uint8Flags))
                elif choice == 3:
                    check(table, '<i2', self.int16Val, table.GetSlot(f, 0, N.Int16Flags))
                elif choice == 4:
                    check(table, '<u2', self.uint16Val, table.GetSlot(f, 0, N.Uint16Flags))
                elif choice == 5:
                    check(table, '<i4', self.int32Val, table.GetSlot(f, 0, N.Int32Flags))
                elif choice == 6:
                    check(table, '<u4', self.uint32Val, table.GetSlot(f, 0, N.Uint32Flags))
                elif choice == 7:
                    check(table, '<i8', self.int64Val, table.GetSlot(f, 0, N.Int64Flags))
                elif choice == 8:
                    check(table, '<u8', self.uint64Val, table.GetSlot(f, 0, N.Uint64Flags))
                elif choice == 9:
                    check(table, '<f4', self.float32Val, table.GetSlot(f, 0, N.Float32Flags))
                elif choice == 10:
                    check(table, '<f8', self.float64Val, table.GetSlot(f, 0, N.Float64Flags))
                else:
                    raise RuntimeError('unreachable')
        self.assertEqual(testValuesMax, len(stats), 'fuzzing failed to test all scalar types: %s' % stats)

class TestByteLayout(unittest.TestCase):
    """ TestByteLayout checks the bytes of a Builder in various scenarios. """

    def assertBuilderEquals(self, builder, want_chars_or_ints):
        if False:
            while True:
                i = 10

        def integerize(x):
            if False:
                print('Hello World!')
            if isinstance(x, compat.string_types):
                return ord(x)
            return x
        want_ints = list(map(integerize, want_chars_or_ints))
        want = bytearray(want_ints)
        got = builder.Bytes[builder.Head():]
        self.assertEqual(want, got)

    def test_numbers(self):
        if False:
            while True:
                i = 10
        b = flatbuffers.Builder(0)
        self.assertBuilderEquals(b, [])
        b.PrependBool(True)
        self.assertBuilderEquals(b, [1])
        b.PrependInt8(-127)
        self.assertBuilderEquals(b, [129, 1])
        b.PrependUint8(255)
        self.assertBuilderEquals(b, [255, 129, 1])
        b.PrependInt16(-32222)
        self.assertBuilderEquals(b, [34, 130, 0, 255, 129, 1])
        b.PrependUint16(65262)
        self.assertBuilderEquals(b, [238, 254, 34, 130, 0, 255, 129, 1])
        b.PrependInt32(-53687092)
        self.assertBuilderEquals(b, [204, 204, 204, 252, 238, 254, 34, 130, 0, 255, 129, 1])
        b.PrependUint32(2557891634)
        self.assertBuilderEquals(b, [50, 84, 118, 152, 204, 204, 204, 252, 238, 254, 34, 130, 0, 255, 129, 1])

    def test_numbers64(self):
        if False:
            i = 10
            return i + 15
        b = flatbuffers.Builder(0)
        b.PrependUint64(1234605616436508552)
        self.assertBuilderEquals(b, [136, 119, 102, 85, 68, 51, 34, 17])
        b = flatbuffers.Builder(0)
        b.PrependInt64(1234605616436508552)
        self.assertBuilderEquals(b, [136, 119, 102, 85, 68, 51, 34, 17])

    def test_1xbyte_vector(self):
        if False:
            while True:
                i = 10
        b = flatbuffers.Builder(0)
        self.assertBuilderEquals(b, [])
        b.StartVector(flatbuffers.number_types.Uint8Flags.bytewidth, 1, 1)
        self.assertBuilderEquals(b, [0, 0, 0])
        b.PrependByte(1)
        self.assertBuilderEquals(b, [1, 0, 0, 0])
        b.EndVector()
        self.assertBuilderEquals(b, [1, 0, 0, 0, 1, 0, 0, 0])

    def test_2xbyte_vector(self):
        if False:
            for i in range(10):
                print('nop')
        b = flatbuffers.Builder(0)
        b.StartVector(flatbuffers.number_types.Uint8Flags.bytewidth, 2, 1)
        self.assertBuilderEquals(b, [0, 0])
        b.PrependByte(1)
        self.assertBuilderEquals(b, [1, 0, 0])
        b.PrependByte(2)
        self.assertBuilderEquals(b, [2, 1, 0, 0])
        b.EndVector()
        self.assertBuilderEquals(b, [2, 0, 0, 0, 2, 1, 0, 0])

    def test_1xuint16_vector(self):
        if False:
            print('Hello World!')
        b = flatbuffers.Builder(0)
        b.StartVector(flatbuffers.number_types.Uint16Flags.bytewidth, 1, 1)
        self.assertBuilderEquals(b, [0, 0])
        b.PrependUint16(1)
        self.assertBuilderEquals(b, [1, 0, 0, 0])
        b.EndVector()
        self.assertBuilderEquals(b, [1, 0, 0, 0, 1, 0, 0, 0])

    def test_2xuint16_vector(self):
        if False:
            i = 10
            return i + 15
        b = flatbuffers.Builder(0)
        b.StartVector(flatbuffers.number_types.Uint16Flags.bytewidth, 2, 1)
        self.assertBuilderEquals(b, [])
        b.PrependUint16(43981)
        self.assertBuilderEquals(b, [205, 171])
        b.PrependUint16(56506)
        self.assertBuilderEquals(b, [186, 220, 205, 171])
        b.EndVector()
        self.assertBuilderEquals(b, [2, 0, 0, 0, 186, 220, 205, 171])

    def test_create_ascii_shared_string(self):
        if False:
            i = 10
            return i + 15
        b = flatbuffers.Builder(0)
        b.CreateSharedString(u'foo', encoding='ascii')
        b.CreateSharedString(u'foo', encoding='ascii')
        self.assertBuilderEquals(b, [3, 0, 0, 0, 'f', 'o', 'o', 0])
        b.CreateSharedString(u'moop', encoding='ascii')
        b.CreateSharedString(u'moop', encoding='ascii')
        self.assertBuilderEquals(b, [4, 0, 0, 0, 'm', 'o', 'o', 'p', 0, 0, 0, 0, 3, 0, 0, 0, 'f', 'o', 'o', 0])

    def test_create_utf8_shared_string(self):
        if False:
            print('Hello World!')
        b = flatbuffers.Builder(0)
        b.CreateSharedString(u'Цлїςσδε')
        b.CreateSharedString(u'Цлїςσδε')
        self.assertBuilderEquals(b, '\x0e\x00\x00\x00Ð¦Ð»Ñ\x97Ï\x82Ï\x83Î´Îµ\x00\x00')
        b.CreateSharedString(u'ﾌﾑｱﾑｶﾓｹﾓ')
        b.CreateSharedString(u'ﾌﾑｱﾑｶﾓｹﾓ')
        self.assertBuilderEquals(b, '\x18\x00\x00\x00ï¾\x8cï¾\x91ï½±ï¾\x91ï½¶ï¾\x93ï½¹ï¾\x93\x00\x00\x00\x00\x0e\x00\x00\x00Ð¦Ð»Ñ\x97Ï\x82Ï\x83Î´Îµ\x00\x00')

    def test_create_arbitrary_shared_string(self):
        if False:
            for i in range(10):
                print('nop')
        b = flatbuffers.Builder(0)
        s = '\x01\x02\x03'
        b.CreateSharedString(s)
        b.CreateSharedString(s)
        self.assertBuilderEquals(b, [3, 0, 0, 0, 1, 2, 3, 0])
        s2 = '\x04\x05\x06\x07'
        b.CreateSharedString(s2)
        b.CreateSharedString(s2)
        self.assertBuilderEquals(b, [4, 0, 0, 0, 4, 5, 6, 7, 0, 0, 0, 0, 3, 0, 0, 0, 1, 2, 3, 0])

    def test_create_ascii_string(self):
        if False:
            i = 10
            return i + 15
        b = flatbuffers.Builder(0)
        b.CreateString(u'foo', encoding='ascii')
        self.assertBuilderEquals(b, [3, 0, 0, 0, 'f', 'o', 'o', 0])
        b.CreateString(u'moop', encoding='ascii')
        self.assertBuilderEquals(b, [4, 0, 0, 0, 'm', 'o', 'o', 'p', 0, 0, 0, 0, 3, 0, 0, 0, 'f', 'o', 'o', 0])

    def test_create_utf8_string(self):
        if False:
            return 10
        b = flatbuffers.Builder(0)
        b.CreateString(u'Цлїςσδε')
        self.assertBuilderEquals(b, '\x0e\x00\x00\x00Ð¦Ð»Ñ\x97Ï\x82Ï\x83Î´Îµ\x00\x00')
        b.CreateString(u'ﾌﾑｱﾑｶﾓｹﾓ')
        self.assertBuilderEquals(b, '\x18\x00\x00\x00ï¾\x8cï¾\x91ï½±ï¾\x91ï½¶ï¾\x93ï½¹ï¾\x93\x00\x00\x00\x00\x0e\x00\x00\x00Ð¦Ð»Ñ\x97Ï\x82Ï\x83Î´Îµ\x00\x00')

    def test_create_arbitrary_string(self):
        if False:
            i = 10
            return i + 15
        b = flatbuffers.Builder(0)
        s = '\x01\x02\x03'
        b.CreateString(s)
        self.assertBuilderEquals(b, [3, 0, 0, 0, 1, 2, 3, 0])
        s2 = '\x04\x05\x06\x07'
        b.CreateString(s2)
        self.assertBuilderEquals(b, [4, 0, 0, 0, 4, 5, 6, 7, 0, 0, 0, 0, 3, 0, 0, 0, 1, 2, 3, 0])

    def test_create_byte_vector(self):
        if False:
            i = 10
            return i + 15
        b = flatbuffers.Builder(0)
        b.CreateByteVector(b'')
        self.assertBuilderEquals(b, [0, 0, 0, 0])
        b = flatbuffers.Builder(0)
        b.CreateByteVector(b'\x01\x02\x03')
        self.assertBuilderEquals(b, [3, 0, 0, 0, 1, 2, 3, 0])

    def test_create_numpy_vector_int8(self):
        if False:
            while True:
                i = 10
        try:
            import numpy as np
            b = flatbuffers.Builder(0)
            x = np.array([1, 2, -3], dtype=np.int8)
            b.CreateNumpyVector(x)
            self.assertBuilderEquals(b, [3, 0, 0, 0, 1, 2, 256 - 3, 0])
            b = flatbuffers.Builder(0)
            x_other_endian = x.byteswap().newbyteorder()
            b.CreateNumpyVector(x_other_endian)
            self.assertBuilderEquals(b, [3, 0, 0, 0, 1, 2, 256 - 3, 0])
        except ImportError:
            b = flatbuffers.Builder(0)
            x = 0
            assertRaises(self, lambda : b.CreateNumpyVector(x), NumpyRequiredForThisFeature)

    def test_create_numpy_vector_uint16(self):
        if False:
            return 10
        try:
            import numpy as np
            b = flatbuffers.Builder(0)
            x = np.array([1, 2, 312], dtype=np.uint16)
            b.CreateNumpyVector(x)
            self.assertBuilderEquals(b, [3, 0, 0, 0, 1, 0, 2, 0, 312 - 256, 1, 0, 0])
            b = flatbuffers.Builder(0)
            x_other_endian = x.byteswap().newbyteorder()
            b.CreateNumpyVector(x_other_endian)
            self.assertBuilderEquals(b, [3, 0, 0, 0, 1, 0, 2, 0, 312 - 256, 1, 0, 0])
        except ImportError:
            b = flatbuffers.Builder(0)
            x = 0
            assertRaises(self, lambda : b.CreateNumpyVector(x), NumpyRequiredForThisFeature)

    def test_create_numpy_vector_int64(self):
        if False:
            i = 10
            return i + 15
        try:
            import numpy as np
            b = flatbuffers.Builder(0)
            x = np.array([1, 2, -12], dtype=np.int64)
            b.CreateNumpyVector(x)
            self.assertBuilderEquals(b, [3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 256 - 12, 255, 255, 255, 255, 255, 255, 255])
            b = flatbuffers.Builder(0)
            x_other_endian = x.byteswap().newbyteorder()
            b.CreateNumpyVector(x_other_endian)
            self.assertBuilderEquals(b, [3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 256 - 12, 255, 255, 255, 255, 255, 255, 255])
        except ImportError:
            b = flatbuffers.Builder(0)
            x = 0
            assertRaises(self, lambda : b.CreateNumpyVector(x), NumpyRequiredForThisFeature)

    def test_create_numpy_vector_float32(self):
        if False:
            while True:
                i = 10
        try:
            import numpy as np
            b = flatbuffers.Builder(0)
            x = np.array([1, 2, -12], dtype=np.float32)
            b.CreateNumpyVector(x)
            self.assertBuilderEquals(b, [3, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 193])
            b = flatbuffers.Builder(0)
            x_other_endian = x.byteswap().newbyteorder()
            b.CreateNumpyVector(x_other_endian)
            self.assertBuilderEquals(b, [3, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 193])
        except ImportError:
            b = flatbuffers.Builder(0)
            x = 0
            assertRaises(self, lambda : b.CreateNumpyVector(x), NumpyRequiredForThisFeature)

    def test_create_numpy_vector_float64(self):
        if False:
            i = 10
            return i + 15
        try:
            import numpy as np
            b = flatbuffers.Builder(0)
            x = np.array([1, 2, -12], dtype=np.float64)
            b.CreateNumpyVector(x)
            self.assertBuilderEquals(b, [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 40, 192])
            b = flatbuffers.Builder(0)
            x_other_endian = x.byteswap().newbyteorder()
            b.CreateNumpyVector(x_other_endian)
            self.assertBuilderEquals(b, [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 40, 192])
        except ImportError:
            b = flatbuffers.Builder(0)
            x = 0
            assertRaises(self, lambda : b.CreateNumpyVector(x), NumpyRequiredForThisFeature)

    def test_create_numpy_vector_bool(self):
        if False:
            return 10
        try:
            import numpy as np
            b = flatbuffers.Builder(0)
            x = np.array([True, False, True], dtype=bool)
            b.CreateNumpyVector(x)
            self.assertBuilderEquals(b, [3, 0, 0, 0, 1, 0, 1, 0])
            b = flatbuffers.Builder(0)
            x_other_endian = x.byteswap().newbyteorder()
            b.CreateNumpyVector(x_other_endian)
            self.assertBuilderEquals(b, [3, 0, 0, 0, 1, 0, 1, 0])
        except ImportError:
            b = flatbuffers.Builder(0)
            x = 0
            assertRaises(self, lambda : b.CreateNumpyVector(x), NumpyRequiredForThisFeature)

    def test_create_numpy_vector_reject_strings(self):
        if False:
            while True:
                i = 10
        try:
            import numpy as np
            b = flatbuffers.Builder(0)
            x = np.array(['hello', 'fb', 'testing'])
            assertRaises(self, lambda : b.CreateNumpyVector(x), TypeError)
        except ImportError:
            b = flatbuffers.Builder(0)
            x = 0
            assertRaises(self, lambda : b.CreateNumpyVector(x), NumpyRequiredForThisFeature)

    def test_create_numpy_vector_reject_object(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            import numpy as np
            b = flatbuffers.Builder(0)
            x = np.array([{'m': 0}, {'as': -2.1, 'c': 'c'}])
            assertRaises(self, lambda : b.CreateNumpyVector(x), TypeError)
        except ImportError:
            b = flatbuffers.Builder(0)
            x = 0
            assertRaises(self, lambda : b.CreateNumpyVector(x), NumpyRequiredForThisFeature)

    def test_empty_vtable(self):
        if False:
            i = 10
            return i + 15
        b = flatbuffers.Builder(0)
        b.StartObject(0)
        self.assertBuilderEquals(b, [])
        b.EndObject()
        self.assertBuilderEquals(b, [4, 0, 4, 0, 4, 0, 0, 0])

    def test_vtable_with_one_true_bool(self):
        if False:
            return 10
        b = flatbuffers.Builder(0)
        self.assertBuilderEquals(b, [])
        b.StartObject(1)
        self.assertBuilderEquals(b, [])
        b.PrependBoolSlot(0, True, False)
        b.EndObject()
        self.assertBuilderEquals(b, [6, 0, 8, 0, 7, 0, 6, 0, 0, 0, 0, 0, 0, 1])

    def test_vtable_with_one_default_bool(self):
        if False:
            while True:
                i = 10
        b = flatbuffers.Builder(0)
        self.assertBuilderEquals(b, [])
        b.StartObject(1)
        self.assertBuilderEquals(b, [])
        b.PrependBoolSlot(0, False, False)
        b.EndObject()
        self.assertBuilderEquals(b, [4, 0, 4, 0, 4, 0, 0, 0])

    def test_vtable_with_one_int16(self):
        if False:
            i = 10
            return i + 15
        b = flatbuffers.Builder(0)
        b.StartObject(1)
        b.PrependInt16Slot(0, 30874, 0)
        b.EndObject()
        self.assertBuilderEquals(b, [6, 0, 8, 0, 6, 0, 6, 0, 0, 0, 0, 0, 154, 120])

    def test_vtable_with_two_int16(self):
        if False:
            i = 10
            return i + 15
        b = flatbuffers.Builder(0)
        b.StartObject(2)
        b.PrependInt16Slot(0, 13398, 0)
        b.PrependInt16Slot(1, 30874, 0)
        b.EndObject()
        self.assertBuilderEquals(b, [8, 0, 8, 0, 6, 0, 4, 0, 8, 0, 0, 0, 154, 120, 86, 52])

    def test_vtable_with_int16_and_bool(self):
        if False:
            for i in range(10):
                print('nop')
        b = flatbuffers.Builder(0)
        b.StartObject(2)
        b.PrependInt16Slot(0, 13398, 0)
        b.PrependBoolSlot(1, True, False)
        b.EndObject()
        self.assertBuilderEquals(b, [8, 0, 8, 0, 6, 0, 5, 0, 8, 0, 0, 0, 0, 1, 86, 52])

    def test_vtable_with_empty_vector(self):
        if False:
            for i in range(10):
                print('nop')
        b = flatbuffers.Builder(0)
        b.StartVector(flatbuffers.number_types.Uint8Flags.bytewidth, 0, 1)
        vecend = b.EndVector()
        b.StartObject(1)
        b.PrependUOffsetTRelativeSlot(0, vecend, 0)
        b.EndObject()
        self.assertBuilderEquals(b, [6, 0, 8, 0, 4, 0, 6, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0])

    def test_vtable_with_empty_vector_of_byte_and_some_scalars(self):
        if False:
            return 10
        b = flatbuffers.Builder(0)
        b.StartVector(flatbuffers.number_types.Uint8Flags.bytewidth, 0, 1)
        vecend = b.EndVector()
        b.StartObject(2)
        b.PrependInt16Slot(0, 55, 0)
        b.PrependUOffsetTRelativeSlot(1, vecend, 0)
        b.EndObject()
        self.assertBuilderEquals(b, [8, 0, 12, 0, 10, 0, 4, 0, 8, 0, 0, 0, 8, 0, 0, 0, 0, 0, 55, 0, 0, 0, 0, 0])

    def test_vtable_with_1_int16_and_2vector_of_int16(self):
        if False:
            return 10
        b = flatbuffers.Builder(0)
        b.StartVector(flatbuffers.number_types.Int16Flags.bytewidth, 2, 1)
        b.PrependInt16(4660)
        b.PrependInt16(22136)
        vecend = b.EndVector()
        b.StartObject(2)
        b.PrependUOffsetTRelativeSlot(1, vecend, 0)
        b.PrependInt16Slot(0, 55, 0)
        b.EndObject()
        self.assertBuilderEquals(b, [8, 0, 12, 0, 6, 0, 8, 0, 8, 0, 0, 0, 0, 0, 55, 0, 4, 0, 0, 0, 2, 0, 0, 0, 120, 86, 52, 18])

    def test_vtable_with_1_struct_of_1_int8__1_int16__1_int32(self):
        if False:
            print('Hello World!')
        b = flatbuffers.Builder(0)
        b.StartObject(1)
        b.Prep(4 + 4 + 4, 0)
        b.PrependInt8(55)
        b.Pad(3)
        b.PrependInt16(4660)
        b.Pad(2)
        b.PrependInt32(305419896)
        structStart = b.Offset()
        b.PrependStructSlot(0, structStart, 0)
        b.EndObject()
        self.assertBuilderEquals(b, [6, 0, 16, 0, 4, 0, 6, 0, 0, 0, 120, 86, 52, 18, 0, 0, 52, 18, 0, 0, 0, 55])

    def test_vtable_with_1_vector_of_2_struct_of_2_int8(self):
        if False:
            return 10
        b = flatbuffers.Builder(0)
        b.StartVector(flatbuffers.number_types.Int8Flags.bytewidth * 2, 2, 1)
        b.PrependInt8(33)
        b.PrependInt8(44)
        b.PrependInt8(55)
        b.PrependInt8(66)
        vecend = b.EndVector()
        b.StartObject(1)
        b.PrependUOffsetTRelativeSlot(0, vecend, 0)
        b.EndObject()
        self.assertBuilderEquals(b, [6, 0, 8, 0, 4, 0, 6, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 66, 55, 44, 33])

    def test_table_with_some_elements(self):
        if False:
            return 10
        b = flatbuffers.Builder(0)
        b.StartObject(2)
        b.PrependInt8Slot(0, 33, 0)
        b.PrependInt16Slot(1, 66, 0)
        off = b.EndObject()
        b.Finish(off)
        self.assertBuilderEquals(b, [12, 0, 0, 0, 8, 0, 8, 0, 7, 0, 4, 0, 8, 0, 0, 0, 66, 0, 0, 33])

    def test__one_unfinished_table_and_one_finished_table(self):
        if False:
            print('Hello World!')
        b = flatbuffers.Builder(0)
        b.StartObject(2)
        b.PrependInt8Slot(0, 33, 0)
        b.PrependInt8Slot(1, 44, 0)
        off = b.EndObject()
        b.Finish(off)
        b.StartObject(3)
        b.PrependInt8Slot(0, 55, 0)
        b.PrependInt8Slot(1, 66, 0)
        b.PrependInt8Slot(2, 77, 0)
        off = b.EndObject()
        b.Finish(off)
        self.assertBuilderEquals(b, [16, 0, 0, 0, 0, 0, 10, 0, 8, 0, 7, 0, 6, 0, 5, 0, 10, 0, 0, 0, 0, 77, 66, 55, 12, 0, 0, 0, 8, 0, 8, 0, 7, 0, 6, 0, 8, 0, 0, 0, 0, 0, 44, 33])

    def test_a_bunch_of_bools(self):
        if False:
            print('Hello World!')
        b = flatbuffers.Builder(0)
        b.StartObject(8)
        b.PrependBoolSlot(0, True, False)
        b.PrependBoolSlot(1, True, False)
        b.PrependBoolSlot(2, True, False)
        b.PrependBoolSlot(3, True, False)
        b.PrependBoolSlot(4, True, False)
        b.PrependBoolSlot(5, True, False)
        b.PrependBoolSlot(6, True, False)
        b.PrependBoolSlot(7, True, False)
        off = b.EndObject()
        b.Finish(off)
        self.assertBuilderEquals(b, [24, 0, 0, 0, 20, 0, 12, 0, 11, 0, 10, 0, 9, 0, 8, 0, 7, 0, 6, 0, 5, 0, 4, 0, 20, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

    def test_three_bools(self):
        if False:
            print('Hello World!')
        b = flatbuffers.Builder(0)
        b.StartObject(3)
        b.PrependBoolSlot(0, True, False)
        b.PrependBoolSlot(1, True, False)
        b.PrependBoolSlot(2, True, False)
        off = b.EndObject()
        b.Finish(off)
        self.assertBuilderEquals(b, [16, 0, 0, 0, 0, 0, 10, 0, 8, 0, 7, 0, 6, 0, 5, 0, 10, 0, 0, 0, 0, 1, 1, 1])

    def test_some_floats(self):
        if False:
            return 10
        b = flatbuffers.Builder(0)
        b.StartObject(1)
        b.PrependFloat32Slot(0, 1.0, 0.0)
        off = b.EndObject()
        self.assertBuilderEquals(b, [6, 0, 8, 0, 4, 0, 6, 0, 0, 0, 0, 0, 128, 63])

def make_monster_from_generated_code(sizePrefix=False, file_identifier=None):
    if False:
        print('Hello World!')
    ' Use generated code to build the example Monster. '
    b = flatbuffers.Builder(0)
    string = b.CreateString('MyMonster')
    test1 = b.CreateString('test1')
    test2 = b.CreateString('test2')
    fred = b.CreateString('Fred')
    _MONSTER.MonsterStartInventoryVector(b, 5)
    b.PrependByte(4)
    b.PrependByte(3)
    b.PrependByte(2)
    b.PrependByte(1)
    b.PrependByte(0)
    inv = b.EndVector()
    _MONSTER.MonsterStart(b)
    _MONSTER.MonsterAddName(b, fred)
    mon2 = _MONSTER.MonsterEnd(b)
    _MONSTER.MonsterStartTest4Vector(b, 2)
    _TEST.CreateTest(b, 10, 20)
    _TEST.CreateTest(b, 30, 40)
    test4 = b.EndVector()
    _MONSTER.MonsterStartTestarrayofstringVector(b, 2)
    b.PrependUOffsetTRelative(test2)
    b.PrependUOffsetTRelative(test1)
    testArrayOfString = b.EndVector()
    _MONSTER.MonsterStartVectorOfLongsVector(b, 5)
    b.PrependInt64(100000000)
    b.PrependInt64(1000000)
    b.PrependInt64(10000)
    b.PrependInt64(100)
    b.PrependInt64(1)
    VectorOfLongs = b.EndVector()
    _MONSTER.MonsterStartVectorOfDoublesVector(b, 3)
    b.PrependFloat64(1.7976931348623157e+308)
    b.PrependFloat64(0)
    b.PrependFloat64(-1.7976931348623157e+308)
    VectorOfDoubles = b.EndVector()
    _MONSTER.MonsterStart(b)
    pos = _VEC3.CreateVec3(b, 1.0, 2.0, 3.0, 3.0, 2, 5, 6)
    _MONSTER.MonsterAddPos(b, pos)
    _MONSTER.MonsterAddHp(b, 80)
    _MONSTER.MonsterAddName(b, string)
    _MONSTER.MonsterAddInventory(b, inv)
    _MONSTER.MonsterAddTestType(b, 1)
    _MONSTER.MonsterAddTest(b, mon2)
    _MONSTER.MonsterAddTest4(b, test4)
    _MONSTER.MonsterAddTestarrayofstring(b, testArrayOfString)
    _MONSTER.MonsterAddVectorOfLongs(b, VectorOfLongs)
    _MONSTER.MonsterAddVectorOfDoubles(b, VectorOfDoubles)
    mon = _MONSTER.MonsterEnd(b)
    if sizePrefix:
        b.FinishSizePrefixed(mon, file_identifier)
    else:
        b.Finish(mon, file_identifier)
    return (b.Bytes, b.Head())

class TestBuilderForceDefaults(unittest.TestCase):
    """Verify that the builder adds default values when forced."""
    test_flags = [N.BoolFlags(), N.Uint8Flags(), N.Uint16Flags(), N.Uint32Flags(), N.Uint64Flags(), N.Int8Flags(), N.Int16Flags(), N.Int32Flags(), N.Int64Flags(), N.Float32Flags(), N.Float64Flags(), N.UOffsetTFlags()]

    def test_default_force_defaults(self):
        if False:
            i = 10
            return i + 15
        for flag in self.test_flags:
            b = flatbuffers.Builder(0)
            b.StartObject(1)
            stored_offset = b.Offset()
            if flag != N.UOffsetTFlags():
                b.PrependSlot(flag, 0, 0, 0)
            else:
                b.PrependUOffsetTRelativeSlot(0, 0, 0)
            end_offset = b.Offset()
            b.EndObject()
            self.assertEqual(0, end_offset - stored_offset)

    def test_force_defaults_true(self):
        if False:
            i = 10
            return i + 15
        for flag in self.test_flags:
            b = flatbuffers.Builder(0)
            b.ForceDefaults(True)
            b.StartObject(1)
            stored_offset = b.Offset()
            if flag != N.UOffsetTFlags():
                b.PrependSlot(flag, 0, 0, 0)
            else:
                b.PrependUOffsetTRelativeSlot(0, 0, 0)
            end_offset = b.Offset()
            b.EndObject()
            self.assertEqual(flag.bytewidth, end_offset - stored_offset)

class TestAllCodePathsOfExampleSchema(unittest.TestCase):

    def setUp(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(TestAllCodePathsOfExampleSchema, self).setUp(*args, **kwargs)
        b = flatbuffers.Builder(0)
        _MONSTER.MonsterStart(b)
        gen_mon = _MONSTER.MonsterEnd(b)
        b.Finish(gen_mon)
        self.mon = _MONSTER.Monster.GetRootAs(b.Bytes, b.Head())

    def test_default_monster_pos(self):
        if False:
            print('Hello World!')
        self.assertTrue(self.mon.Pos() is None)

    def test_nondefault_monster_mana(self):
        if False:
            i = 10
            return i + 15
        b = flatbuffers.Builder(0)
        _MONSTER.MonsterStart(b)
        _MONSTER.MonsterAddMana(b, 50)
        mon = _MONSTER.MonsterEnd(b)
        b.Finish(mon)
        got_mon = _MONSTER.Monster.GetRootAs(b.Bytes, b.Head())
        self.assertEqual(50, got_mon.Mana())

    def test_default_monster_hp(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(100, self.mon.Hp())

    def test_default_monster_name(self):
        if False:
            print('Hello World!')
        self.assertEqual(None, self.mon.Name())

    def test_default_monster_inventory_item(self):
        if False:
            print('Hello World!')
        self.assertEqual(0, self.mon.Inventory(0))

    def test_default_monster_inventory_length(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(0, self.mon.InventoryLength())
        self.assertTrue(self.mon.InventoryIsNone())

    def test_empty_monster_inventory_vector(self):
        if False:
            while True:
                i = 10
        b = flatbuffers.Builder(0)
        _MONSTER.MonsterStartInventoryVector(b, 0)
        inv = b.EndVector()
        _MONSTER.MonsterStart(b)
        _MONSTER.MonsterAddInventory(b, inv)
        mon = _MONSTER.MonsterEnd(b)
        b.Finish(mon)
        mon2 = _MONSTER.Monster.GetRootAs(b.Bytes, b.Head())
        self.assertFalse(mon2.InventoryIsNone())

    def test_default_monster_color(self):
        if False:
            while True:
                i = 10
        self.assertEqual(_COLOR.Color.Blue, self.mon.Color())

    def test_nondefault_monster_color(self):
        if False:
            while True:
                i = 10
        b = flatbuffers.Builder(0)
        color = _COLOR.Color.Red
        _MONSTER.MonsterStart(b)
        _MONSTER.MonsterAddColor(b, color)
        mon = _MONSTER.MonsterEnd(b)
        b.Finish(mon)
        mon2 = _MONSTER.Monster.GetRootAs(b.Bytes, b.Head())
        self.assertEqual(_COLOR.Color.Red, mon2.Color())

    def test_default_monster_testtype(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(0, self.mon.TestType())

    def test_default_monster_test_field(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(None, self.mon.Test())

    def test_default_monster_test4_item(self):
        if False:
            print('Hello World!')
        self.assertEqual(None, self.mon.Test4(0))

    def test_default_monster_test4_length(self):
        if False:
            while True:
                i = 10
        self.assertEqual(0, self.mon.Test4Length())
        self.assertTrue(self.mon.Test4IsNone())

    def test_empty_monster_test4_vector(self):
        if False:
            while True:
                i = 10
        b = flatbuffers.Builder(0)
        _MONSTER.MonsterStartTest4Vector(b, 0)
        test4 = b.EndVector()
        _MONSTER.MonsterStart(b)
        _MONSTER.MonsterAddTest4(b, test4)
        mon = _MONSTER.MonsterEnd(b)
        b.Finish(mon)
        mon2 = _MONSTER.Monster.GetRootAs(b.Bytes, b.Head())
        self.assertFalse(mon2.Test4IsNone())

    def test_default_monster_testarrayofstring(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual('', self.mon.Testarrayofstring(0))

    def test_default_monster_testarrayofstring_length(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(0, self.mon.TestarrayofstringLength())
        self.assertTrue(self.mon.TestarrayofstringIsNone())

    def test_empty_monster_testarrayofstring_vector(self):
        if False:
            for i in range(10):
                print('nop')
        b = flatbuffers.Builder(0)
        _MONSTER.MonsterStartTestarrayofstringVector(b, 0)
        testarrayofstring = b.EndVector()
        _MONSTER.MonsterStart(b)
        _MONSTER.MonsterAddTestarrayofstring(b, testarrayofstring)
        mon = _MONSTER.MonsterEnd(b)
        b.Finish(mon)
        mon2 = _MONSTER.Monster.GetRootAs(b.Bytes, b.Head())
        self.assertFalse(mon2.TestarrayofstringIsNone())

    def test_default_monster_testarrayoftables(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(None, self.mon.Testarrayoftables(0))

    def test_nondefault_monster_testarrayoftables(self):
        if False:
            for i in range(10):
                print('nop')
        b = flatbuffers.Builder(0)
        _MONSTER.MonsterStart(b)
        _MONSTER.MonsterAddHp(b, 99)
        sub_monster = _MONSTER.MonsterEnd(b)
        _MONSTER.MonsterStartTestarrayoftablesVector(b, 1)
        b.PrependUOffsetTRelative(sub_monster)
        vec = b.EndVector()
        _MONSTER.MonsterStart(b)
        _MONSTER.MonsterAddTestarrayoftables(b, vec)
        mon = _MONSTER.MonsterEnd(b)
        b.Finish(mon)
        mon2 = _MONSTER.Monster.GetRootAs(b.Output(), 0)
        self.assertEqual(99, mon2.Testarrayoftables(0).Hp())
        self.assertEqual(1, mon2.TestarrayoftablesLength())
        self.assertFalse(mon2.TestarrayoftablesIsNone())

    def test_default_monster_testarrayoftables_length(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(0, self.mon.TestarrayoftablesLength())
        self.assertTrue(self.mon.TestarrayoftablesIsNone())

    def test_empty_monster_testarrayoftables_vector(self):
        if False:
            print('Hello World!')
        b = flatbuffers.Builder(0)
        _MONSTER.MonsterStartTestarrayoftablesVector(b, 0)
        testarrayoftables = b.EndVector()
        _MONSTER.MonsterStart(b)
        _MONSTER.MonsterAddTestarrayoftables(b, testarrayoftables)
        mon = _MONSTER.MonsterEnd(b)
        b.Finish(mon)
        mon2 = _MONSTER.Monster.GetRootAs(b.Bytes, b.Head())
        self.assertFalse(mon2.TestarrayoftablesIsNone())

    def test_default_monster_testarrayoftables_length(self):
        if False:
            print('Hello World!')
        self.assertEqual(0, self.mon.TestarrayoftablesLength())

    def test_nondefault_monster_enemy(self):
        if False:
            print('Hello World!')
        b = flatbuffers.Builder(0)
        _MONSTER.MonsterStart(b)
        _MONSTER.MonsterAddHp(b, 88)
        enemy = _MONSTER.MonsterEnd(b)
        b.Finish(enemy)
        _MONSTER.MonsterStart(b)
        _MONSTER.MonsterAddEnemy(b, enemy)
        mon = _MONSTER.MonsterEnd(b)
        b.Finish(mon)
        mon2 = _MONSTER.Monster.GetRootAs(b.Bytes, b.Head())
        self.assertEqual(88, mon2.Enemy().Hp())

    def test_default_monster_testnestedflatbuffer(self):
        if False:
            while True:
                i = 10
        self.assertEqual(0, self.mon.Testnestedflatbuffer(0))

    def test_default_monster_testnestedflatbuffer_length(self):
        if False:
            print('Hello World!')
        self.assertEqual(0, self.mon.TestnestedflatbufferLength())
        self.assertTrue(self.mon.TestnestedflatbufferIsNone())

    def test_empty_monster_testnestedflatbuffer_vector(self):
        if False:
            print('Hello World!')
        b = flatbuffers.Builder(0)
        _MONSTER.MonsterStartTestnestedflatbufferVector(b, 0)
        testnestedflatbuffer = b.EndVector()
        _MONSTER.MonsterStart(b)
        _MONSTER.MonsterAddTestnestedflatbuffer(b, testnestedflatbuffer)
        mon = _MONSTER.MonsterEnd(b)
        b.Finish(mon)
        mon2 = _MONSTER.Monster.GetRootAs(b.Bytes, b.Head())
        self.assertFalse(mon2.TestnestedflatbufferIsNone())

    def test_nondefault_monster_testnestedflatbuffer(self):
        if False:
            return 10
        b = flatbuffers.Builder(0)
        _MONSTER.MonsterStartTestnestedflatbufferVector(b, 3)
        b.PrependByte(4)
        b.PrependByte(2)
        b.PrependByte(0)
        sub_buf = b.EndVector()
        _MONSTER.MonsterStart(b)
        _MONSTER.MonsterAddTestnestedflatbuffer(b, sub_buf)
        mon = _MONSTER.MonsterEnd(b)
        b.Finish(mon)
        mon2 = _MONSTER.Monster.GetRootAs(b.Bytes, b.Head())
        self.assertEqual(3, mon2.TestnestedflatbufferLength())
        self.assertFalse(mon2.TestnestedflatbufferIsNone())
        self.assertEqual(0, mon2.Testnestedflatbuffer(0))
        self.assertEqual(2, mon2.Testnestedflatbuffer(1))
        self.assertEqual(4, mon2.Testnestedflatbuffer(2))
        try:
            import numpy as np
            self.assertEqual([0, 2, 4], mon2.TestnestedflatbufferAsNumpy().tolist())
        except ImportError:
            assertRaises(self, lambda : mon2.TestnestedflatbufferAsNumpy(), NumpyRequiredForThisFeature)

    def test_nested_monster_testnestedflatbuffer(self):
        if False:
            while True:
                i = 10
        b = flatbuffers.Builder(0)
        nestedB = flatbuffers.Builder(0)
        nameStr = nestedB.CreateString('Nested Monster')
        _MONSTER.MonsterStart(nestedB)
        _MONSTER.MonsterAddHp(nestedB, 30)
        _MONSTER.MonsterAddName(nestedB, nameStr)
        nestedMon = _MONSTER.MonsterEnd(nestedB)
        nestedB.Finish(nestedMon)
        sub_buf = _MONSTER.MonsterMakeTestnestedflatbufferVectorFromBytes(b, nestedB.Output())
        _MONSTER.MonsterStart(b)
        _MONSTER.MonsterAddTestnestedflatbuffer(b, sub_buf)
        mon = _MONSTER.MonsterEnd(b)
        b.Finish(mon)
        mon2 = _MONSTER.Monster.GetRootAs(b.Bytes, b.Head())
        nestedMon2 = mon2.TestnestedflatbufferNestedRoot()
        self.assertEqual(b'Nested Monster', nestedMon2.Name())
        self.assertEqual(30, nestedMon2.Hp())

    def test_nondefault_monster_testempty(self):
        if False:
            return 10
        b = flatbuffers.Builder(0)
        _STAT.StatStart(b)
        _STAT.StatAddVal(b, 123)
        my_stat = _STAT.StatEnd(b)
        b.Finish(my_stat)
        _MONSTER.MonsterStart(b)
        _MONSTER.MonsterAddTestempty(b, my_stat)
        mon = _MONSTER.MonsterEnd(b)
        b.Finish(mon)
        mon2 = _MONSTER.Monster.GetRootAs(b.Bytes, b.Head())
        self.assertEqual(123, mon2.Testempty().Val())

    def test_default_monster_testbool(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertFalse(self.mon.Testbool())

    def test_nondefault_monster_testbool(self):
        if False:
            while True:
                i = 10
        b = flatbuffers.Builder(0)
        _MONSTER.MonsterStart(b)
        _MONSTER.MonsterAddTestbool(b, True)
        mon = _MONSTER.MonsterEnd(b)
        b.Finish(mon)
        mon2 = _MONSTER.Monster.GetRootAs(b.Bytes, b.Head())
        self.assertTrue(mon2.Testbool())

    def test_default_monster_testhashes(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(0, self.mon.Testhashs32Fnv1())
        self.assertEqual(0, self.mon.Testhashu32Fnv1())
        self.assertEqual(0, self.mon.Testhashs64Fnv1())
        self.assertEqual(0, self.mon.Testhashu64Fnv1())
        self.assertEqual(0, self.mon.Testhashs32Fnv1a())
        self.assertEqual(0, self.mon.Testhashu32Fnv1a())
        self.assertEqual(0, self.mon.Testhashs64Fnv1a())
        self.assertEqual(0, self.mon.Testhashu64Fnv1a())

    def test_nondefault_monster_testhashes(self):
        if False:
            return 10
        b = flatbuffers.Builder(0)
        _MONSTER.MonsterStart(b)
        _MONSTER.MonsterAddTesthashs32Fnv1(b, 1)
        _MONSTER.MonsterAddTesthashu32Fnv1(b, 2)
        _MONSTER.MonsterAddTesthashs64Fnv1(b, 3)
        _MONSTER.MonsterAddTesthashu64Fnv1(b, 4)
        _MONSTER.MonsterAddTesthashs32Fnv1a(b, 5)
        _MONSTER.MonsterAddTesthashu32Fnv1a(b, 6)
        _MONSTER.MonsterAddTesthashs64Fnv1a(b, 7)
        _MONSTER.MonsterAddTesthashu64Fnv1a(b, 8)
        mon = _MONSTER.MonsterEnd(b)
        b.Finish(mon)
        mon2 = _MONSTER.Monster.GetRootAs(b.Bytes, b.Head())
        self.assertEqual(1, mon2.Testhashs32Fnv1())
        self.assertEqual(2, mon2.Testhashu32Fnv1())
        self.assertEqual(3, mon2.Testhashs64Fnv1())
        self.assertEqual(4, mon2.Testhashu64Fnv1())
        self.assertEqual(5, mon2.Testhashs32Fnv1a())
        self.assertEqual(6, mon2.Testhashu32Fnv1a())
        self.assertEqual(7, mon2.Testhashs64Fnv1a())
        self.assertEqual(8, mon2.Testhashu64Fnv1a())

    def test_default_monster_parent_namespace_test(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(None, self.mon.ParentNamespaceTest())

    def test_nondefault_monster_parent_namespace_test(self):
        if False:
            print('Hello World!')
        b = flatbuffers.Builder(0)
        _IN_PARENT_NAMESPACE.InParentNamespaceStart(b)
        parent = _IN_PARENT_NAMESPACE.InParentNamespaceEnd(b)
        _MONSTER.MonsterStart(b)
        _MONSTER.MonsterAddParentNamespaceTest(b, parent)
        mon = _MONSTER.MonsterEnd(b)
        b.Finish(mon)
        monster = _MONSTER.Monster.GetRootAs(b.Bytes, b.Head())
        self.assertTrue(isinstance(monster.ParentNamespaceTest(), _IN_PARENT_NAMESPACE.InParentNamespace))

    def test_getrootas_for_nonroot_table(self):
        if False:
            print('Hello World!')
        b = flatbuffers.Builder(0)
        string = b.CreateString('MyStat')
        _STAT.StatStart(b)
        _STAT.StatAddId(b, string)
        _STAT.StatAddVal(b, 12345678)
        _STAT.StatAddCount(b, 12345)
        stat = _STAT.StatEnd(b)
        b.Finish(stat)
        stat2 = _STAT.Stat.GetRootAs(b.Bytes, b.Head())
        self.assertEqual(b'MyStat', stat2.Id())
        self.assertEqual(12345678, stat2.Val())
        self.assertEqual(12345, stat2.Count())

class TestAllCodePathsOfMonsterExtraSchema(unittest.TestCase):

    def setUp(self, *args, **kwargs):
        if False:
            return 10
        super(TestAllCodePathsOfMonsterExtraSchema, self).setUp(*args, **kwargs)
        b = flatbuffers.Builder(0)
        MyGame.MonsterExtra.Start(b)
        gen_mon = MyGame.MonsterExtra.End(b)
        b.Finish(gen_mon)
        self.mon = MyGame.MonsterExtra.MonsterExtra.GetRootAs(b.Bytes, b.Head())

    def test_default_nan_inf(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(math.isnan(self.mon.F1()))
        self.assertEqual(self.mon.F2(), float('inf'))
        self.assertEqual(self.mon.F3(), float('-inf'))
        self.assertTrue(math.isnan(self.mon.D1()))
        self.assertEqual(self.mon.D2(), float('inf'))
        self.assertEqual(self.mon.D3(), float('-inf'))

class TestVtableDeduplication(unittest.TestCase):
    """ TestVtableDeduplication verifies that vtables are deduplicated. """

    def test_vtable_deduplication(self):
        if False:
            for i in range(10):
                print('nop')
        b = flatbuffers.Builder(0)
        b.StartObject(4)
        b.PrependByteSlot(0, 0, 0)
        b.PrependByteSlot(1, 11, 0)
        b.PrependByteSlot(2, 22, 0)
        b.PrependInt16Slot(3, 33, 0)
        obj0 = b.EndObject()
        b.StartObject(4)
        b.PrependByteSlot(0, 0, 0)
        b.PrependByteSlot(1, 44, 0)
        b.PrependByteSlot(2, 55, 0)
        b.PrependInt16Slot(3, 66, 0)
        obj1 = b.EndObject()
        b.StartObject(4)
        b.PrependByteSlot(0, 0, 0)
        b.PrependByteSlot(1, 77, 0)
        b.PrependByteSlot(2, 88, 0)
        b.PrependInt16Slot(3, 99, 0)
        obj2 = b.EndObject()
        got = b.Bytes[b.Head():]
        want = bytearray([240, 255, 255, 255, 99, 0, 88, 77, 248, 255, 255, 255, 66, 0, 55, 44, 12, 0, 8, 0, 0, 0, 7, 0, 6, 0, 4, 0, 12, 0, 0, 0, 33, 0, 22, 11])
        self.assertEqual((len(want), want), (len(got), got))
        table0 = flatbuffers.table.Table(b.Bytes, len(b.Bytes) - obj0)
        table1 = flatbuffers.table.Table(b.Bytes, len(b.Bytes) - obj1)
        table2 = flatbuffers.table.Table(b.Bytes, len(b.Bytes) - obj2)

        def _checkTable(tab, voffsett_value, b, c, d):
            if False:
                print('Hello World!')
            got = tab.GetVOffsetTSlot(0, 0)
            self.assertEqual(12, got, 'case 0, 0')
            got = tab.GetVOffsetTSlot(2, 0)
            self.assertEqual(8, got, 'case 2, 0')
            got = tab.GetVOffsetTSlot(4, 0)
            self.assertEqual(voffsett_value, got, 'case 4, 0')
            got = tab.GetSlot(6, 0, N.Uint8Flags)
            self.assertEqual(b, got, 'case 6, 0')
            val = tab.GetSlot(8, 0, N.Uint8Flags)
            self.assertEqual(c, val, 'failed 8, 0')
            got = tab.GetSlot(10, 0, N.Uint8Flags)
            self.assertEqual(d, got, 'failed 10, 0')
        _checkTable(table0, 0, 11, 22, 33)
        _checkTable(table1, 0, 44, 55, 66)
        _checkTable(table2, 0, 77, 88, 99)

class TestExceptions(unittest.TestCase):

    def test_object_is_nested_error(self):
        if False:
            return 10
        b = flatbuffers.Builder(0)
        b.StartObject(0)
        assertRaises(self, lambda : b.StartObject(0), flatbuffers.builder.IsNestedError)

    def test_object_is_not_nested_error(self):
        if False:
            return 10
        b = flatbuffers.Builder(0)
        assertRaises(self, lambda : b.EndObject(), flatbuffers.builder.IsNotNestedError)

    def test_struct_is_not_inline_error(self):
        if False:
            print('Hello World!')
        b = flatbuffers.Builder(0)
        b.StartObject(0)
        assertRaises(self, lambda : b.PrependStructSlot(0, 1, 0), flatbuffers.builder.StructIsNotInlineError)

    def test_unreachable_error(self):
        if False:
            i = 10
            return i + 15
        b = flatbuffers.Builder(0)
        assertRaises(self, lambda : b.PrependUOffsetTRelative(1), flatbuffers.builder.OffsetArithmeticError)

    def test_create_shared_string_is_nested_error(self):
        if False:
            for i in range(10):
                print('nop')
        b = flatbuffers.Builder(0)
        b.StartObject(0)
        s = 'test1'
        assertRaises(self, lambda : b.CreateSharedString(s), flatbuffers.builder.IsNestedError)

    def test_create_string_is_nested_error(self):
        if False:
            print('Hello World!')
        b = flatbuffers.Builder(0)
        b.StartObject(0)
        s = 'test1'
        assertRaises(self, lambda : b.CreateString(s), flatbuffers.builder.IsNestedError)

    def test_create_byte_vector_is_nested_error(self):
        if False:
            i = 10
            return i + 15
        b = flatbuffers.Builder(0)
        b.StartObject(0)
        s = b'test1'
        assertRaises(self, lambda : b.CreateByteVector(s), flatbuffers.builder.IsNestedError)

    def test_finished_bytes_error(self):
        if False:
            while True:
                i = 10
        b = flatbuffers.Builder(0)
        assertRaises(self, lambda : b.Output(), flatbuffers.builder.BuilderNotFinishedError)

class TestFixedLengthArrays(unittest.TestCase):

    def test_fixed_length_array(self):
        if False:
            print('Hello World!')
        builder = flatbuffers.Builder(0)
        a = 0.5
        b = range(0, 15)
        c = 1
        d_a = [[1, 2], [3, 4]]
        d_b = [MyGame.Example.TestEnum.TestEnum.B, MyGame.Example.TestEnum.TestEnum.C]
        d_c = [[MyGame.Example.TestEnum.TestEnum.A, MyGame.Example.TestEnum.TestEnum.B], [MyGame.Example.TestEnum.TestEnum.C, MyGame.Example.TestEnum.TestEnum.B]]
        d_d = [[-1, 1], [-2, 2]]
        e = 2
        f = [-1, 1]
        arrayOffset = MyGame.Example.ArrayStruct.CreateArrayStruct(builder, a, b, c, d_a, d_b, d_c, d_d, e, f)
        MyGame.Example.ArrayTable.Start(builder)
        MyGame.Example.ArrayTable.AddA(builder, arrayOffset)
        tableOffset = MyGame.Example.ArrayTable.End(builder)
        builder.Finish(tableOffset)
        buf = builder.Output()
        table = MyGame.Example.ArrayTable.ArrayTable.GetRootAs(buf)
        nested = MyGame.Example.NestedStruct.NestedStruct()
        self.assertEqual(table.A().A(), 0.5)
        self.assertEqual(table.A().B(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        self.assertEqual(table.A().C(), 1)
        self.assertEqual(table.A().D(0).A(), [1, 2])
        self.assertEqual(table.A().D(1).A(), [3, 4])
        self.assertEqual(table.A().D(0).B(), MyGame.Example.TestEnum.TestEnum.B)
        self.assertEqual(table.A().D(1).B(), MyGame.Example.TestEnum.TestEnum.C)
        self.assertEqual(table.A().D(0).C(), [MyGame.Example.TestEnum.TestEnum.A, MyGame.Example.TestEnum.TestEnum.B])
        self.assertEqual(table.A().D(1).C(), [MyGame.Example.TestEnum.TestEnum.C, MyGame.Example.TestEnum.TestEnum.B])
        self.assertEqual(table.A().D(0).D(), [-1, 1])
        self.assertEqual(table.A().D(1).D(), [-2, 2])
        self.assertEqual(table.A().E(), 2)
        self.assertEqual(table.A().F(), [-1, 1])
        self.assertEqual(table.A().D(0).D(0), -1)
        self.assertEqual(table.A().D(0).D(1), 1)
        self.assertEqual(table.A().D(1).D(0), -2)
        self.assertEqual(table.A().D(1).D(1), 2)

class TestNestedUnionTables(unittest.TestCase):

    def test_nested_union_tables(self):
        if False:
            for i in range(10):
                print('nop')
        nestUnion = MyGame.Example.NestedUnion.NestedUnionTest.NestedUnionTestT()
        nestUnion.name = b'testUnion1'
        nestUnion.id = 1
        nestUnion.data = MyGame.Example.NestedUnion.Vec3.Vec3T()
        nestUnion.dataType = MyGame.Example.NestedUnion.Any.Any.Vec3
        nestUnion.data.x = 4.278975356
        nestUnion.data.y = 5.32
        nestUnion.data.z = -6.464
        nestUnion.data.test1 = 0.9
        nestUnion.data.test2 = MyGame.Example.NestedUnion.Color.Color.Red
        nestUnion.data.test3 = MyGame.Example.NestedUnion.Test.TestT()
        nestUnion.data.test3.a = 5
        nestUnion.data.test3.b = 2
        b = flatbuffers.Builder(0)
        b.Finish(nestUnion.Pack(b))
        nestUnionDecode = MyGame.Example.NestedUnion.NestedUnionTest.NestedUnionTest.GetRootAs(b.Bytes, b.Head())
        nestUnionDecodeT = MyGame.Example.NestedUnion.NestedUnionTest.NestedUnionTestT.InitFromObj(nestUnionDecode)
        self.assertEqual(nestUnionDecodeT.name, nestUnion.name)
        self.assertEqual(nestUnionDecodeT.id, nestUnion.id)
        self.assertEqual(nestUnionDecodeT.dataType, nestUnion.dataType)
        self.assertEqual(nestUnionDecodeT.data.x, nestUnion.data.x)
        self.assertEqual(nestUnionDecodeT.data.y, nestUnion.data.y)
        self.assertEqual(nestUnionDecodeT.data.z, nestUnion.data.z)
        self.assertEqual(nestUnionDecodeT.data.test1, nestUnion.data.test1)
        self.assertEqual(nestUnionDecodeT.data.test2, nestUnion.data.test2)
        self.assertEqual(nestUnionDecodeT.data.test3.a, nestUnion.data.test3.a)
        self.assertEqual(nestUnionDecodeT.data.test3.b, nestUnion.data.test3.b)
        nestUnionDecodeTFromBuf = MyGame.Example.NestedUnion.NestedUnionTest.NestedUnionTestT.InitFromPackedBuf(b.Bytes, b.Head())
        self.assertEqual(nestUnionDecodeTFromBuf.name, nestUnion.name)
        self.assertEqual(nestUnionDecodeTFromBuf.id, nestUnion.id)
        self.assertEqual(nestUnionDecodeTFromBuf.dataType, nestUnion.dataType)
        self.assertEqual(nestUnionDecodeTFromBuf.data.x, nestUnion.data.x)
        self.assertEqual(nestUnionDecodeTFromBuf.data.y, nestUnion.data.y)
        self.assertEqual(nestUnionDecodeTFromBuf.data.z, nestUnion.data.z)
        self.assertEqual(nestUnionDecodeTFromBuf.data.test1, nestUnion.data.test1)
        self.assertEqual(nestUnionDecodeTFromBuf.data.test2, nestUnion.data.test2)
        self.assertEqual(nestUnionDecodeTFromBuf.data.test3.a, nestUnion.data.test3.a)
        self.assertEqual(nestUnionDecodeTFromBuf.data.test3.b, nestUnion.data.test3.b)
        nestUnionDecodeTFromBuf2 = MyGame.Example.NestedUnion.NestedUnionTest.NestedUnionTestT.InitFromPackedBuf(b.Output())
        self.assertEqual(nestUnionDecodeTFromBuf2.name, nestUnion.name)
        self.assertEqual(nestUnionDecodeTFromBuf2.id, nestUnion.id)
        self.assertEqual(nestUnionDecodeTFromBuf2.dataType, nestUnion.dataType)
        self.assertEqual(nestUnionDecodeTFromBuf2.data.x, nestUnion.data.x)
        self.assertEqual(nestUnionDecodeTFromBuf2.data.y, nestUnion.data.y)
        self.assertEqual(nestUnionDecodeTFromBuf2.data.z, nestUnion.data.z)
        self.assertEqual(nestUnionDecodeTFromBuf2.data.test1, nestUnion.data.test1)
        self.assertEqual(nestUnionDecodeTFromBuf2.data.test2, nestUnion.data.test2)
        self.assertEqual(nestUnionDecodeTFromBuf2.data.test3.a, nestUnion.data.test3.a)
        self.assertEqual(nestUnionDecodeTFromBuf2.data.test3.b, nestUnion.data.test3.b)

def CheckAgainstGoldDataGo():
    if False:
        i = 10
        return i + 15
    try:
        (gen_buf, gen_off) = make_monster_from_generated_code()
        fn = 'monsterdata_go_wire.mon'
        if not os.path.exists(fn):
            print('Go-generated data does not exist, failed.')
            return False
        f = open(fn, 'rb')
        go_wire_data = f.read()
        f.close()
        CheckReadBuffer(bytearray(go_wire_data), 0)
        if not bytearray(gen_buf[gen_off:]) == bytearray(go_wire_data):
            raise AssertionError('CheckAgainstGoldDataGo failed')
    except:
        print('Failed to test against Go-generated test data.')
        return False
    print('Can read Go-generated test data, and Python generates bytewise identical data.')
    return True

def CheckAgainstGoldDataJava():
    if False:
        while True:
            i = 10
    try:
        (gen_buf, gen_off) = make_monster_from_generated_code()
        fn = 'monsterdata_java_wire.mon'
        if not os.path.exists(fn):
            print('Java-generated data does not exist, failed.')
            return False
        f = open(fn, 'rb')
        java_wire_data = f.read()
        f.close()
        CheckReadBuffer(bytearray(java_wire_data), 0)
    except:
        print('Failed to read Java-generated test data.')
        return False
    print('Can read Java-generated test data.')
    return True

class LCG(object):
    """ Include simple random number generator to ensure results will be the

        same cross platform.
        http://en.wikipedia.org/wiki/Park%E2%80%93Miller_random_number_generator
        """
    __slots__ = ['n']
    InitialLCGSeed = 48271

    def __init__(self):
        if False:
            return 10
        self.n = self.InitialLCGSeed

    def Reset(self):
        if False:
            print('Hello World!')
        self.n = self.InitialLCGSeed

    def Next(self):
        if False:
            return 10
        self.n = self.n * 279470273 % 4294967291 & 4294967295
        return self.n

def BenchmarkVtableDeduplication(count):
    if False:
        for i in range(10):
            print('nop')
    '\n    BenchmarkVtableDeduplication measures the speed of vtable deduplication\n    by creating `prePop` vtables, then populating `count` objects with a\n    different single vtable.\n\n    When count is large (as in long benchmarks), memory usage may be high.\n    '
    for prePop in (1, 10, 100, 1000):
        builder = flatbuffers.Builder(0)
        n = 1 + int(math.log(prePop, 1.5))
        layouts = set()
        r = list(compat_range(n))
        while len(layouts) < prePop:
            layouts.add(tuple(sorted(random.sample(r, int(max(1, n / 2))))))
        layouts = list(layouts)
        for layout in layouts:
            builder.StartObject(n)
            for j in layout:
                builder.PrependInt16Slot(j, j, 0)
            builder.EndObject()

        def f():
            if False:
                i = 10
                return i + 15
            layout = random.choice(layouts)
            builder.StartObject(n)
            for j in layout:
                builder.PrependInt16Slot(j, j, 0)
            builder.EndObject()
        duration = timeit.timeit(stmt=f, number=count)
        rate = float(count) / duration
        print('vtable deduplication rate (n=%d, vtables=%d): %.2f sec' % (prePop, len(builder.vtables), rate))

def BenchmarkCheckReadBuffer(count, buf, off):
    if False:
        while True:
            i = 10
    '\n    BenchmarkCheckReadBuffer measures the speed of flatbuffer reading\n    by re-using the CheckReadBuffer function with the gold data.\n    '

    def f():
        if False:
            for i in range(10):
                print('nop')
        CheckReadBuffer(buf, off)
    duration = timeit.timeit(stmt=f, number=count)
    rate = float(count) / duration
    data = float(len(buf) * count) / float(1024 * 1024)
    data_rate = data / float(duration)
    print('traversed %d %d-byte flatbuffers in %.2fsec: %.2f/sec, %.2fMB/sec' % (count, len(buf), duration, rate, data_rate))

def BenchmarkMakeMonsterFromGeneratedCode(count, length):
    if False:
        return 10
    '\n    BenchmarkMakeMonsterFromGeneratedCode measures the speed of flatbuffer\n    creation by re-using the make_monster_from_generated_code function for\n    generating gold data examples.\n    '
    duration = timeit.timeit(stmt=make_monster_from_generated_code, number=count)
    rate = float(count) / duration
    data = float(length * count) / float(1024 * 1024)
    data_rate = data / float(duration)
    print('built %d %d-byte flatbuffers in %.2fsec: %.2f/sec, %.2fMB/sec' % (count, length, duration, rate, data_rate))

def backward_compatible_run_tests(**kwargs):
    if False:
        while True:
            i = 10
    if PY_VERSION < (2, 6):
        sys.stderr.write('Python version less than 2.6 are not supported')
        sys.stderr.flush()
        return False
    if PY_VERSION == (2, 6):
        try:
            unittest.main(**kwargs)
        except SystemExit as e:
            if not e.code == 0:
                return False
        return True
    kwargs['exit'] = False
    kwargs['verbosity'] = 0
    ret = unittest.main(**kwargs)
    if ret.result.errors or ret.result.failures:
        return False
    return True

def main():
    if False:
        while True:
            i = 10
    import os
    import sys
    if not len(sys.argv) == 5:
        sys.stderr.write('Usage: %s <benchmark vtable count> <benchmark read count> <benchmark build count> <is_onefile>\n' % sys.argv[0])
        sys.stderr.write('       Provide COMPARE_GENERATED_TO_GO=1   to checkfor bytewise comparison to Go data.\n')
        sys.stderr.write('       Provide COMPARE_GENERATED_TO_JAVA=1 to checkfor bytewise comparison to Java data.\n')
        sys.stderr.flush()
        sys.exit(1)
    kwargs = dict(argv=sys.argv[:-4])
    create_namespace_shortcut(sys.argv[4].lower() == 'true')
    try:
        import numpy
        print('numpy available')
    except ImportError:
        print('numpy not available')
    success = backward_compatible_run_tests(**kwargs)
    if success and os.environ.get('COMPARE_GENERATED_TO_GO', 0) == '1':
        success = success and CheckAgainstGoldDataGo()
    if success and os.environ.get('COMPARE_GENERATED_TO_JAVA', 0) == '1':
        success = success and CheckAgainstGoldDataJava()
    if not success:
        sys.stderr.write('Tests failed, skipping benchmarks.\n')
        sys.stderr.flush()
        sys.exit(1)
    bench_vtable = int(sys.argv[1])
    bench_traverse = int(sys.argv[2])
    bench_build = int(sys.argv[3])
    if bench_vtable:
        BenchmarkVtableDeduplication(bench_vtable)
    if bench_traverse:
        (buf, off) = make_monster_from_generated_code()
        BenchmarkCheckReadBuffer(bench_traverse, buf, off)
    if bench_build:
        (buf, off) = make_monster_from_generated_code()
        BenchmarkMakeMonsterFromGeneratedCode(bench_build, len(buf))
if __name__ == '__main__':
    main()