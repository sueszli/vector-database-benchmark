import flatbuffers
from flatbuffers.compat import import_numpy
from typing import Any
from MyGame.Example.NestedUnion.Test import Test
from typing import Optional
np = import_numpy()

class Vec3(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset: int=0):
        if False:
            while True:
                i = 10
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Vec3()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsVec3(cls, buf, offset=0):
        if False:
            return 10
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    def Init(self, buf: bytes, pos: int):
        if False:
            print('Hello World!')
        self._tab = flatbuffers.table.Table(buf, pos)

    def X(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float64Flags, o + self._tab.Pos)
        return 0.0

    def Y(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float64Flags, o + self._tab.Pos)
        return 0.0

    def Z(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float64Flags, o + self._tab.Pos)
        return 0.0

    def Test1(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float64Flags, o + self._tab.Pos)
        return 0.0

    def Test2(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    def Test3(self) -> Optional[Test]:
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            x = o + self._tab.Pos
            obj = Test()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def Vec3Start(builder: flatbuffers.Builder):
    if False:
        for i in range(10):
            print('nop')
    builder.StartObject(6)

def Start(builder: flatbuffers.Builder):
    if False:
        for i in range(10):
            print('nop')
    Vec3Start(builder)

def Vec3AddX(builder: flatbuffers.Builder, x: float):
    if False:
        print('Hello World!')
    builder.PrependFloat64Slot(0, x, 0.0)

def AddX(builder: flatbuffers.Builder, x: float):
    if False:
        for i in range(10):
            print('nop')
    Vec3AddX(builder, x)

def Vec3AddY(builder: flatbuffers.Builder, y: float):
    if False:
        i = 10
        return i + 15
    builder.PrependFloat64Slot(1, y, 0.0)

def AddY(builder: flatbuffers.Builder, y: float):
    if False:
        i = 10
        return i + 15
    Vec3AddY(builder, y)

def Vec3AddZ(builder: flatbuffers.Builder, z: float):
    if False:
        i = 10
        return i + 15
    builder.PrependFloat64Slot(2, z, 0.0)

def AddZ(builder: flatbuffers.Builder, z: float):
    if False:
        print('Hello World!')
    Vec3AddZ(builder, z)

def Vec3AddTest1(builder: flatbuffers.Builder, test1: float):
    if False:
        i = 10
        return i + 15
    builder.PrependFloat64Slot(3, test1, 0.0)

def AddTest1(builder: flatbuffers.Builder, test1: float):
    if False:
        while True:
            i = 10
    Vec3AddTest1(builder, test1)

def Vec3AddTest2(builder: flatbuffers.Builder, test2: int):
    if False:
        print('Hello World!')
    builder.PrependUint8Slot(4, test2, 0)

def AddTest2(builder: flatbuffers.Builder, test2: int):
    if False:
        for i in range(10):
            print('nop')
    Vec3AddTest2(builder, test2)

def Vec3AddTest3(builder: flatbuffers.Builder, test3: Any):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependStructSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(test3), 0)

def AddTest3(builder: flatbuffers.Builder, test3: Any):
    if False:
        i = 10
        return i + 15
    Vec3AddTest3(builder, test3)

def Vec3End(builder: flatbuffers.Builder) -> int:
    if False:
        print('Hello World!')
    return builder.EndObject()

def End(builder: flatbuffers.Builder) -> int:
    if False:
        for i in range(10):
            print('nop')
    return Vec3End(builder)
import MyGame.Example.NestedUnion.Test
try:
    from typing import Optional
except:
    pass

class Vec3T(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.test1 = 0.0
        self.test2 = 0
        self.test3 = None

    @classmethod
    def InitFromBuf(cls, buf, pos):
        if False:
            i = 10
            return i + 15
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
            for i in range(10):
                print('nop')
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
        if vec3.Test3() is not None:
            self.test3 = MyGame.Example.NestedUnion.Test.TestT.InitFromObj(vec3.Test3())

    def Pack(self, builder):
        if False:
            while True:
                i = 10
        Vec3Start(builder)
        Vec3AddX(builder, self.x)
        Vec3AddY(builder, self.y)
        Vec3AddZ(builder, self.z)
        Vec3AddTest1(builder, self.test1)
        Vec3AddTest2(builder, self.test2)
        if self.test3 is not None:
            test3 = self.test3.Pack(builder)
            Vec3AddTest3(builder, test3)
        vec3 = Vec3End(builder)
        return vec3