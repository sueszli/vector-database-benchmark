import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class StructOfStructs(object):
    __slots__ = ['_tab']

    @classmethod
    def SizeOf(cls):
        if False:
            while True:
                i = 10
        return 20

    def Init(self, buf, pos):
        if False:
            while True:
                i = 10
        self._tab = flatbuffers.table.Table(buf, pos)

    def A(self, obj):
        if False:
            i = 10
            return i + 15
        obj.Init(self._tab.Bytes, self._tab.Pos + 0)
        return obj

    def B(self, obj):
        if False:
            i = 10
            return i + 15
        obj.Init(self._tab.Bytes, self._tab.Pos + 8)
        return obj

    def C(self, obj):
        if False:
            i = 10
            return i + 15
        obj.Init(self._tab.Bytes, self._tab.Pos + 12)
        return obj

def CreateStructOfStructs(builder, a_id, a_distance, b_a, b_b, c_id, c_distance):
    if False:
        i = 10
        return i + 15
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
import MyGame.Example.Ability
import MyGame.Example.Test
try:
    from typing import Optional
except:
    pass

class StructOfStructsT(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.a = None
        self.b = None
        self.c = None

    @classmethod
    def InitFromBuf(cls, buf, pos):
        if False:
            print('Hello World!')
        structOfStructs = StructOfStructs()
        structOfStructs.Init(buf, pos)
        return cls.InitFromObj(structOfStructs)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        if False:
            while True:
                i = 10
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, structOfStructs):
        if False:
            while True:
                i = 10
        x = StructOfStructsT()
        x._UnPack(structOfStructs)
        return x

    def _UnPack(self, structOfStructs):
        if False:
            print('Hello World!')
        if structOfStructs is None:
            return
        if structOfStructs.A(MyGame.Example.Ability.Ability()) is not None:
            self.a = MyGame.Example.Ability.AbilityT.InitFromObj(structOfStructs.A(MyGame.Example.Ability.Ability()))
        if structOfStructs.B(MyGame.Example.Test.Test()) is not None:
            self.b = MyGame.Example.Test.TestT.InitFromObj(structOfStructs.B(MyGame.Example.Test.Test()))
        if structOfStructs.C(MyGame.Example.Ability.Ability()) is not None:
            self.c = MyGame.Example.Ability.AbilityT.InitFromObj(structOfStructs.C(MyGame.Example.Ability.Ability()))

    def Pack(self, builder):
        if False:
            for i in range(10):
                print('nop')
        return CreateStructOfStructs(builder, self.a.id, self.a.distance, self.b.a, self.b.b, self.c.id, self.c.distance)