import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class StructOfStructsOfStructs(object):
    __slots__ = ['_tab']

    @classmethod
    def SizeOf(cls):
        if False:
            return 10
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
        i = 10
        return i + 15
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
import MyGame.Example.StructOfStructs
try:
    from typing import Optional
except:
    pass

class StructOfStructsOfStructsT(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.a = None

    @classmethod
    def InitFromBuf(cls, buf, pos):
        if False:
            for i in range(10):
                print('nop')
        structOfStructsOfStructs = StructOfStructsOfStructs()
        structOfStructsOfStructs.Init(buf, pos)
        return cls.InitFromObj(structOfStructsOfStructs)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        if False:
            print('Hello World!')
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, structOfStructsOfStructs):
        if False:
            for i in range(10):
                print('nop')
        x = StructOfStructsOfStructsT()
        x._UnPack(structOfStructsOfStructs)
        return x

    def _UnPack(self, structOfStructsOfStructs):
        if False:
            while True:
                i = 10
        if structOfStructsOfStructs is None:
            return
        if structOfStructsOfStructs.A(MyGame.Example.StructOfStructs.StructOfStructs()) is not None:
            self.a = MyGame.Example.StructOfStructs.StructOfStructsT.InitFromObj(structOfStructsOfStructs.A(MyGame.Example.StructOfStructs.StructOfStructs()))

    def Pack(self, builder):
        if False:
            for i in range(10):
                print('nop')
        return CreateStructOfStructsOfStructs(builder, self.a.a.id, self.a.a.distance, self.a.b.a, self.a.b.b, self.a.c.id, self.a.c.distance)