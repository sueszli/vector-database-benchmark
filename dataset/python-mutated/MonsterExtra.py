import flatbuffers
from flatbuffers.compat import import_numpy
from typing import Any
np = import_numpy()

class MonsterExtra(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset: int=0):
        if False:
            for i in range(10):
                print('nop')
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = MonsterExtra()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsMonsterExtra(cls, buf, offset=0):
        if False:
            print('Hello World!')
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    @classmethod
    def MonsterExtraBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        if False:
            for i in range(10):
                print('nop')
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b'MONE', size_prefixed=size_prefixed)

    def Init(self, buf: bytes, pos: int):
        if False:
            i = 10
            return i + 15
        self._tab = flatbuffers.table.Table(buf, pos)

    def D0(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float64Flags, o + self._tab.Pos)
        return float('nan')

    def D1(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float64Flags, o + self._tab.Pos)
        return float('nan')

    def D2(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float64Flags, o + self._tab.Pos)
        return float('inf')

    def D3(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float64Flags, o + self._tab.Pos)
        return float('-inf')

    def F0(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return float('nan')

    def F1(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return float('nan')

    def F2(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return float('inf')

    def F3(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return float('-inf')

    def Dvec(self, j: int):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Float64Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 8))
        return 0

    def DvecAsNumpy(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Float64Flags, o)
        return 0

    def DvecLength(self) -> int:
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def DvecIsNone(self) -> bool:
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        return o == 0

    def Fvec(self, j: int):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Float32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    def FvecAsNumpy(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Float32Flags, o)
        return 0

    def FvecLength(self) -> int:
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def FvecIsNone(self) -> bool:
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        return o == 0

def MonsterExtraStart(builder: flatbuffers.Builder):
    if False:
        while True:
            i = 10
    builder.StartObject(11)

def Start(builder: flatbuffers.Builder):
    if False:
        i = 10
        return i + 15
    MonsterExtraStart(builder)

def MonsterExtraAddD0(builder: flatbuffers.Builder, d0: float):
    if False:
        return 10
    builder.PrependFloat64Slot(0, d0, float('nan'))

def AddD0(builder: flatbuffers.Builder, d0: float):
    if False:
        for i in range(10):
            print('nop')
    MonsterExtraAddD0(builder, d0)

def MonsterExtraAddD1(builder: flatbuffers.Builder, d1: float):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependFloat64Slot(1, d1, float('nan'))

def AddD1(builder: flatbuffers.Builder, d1: float):
    if False:
        return 10
    MonsterExtraAddD1(builder, d1)

def MonsterExtraAddD2(builder: flatbuffers.Builder, d2: float):
    if False:
        while True:
            i = 10
    builder.PrependFloat64Slot(2, d2, float('inf'))

def AddD2(builder: flatbuffers.Builder, d2: float):
    if False:
        while True:
            i = 10
    MonsterExtraAddD2(builder, d2)

def MonsterExtraAddD3(builder: flatbuffers.Builder, d3: float):
    if False:
        while True:
            i = 10
    builder.PrependFloat64Slot(3, d3, float('-inf'))

def AddD3(builder: flatbuffers.Builder, d3: float):
    if False:
        i = 10
        return i + 15
    MonsterExtraAddD3(builder, d3)

def MonsterExtraAddF0(builder: flatbuffers.Builder, f0: float):
    if False:
        i = 10
        return i + 15
    builder.PrependFloat32Slot(4, f0, float('nan'))

def AddF0(builder: flatbuffers.Builder, f0: float):
    if False:
        while True:
            i = 10
    MonsterExtraAddF0(builder, f0)

def MonsterExtraAddF1(builder: flatbuffers.Builder, f1: float):
    if False:
        print('Hello World!')
    builder.PrependFloat32Slot(5, f1, float('nan'))

def AddF1(builder: flatbuffers.Builder, f1: float):
    if False:
        return 10
    MonsterExtraAddF1(builder, f1)

def MonsterExtraAddF2(builder: flatbuffers.Builder, f2: float):
    if False:
        print('Hello World!')
    builder.PrependFloat32Slot(6, f2, float('inf'))

def AddF2(builder: flatbuffers.Builder, f2: float):
    if False:
        while True:
            i = 10
    MonsterExtraAddF2(builder, f2)

def MonsterExtraAddF3(builder: flatbuffers.Builder, f3: float):
    if False:
        while True:
            i = 10
    builder.PrependFloat32Slot(7, f3, float('-inf'))

def AddF3(builder: flatbuffers.Builder, f3: float):
    if False:
        for i in range(10):
            print('nop')
    MonsterExtraAddF3(builder, f3)

def MonsterExtraAddDvec(builder: flatbuffers.Builder, dvec: int):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependUOffsetTRelativeSlot(8, flatbuffers.number_types.UOffsetTFlags.py_type(dvec), 0)

def AddDvec(builder: flatbuffers.Builder, dvec: int):
    if False:
        for i in range(10):
            print('nop')
    MonsterExtraAddDvec(builder, dvec)

def MonsterExtraStartDvecVector(builder, numElems: int) -> int:
    if False:
        return 10
    return builder.StartVector(8, numElems, 8)

def StartDvecVector(builder, numElems: int) -> int:
    if False:
        i = 10
        return i + 15
    return MonsterExtraStartDvecVector(builder, numElems)

def MonsterExtraAddFvec(builder: flatbuffers.Builder, fvec: int):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependUOffsetTRelativeSlot(9, flatbuffers.number_types.UOffsetTFlags.py_type(fvec), 0)

def AddFvec(builder: flatbuffers.Builder, fvec: int):
    if False:
        i = 10
        return i + 15
    MonsterExtraAddFvec(builder, fvec)

def MonsterExtraStartFvecVector(builder, numElems: int) -> int:
    if False:
        return 10
    return builder.StartVector(4, numElems, 4)

def StartFvecVector(builder, numElems: int) -> int:
    if False:
        for i in range(10):
            print('nop')
    return MonsterExtraStartFvecVector(builder, numElems)

def MonsterExtraEnd(builder: flatbuffers.Builder) -> int:
    if False:
        i = 10
        return i + 15
    return builder.EndObject()

def End(builder: flatbuffers.Builder) -> int:
    if False:
        print('Hello World!')
    return MonsterExtraEnd(builder)
try:
    from typing import List
except:
    pass

class MonsterExtraT(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.d0 = float('nan')
        self.d1 = float('nan')
        self.d2 = float('inf')
        self.d3 = float('-inf')
        self.f0 = float('nan')
        self.f1 = float('nan')
        self.f2 = float('inf')
        self.f3 = float('-inf')
        self.dvec = None
        self.fvec = None

    @classmethod
    def InitFromBuf(cls, buf, pos):
        if False:
            for i in range(10):
                print('nop')
        monsterExtra = MonsterExtra()
        monsterExtra.Init(buf, pos)
        return cls.InitFromObj(monsterExtra)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        if False:
            print('Hello World!')
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, monsterExtra):
        if False:
            return 10
        x = MonsterExtraT()
        x._UnPack(monsterExtra)
        return x

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return type(self) == type(other) and self.d0 == other.d0 and (self.d1 == other.d1) and (self.d2 == other.d2) and (self.d3 == other.d3) and (self.f0 == other.f0) and (self.f1 == other.f1) and (self.f2 == other.f2) and (self.f3 == other.f3) and (self.dvec == other.dvec) and (self.fvec == other.fvec)

    def _UnPack(self, monsterExtra):
        if False:
            i = 10
            return i + 15
        if monsterExtra is None:
            return
        self.d0 = monsterExtra.D0()
        self.d1 = monsterExtra.D1()
        self.d2 = monsterExtra.D2()
        self.d3 = monsterExtra.D3()
        self.f0 = monsterExtra.F0()
        self.f1 = monsterExtra.F1()
        self.f2 = monsterExtra.F2()
        self.f3 = monsterExtra.F3()
        if not monsterExtra.DvecIsNone():
            if np is None:
                self.dvec = []
                for i in range(monsterExtra.DvecLength()):
                    self.dvec.append(monsterExtra.Dvec(i))
            else:
                self.dvec = monsterExtra.DvecAsNumpy()
        if not monsterExtra.FvecIsNone():
            if np is None:
                self.fvec = []
                for i in range(monsterExtra.FvecLength()):
                    self.fvec.append(monsterExtra.Fvec(i))
            else:
                self.fvec = monsterExtra.FvecAsNumpy()

    def Pack(self, builder):
        if False:
            print('Hello World!')
        if self.dvec is not None:
            if np is not None and type(self.dvec) is np.ndarray:
                dvec = builder.CreateNumpyVector(self.dvec)
            else:
                MonsterExtraStartDvecVector(builder, len(self.dvec))
                for i in reversed(range(len(self.dvec))):
                    builder.PrependFloat64(self.dvec[i])
                dvec = builder.EndVector()
        if self.fvec is not None:
            if np is not None and type(self.fvec) is np.ndarray:
                fvec = builder.CreateNumpyVector(self.fvec)
            else:
                MonsterExtraStartFvecVector(builder, len(self.fvec))
                for i in reversed(range(len(self.fvec))):
                    builder.PrependFloat32(self.fvec[i])
                fvec = builder.EndVector()
        MonsterExtraStart(builder)
        MonsterExtraAddD0(builder, self.d0)
        MonsterExtraAddD1(builder, self.d1)
        MonsterExtraAddD2(builder, self.d2)
        MonsterExtraAddD3(builder, self.d3)
        MonsterExtraAddF0(builder, self.f0)
        MonsterExtraAddF1(builder, self.f1)
        MonsterExtraAddF2(builder, self.f2)
        MonsterExtraAddF3(builder, self.f3)
        if self.dvec is not None:
            MonsterExtraAddDvec(builder, dvec)
        if self.fvec is not None:
            MonsterExtraAddFvec(builder, fvec)
        monsterExtra = MonsterExtraEnd(builder)
        return monsterExtra