import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Universe(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            while True:
                i = 10
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Universe()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsUniverse(cls, buf, offset=0):
        if False:
            i = 10
            return i + 15
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    def Init(self, buf, pos):
        if False:
            return 10
        self._tab = flatbuffers.table.Table(buf, pos)

    def Age(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float64Flags, o + self._tab.Pos)
        return 0.0

    def Galaxies(self, j):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from .Galaxy import Galaxy
            obj = Galaxy()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def GalaxiesLength(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def GalaxiesIsNone(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

def UniverseStart(builder):
    if False:
        print('Hello World!')
    builder.StartObject(2)

def Start(builder):
    if False:
        return 10
    UniverseStart(builder)

def UniverseAddAge(builder, age):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependFloat64Slot(0, age, 0.0)

def AddAge(builder: flatbuffers.Builder, age: float):
    if False:
        for i in range(10):
            print('nop')
    UniverseAddAge(builder, age)

def UniverseAddGalaxies(builder, galaxies):
    if False:
        i = 10
        return i + 15
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(galaxies), 0)

def AddGalaxies(builder: flatbuffers.Builder, galaxies: int):
    if False:
        return 10
    UniverseAddGalaxies(builder, galaxies)

def UniverseStartGalaxiesVector(builder, numElems):
    if False:
        i = 10
        return i + 15
    return builder.StartVector(4, numElems, 4)

def StartGalaxiesVector(builder, numElems: int) -> int:
    if False:
        return 10
    return UniverseStartGalaxiesVector(builder, numElems)

def UniverseEnd(builder):
    if False:
        i = 10
        return i + 15
    return builder.EndObject()

def End(builder):
    if False:
        while True:
            i = 10
    return UniverseEnd(builder)