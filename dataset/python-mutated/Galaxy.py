import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Galaxy(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            return 10
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Galaxy()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsGalaxy(cls, buf, offset=0):
        if False:
            for i in range(10):
                print('nop')
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    def Init(self, buf, pos):
        if False:
            i = 10
            return i + 15
        self._tab = flatbuffers.table.Table(buf, pos)

    def NumStars(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int64Flags, o + self._tab.Pos)
        return 0

def GalaxyStart(builder):
    if False:
        i = 10
        return i + 15
    builder.StartObject(1)

def Start(builder):
    if False:
        while True:
            i = 10
    GalaxyStart(builder)

def GalaxyAddNumStars(builder, numStars):
    if False:
        while True:
            i = 10
    builder.PrependInt64Slot(0, numStars, 0)

def AddNumStars(builder: flatbuffers.Builder, numStars: int):
    if False:
        return 10
    GalaxyAddNumStars(builder, numStars)

def GalaxyEnd(builder):
    if False:
        for i in range(10):
            print('nop')
    return builder.EndObject()

def End(builder):
    if False:
        for i in range(10):
            print('nop')
    return GalaxyEnd(builder)