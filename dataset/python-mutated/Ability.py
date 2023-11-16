import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Ability(object):
    __slots__ = ['_tab']

    @classmethod
    def SizeOf(cls):
        if False:
            for i in range(10):
                print('nop')
        return 8

    def Init(self, buf, pos):
        if False:
            for i in range(10):
                print('nop')
        self._tab = flatbuffers.table.Table(buf, pos)

    def Id(self):
        if False:
            return 10
        return self._tab.Get(flatbuffers.number_types.Uint32Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(0))

    def Distance(self):
        if False:
            return 10
        return self._tab.Get(flatbuffers.number_types.Uint32Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(4))

def CreateAbility(builder, id, distance):
    if False:
        while True:
            i = 10
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
            return 10
        ability = Ability()
        ability.Init(buf, pos)
        return cls.InitFromObj(ability)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        if False:
            while True:
                i = 10
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, ability):
        if False:
            return 10
        x = AbilityT()
        x._UnPack(ability)
        return x

    def _UnPack(self, ability):
        if False:
            print('Hello World!')
        if ability is None:
            return
        self.id = ability.Id()
        self.distance = ability.Distance()

    def Pack(self, builder):
        if False:
            while True:
                i = 10
        return CreateAbility(builder, self.id, self.distance)