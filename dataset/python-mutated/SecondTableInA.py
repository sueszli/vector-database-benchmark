import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class SecondTableInA(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            print('Hello World!')
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SecondTableInA()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsSecondTableInA(cls, buf, offset=0):
        if False:
            print('Hello World!')
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    def Init(self, buf, pos):
        if False:
            print('Hello World!')
        self._tab = flatbuffers.table.Table(buf, pos)

    def ReferToC(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            obj = TableInC()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def SecondTableInAStart(builder):
    if False:
        while True:
            i = 10
    builder.StartObject(1)

def Start(builder):
    if False:
        return 10
    return SecondTableInAStart(builder)

def SecondTableInAAddReferToC(builder, referToC):
    if False:
        print('Hello World!')
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(referToC), 0)

def AddReferToC(builder, referToC):
    if False:
        for i in range(10):
            print('nop')
    return SecondTableInAAddReferToC(builder, referToC)

def SecondTableInAEnd(builder):
    if False:
        for i in range(10):
            print('nop')
    return builder.EndObject()

def End(builder):
    if False:
        for i in range(10):
            print('nop')
    return SecondTableInAEnd(builder)
try:
    from typing import Optional
except:
    pass

class SecondTableInAT(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.referToC = None

    @classmethod
    def InitFromBuf(cls, buf, pos):
        if False:
            while True:
                i = 10
        secondTableInA = SecondTableInA()
        secondTableInA.Init(buf, pos)
        return cls.InitFromObj(secondTableInA)

    @classmethod
    def InitFromObj(cls, secondTableInA):
        if False:
            return 10
        x = SecondTableInAT()
        x._UnPack(secondTableInA)
        return x

    def _UnPack(self, secondTableInA):
        if False:
            while True:
                i = 10
        if secondTableInA is None:
            return
        if secondTableInA.ReferToC() is not None:
            self.referToC = TableInCT.InitFromObj(secondTableInA.ReferToC())

    def Pack(self, builder):
        if False:
            print('Hello World!')
        if self.referToC is not None:
            referToC = self.referToC.Pack(builder)
        SecondTableInAStart(builder)
        if self.referToC is not None:
            SecondTableInAAddReferToC(builder, referToC)
        secondTableInA = SecondTableInAEnd(builder)
        return secondTableInA