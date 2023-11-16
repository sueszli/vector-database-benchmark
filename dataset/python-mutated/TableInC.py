import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class TableInC(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            while True:
                i = 10
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = TableInC()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsTableInC(cls, buf, offset=0):
        if False:
            return 10
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    def Init(self, buf, pos):
        if False:
            for i in range(10):
                print('nop')
        self._tab = flatbuffers.table.Table(buf, pos)

    def ReferToA1(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            obj = TableInFirstNS()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    def ReferToA2(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            obj = SecondTableInA()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def TableInCStart(builder):
    if False:
        i = 10
        return i + 15
    builder.StartObject(2)

def Start(builder):
    if False:
        for i in range(10):
            print('nop')
    return TableInCStart(builder)

def TableInCAddReferToA1(builder, referToA1):
    if False:
        return 10
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(referToA1), 0)

def AddReferToA1(builder, referToA1):
    if False:
        for i in range(10):
            print('nop')
    return TableInCAddReferToA1(builder, referToA1)

def TableInCAddReferToA2(builder, referToA2):
    if False:
        print('Hello World!')
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(referToA2), 0)

def AddReferToA2(builder, referToA2):
    if False:
        i = 10
        return i + 15
    return TableInCAddReferToA2(builder, referToA2)

def TableInCEnd(builder):
    if False:
        return 10
    return builder.EndObject()

def End(builder):
    if False:
        for i in range(10):
            print('nop')
    return TableInCEnd(builder)
try:
    from typing import Optional
except:
    pass

class TableInCT(object):

    def __init__(self):
        if False:
            return 10
        self.referToA1 = None
        self.referToA2 = None

    @classmethod
    def InitFromBuf(cls, buf, pos):
        if False:
            while True:
                i = 10
        tableInC = TableInC()
        tableInC.Init(buf, pos)
        return cls.InitFromObj(tableInC)

    @classmethod
    def InitFromObj(cls, tableInC):
        if False:
            return 10
        x = TableInCT()
        x._UnPack(tableInC)
        return x

    def _UnPack(self, tableInC):
        if False:
            return 10
        if tableInC is None:
            return
        if tableInC.ReferToA1() is not None:
            self.referToA1 = TableInFirstNST.InitFromObj(tableInC.ReferToA1())
        if tableInC.ReferToA2() is not None:
            self.referToA2 = SecondTableInAT.InitFromObj(tableInC.ReferToA2())

    def Pack(self, builder):
        if False:
            print('Hello World!')
        if self.referToA1 is not None:
            referToA1 = self.referToA1.Pack(builder)
        if self.referToA2 is not None:
            referToA2 = self.referToA2.Pack(builder)
        TableInCStart(builder)
        if self.referToA1 is not None:
            TableInCAddReferToA1(builder, referToA1)
        if self.referToA2 is not None:
            TableInCAddReferToA2(builder, referToA2)
        tableInC = TableInCEnd(builder)
        return tableInC