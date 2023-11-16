import flatbuffers
from flatbuffers.compat import import_numpy
from typing import Any
np = import_numpy()

class NestedStruct(object):
    __slots__ = ['_tab']

    @classmethod
    def SizeOf(cls) -> int:
        if False:
            return 10
        return 32

    def Init(self, buf: bytes, pos: int):
        if False:
            for i in range(10):
                print('nop')
        self._tab = flatbuffers.table.Table(buf, pos)

    def A(self, j=None):
        if False:
            for i in range(10):
                print('nop')
        if j is None:
            return [self._tab.Get(flatbuffers.number_types.Int32Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(0 + i * 4)) for i in range(self.ALength())]
        elif j >= 0 and j < self.ALength():
            return self._tab.Get(flatbuffers.number_types.Int32Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(0 + j * 4))
        else:
            return None

    def AAsNumpy(self):
        if False:
            print('Hello World!')
        return self._tab.GetArrayAsNumpy(flatbuffers.number_types.Int32Flags, self._tab.Pos + 0, self.ALength())

    def ALength(self) -> int:
        if False:
            return 10
        return 2

    def AIsNone(self) -> bool:
        if False:
            print('Hello World!')
        return False

    def B(self):
        if False:
            return 10
        return self._tab.Get(flatbuffers.number_types.Int8Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(8))

    def C(self, j=None):
        if False:
            return 10
        if j is None:
            return [self._tab.Get(flatbuffers.number_types.Int8Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(9 + i * 1)) for i in range(self.CLength())]
        elif j >= 0 and j < self.CLength():
            return self._tab.Get(flatbuffers.number_types.Int8Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(9 + j * 1))
        else:
            return None

    def CAsNumpy(self):
        if False:
            while True:
                i = 10
        return self._tab.GetArrayAsNumpy(flatbuffers.number_types.Int8Flags, self._tab.Pos + 9, self.CLength())

    def CLength(self) -> int:
        if False:
            print('Hello World!')
        return 2

    def CIsNone(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return False

    def D(self, j=None):
        if False:
            for i in range(10):
                print('nop')
        if j is None:
            return [self._tab.Get(flatbuffers.number_types.Int64Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(16 + i * 8)) for i in range(self.DLength())]
        elif j >= 0 and j < self.DLength():
            return self._tab.Get(flatbuffers.number_types.Int64Flags, self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(16 + j * 8))
        else:
            return None

    def DAsNumpy(self):
        if False:
            i = 10
            return i + 15
        return self._tab.GetArrayAsNumpy(flatbuffers.number_types.Int64Flags, self._tab.Pos + 16, self.DLength())

    def DLength(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return 2

    def DIsNone(self) -> bool:
        if False:
            while True:
                i = 10
        return False

def CreateNestedStruct(builder, a, b, c, d):
    if False:
        while True:
            i = 10
    builder.Prep(8, 32)
    for _idx0 in range(2, 0, -1):
        builder.PrependInt64(d[_idx0 - 1])
    builder.Pad(5)
    for _idx0 in range(2, 0, -1):
        builder.PrependInt8(c[_idx0 - 1])
    builder.PrependInt8(b)
    for _idx0 in range(2, 0, -1):
        builder.PrependInt32(a[_idx0 - 1])
    return builder.Offset()
try:
    from typing import List
except:
    pass

class NestedStructT(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.a = None
        self.b = 0
        self.c = None
        self.d = None

    @classmethod
    def InitFromBuf(cls, buf, pos):
        if False:
            print('Hello World!')
        nestedStruct = NestedStruct()
        nestedStruct.Init(buf, pos)
        return cls.InitFromObj(nestedStruct)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        if False:
            for i in range(10):
                print('nop')
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, nestedStruct):
        if False:
            return 10
        x = NestedStructT()
        x._UnPack(nestedStruct)
        return x

    def _UnPack(self, nestedStruct):
        if False:
            for i in range(10):
                print('nop')
        if nestedStruct is None:
            return
        if not nestedStruct.AIsNone():
            if np is None:
                self.a = []
                for i in range(nestedStruct.ALength()):
                    self.a.append(nestedStruct.A(i))
            else:
                self.a = nestedStruct.AAsNumpy()
        self.b = nestedStruct.B()
        if not nestedStruct.CIsNone():
            if np is None:
                self.c = []
                for i in range(nestedStruct.CLength()):
                    self.c.append(nestedStruct.C(i))
            else:
                self.c = nestedStruct.CAsNumpy()
        if not nestedStruct.DIsNone():
            if np is None:
                self.d = []
                for i in range(nestedStruct.DLength()):
                    self.d.append(nestedStruct.D(i))
            else:
                self.d = nestedStruct.DAsNumpy()

    def Pack(self, builder):
        if False:
            return 10
        return CreateNestedStruct(builder, self.a, self.b, self.c, self.d)