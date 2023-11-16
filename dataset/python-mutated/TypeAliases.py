import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class TypeAliases(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        if False:
            i = 10
            return i + 15
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = TypeAliases()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsTypeAliases(cls, buf, offset=0):
        if False:
            while True:
                i = 10
        'This method is deprecated. Please switch to GetRootAs.'
        return cls.GetRootAs(buf, offset)

    @classmethod
    def TypeAliasesBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        if False:
            print('Hello World!')
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b'MONS', size_prefixed=size_prefixed)

    def Init(self, buf, pos):
        if False:
            for i in range(10):
                print('nop')
        self._tab = flatbuffers.table.Table(buf, pos)

    def I8(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    def U8(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    def I16(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int16Flags, o + self._tab.Pos)
        return 0

    def U16(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint16Flags, o + self._tab.Pos)
        return 0

    def I32(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    def U32(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    def I64(self):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int64Flags, o + self._tab.Pos)
        return 0

    def U64(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, o + self._tab.Pos)
        return 0

    def F32(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

    def F64(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float64Flags, o + self._tab.Pos)
        return 0.0

    def V8(self, j):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    def V8AsNumpy(self):
        if False:
            i = 10
            return i + 15
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int8Flags, o)
        return 0

    def V8Length(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def V8IsNone(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        return o == 0

    def Vf64(self, j):
        if False:
            return 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Float64Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 8))
        return 0

    def Vf64AsNumpy(self):
        if False:
            while True:
                i = 10
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Float64Flags, o)
        return 0

    def Vf64Length(self):
        if False:
            print('Hello World!')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    def Vf64IsNone(self):
        if False:
            for i in range(10):
                print('nop')
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        return o == 0

def TypeAliasesStart(builder):
    if False:
        while True:
            i = 10
    builder.StartObject(12)

def Start(builder):
    if False:
        for i in range(10):
            print('nop')
    TypeAliasesStart(builder)

def TypeAliasesAddI8(builder, i8):
    if False:
        return 10
    builder.PrependInt8Slot(0, i8, 0)

def AddI8(builder, i8):
    if False:
        print('Hello World!')
    TypeAliasesAddI8(builder, i8)

def TypeAliasesAddU8(builder, u8):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependUint8Slot(1, u8, 0)

def AddU8(builder, u8):
    if False:
        i = 10
        return i + 15
    TypeAliasesAddU8(builder, u8)

def TypeAliasesAddI16(builder, i16):
    if False:
        while True:
            i = 10
    builder.PrependInt16Slot(2, i16, 0)

def AddI16(builder, i16):
    if False:
        for i in range(10):
            print('nop')
    TypeAliasesAddI16(builder, i16)

def TypeAliasesAddU16(builder, u16):
    if False:
        i = 10
        return i + 15
    builder.PrependUint16Slot(3, u16, 0)

def AddU16(builder, u16):
    if False:
        i = 10
        return i + 15
    TypeAliasesAddU16(builder, u16)

def TypeAliasesAddI32(builder, i32):
    if False:
        while True:
            i = 10
    builder.PrependInt32Slot(4, i32, 0)

def AddI32(builder, i32):
    if False:
        for i in range(10):
            print('nop')
    TypeAliasesAddI32(builder, i32)

def TypeAliasesAddU32(builder, u32):
    if False:
        i = 10
        return i + 15
    builder.PrependUint32Slot(5, u32, 0)

def AddU32(builder, u32):
    if False:
        print('Hello World!')
    TypeAliasesAddU32(builder, u32)

def TypeAliasesAddI64(builder, i64):
    if False:
        return 10
    builder.PrependInt64Slot(6, i64, 0)

def AddI64(builder, i64):
    if False:
        return 10
    TypeAliasesAddI64(builder, i64)

def TypeAliasesAddU64(builder, u64):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependUint64Slot(7, u64, 0)

def AddU64(builder, u64):
    if False:
        i = 10
        return i + 15
    TypeAliasesAddU64(builder, u64)

def TypeAliasesAddF32(builder, f32):
    if False:
        i = 10
        return i + 15
    builder.PrependFloat32Slot(8, f32, 0.0)

def AddF32(builder, f32):
    if False:
        return 10
    TypeAliasesAddF32(builder, f32)

def TypeAliasesAddF64(builder, f64):
    if False:
        return 10
    builder.PrependFloat64Slot(9, f64, 0.0)

def AddF64(builder, f64):
    if False:
        while True:
            i = 10
    TypeAliasesAddF64(builder, f64)

def TypeAliasesAddV8(builder, v8):
    if False:
        i = 10
        return i + 15
    builder.PrependUOffsetTRelativeSlot(10, flatbuffers.number_types.UOffsetTFlags.py_type(v8), 0)

def AddV8(builder, v8):
    if False:
        i = 10
        return i + 15
    TypeAliasesAddV8(builder, v8)

def TypeAliasesStartV8Vector(builder, numElems):
    if False:
        print('Hello World!')
    return builder.StartVector(1, numElems, 1)

def StartV8Vector(builder, numElems: int) -> int:
    if False:
        print('Hello World!')
    return TypeAliasesStartV8Vector(builder, numElems)

def TypeAliasesAddVf64(builder, vf64):
    if False:
        for i in range(10):
            print('nop')
    builder.PrependUOffsetTRelativeSlot(11, flatbuffers.number_types.UOffsetTFlags.py_type(vf64), 0)

def AddVf64(builder, vf64):
    if False:
        print('Hello World!')
    TypeAliasesAddVf64(builder, vf64)

def TypeAliasesStartVf64Vector(builder, numElems):
    if False:
        for i in range(10):
            print('nop')
    return builder.StartVector(8, numElems, 8)

def StartVf64Vector(builder, numElems: int) -> int:
    if False:
        while True:
            i = 10
    return TypeAliasesStartVf64Vector(builder, numElems)

def TypeAliasesEnd(builder):
    if False:
        for i in range(10):
            print('nop')
    return builder.EndObject()

def End(builder):
    if False:
        i = 10
        return i + 15
    return TypeAliasesEnd(builder)
try:
    from typing import List
except:
    pass

class TypeAliasesT(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.i8 = 0
        self.u8 = 0
        self.i16 = 0
        self.u16 = 0
        self.i32 = 0
        self.u32 = 0
        self.i64 = 0
        self.u64 = 0
        self.f32 = 0.0
        self.f64 = 0.0
        self.v8 = None
        self.vf64 = None

    @classmethod
    def InitFromBuf(cls, buf, pos):
        if False:
            print('Hello World!')
        typeAliases = TypeAliases()
        typeAliases.Init(buf, pos)
        return cls.InitFromObj(typeAliases)

    @classmethod
    def InitFromPackedBuf(cls, buf, pos=0):
        if False:
            while True:
                i = 10
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, pos)
        return cls.InitFromBuf(buf, pos + n)

    @classmethod
    def InitFromObj(cls, typeAliases):
        if False:
            return 10
        x = TypeAliasesT()
        x._UnPack(typeAliases)
        return x

    def _UnPack(self, typeAliases):
        if False:
            return 10
        if typeAliases is None:
            return
        self.i8 = typeAliases.I8()
        self.u8 = typeAliases.U8()
        self.i16 = typeAliases.I16()
        self.u16 = typeAliases.U16()
        self.i32 = typeAliases.I32()
        self.u32 = typeAliases.U32()
        self.i64 = typeAliases.I64()
        self.u64 = typeAliases.U64()
        self.f32 = typeAliases.F32()
        self.f64 = typeAliases.F64()
        if not typeAliases.V8IsNone():
            if np is None:
                self.v8 = []
                for i in range(typeAliases.V8Length()):
                    self.v8.append(typeAliases.V8(i))
            else:
                self.v8 = typeAliases.V8AsNumpy()
        if not typeAliases.Vf64IsNone():
            if np is None:
                self.vf64 = []
                for i in range(typeAliases.Vf64Length()):
                    self.vf64.append(typeAliases.Vf64(i))
            else:
                self.vf64 = typeAliases.Vf64AsNumpy()

    def Pack(self, builder):
        if False:
            i = 10
            return i + 15
        if self.v8 is not None:
            if np is not None and type(self.v8) is np.ndarray:
                v8 = builder.CreateNumpyVector(self.v8)
            else:
                TypeAliasesStartV8Vector(builder, len(self.v8))
                for i in reversed(range(len(self.v8))):
                    builder.PrependByte(self.v8[i])
                v8 = builder.EndVector()
        if self.vf64 is not None:
            if np is not None and type(self.vf64) is np.ndarray:
                vf64 = builder.CreateNumpyVector(self.vf64)
            else:
                TypeAliasesStartVf64Vector(builder, len(self.vf64))
                for i in reversed(range(len(self.vf64))):
                    builder.PrependFloat64(self.vf64[i])
                vf64 = builder.EndVector()
        TypeAliasesStart(builder)
        TypeAliasesAddI8(builder, self.i8)
        TypeAliasesAddU8(builder, self.u8)
        TypeAliasesAddI16(builder, self.i16)
        TypeAliasesAddU16(builder, self.u16)
        TypeAliasesAddI32(builder, self.i32)
        TypeAliasesAddU32(builder, self.u32)
        TypeAliasesAddI64(builder, self.i64)
        TypeAliasesAddU64(builder, self.u64)
        TypeAliasesAddF32(builder, self.f32)
        TypeAliasesAddF64(builder, self.f64)
        if self.v8 is not None:
            TypeAliasesAddV8(builder, v8)
        if self.vf64 is not None:
            TypeAliasesAddVf64(builder, vf64)
        typeAliases = TypeAliasesEnd(builder)
        return typeAliases