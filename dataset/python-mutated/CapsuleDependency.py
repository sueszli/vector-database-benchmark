import struct
import json
import sys
import uuid
import re
'\nCapsuleDependency\n'

class OpConvert(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self._DepexOperations = {0: (16, 16, 's', self.Str2Guid, self.Guid2Str), 1: (4, 1, 'I', self.Str2Uint, self.Uint2Str), 2: (1, 0, 's', self.Str2Utf8, self.Byte2Str)}

    def Str2Uint(self, Data):
        if False:
            while True:
                i = 10
        try:
            Value = int(Data, 16)
        except:
            Message = '{Data} is not a valid integer value.'.format(Data=Data)
            raise ValueError(Message)
        if Value < 0 or Value > 4294967295:
            Message = '{Data} is not an UINT32.'.format(Data=Data)
            raise ValueError(Message)
        return Value

    def Uint2Str(self, Data):
        if False:
            i = 10
            return i + 15
        if Data < 0 or Data > 4294967295:
            Message = '{Data} is not an UINT32.'.format(Data=Data)
            raise ValueError(Message)
        return '0x{Data:08x}'.format(Data=Data)

    def Str2Guid(self, Data):
        if False:
            i = 10
            return i + 15
        try:
            Guid = uuid.UUID(Data)
        except:
            Message = '{Data} is not a valid registry format GUID value.'.format(Data=Data)
            raise ValueError(Message)
        return Guid.bytes_le

    def Guid2Str(self, Data):
        if False:
            for i in range(10):
                print('nop')
        try:
            Guid = uuid.UUID(bytes_le=Data)
        except:
            Message = '{Data} is not a valid binary format GUID value.'.format(Data=Data)
            raise ValueError(Message)
        return str(Guid).upper()

    def Str2Utf8(self, Data):
        if False:
            return 10
        if isinstance(Data, str):
            return Data.encode('utf-8')
        else:
            Message = '{Data} is not a valid string.'.format(Data=Data)
            raise ValueError(Message)

    def Byte2Str(self, Data):
        if False:
            print('Hello World!')
        if isinstance(Data, bytes):
            if Data[-1:] == b'\x00':
                return str(Data[:-1], 'utf-8')
            else:
                return str(Data, 'utf-8')
        else:
            Message = '{Data} is not a valid binary string.'.format(Data=Data)
            raise ValueError(Message)

    def OpEncode(self, Opcode, Operand=None):
        if False:
            while True:
                i = 10
        BinTemp = struct.pack('<b', Opcode)
        if Opcode <= 2 and Operand != None:
            (OperandSize, PackSize, PackFmt, EncodeConvert, DecodeConvert) = self._DepexOperations[Opcode]
            Value = EncodeConvert(Operand)
            if Opcode == 2:
                PackSize = len(Value) + 1
            BinTemp += struct.pack('<{PackSize}{PackFmt}'.format(PackSize=PackSize, PackFmt=PackFmt), Value)
        return BinTemp

    def OpDecode(self, Buffer):
        if False:
            return 10
        Opcode = struct.unpack('<b', Buffer[0:1])[0]
        if Opcode <= 2:
            (OperandSize, PackSize, PackFmt, EncodeConvert, DecodeConvert) = self._DepexOperations[Opcode]
            if Opcode == 2:
                try:
                    PackSize = Buffer[1:].index(b'\x00') + 1
                    OperandSize = PackSize
                except:
                    Message = 'CapsuleDependency: OpConvert: error: decode failed with wrong opcode/string.'
                    raise ValueError(Message)
            try:
                Operand = DecodeConvert(struct.unpack('<{PackSize}{PackFmt}'.format(PackSize=PackSize, PackFmt=PackFmt), Buffer[1:1 + OperandSize])[0])
            except:
                Message = 'CapsuleDependency: OpConvert: error: decode failed with unpack failure.'
                raise ValueError(Message)
        else:
            Operand = None
            OperandSize = 0
        return (Opcode, Operand, OperandSize)

class CapsuleDependencyClass(object):
    _opReference = {'&&': [2, 3, 2], '||': [1, 4, 2], '~': [5, 5, 1], '==': [3, 8, 2], '>': [4, 9, 2], '>=': [4, 10, 2], '<': [4, 11, 2], '<=': [4, 12, 2]}

    def __init__(self):
        if False:
            while True:
                i = 10
        self.Payload = b''
        self._DepexExp = None
        self._DepexList = []
        self._DepexDump = []
        self.Depex = b''
        self._Valid = False
        self._DepexSize = 0
        self._opReferenceReverse = {v[1]: k for (k, v) in self._opReference.items()}
        self.OpConverter = OpConvert()

    @property
    def DepexExp(self):
        if False:
            i = 10
            return i + 15
        return self._DepexExp

    @DepexExp.setter
    def DepexExp(self, DepexExp=''):
        if False:
            i = 10
            return i + 15
        if isinstance(DepexExp, str):
            DepexExp = re.sub('\\n', ' ', DepexExp)
            DepexExp = re.sub('\\(', ' ( ', DepexExp)
            DepexExp = re.sub('\\)', ' ) ', DepexExp)
            DepexExp = re.sub('~', ' ~ ', DepexExp)
            self._DepexList = re.findall('[^\\s\\"\\\']+|\\"[^\\"]*\\"|\\\'[^\\\']*\\\'', DepexExp)
            self._DepexExp = ' '.join(self._DepexList)
        else:
            Msg = 'Input Depex Expression is not valid string.'
            raise ValueError(Msg)

    def IsValidOperator(self, op):
        if False:
            print('Hello World!')
        return op in self._opReference.keys()

    def IsValidUnaryOperator(self, op):
        if False:
            i = 10
            return i + 15
        return op in self._opReference.keys() and self._opReference[op][2] == 1

    def IsValidBinocularOperator(self, op):
        if False:
            i = 10
            return i + 15
        return op in self._opReference.keys() and self._opReference[op][2] == 2

    def IsValidGuid(self, operand):
        if False:
            while True:
                i = 10
        try:
            uuid.UUID(operand)
        except:
            return False
        return True

    def IsValidVersion(self, operand):
        if False:
            while True:
                i = 10
        try:
            Value = int(operand, 16)
            if Value < 0 or Value > 4294967295:
                return False
        except:
            return False
        return True

    def IsValidBoolean(self, operand):
        if False:
            for i in range(10):
                print('nop')
        try:
            return operand.upper() in ['TRUE', 'FALSE']
        except:
            return False

    def IsValidOperand(self, operand):
        if False:
            for i in range(10):
                print('nop')
        return self.IsValidVersion(operand) or self.IsValidGuid(operand) or self.IsValidBoolean(operand)

    def IsValidString(self, operand):
        if False:
            print('Hello World!')
        return operand[0] == '"' and operand[-1] == '"' and (len(operand) >= 2)

    def PriorityNotGreater(self, prevOp, currOp):
        if False:
            return 10
        return self._opReference[currOp][0] <= self._opReference[prevOp][0]

    def ValidateDepex(self):
        if False:
            while True:
                i = 10
        OpList = self._DepexList
        i = 0
        while i < len(OpList):
            Op = OpList[i]
            if Op == 'DECLARE':
                i += 1
                if i >= len(OpList):
                    Msg = 'No more Operand after {Op}.'.format(Op=OpList[i - 1])
                    raise IndexError(Msg)
                if not self.IsValidString(OpList[i]):
                    Msg = '{Operand} after {Op} is not a valid expression input.'.format(Operand=OpList[i], Op=OpList[i - 1])
                    raise ValueError(Msg)
            elif Op == '(':
                if i == len(OpList) - 1:
                    Msg = "Expression cannot end with '('"
                    raise ValueError(Msg)
                if self.IsValidBinocularOperator(OpList[i + 1]):
                    Msg = "{Op} after '(' is not a valid expression input.".format(Op=OpList[i + 1])
                    raise ValueError(Msg)
            elif Op == ')':
                if i == 0:
                    Msg = "Expression cannot start with ')'"
                    raise ValueError(Msg)
                if self.IsValidOperator(OpList[i - 1]):
                    Msg = "{Op} before ')' is not a valid expression input.".format(Op=OpList[i - 1])
                    raise ValueError(Msg)
                if i + 1 < len(OpList) and (self.IsValidOperand(OpList[i + 1]) or self.IsValidUnaryOperator(OpList[i + 1])):
                    Msg = "{Op} after ')' is not a valid expression input.".format(Op=OpList[i + 1])
                    raise ValueError(Msg)
            elif self.IsValidOperand(Op):
                if i + 1 < len(OpList) and (self.IsValidOperand(OpList[i + 1]) or self.IsValidUnaryOperator(OpList[i + 1])):
                    Msg = '{Op} after {PrevOp} is not a valid expression input.'.format(Op=OpList[i + 1], PrevOp=Op)
                    raise ValueError(Msg)
            elif self.IsValidOperator(Op):
                if i + 1 < len(OpList) and self.IsValidBinocularOperator(OpList[i + 1]):
                    Msg = '{Op} after {PrevOp} is not a valid expression input.'.format(Op=OpList[i + 1], PrevOp=Op)
                    raise ValueError(Msg)
                if i == 0 and self.IsValidBinocularOperator(Op):
                    Msg = 'Expression cannot start with an operator {Op}.'.format(Op=Op)
                    raise ValueError(Msg)
                if i == len(OpList) - 1:
                    Msg = 'Expression cannot ended with an operator {Op}.'.format(Op=Op)
                    raise ValueError(Msg)
                if self.IsValidUnaryOperator(Op) and (self.IsValidGuid(OpList[i + 1]) or self.IsValidVersion(OpList[i + 1])):
                    Msg = '{Op} after {PrevOp} is not a valid expression input.'.format(Op=OpList[i + 1], PrevOp=Op)
                    raise ValueError(Msg)
            else:
                Msg = '{Op} is not a valid expression input.'.format(Op=Op)
                raise ValueError(Msg)
            i += 1

    def Encode(self):
        if False:
            while True:
                i = 10
        self.Depex = b''
        self._DepexDump = []
        OperandStack = []
        OpeartorStack = []
        OpList = self._DepexList
        self.ValidateDepex()
        i = 0
        while i < len(OpList):
            Op = OpList[i]
            if Op == 'DECLARE':
                i += 1
                self.Depex += self.OpConverter.OpEncode(2, OpList[i][1:-1])
            elif Op == '(':
                OpeartorStack.append(Op)
            elif Op == ')':
                while OpeartorStack and OpeartorStack[-1] != '(':
                    Operator = OpeartorStack.pop()
                    self.Depex += self.OpConverter.OpEncode(self._opReference[Operator][1])
                try:
                    OpeartorStack.pop()
                except:
                    Msg = "Pop out '(' failed, too many ')'"
                    raise ValueError(Msg)
            elif self.IsValidGuid(Op):
                if not OperandStack:
                    OperandStack.append(self.OpConverter.OpEncode(0, Op))
                else:
                    self.Depex += self.OpConverter.OpEncode(0, Op)
                    self.Depex += OperandStack.pop()
            elif self.IsValidVersion(Op):
                if not OperandStack:
                    OperandStack.append(self.OpConverter.OpEncode(1, Op))
                else:
                    self.Depex += self.OpConverter.OpEncode(1, Op)
                    self.Depex += OperandStack.pop()
            elif self.IsValidBoolean(Op):
                if Op.upper() == 'FALSE':
                    self.Depex += self.OpConverter.OpEncode(7)
                elif Op.upper() == 'TRUE':
                    self.Depex += self.OpConverter.OpEncode(6)
            elif self.IsValidOperator(Op):
                while OpeartorStack and OpeartorStack[-1] != '(' and self.PriorityNotGreater(OpeartorStack[-1], Op):
                    Operator = OpeartorStack.pop()
                    self.Depex += self.OpConverter.OpEncode(self._opReference[Operator][1])
                OpeartorStack.append(Op)
            i += 1
        while OpeartorStack:
            Operator = OpeartorStack.pop()
            if Operator == '(':
                Msg = "Too many '('."
                raise ValueError(Msg)
            self.Depex += self.OpConverter.OpEncode(self._opReference[Operator][1])
        self.Depex += self.OpConverter.OpEncode(13)
        self._Valid = True
        self._DepexSize = len(self.Depex)
        return self.Depex + self.Payload

    def Decode(self, Buffer):
        if False:
            i = 10
            return i + 15
        self.Depex = Buffer
        OperandStack = []
        DepexLen = 0
        while True:
            (Opcode, Operand, OperandSize) = self.OpConverter.OpDecode(Buffer[DepexLen:])
            DepexLen += OperandSize + 1
            if Opcode == 13:
                break
            elif Opcode == 2:
                if not OperandStack:
                    OperandStack.append('DECLARE "{String}"'.format(String=Operand))
                else:
                    PrevOperand = OperandStack.pop()
                    OperandStack.append('{Operand} DECLARE "{String}"'.format(Operand=PrevOperand, String=Operand))
            elif Opcode in [0, 1]:
                OperandStack.append(Operand)
            elif Opcode == 6:
                OperandStack.append('TRUE')
            elif Opcode == 7:
                OperandStack.append('FALSE')
            elif self.IsValidOperator(self._opReferenceReverse[Opcode]):
                Operator = self._opReferenceReverse[Opcode]
                if self.IsValidUnaryOperator(self._opReferenceReverse[Opcode]) and len(OperandStack) >= 1:
                    Oprand = OperandStack.pop()
                    OperandStack.append(' ( {Operator} {Oprand} )'.format(Operator=Operator, Oprand=Oprand))
                elif self.IsValidBinocularOperator(self._opReferenceReverse[Opcode]) and len(OperandStack) >= 2:
                    Oprand1 = OperandStack.pop()
                    Oprand2 = OperandStack.pop()
                    OperandStack.append(' ( {Oprand1} {Operator} {Oprand2} )'.format(Operator=Operator, Oprand1=Oprand1, Oprand2=Oprand2))
                else:
                    Msg = 'No enough Operands for {Opcode:02X}.'.format(Opcode=Opcode)
                    raise ValueError(Msg)
            else:
                Msg = '{Opcode:02X} is not a valid OpCode.'.format(Opcode=Opcode)
                raise ValueError(Msg)
        self.DepexExp = OperandStack[0].strip(' ')
        self.Payload = Buffer[DepexLen:]
        self._Valid = True
        self._DepexSize = DepexLen
        return self.Payload

    def DumpInfo(self):
        if False:
            for i in range(10):
                print('nop')
        DepexLen = 0
        Opcode = None
        Buffer = self.Depex
        if self._Valid == True:
            print('EFI_FIRMWARE_IMAGE_DEP.Dependencies = {')
            while Opcode != 13:
                (Opcode, Operand, OperandSize) = self.OpConverter.OpDecode(Buffer[DepexLen:])
                DepexLen += OperandSize + 1
                if Operand:
                    print('    {Opcode:02X}, {Operand},'.format(Opcode=Opcode, Operand=Operand))
                else:
                    print('    {Opcode:02X},'.format(Opcode=Opcode))
            print('}')
            print('sizeof (EFI_FIRMWARE_IMAGE_DEP.Dependencies)    = {Size:08X}'.format(Size=self._DepexSize))
            print('sizeof (Payload)                                = {Size:08X}'.format(Size=len(self.Payload)))