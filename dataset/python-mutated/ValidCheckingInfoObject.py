import os
from Common.RangeExpression import RangeExpression
from Common.Misc import *
from io import BytesIO
from struct import pack
from Common.DataType import *

class VAR_CHECK_PCD_VARIABLE_TAB_CONTAINER(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.var_check_info = []

    def push_back(self, var_check_tab):
        if False:
            print('Hello World!')
        for tab in self.var_check_info:
            if tab.equal(var_check_tab):
                tab.merge(var_check_tab)
                break
        else:
            self.var_check_info.append(var_check_tab)

    def dump(self, dest, Phase):
        if False:
            while True:
                i = 10
        if not os.path.isabs(dest):
            return
        if not os.path.exists(dest):
            os.mkdir(dest)
        BinFileName = 'PcdVarCheck.bin'
        BinFilePath = os.path.join(dest, BinFileName)
        Buffer = bytearray()
        index = 0
        for var_check_tab in self.var_check_info:
            index += 1
            realLength = 0
            realLength += 32
            Name = var_check_tab.Name[1:-1]
            NameChars = Name.split(',')
            realLength += len(NameChars)
            if index < len(self.var_check_info) and realLength % 4 or (index == len(self.var_check_info) and len(var_check_tab.validtab) > 0 and realLength % 4):
                realLength += 4 - realLength % 4
            itemIndex = 0
            for item in var_check_tab.validtab:
                itemIndex += 1
                realLength += 5
                for v_data in item.data:
                    if isinstance(v_data, int):
                        realLength += item.StorageWidth
                    else:
                        realLength += item.StorageWidth
                        realLength += item.StorageWidth
                if index == len(self.var_check_info):
                    if itemIndex < len(var_check_tab.validtab) and realLength % 4:
                        realLength += 4 - realLength % 4
                elif realLength % 4:
                    realLength += 4 - realLength % 4
            var_check_tab.Length = realLength
        realLength = 0
        index = 0
        for var_check_tab in self.var_check_info:
            index += 1
            b = pack('=H', var_check_tab.Revision)
            Buffer += b
            realLength += 2
            b = pack('=H', var_check_tab.HeaderLength)
            Buffer += b
            realLength += 2
            b = pack('=L', var_check_tab.Length)
            Buffer += b
            realLength += 4
            b = pack('=B', var_check_tab.Type)
            Buffer += b
            realLength += 1
            for i in range(0, 3):
                b = pack('=B', var_check_tab.Reserved)
                Buffer += b
                realLength += 1
            b = pack('=L', var_check_tab.Attributes)
            Buffer += b
            realLength += 4
            Guid = var_check_tab.Guid
            b = PackByteFormatGUID(Guid)
            Buffer += b
            realLength += 16
            Name = var_check_tab.Name[1:-1]
            NameChars = Name.split(',')
            for NameChar in NameChars:
                NameCharNum = int(NameChar, 16)
                b = pack('=B', NameCharNum)
                Buffer += b
                realLength += 1
            if index < len(self.var_check_info) and realLength % 4 or (index == len(self.var_check_info) and len(var_check_tab.validtab) > 0 and realLength % 4):
                for i in range(4 - realLength % 4):
                    b = pack('=B', var_check_tab.pad)
                    Buffer += b
                    realLength += 1
            itemIndex = 0
            for item in var_check_tab.validtab:
                itemIndex += 1
                b = pack('=B', item.Type)
                Buffer += b
                realLength += 1
                b = pack('=B', item.Length)
                Buffer += b
                realLength += 1
                b = pack('=H', int(item.VarOffset, 16))
                Buffer += b
                realLength += 2
                b = pack('=B', item.StorageWidth)
                Buffer += b
                realLength += 1
                for v_data in item.data:
                    if isinstance(v_data, int):
                        b = pack(PACK_CODE_BY_SIZE[item.StorageWidth], v_data)
                        Buffer += b
                        realLength += item.StorageWidth
                    else:
                        b = pack(PACK_CODE_BY_SIZE[item.StorageWidth], v_data[0])
                        Buffer += b
                        realLength += item.StorageWidth
                        b = pack(PACK_CODE_BY_SIZE[item.StorageWidth], v_data[1])
                        Buffer += b
                        realLength += item.StorageWidth
                if index == len(self.var_check_info):
                    if itemIndex < len(var_check_tab.validtab) and realLength % 4:
                        for i in range(4 - realLength % 4):
                            b = pack('=B', var_check_tab.pad)
                            Buffer += b
                            realLength += 1
                elif realLength % 4:
                    for i in range(4 - realLength % 4):
                        b = pack('=B', var_check_tab.pad)
                        Buffer += b
                        realLength += 1
        DbFile = BytesIO()
        if Phase == 'DXE' and os.path.exists(BinFilePath):
            BinFile = open(BinFilePath, 'rb')
            BinBuffer = BinFile.read()
            BinFile.close()
            BinBufferSize = len(BinBuffer)
            if BinBufferSize % 4:
                for i in range(4 - BinBufferSize % 4):
                    b = pack('=B', VAR_CHECK_PCD_VARIABLE_TAB.pad)
                    BinBuffer += b
            Buffer = BinBuffer + Buffer
        DbFile.write(Buffer)
        SaveFileOnChange(BinFilePath, DbFile.getvalue(), True)

class VAR_CHECK_PCD_VARIABLE_TAB(object):
    pad = 218

    def __init__(self, TokenSpaceGuid, PcdCName):
        if False:
            while True:
                i = 10
        self.Revision = 1
        self.HeaderLength = 0
        self.Length = 0
        self.Type = 0
        self.Reserved = 0
        self.Attributes = 0
        self.Guid = eval('[' + TokenSpaceGuid.replace('{', '').replace('}', '') + ']')
        self.Name = PcdCName
        self.validtab = []

    def UpdateSize(self):
        if False:
            return 10
        self.HeaderLength = 32 + len(self.Name.split(','))
        self.Length = 32 + len(self.Name.split(',')) + self.GetValidTabLen()

    def GetValidTabLen(self):
        if False:
            i = 10
            return i + 15
        validtablen = 0
        for item in self.validtab:
            validtablen += item.Length
        return validtablen

    def SetAttributes(self, attributes):
        if False:
            for i in range(10):
                print('nop')
        self.Attributes = attributes

    def push_back(self, valid_obj):
        if False:
            return 10
        if valid_obj is not None:
            self.validtab.append(valid_obj)

    def equal(self, varchecktab):
        if False:
            for i in range(10):
                print('nop')
        if self.Guid == varchecktab.Guid and self.Name == varchecktab.Name:
            return True
        else:
            return False

    def merge(self, varchecktab):
        if False:
            print('Hello World!')
        for validobj in varchecktab.validtab:
            if validobj in self.validtab:
                continue
            self.validtab.append(validobj)
        self.UpdateSize()

class VAR_CHECK_PCD_VALID_OBJ(object):

    def __init__(self, VarOffset, data, PcdDataType):
        if False:
            while True:
                i = 10
        self.Type = 1
        self.Length = 0
        self.VarOffset = VarOffset
        self.PcdDataType = PcdDataType.strip()
        self.rawdata = data
        self.data = set()
        try:
            self.StorageWidth = MAX_SIZE_TYPE[self.PcdDataType]
            self.ValidData = True
        except:
            self.StorageWidth = 0
            self.ValidData = False

    def __eq__(self, validObj):
        if False:
            while True:
                i = 10
        return validObj and self.VarOffset == validObj.VarOffset

class VAR_CHECK_PCD_VALID_LIST(VAR_CHECK_PCD_VALID_OBJ):

    def __init__(self, VarOffset, validlist, PcdDataType):
        if False:
            while True:
                i = 10
        super(VAR_CHECK_PCD_VALID_LIST, self).__init__(VarOffset, validlist, PcdDataType)
        self.Type = 1
        valid_num_list = []
        for item in self.rawdata:
            valid_num_list.extend(item.split(','))
        for valid_num in valid_num_list:
            valid_num = valid_num.strip()
            if valid_num.startswith('0x') or valid_num.startswith('0X'):
                self.data.add(int(valid_num, 16))
            else:
                self.data.add(int(valid_num))
        self.Length = 5 + len(self.data) * self.StorageWidth

class VAR_CHECK_PCD_VALID_RANGE(VAR_CHECK_PCD_VALID_OBJ):

    def __init__(self, VarOffset, validrange, PcdDataType):
        if False:
            print('Hello World!')
        super(VAR_CHECK_PCD_VALID_RANGE, self).__init__(VarOffset, validrange, PcdDataType)
        self.Type = 2
        RangeExpr = ''
        i = 0
        for item in self.rawdata:
            if i == 0:
                RangeExpr = '( ' + item + ' )'
            else:
                RangeExpr = RangeExpr + 'OR ( ' + item + ' )'
        range_result = RangeExpression(RangeExpr, self.PcdDataType)(True)
        for rangelist in range_result:
            for obj in rangelist.pop():
                self.data.add((obj.start, obj.end))
        self.Length = 5 + len(self.data) * 2 * self.StorageWidth

def GetValidationObject(PcdClass, VarOffset):
    if False:
        i = 10
        return i + 15
    if PcdClass.validateranges:
        return VAR_CHECK_PCD_VALID_RANGE(VarOffset, PcdClass.validateranges, PcdClass.DatumType)
    if PcdClass.validlists:
        return VAR_CHECK_PCD_VALID_LIST(VarOffset, PcdClass.validlists, PcdClass.DatumType)
    else:
        return None