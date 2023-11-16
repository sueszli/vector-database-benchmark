from __future__ import absolute_import
import Common.LongFilePathOs as os
from io import BytesIO
from . import StringTable as st
import array
import re
from Common.LongFilePathSupport import OpenLongFilePath as open
from struct import *
from Common.DataType import MAX_SIZE_TYPE, MAX_VAL_TYPE, TAB_STAR
import Common.EdkLogger as EdkLogger
import Common.BuildToolError as BuildToolError
_FORMAT_CHAR = {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}

class PcdEntry:

    def __init__(self, PcdCName, SkuId, PcdOffset, PcdSize, PcdValue, Lineno=None, FileName=None, PcdUnpackValue=None, PcdBinOffset=None, PcdBinSize=None, Alignment=None):
        if False:
            i = 10
            return i + 15
        self.PcdCName = PcdCName.strip()
        self.SkuId = SkuId.strip()
        self.PcdOffset = PcdOffset.strip()
        self.PcdSize = PcdSize.strip()
        self.PcdValue = PcdValue.strip()
        self.Lineno = Lineno.strip()
        self.FileName = FileName.strip()
        self.PcdUnpackValue = PcdUnpackValue
        self.PcdBinOffset = PcdBinOffset
        self.PcdBinSize = PcdBinSize
        self.Alignment = Alignment
        if self.PcdValue == '':
            EdkLogger.error('BPDG', BuildToolError.FORMAT_INVALID, 'Invalid PCD format(Name: %s File: %s line: %s) , no Value specified!' % (self.PcdCName, self.FileName, self.Lineno))
        if self.PcdOffset == '':
            EdkLogger.error('BPDG', BuildToolError.FORMAT_INVALID, 'Invalid PCD format(Name: %s File: %s Line: %s) , no Offset specified!' % (self.PcdCName, self.FileName, self.Lineno))
        if self.PcdSize == '':
            EdkLogger.error('BPDG', BuildToolError.FORMAT_INVALID, 'Invalid PCD format(Name: %s File: %s Line: %s), no PcdSize specified!' % (self.PcdCName, self.FileName, self.Lineno))
        self._GenOffsetValue()

    def _IsBoolean(self, ValueString, Size):
        if False:
            print('Hello World!')
        if Size == '1':
            if ValueString.upper() in ['TRUE', 'FALSE']:
                return True
            elif ValueString in ['0', '1', '0x0', '0x1', '0x00', '0x01']:
                return True
        return False

    def _GenOffsetValue(self):
        if False:
            return 10
        if self.PcdOffset != TAB_STAR:
            try:
                self.PcdBinOffset = int(self.PcdOffset)
            except:
                try:
                    self.PcdBinOffset = int(self.PcdOffset, 16)
                except:
                    EdkLogger.error('BPDG', BuildToolError.FORMAT_INVALID, 'Invalid offset value %s for PCD %s (File: %s Line: %s)' % (self.PcdOffset, self.PcdCName, self.FileName, self.Lineno))

    def _PackBooleanValue(self, ValueString):
        if False:
            i = 10
            return i + 15
        if ValueString.upper() == 'TRUE' or ValueString in ['1', '0x1', '0x01']:
            try:
                self.PcdValue = pack(_FORMAT_CHAR[1], 1)
            except:
                EdkLogger.error('BPDG', BuildToolError.FORMAT_INVALID, 'Invalid size or value for PCD %s to pack(File: %s Line: %s).' % (self.PcdCName, self.FileName, self.Lineno))
        else:
            try:
                self.PcdValue = pack(_FORMAT_CHAR[1], 0)
            except:
                EdkLogger.error('BPDG', BuildToolError.FORMAT_INVALID, 'Invalid size or value for PCD %s to pack(File: %s Line: %s).' % (self.PcdCName, self.FileName, self.Lineno))

    def _PackIntValue(self, IntValue, Size):
        if False:
            while True:
                i = 10
        if Size not in _FORMAT_CHAR:
            EdkLogger.error('BPDG', BuildToolError.FORMAT_INVALID, 'Invalid size %d for PCD %s in integer datum size(File: %s Line: %s).' % (Size, self.PcdCName, self.FileName, self.Lineno))
        for (Type, MaxSize) in MAX_SIZE_TYPE.items():
            if Type == 'BOOLEAN':
                continue
            if Size == MaxSize:
                if IntValue < 0:
                    EdkLogger.error('BPDG', BuildToolError.FORMAT_INVALID, "PCD can't be set to negative value %d for PCD %s in %s datum type(File: %s Line: %s)." % (IntValue, self.PcdCName, Type, self.FileName, self.Lineno))
                elif IntValue > MAX_VAL_TYPE[Type]:
                    EdkLogger.error('BPDG', BuildToolError.FORMAT_INVALID, 'Too large PCD value %d for datum type %s for PCD %s(File: %s Line: %s).' % (IntValue, Type, self.PcdCName, self.FileName, self.Lineno))
        try:
            self.PcdValue = pack(_FORMAT_CHAR[Size], IntValue)
        except:
            EdkLogger.error('BPDG', BuildToolError.FORMAT_INVALID, 'Invalid size or value for PCD %s to pack(File: %s Line: %s).' % (self.PcdCName, self.FileName, self.Lineno))

    def _PackPtrValue(self, ValueString, Size):
        if False:
            print('Hello World!')
        if ValueString.startswith('L"') or ValueString.startswith("L'"):
            self._PackUnicode(ValueString, Size)
        elif ValueString.startswith('{') and ValueString.endswith('}'):
            self._PackByteArray(ValueString, Size)
        elif ValueString.startswith('"') and ValueString.endswith('"') or (ValueString.startswith("'") and ValueString.endswith("'")):
            self._PackString(ValueString, Size)
        else:
            EdkLogger.error('BPDG', BuildToolError.FORMAT_INVALID, 'Invalid VOID* type PCD %s value %s (File: %s Line: %s)' % (self.PcdCName, ValueString, self.FileName, self.Lineno))

    def _PackString(self, ValueString, Size):
        if False:
            print('Hello World!')
        if Size < 0:
            EdkLogger.error('BPDG', BuildToolError.FORMAT_INVALID, 'Invalid parameter Size %s of PCD %s!(File: %s Line: %s)' % (self.PcdBinSize, self.PcdCName, self.FileName, self.Lineno))
        if ValueString == '':
            EdkLogger.error('BPDG', BuildToolError.FORMAT_INVALID, 'Invalid parameter ValueString %s of PCD %s!(File: %s Line: %s)' % (self.PcdUnpackValue, self.PcdCName, self.FileName, self.Lineno))
        QuotedFlag = True
        if ValueString.startswith("'"):
            QuotedFlag = False
        ValueString = ValueString[1:-1]
        if QuotedFlag and len(ValueString) + 1 > Size or (not QuotedFlag and len(ValueString) > Size):
            EdkLogger.error('BPDG', BuildToolError.RESOURCE_OVERFLOW, 'PCD value string %s is exceed to size %d(File: %s Line: %s)' % (ValueString, Size, self.FileName, self.Lineno))
        try:
            self.PcdValue = pack('%ds' % Size, ValueString.encode('utf-8'))
        except:
            EdkLogger.error('BPDG', BuildToolError.FORMAT_INVALID, 'Invalid size or value for PCD %s to pack(File: %s Line: %s).' % (self.PcdCName, self.FileName, self.Lineno))

    def _PackByteArray(self, ValueString, Size):
        if False:
            for i in range(10):
                print('nop')
        if Size < 0:
            EdkLogger.error('BPDG', BuildToolError.FORMAT_INVALID, 'Invalid parameter Size %s of PCD %s!(File: %s Line: %s)' % (self.PcdBinSize, self.PcdCName, self.FileName, self.Lineno))
        if ValueString == '':
            EdkLogger.error('BPDG', BuildToolError.FORMAT_INVALID, 'Invalid parameter ValueString %s of PCD %s!(File: %s Line: %s)' % (self.PcdUnpackValue, self.PcdCName, self.FileName, self.Lineno))
        ValueString = ValueString.strip()
        ValueString = ValueString.lstrip('{').strip('}')
        ValueList = ValueString.split(',')
        ValueList = [item.strip() for item in ValueList]
        if len(ValueList) > Size:
            EdkLogger.error('BPDG', BuildToolError.RESOURCE_OVERFLOW, 'The byte array %s is too large for size %d(File: %s Line: %s)' % (ValueString, Size, self.FileName, self.Lineno))
        ReturnArray = array.array('B')
        for Index in range(len(ValueList)):
            Value = None
            if ValueList[Index].lower().startswith('0x'):
                try:
                    Value = int(ValueList[Index], 16)
                except:
                    EdkLogger.error('BPDG', BuildToolError.FORMAT_INVALID, 'The value item %s in byte array %s is an invalid HEX value.(File: %s Line: %s)' % (ValueList[Index], ValueString, self.FileName, self.Lineno))
            else:
                try:
                    Value = int(ValueList[Index], 10)
                except:
                    EdkLogger.error('BPDG', BuildToolError.FORMAT_INVALID, 'The value item %s in byte array %s is an invalid DECIMAL value.(File: %s Line: %s)' % (ValueList[Index], ValueString, self.FileName, self.Lineno))
            if Value > 255:
                EdkLogger.error('BPDG', BuildToolError.FORMAT_INVALID, 'The value item %s in byte array %s do not in range 0 ~ 0xFF(File: %s Line: %s)' % (ValueList[Index], ValueString, self.FileName, self.Lineno))
            ReturnArray.append(Value)
        for Index in range(len(ValueList), Size):
            ReturnArray.append(0)
        self.PcdValue = ReturnArray.tolist()

    def _PackUnicode(self, UnicodeString, Size):
        if False:
            for i in range(10):
                print('nop')
        if Size < 0:
            EdkLogger.error('BPDG', BuildToolError.FORMAT_INVALID, 'Invalid parameter Size %s of PCD %s!(File: %s Line: %s)' % (self.PcdBinSize, self.PcdCName, self.FileName, self.Lineno))
        QuotedFlag = True
        if UnicodeString.startswith("L'"):
            QuotedFlag = False
        UnicodeString = UnicodeString[2:-1]
        if QuotedFlag and (len(UnicodeString) + 1) * 2 > Size or (not QuotedFlag and len(UnicodeString) * 2 > Size):
            EdkLogger.error('BPDG', BuildToolError.RESOURCE_OVERFLOW, 'The size of unicode string %s is too larger for size %s(File: %s Line: %s)' % (UnicodeString, Size, self.FileName, self.Lineno))
        ReturnArray = array.array('B')
        for Value in UnicodeString:
            try:
                ReturnArray.append(ord(Value))
                ReturnArray.append(0)
            except:
                EdkLogger.error('BPDG', BuildToolError.FORMAT_INVALID, 'Invalid unicode character %s in unicode string %s(File: %s Line: %s)' % (Value, UnicodeString, self.FileName, self.Lineno))
        for Index in range(len(UnicodeString) * 2, Size):
            ReturnArray.append(0)
        self.PcdValue = ReturnArray.tolist()

class GenVPD:

    def __init__(self, InputFileName, MapFileName, VpdFileName):
        if False:
            i = 10
            return i + 15
        self.InputFileName = InputFileName
        self.MapFileName = MapFileName
        self.VpdFileName = VpdFileName
        self.FileLinesList = []
        self.PcdFixedOffsetSizeList = []
        self.PcdUnknownOffsetList = []
        try:
            fInputfile = open(InputFileName, 'r')
            try:
                self.FileLinesList = fInputfile.readlines()
            except:
                EdkLogger.error('BPDG', BuildToolError.FILE_READ_FAILURE, 'File read failed for %s' % InputFileName, None)
            finally:
                fInputfile.close()
        except:
            EdkLogger.error('BPDG', BuildToolError.FILE_OPEN_FAILURE, 'File open failed for %s' % InputFileName, None)

    def ParserInputFile(self):
        if False:
            while True:
                i = 10
        count = 0
        for line in self.FileLinesList:
            line = line.strip()
            line = line.rstrip(os.linesep)
            if not line.startswith('#') and len(line) > 1:
                ValueList = ['', '', '', '', '']
                ValueRe = re.compile('\\s*L?\\".*\\|.*\\"\\s*$')
                PtrValue = ValueRe.findall(line)
                ValueUpdateFlag = False
                if len(PtrValue) >= 1:
                    line = re.sub(ValueRe, '', line)
                    ValueUpdateFlag = True
                TokenList = line.split('|')
                ValueList[0:len(TokenList)] = TokenList
                if ValueUpdateFlag:
                    ValueList[4] = PtrValue[0]
                self.FileLinesList[count] = ValueList
                self.FileLinesList[count].append(str(count + 1))
            elif len(line) <= 1:
                self.FileLinesList[count] = None
            else:
                self.FileLinesList[count] = None
            count += 1
        count = 0
        while True:
            try:
                if self.FileLinesList[count] is None:
                    del self.FileLinesList[count]
                else:
                    count += 1
            except:
                break
        if len(self.FileLinesList) == 0:
            EdkLogger.warn('BPDG', BuildToolError.RESOURCE_NOT_AVAILABLE, 'There are no VPD type pcds defined in DSC file, Please check it.')
        count = 0
        for line in self.FileLinesList:
            if line is not None:
                PCD = PcdEntry(line[0], line[1], line[2], line[3], line[4], line[5], self.InputFileName)
                PCD.PcdCName = PCD.PcdCName.strip(' ')
                PCD.SkuId = PCD.SkuId.strip(' ')
                PCD.PcdOffset = PCD.PcdOffset.strip(' ')
                PCD.PcdSize = PCD.PcdSize.strip(' ')
                PCD.PcdValue = PCD.PcdValue.strip(' ')
                PCD.Lineno = PCD.Lineno.strip(' ')
                PCD.PcdUnpackValue = str(PCD.PcdValue)
                PackSize = None
                try:
                    PackSize = int(PCD.PcdSize, 10)
                    PCD.PcdBinSize = PackSize
                except:
                    try:
                        PackSize = int(PCD.PcdSize, 16)
                        PCD.PcdBinSize = PackSize
                    except:
                        EdkLogger.error('BPDG', BuildToolError.FORMAT_INVALID, 'Invalid PCD size value %s at file: %s line: %s' % (PCD.PcdSize, self.InputFileName, PCD.Lineno))
                PCD.PcdOccupySize = PCD.PcdBinSize
                if PCD.PcdUnpackValue.startswith('{'):
                    Alignment = 8
                elif PCD.PcdUnpackValue.startswith('L'):
                    Alignment = 2
                else:
                    Alignment = 1
                PCD.Alignment = Alignment
                if PCD.PcdOffset != TAB_STAR:
                    if PCD.PcdOccupySize % Alignment != 0:
                        if PCD.PcdUnpackValue.startswith('{'):
                            EdkLogger.warn('BPDG', 'The offset value of PCD %s is not 8-byte aligned!' % PCD.PcdCName, File=self.InputFileName)
                        else:
                            EdkLogger.error('BPDG', BuildToolError.FORMAT_INVALID, 'The offset value of PCD %s should be %s-byte aligned.' % (PCD.PcdCName, Alignment))
                elif PCD.PcdOccupySize % Alignment != 0:
                    PCD.PcdOccupySize = (PCD.PcdOccupySize // Alignment + 1) * Alignment
                PackSize = PCD.PcdOccupySize
                if PCD._IsBoolean(PCD.PcdValue, PCD.PcdSize):
                    PCD._PackBooleanValue(PCD.PcdValue)
                    self.FileLinesList[count] = PCD
                    count += 1
                    continue
                IsInteger = True
                PackValue = None
                try:
                    PackValue = int(PCD.PcdValue)
                except:
                    try:
                        PackValue = int(PCD.PcdValue, 16)
                    except:
                        IsInteger = False
                if IsInteger:
                    PCD._PackIntValue(PackValue, PackSize)
                else:
                    PCD._PackPtrValue(PCD.PcdValue, PackSize)
                self.FileLinesList[count] = PCD
                count += 1
            else:
                continue

    def FormatFileLine(self):
        if False:
            for i in range(10):
                print('nop')
        for eachPcd in self.FileLinesList:
            if eachPcd.PcdOffset != TAB_STAR:
                self.PcdFixedOffsetSizeList.append(eachPcd)
            else:
                self.PcdUnknownOffsetList.append(eachPcd)

    def FixVpdOffset(self):
        if False:
            return 10
        self.PcdFixedOffsetSizeList.sort(key=lambda x: x.PcdBinOffset)
        self.PcdUnknownOffsetList.sort(key=lambda x: x.PcdBinSize)
        index = 0
        for pcd in self.PcdUnknownOffsetList:
            index += 1
            if pcd.PcdCName == '.'.join(('gEfiMdeModulePkgTokenSpaceGuid', 'PcdNvStoreDefaultValueBuffer')):
                if index != len(self.PcdUnknownOffsetList):
                    for i in range(len(self.PcdUnknownOffsetList) - index):
                        (self.PcdUnknownOffsetList[index + i - 1], self.PcdUnknownOffsetList[index + i]) = (self.PcdUnknownOffsetList[index + i], self.PcdUnknownOffsetList[index + i - 1])
        if len(self.PcdFixedOffsetSizeList) == 0 and len(self.PcdUnknownOffsetList) != 0:
            NowOffset = 0
            for Pcd in self.PcdUnknownOffsetList:
                if NowOffset % Pcd.Alignment != 0:
                    NowOffset = (NowOffset // Pcd.Alignment + 1) * Pcd.Alignment
                Pcd.PcdBinOffset = NowOffset
                Pcd.PcdOffset = str(hex(Pcd.PcdBinOffset))
                NowOffset += Pcd.PcdOccupySize
            self.PcdFixedOffsetSizeList = self.PcdUnknownOffsetList
            return
        if self.PcdFixedOffsetSizeList[0].PcdBinOffset != 0:
            EdkLogger.warn('BPDG', 'The offset of VPD type pcd should start with 0, please check it.', None)
        lenOfList = len(self.PcdFixedOffsetSizeList)
        count = 0
        while count < lenOfList - 1:
            PcdNow = self.PcdFixedOffsetSizeList[count]
            PcdNext = self.PcdFixedOffsetSizeList[count + 1]
            if PcdNow.PcdBinOffset == PcdNext.PcdBinOffset:
                EdkLogger.error('BPDG', BuildToolError.ATTRIBUTE_GET_FAILURE, 'The offset of %s at line: %s is same with %s at line: %s in file %s' % (PcdNow.PcdCName, PcdNow.Lineno, PcdNext.PcdCName, PcdNext.Lineno, PcdNext.FileName), None)
            if PcdNow.PcdBinOffset + PcdNow.PcdOccupySize > PcdNext.PcdBinOffset:
                EdkLogger.error('BPDG', BuildToolError.ATTRIBUTE_GET_FAILURE, 'The offset of %s at line: %s is overlapped with %s at line: %s in file %s' % (PcdNow.PcdCName, PcdNow.Lineno, PcdNext.PcdCName, PcdNext.Lineno, PcdNext.FileName), None)
            if PcdNow.PcdBinOffset + PcdNow.PcdOccupySize < PcdNext.PcdBinOffset:
                EdkLogger.warn('BPDG', BuildToolError.ATTRIBUTE_GET_FAILURE, 'The offsets have free space of between %s at line: %s and %s at line: %s in file %s' % (PcdNow.PcdCName, PcdNow.Lineno, PcdNext.PcdCName, PcdNext.Lineno, PcdNext.FileName), None)
            count += 1
        LastOffset = self.PcdFixedOffsetSizeList[0].PcdBinOffset
        FixOffsetSizeListCount = 0
        lenOfList = len(self.PcdFixedOffsetSizeList)
        lenOfUnfixedList = len(self.PcdUnknownOffsetList)
        while FixOffsetSizeListCount < lenOfList:
            eachFixedPcd = self.PcdFixedOffsetSizeList[FixOffsetSizeListCount]
            NowOffset = eachFixedPcd.PcdBinOffset
            if LastOffset < NowOffset:
                if lenOfUnfixedList != 0:
                    countOfUnfixedList = 0
                    while countOfUnfixedList < lenOfUnfixedList:
                        eachUnfixedPcd = self.PcdUnknownOffsetList[countOfUnfixedList]
                        needFixPcdSize = eachUnfixedPcd.PcdOccupySize
                        if eachUnfixedPcd.PcdOffset == TAB_STAR:
                            if LastOffset % eachUnfixedPcd.Alignment != 0:
                                LastOffset = (LastOffset // eachUnfixedPcd.Alignment + 1) * eachUnfixedPcd.Alignment
                            if needFixPcdSize <= NowOffset - LastOffset:
                                eachUnfixedPcd.PcdOffset = str(hex(LastOffset))
                                eachUnfixedPcd.PcdBinOffset = LastOffset
                                self.PcdFixedOffsetSizeList.insert(FixOffsetSizeListCount, eachUnfixedPcd)
                                self.PcdUnknownOffsetList.pop(countOfUnfixedList)
                                lenOfList += 1
                                FixOffsetSizeListCount += 1
                                lenOfUnfixedList -= 1
                                LastOffset += needFixPcdSize
                            else:
                                LastOffset = NowOffset + self.PcdFixedOffsetSizeList[FixOffsetSizeListCount].PcdOccupySize
                                FixOffsetSizeListCount += 1
                                break
                else:
                    FixOffsetSizeListCount = lenOfList
            elif LastOffset == NowOffset:
                LastOffset = NowOffset + eachFixedPcd.PcdOccupySize
                FixOffsetSizeListCount += 1
            else:
                EdkLogger.error('BPDG', BuildToolError.ATTRIBUTE_NOT_AVAILABLE, 'The offset value definition has overlapped at pcd: %s, its offset is: %s, in file: %s line: %s' % (eachFixedPcd.PcdCName, eachFixedPcd.PcdOffset, eachFixedPcd.InputFileName, eachFixedPcd.Lineno), None)
                FixOffsetSizeListCount += 1
        lenOfUnfixedList = len(self.PcdUnknownOffsetList)
        lenOfList = len(self.PcdFixedOffsetSizeList)
        while lenOfUnfixedList > 0:
            LastPcd = self.PcdFixedOffsetSizeList[lenOfList - 1]
            NeedFixPcd = self.PcdUnknownOffsetList[0]
            NeedFixPcd.PcdBinOffset = LastPcd.PcdBinOffset + LastPcd.PcdOccupySize
            if NeedFixPcd.PcdBinOffset % NeedFixPcd.Alignment != 0:
                NeedFixPcd.PcdBinOffset = (NeedFixPcd.PcdBinOffset // NeedFixPcd.Alignment + 1) * NeedFixPcd.Alignment
            NeedFixPcd.PcdOffset = str(hex(NeedFixPcd.PcdBinOffset))
            self.PcdFixedOffsetSizeList.insert(lenOfList, NeedFixPcd)
            self.PcdUnknownOffsetList.pop(0)
            lenOfList += 1
            lenOfUnfixedList -= 1

    def GenerateVpdFile(self, MapFileName, BinFileName):
        if False:
            for i in range(10):
                print('nop')
        try:
            fVpdFile = open(BinFileName, 'wb')
        except:
            EdkLogger.error('BPDG', BuildToolError.FILE_OPEN_FAILURE, 'File open failed for %s' % self.VpdFileName, None)
        try:
            fMapFile = open(MapFileName, 'w')
        except:
            EdkLogger.error('BPDG', BuildToolError.FILE_OPEN_FAILURE, 'File open failed for %s' % self.MapFileName, None)
        fStringIO = BytesIO()
        try:
            fMapFile.write(st.MAP_FILE_COMMENT_TEMPLATE + '\n')
        except:
            EdkLogger.error('BPDG', BuildToolError.FILE_WRITE_FAILURE, 'Write data to file %s failed, please check whether the file been locked or using by other applications.' % self.MapFileName, None)
        for eachPcd in self.PcdFixedOffsetSizeList:
            try:
                fMapFile.write('%s | %s | %s | %s | %s  \n' % (eachPcd.PcdCName, eachPcd.SkuId, eachPcd.PcdOffset, eachPcd.PcdSize, eachPcd.PcdUnpackValue))
            except:
                EdkLogger.error('BPDG', BuildToolError.FILE_WRITE_FAILURE, 'Write data to file %s failed, please check whether the file been locked or using by other applications.' % self.MapFileName, None)
            fStringIO.seek(eachPcd.PcdBinOffset)
            if isinstance(eachPcd.PcdValue, list):
                for i in range(len(eachPcd.PcdValue)):
                    Value = eachPcd.PcdValue[i:i + 1]
                    if isinstance(bytes(Value), str):
                        fStringIO.write(chr(Value[0]))
                    else:
                        fStringIO.write(bytes(Value))
            else:
                fStringIO.write(eachPcd.PcdValue)
        try:
            fVpdFile.write(fStringIO.getvalue())
        except:
            EdkLogger.error('BPDG', BuildToolError.FILE_WRITE_FAILURE, 'Write data to file %s failed, please check whether the file been locked or using by other applications.' % self.VpdFileName, None)
        fStringIO.close()
        fVpdFile.close()
        fMapFile.close()