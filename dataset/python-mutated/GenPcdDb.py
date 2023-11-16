from __future__ import absolute_import
from io import BytesIO
from Common.Misc import *
from Common.StringUtils import StringToArray
from struct import pack
from .ValidCheckingInfoObject import VAR_CHECK_PCD_VARIABLE_TAB_CONTAINER
from .ValidCheckingInfoObject import VAR_CHECK_PCD_VARIABLE_TAB
from .ValidCheckingInfoObject import GetValidationObject
from Common.VariableAttributes import VariableAttributes
import copy
from struct import unpack
from Common.DataType import *
from Common import GlobalData
from Common import EdkLogger
import Common.LongFilePathOs as os
DATABASE_VERSION = 7
gPcdDatabaseAutoGenC = TemplateString('\n//\n// External PCD database debug information\n//\n#if 0\n${PHASE}_PCD_DATABASE_INIT g${PHASE}PcdDbInit = {\n  /* SkuIdTable */\n  { ${BEGIN}${SKUID_VALUE}, ${END} },\n${BEGIN}  { ${INIT_VALUE_UINT64} }, /*  ${INIT_CNAME_DECL_UINT64}_${INIT_GUID_DECL_UINT64}[${INIT_NUMSKUS_DECL_UINT64}] */\n${END}\n${BEGIN}  ${VARDEF_VALUE_UINT64}, /* ${VARDEF_CNAME_UINT64}_${VARDEF_GUID_UINT64}_VariableDefault_${VARDEF_SKUID_UINT64} */\n${END}\n${BEGIN}  { ${INIT_VALUE_UINT32} }, /*  ${INIT_CNAME_DECL_UINT32}_${INIT_GUID_DECL_UINT32}[${INIT_NUMSKUS_DECL_UINT32}] */\n${END}\n${BEGIN}  ${VARDEF_VALUE_UINT32}, /* ${VARDEF_CNAME_UINT32}_${VARDEF_GUID_UINT32}_VariableDefault_${VARDEF_SKUID_UINT32} */\n${END}\n  /* VPD */\n${BEGIN}  { ${VPD_HEAD_VALUE} }, /* ${VPD_HEAD_CNAME_DECL}_${VPD_HEAD_GUID_DECL}[${VPD_HEAD_NUMSKUS_DECL}] */\n${END}\n  /* ExMapTable */\n  {\n${BEGIN}    { ${EXMAPPING_TABLE_EXTOKEN}, ${EXMAPPING_TABLE_LOCAL_TOKEN}, ${EXMAPPING_TABLE_GUID_INDEX} },\n${END}\n  },\n  /* LocalTokenNumberTable */\n  {\n${BEGIN}    offsetof(${PHASE}_PCD_DATABASE, ${TOKEN_INIT}.${TOKEN_CNAME}_${TOKEN_GUID}${VARDEF_HEADER}) | ${TOKEN_TYPE},\n${END}\n  },\n  /* GuidTable */\n  {\n${BEGIN}    ${GUID_STRUCTURE},\n${END}\n  },\n${BEGIN}  { ${STRING_HEAD_VALUE} }, /* ${STRING_HEAD_CNAME_DECL}_${STRING_HEAD_GUID_DECL}[${STRING_HEAD_NUMSKUS_DECL}] */\n${END}\n${BEGIN}  /* ${VARIABLE_HEAD_CNAME_DECL}_${VARIABLE_HEAD_GUID_DECL}_Variable_Header[${VARIABLE_HEAD_NUMSKUS_DECL}] */\n  {\n    ${VARIABLE_HEAD_VALUE}\n  },\n${END}\n/* SkuHead */\n  {\n  ${BEGIN} offsetof (${PHASE}_PCD_DATABASE, ${TOKEN_INIT}.${TOKEN_CNAME}_${TOKEN_GUID}${VARDEF_HEADER}) | ${TOKEN_TYPE}, /* */\n           offsetof (${PHASE}_PCD_DATABASE, ${TOKEN_INIT}.SkuHead)  /* */\n  ${END}\n  },\n /* StringTable */\n${BEGIN}  ${STRING_TABLE_VALUE}, /* ${STRING_TABLE_CNAME}_${STRING_TABLE_GUID} */\n${END}\n  /* SizeTable */\n  {\n${BEGIN}    ${SIZE_TABLE_MAXIMUM_LENGTH}, ${SIZE_TABLE_CURRENT_LENGTH}, /* ${SIZE_TABLE_CNAME}_${SIZE_TABLE_GUID} */\n${END}\n  },\n${BEGIN}  { ${INIT_VALUE_UINT16} }, /*  ${INIT_CNAME_DECL_UINT16}_${INIT_GUID_DECL_UINT16}[${INIT_NUMSKUS_DECL_UINT16}] */\n${END}\n${BEGIN}  ${VARDEF_VALUE_UINT16}, /* ${VARDEF_CNAME_UINT16}_${VARDEF_GUID_UINT16}_VariableDefault_${VARDEF_SKUID_UINT16} */\n${END}\n${BEGIN}  { ${INIT_VALUE_UINT8} }, /*  ${INIT_CNAME_DECL_UINT8}_${INIT_GUID_DECL_UINT8}[${INIT_NUMSKUS_DECL_UINT8}] */\n${END}\n${BEGIN}  ${VARDEF_VALUE_UINT8}, /* ${VARDEF_CNAME_UINT8}_${VARDEF_GUID_UINT8}_VariableDefault_${VARDEF_SKUID_UINT8} */\n${END}\n${BEGIN}  { ${INIT_VALUE_BOOLEAN} }, /*  ${INIT_CNAME_DECL_BOOLEAN}_${INIT_GUID_DECL_BOOLEAN}[${INIT_NUMSKUS_DECL_BOOLEAN}] */\n${END}\n${BEGIN}  ${VARDEF_VALUE_BOOLEAN}, /* ${VARDEF_CNAME_BOOLEAN}_${VARDEF_GUID_BOOLEAN}_VariableDefault_${VARDEF_SKUID_BOOLEAN} */\n${END}\n  ${SYSTEM_SKU_ID_VALUE}\n};\n#endif\n')
gPcdPhaseMap = {'PEI_PCD_DRIVER': 'PEI', 'DXE_PCD_DRIVER': 'DXE'}
gPcdDatabaseAutoGenH = TemplateString('\n#define PCD_${PHASE}_SERVICE_DRIVER_VERSION         ${SERVICE_DRIVER_VERSION}\n\n//\n// External PCD database debug information\n//\n#if 0\n#define ${PHASE}_GUID_TABLE_SIZE                ${GUID_TABLE_SIZE}\n#define ${PHASE}_STRING_TABLE_SIZE              ${STRING_TABLE_SIZE}\n#define ${PHASE}_SKUID_TABLE_SIZE               ${SKUID_TABLE_SIZE}\n#define ${PHASE}_LOCAL_TOKEN_NUMBER_TABLE_SIZE  ${LOCAL_TOKEN_NUMBER_TABLE_SIZE}\n#define ${PHASE}_LOCAL_TOKEN_NUMBER             ${LOCAL_TOKEN_NUMBER}\n#define ${PHASE}_EXMAPPING_TABLE_SIZE           ${EXMAPPING_TABLE_SIZE}\n#define ${PHASE}_EX_TOKEN_NUMBER                ${EX_TOKEN_NUMBER}\n#define ${PHASE}_SIZE_TABLE_SIZE                ${SIZE_TABLE_SIZE}\n#define ${PHASE}_GUID_TABLE_EMPTY               ${GUID_TABLE_EMPTY}\n#define ${PHASE}_STRING_TABLE_EMPTY             ${STRING_TABLE_EMPTY}\n#define ${PHASE}_SKUID_TABLE_EMPTY              ${SKUID_TABLE_EMPTY}\n#define ${PHASE}_DATABASE_EMPTY                 ${DATABASE_EMPTY}\n#define ${PHASE}_EXMAP_TABLE_EMPTY              ${EXMAP_TABLE_EMPTY}\n\ntypedef struct {\n  UINT64             SkuIdTable[${PHASE}_SKUID_TABLE_SIZE];\n${BEGIN}  UINT64             ${INIT_CNAME_DECL_UINT64}_${INIT_GUID_DECL_UINT64}[${INIT_NUMSKUS_DECL_UINT64}];\n${END}\n${BEGIN}  UINT64             ${VARDEF_CNAME_UINT64}_${VARDEF_GUID_UINT64}_VariableDefault_${VARDEF_SKUID_UINT64};\n${END}\n${BEGIN}  UINT32             ${INIT_CNAME_DECL_UINT32}_${INIT_GUID_DECL_UINT32}[${INIT_NUMSKUS_DECL_UINT32}];\n${END}\n${BEGIN}  UINT32             ${VARDEF_CNAME_UINT32}_${VARDEF_GUID_UINT32}_VariableDefault_${VARDEF_SKUID_UINT32};\n${END}\n${BEGIN}  VPD_HEAD           ${VPD_HEAD_CNAME_DECL}_${VPD_HEAD_GUID_DECL}[${VPD_HEAD_NUMSKUS_DECL}];\n${END}\n  DYNAMICEX_MAPPING  ExMapTable[${PHASE}_EXMAPPING_TABLE_SIZE];\n  UINT32             LocalTokenNumberTable[${PHASE}_LOCAL_TOKEN_NUMBER_TABLE_SIZE];\n  GUID               GuidTable[${PHASE}_GUID_TABLE_SIZE];\n${BEGIN}  STRING_HEAD        ${STRING_HEAD_CNAME_DECL}_${STRING_HEAD_GUID_DECL}[${STRING_HEAD_NUMSKUS_DECL}];\n${END}\n${BEGIN}  VARIABLE_HEAD      ${VARIABLE_HEAD_CNAME_DECL}_${VARIABLE_HEAD_GUID_DECL}_Variable_Header[${VARIABLE_HEAD_NUMSKUS_DECL}];\n${BEGIN}  UINT8              StringTable${STRING_TABLE_INDEX}[${STRING_TABLE_LENGTH}]; /* ${STRING_TABLE_CNAME}_${STRING_TABLE_GUID} */\n${END}\n  SIZE_INFO          SizeTable[${PHASE}_SIZE_TABLE_SIZE];\n${BEGIN}  UINT16             ${INIT_CNAME_DECL_UINT16}_${INIT_GUID_DECL_UINT16}[${INIT_NUMSKUS_DECL_UINT16}];\n${END}\n${BEGIN}  UINT16             ${VARDEF_CNAME_UINT16}_${VARDEF_GUID_UINT16}_VariableDefault_${VARDEF_SKUID_UINT16};\n${END}\n${BEGIN}  UINT8              ${INIT_CNAME_DECL_UINT8}_${INIT_GUID_DECL_UINT8}[${INIT_NUMSKUS_DECL_UINT8}];\n${END}\n${BEGIN}  UINT8              ${VARDEF_CNAME_UINT8}_${VARDEF_GUID_UINT8}_VariableDefault_${VARDEF_SKUID_UINT8};\n${END}\n${BEGIN}  BOOLEAN            ${INIT_CNAME_DECL_BOOLEAN}_${INIT_GUID_DECL_BOOLEAN}[${INIT_NUMSKUS_DECL_BOOLEAN}];\n${END}\n${BEGIN}  BOOLEAN            ${VARDEF_CNAME_BOOLEAN}_${VARDEF_GUID_BOOLEAN}_VariableDefault_${VARDEF_SKUID_BOOLEAN};\n${END}\n${SYSTEM_SKU_ID}\n} ${PHASE}_PCD_DATABASE_INIT;\n\ntypedef struct {\n${PCD_DATABASE_UNINIT_EMPTY}\n${BEGIN}  UINT64   ${UNINIT_CNAME_DECL_UINT64}_${UNINIT_GUID_DECL_UINT64}[${UNINIT_NUMSKUS_DECL_UINT64}];\n${END}\n${BEGIN}  UINT32   ${UNINIT_CNAME_DECL_UINT32}_${UNINIT_GUID_DECL_UINT32}[${UNINIT_NUMSKUS_DECL_UINT32}];\n${END}\n${BEGIN}  UINT16   ${UNINIT_CNAME_DECL_UINT16}_${UNINIT_GUID_DECL_UINT16}[${UNINIT_NUMSKUS_DECL_UINT16}];\n${END}\n${BEGIN}  UINT8    ${UNINIT_CNAME_DECL_UINT8}_${UNINIT_GUID_DECL_UINT8}[${UNINIT_NUMSKUS_DECL_UINT8}];\n${END}\n${BEGIN}  BOOLEAN  ${UNINIT_CNAME_DECL_BOOLEAN}_${UNINIT_GUID_DECL_BOOLEAN}[${UNINIT_NUMSKUS_DECL_BOOLEAN}];\n${END}\n} ${PHASE}_PCD_DATABASE_UNINIT;\n\ntypedef struct {\n  //GUID                  Signature;  // PcdDataBaseGuid\n  //UINT32                BuildVersion;\n  //UINT32                Length;\n  //SKU_ID                SystemSkuId;       // Current SkuId value.\n  //UINT32                LengthForAllSkus;  // Length of all SKU PCD DB\n  //UINT32                UninitDataBaseSize;// Total size for PCD those default value with 0.\n  //TABLE_OFFSET          LocalTokenNumberTableOffset;\n  //TABLE_OFFSET          ExMapTableOffset;\n  //TABLE_OFFSET          GuidTableOffset;\n  //TABLE_OFFSET          StringTableOffset;\n  //TABLE_OFFSET          SizeTableOffset;\n  //TABLE_OFFSET          SkuIdTableOffset;\n  //TABLE_OFFSET          PcdNameTableOffset;\n  //UINT16                LocalTokenCount;  // LOCAL_TOKEN_NUMBER for all\n  //UINT16                ExTokenCount;     // EX_TOKEN_NUMBER for DynamicEx\n  //UINT16                GuidTableCount;   // The Number of Guid in GuidTable\n  //UINT8                 Pad[6];\n  ${PHASE}_PCD_DATABASE_INIT    Init;\n  ${PHASE}_PCD_DATABASE_UNINIT  Uninit;\n} ${PHASE}_PCD_DATABASE;\n\n#define ${PHASE}_NEX_TOKEN_NUMBER (${PHASE}_LOCAL_TOKEN_NUMBER - ${PHASE}_EX_TOKEN_NUMBER)\n#endif\n')
gEmptyPcdDatabaseAutoGenC = TemplateString('\n//\n// External PCD database debug information\n//\n#if 0\n${PHASE}_PCD_DATABASE_INIT g${PHASE}PcdDbInit = {\n  /* SkuIdTable */\n  { 0 },\n  /* ExMapTable */\n  {\n    {0, 0, 0}\n  },\n  /* LocalTokenNumberTable */\n  {\n    0\n  },\n  /* GuidTable */\n  {\n    {0x00000000, 0x0000, 0x0000, {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}}\n  },\n  /* StringTable */\n  { 0 },\n  /* SkuHead */\n  {\n    0, 0\n  },\n  /* SizeTable */\n  {\n    0, 0\n  },\n  ${SYSTEM_SKU_ID_VALUE}\n};\n#endif\n')

class DbItemList:

    def __init__(self, ItemSize, DataList=None, RawDataList=None):
        if False:
            while True:
                i = 10
        self.ItemSize = ItemSize
        self.DataList = DataList if DataList else []
        self.RawDataList = RawDataList if RawDataList else []
        self.ListSize = 0

    def GetInterOffset(self, Index):
        if False:
            for i in range(10):
                print('nop')
        Offset = 0
        if self.ItemSize == 0:
            assert Index < len(self.RawDataList)
            for ItemIndex in range(Index):
                Offset += len(self.RawDataList[ItemIndex])
        else:
            Offset = self.ItemSize * Index
        return Offset

    def GetListSize(self):
        if False:
            i = 10
            return i + 15
        if self.ListSize:
            return self.ListSize
        if len(self.RawDataList) == 0:
            self.ListSize = 0
            return self.ListSize
        if self.ItemSize == 0:
            self.ListSize = self.GetInterOffset(len(self.RawDataList) - 1) + len(self.RawDataList[len(self.RawDataList) - 1])
        else:
            self.ListSize = self.ItemSize * len(self.RawDataList)
        return self.ListSize

    def PackData(self):
        if False:
            print('Hello World!')

        def PackGuid(GuidStructureValue):
            if False:
                i = 10
                return i + 15
            GuidString = GuidStructureStringToGuidString(GuidStructureValue)
            return PackGUID(GuidString.split('-'))
        PackStr = PACK_CODE_BY_SIZE[self.ItemSize]
        Buffer = bytearray()
        for Datas in self.RawDataList:
            if type(Datas) in (list, tuple):
                for Data in Datas:
                    if PackStr:
                        Buffer += pack(PackStr, GetIntegerValue(Data))
                    else:
                        Buffer += PackGuid(Data)
            elif PackStr:
                Buffer += pack(PackStr, GetIntegerValue(Datas))
            else:
                Buffer += PackGuid(Datas)
        return Buffer

class DbExMapTblItemList(DbItemList):

    def __init__(self, ItemSize, DataList=None, RawDataList=None):
        if False:
            for i in range(10):
                print('nop')
        DbItemList.__init__(self, ItemSize, DataList, RawDataList)

    def PackData(self):
        if False:
            i = 10
            return i + 15
        Buffer = bytearray()
        PackStr = '=LHH'
        for Datas in self.RawDataList:
            Buffer += pack(PackStr, GetIntegerValue(Datas[0]), GetIntegerValue(Datas[1]), GetIntegerValue(Datas[2]))
        return Buffer

class DbComItemList(DbItemList):

    def __init__(self, ItemSize, DataList=None, RawDataList=None):
        if False:
            while True:
                i = 10
        DbItemList.__init__(self, ItemSize, DataList, RawDataList)

    def GetInterOffset(self, Index):
        if False:
            for i in range(10):
                print('nop')
        Offset = 0
        if self.ItemSize == 0:
            assert False
        else:
            assert Index < len(self.RawDataList)
            for ItemIndex in range(Index):
                Offset += len(self.RawDataList[ItemIndex]) * self.ItemSize
        return Offset

    def GetListSize(self):
        if False:
            print('Hello World!')
        if self.ListSize:
            return self.ListSize
        if self.ItemSize == 0:
            assert False
        elif len(self.RawDataList) == 0:
            self.ListSize = 0
        else:
            self.ListSize = self.GetInterOffset(len(self.RawDataList) - 1) + len(self.RawDataList[len(self.RawDataList) - 1]) * self.ItemSize
        return self.ListSize

    def PackData(self):
        if False:
            print('Hello World!')
        PackStr = PACK_CODE_BY_SIZE[self.ItemSize]
        Buffer = bytearray()
        for DataList in self.RawDataList:
            for Data in DataList:
                if type(Data) in (list, tuple):
                    for SingleData in Data:
                        Buffer += pack(PackStr, GetIntegerValue(SingleData))
                else:
                    Buffer += pack(PackStr, GetIntegerValue(Data))
        return Buffer

class DbVariableTableItemList(DbComItemList):

    def __init__(self, ItemSize, DataList=None, RawDataList=None):
        if False:
            print('Hello World!')
        DbComItemList.__init__(self, ItemSize, DataList, RawDataList)

    def PackData(self):
        if False:
            i = 10
            return i + 15
        PackStr = '=LLHHLHH'
        Buffer = bytearray()
        for DataList in self.RawDataList:
            for Data in DataList:
                Buffer += pack(PackStr, GetIntegerValue(Data[0]), GetIntegerValue(Data[1]), GetIntegerValue(Data[2]), GetIntegerValue(Data[3]), GetIntegerValue(Data[4]), GetIntegerValue(Data[5]), GetIntegerValue(0))
        return Buffer

class DbStringHeadTableItemList(DbItemList):

    def __init__(self, ItemSize, DataList=None, RawDataList=None):
        if False:
            while True:
                i = 10
        DbItemList.__init__(self, ItemSize, DataList, RawDataList)

    def GetInterOffset(self, Index):
        if False:
            for i in range(10):
                print('nop')
        Offset = 0
        if self.ItemSize == 0:
            assert Index < len(self.RawDataList)
            for ItemIndex in range(Index):
                Offset += len(self.RawDataList[ItemIndex])
        else:
            for innerIndex in range(Index):
                if type(self.RawDataList[innerIndex]) in (list, tuple):
                    Offset += len(self.RawDataList[innerIndex]) * self.ItemSize
                else:
                    Offset += self.ItemSize
        return Offset

    def GetListSize(self):
        if False:
            while True:
                i = 10
        if self.ListSize:
            return self.ListSize
        if len(self.RawDataList) == 0:
            self.ListSize = 0
            return self.ListSize
        if self.ItemSize == 0:
            self.ListSize = self.GetInterOffset(len(self.RawDataList) - 1) + len(self.RawDataList[len(self.RawDataList) - 1])
        else:
            for Datas in self.RawDataList:
                if type(Datas) in (list, tuple):
                    self.ListSize += len(Datas) * self.ItemSize
                else:
                    self.ListSize += self.ItemSize
        return self.ListSize

class DbSkuHeadTableItemList(DbItemList):

    def __init__(self, ItemSize, DataList=None, RawDataList=None):
        if False:
            print('Hello World!')
        DbItemList.__init__(self, ItemSize, DataList, RawDataList)

    def PackData(self):
        if False:
            return 10
        PackStr = '=LL'
        Buffer = bytearray()
        for Data in self.RawDataList:
            Buffer += pack(PackStr, GetIntegerValue(Data[0]), GetIntegerValue(Data[1]))
        return Buffer

class DbSizeTableItemList(DbItemList):

    def __init__(self, ItemSize, DataList=None, RawDataList=None):
        if False:
            return 10
        DbItemList.__init__(self, ItemSize, DataList, RawDataList)

    def GetListSize(self):
        if False:
            for i in range(10):
                print('nop')
        length = 0
        for Data in self.RawDataList:
            length += 1 + len(Data[1])
        return length * self.ItemSize

    def PackData(self):
        if False:
            print('Hello World!')
        PackStr = '=H'
        Buffer = bytearray()
        for Data in self.RawDataList:
            Buffer += pack(PackStr, GetIntegerValue(Data[0]))
            for subData in Data[1]:
                Buffer += pack(PackStr, GetIntegerValue(subData))
        return Buffer

class DbStringItemList(DbComItemList):

    def __init__(self, ItemSize, DataList=None, RawDataList=None, LenList=None):
        if False:
            print('Hello World!')
        if DataList is None:
            DataList = []
        if RawDataList is None:
            RawDataList = []
        if LenList is None:
            LenList = []
        assert len(RawDataList) == len(LenList)
        DataList = []
        for Index in range(len(RawDataList)):
            Len = LenList[Index]
            RawDatas = RawDataList[Index]
            assert Len >= len(RawDatas)
            ActualDatas = []
            for i in range(len(RawDatas)):
                ActualDatas.append(RawDatas[i])
            for i in range(len(RawDatas), Len):
                ActualDatas.append(0)
            DataList.append(ActualDatas)
        self.LenList = LenList
        DbComItemList.__init__(self, ItemSize, DataList, RawDataList)

    def GetInterOffset(self, Index):
        if False:
            i = 10
            return i + 15
        Offset = 0
        assert Index < len(self.LenList)
        for ItemIndex in range(Index):
            Offset += self.LenList[ItemIndex]
        return Offset

    def GetListSize(self):
        if False:
            i = 10
            return i + 15
        if self.ListSize:
            return self.ListSize
        if len(self.LenList) == 0:
            self.ListSize = 0
        else:
            self.ListSize = self.GetInterOffset(len(self.LenList) - 1) + self.LenList[len(self.LenList) - 1]
        return self.ListSize

    def PackData(self):
        if False:
            for i in range(10):
                print('nop')
        self.RawDataList = self.DataList
        return DbComItemList.PackData(self)

def GetMatchedIndex(Key1, List1, Key2, List2):
    if False:
        print('Hello World!')
    StartPos = 0
    while StartPos < len(List1):
        Index = List1.index(Key1, StartPos)
        if List2[Index] == Key2:
            return Index
        else:
            StartPos = Index + 1
    return -1

def StringArrayToList(StringArray):
    if False:
        i = 10
        return i + 15
    StringArray = StringArray[1:-1]
    StringArray = '[' + StringArray + ']'
    return eval(StringArray)

def GetTokenTypeValue(TokenType):
    if False:
        while True:
            i = 10
    TokenTypeDict = {'PCD_TYPE_SHIFT': 28, 'PCD_TYPE_DATA': 0 << 28, 'PCD_TYPE_HII': 8 << 28, 'PCD_TYPE_VPD': 4 << 28, 'PCD_TYPE_STRING': 1 << 28, 'PCD_DATUM_TYPE_SHIFT': 24, 'PCD_DATUM_TYPE_POINTER': 0 << 24, 'PCD_DATUM_TYPE_UINT8': 1 << 24, 'PCD_DATUM_TYPE_UINT16': 2 << 24, 'PCD_DATUM_TYPE_UINT32': 4 << 24, 'PCD_DATUM_TYPE_UINT64': 8 << 24, 'PCD_DATUM_TYPE_SHIFT2': 20, 'PCD_DATUM_TYPE_UINT8_BOOLEAN': 1 << 20 | 1 << 24}
    return eval(TokenType, TokenTypeDict)

def BuildExDataBase(Dict):
    if False:
        i = 10
        return i + 15
    InitValueUint64 = Dict['INIT_DB_VALUE_UINT64']
    DbInitValueUint64 = DbComItemList(8, RawDataList=InitValueUint64)
    VardefValueUint64 = Dict['VARDEF_DB_VALUE_UINT64']
    DbVardefValueUint64 = DbItemList(8, RawDataList=VardefValueUint64)
    InitValueUint32 = Dict['INIT_DB_VALUE_UINT32']
    DbInitValueUint32 = DbComItemList(4, RawDataList=InitValueUint32)
    VardefValueUint32 = Dict['VARDEF_DB_VALUE_UINT32']
    DbVardefValueUint32 = DbItemList(4, RawDataList=VardefValueUint32)
    VpdHeadValue = Dict['VPD_DB_VALUE']
    DbVpdHeadValue = DbComItemList(4, RawDataList=VpdHeadValue)
    ExMapTable = list(zip(Dict['EXMAPPING_TABLE_EXTOKEN'], Dict['EXMAPPING_TABLE_LOCAL_TOKEN'], Dict['EXMAPPING_TABLE_GUID_INDEX']))
    DbExMapTable = DbExMapTblItemList(8, RawDataList=ExMapTable)
    LocalTokenNumberTable = Dict['LOCAL_TOKEN_NUMBER_DB_VALUE']
    DbLocalTokenNumberTable = DbItemList(4, RawDataList=LocalTokenNumberTable)
    GuidTable = Dict['GUID_STRUCTURE']
    DbGuidTable = DbItemList(16, RawDataList=GuidTable)
    StringHeadValue = Dict['STRING_DB_VALUE']
    DbStringHeadValue = DbStringHeadTableItemList(4, RawDataList=StringHeadValue)
    VariableTable = Dict['VARIABLE_DB_VALUE']
    DbVariableTable = DbVariableTableItemList(20, RawDataList=VariableTable)
    NumberOfSkuEnabledPcd = GetIntegerValue(Dict['SKU_HEAD_SIZE'])
    Dict['STRING_TABLE_DB_VALUE'] = [StringArrayToList(x) for x in Dict['STRING_TABLE_VALUE']]
    StringTableValue = Dict['STRING_TABLE_DB_VALUE']
    StringTableLen = Dict['STRING_TABLE_LENGTH']
    DbStringTableLen = DbStringItemList(0, RawDataList=StringTableValue, LenList=StringTableLen)
    PcdTokenTable = Dict['PCD_TOKENSPACE']
    PcdTokenLen = Dict['PCD_TOKENSPACE_LENGTH']
    PcdTokenTableValue = [StringArrayToList(x) for x in Dict['PCD_TOKENSPACE']]
    DbPcdTokenTable = DbStringItemList(0, RawDataList=PcdTokenTableValue, LenList=PcdTokenLen)
    PcdCNameTable = Dict['PCD_CNAME']
    PcdCNameLen = Dict['PCD_CNAME_LENGTH']
    PcdCNameTableValue = [StringArrayToList(x) for x in Dict['PCD_CNAME']]
    DbPcdCNameTable = DbStringItemList(0, RawDataList=PcdCNameTableValue, LenList=PcdCNameLen)
    PcdNameOffsetTable = Dict['PCD_NAME_OFFSET']
    DbPcdNameOffsetTable = DbItemList(4, RawDataList=PcdNameOffsetTable)
    SizeTableValue = list(zip(Dict['SIZE_TABLE_MAXIMUM_LENGTH'], Dict['SIZE_TABLE_CURRENT_LENGTH']))
    DbSizeTableValue = DbSizeTableItemList(2, RawDataList=SizeTableValue)
    InitValueUint16 = Dict['INIT_DB_VALUE_UINT16']
    DbInitValueUint16 = DbComItemList(2, RawDataList=InitValueUint16)
    VardefValueUint16 = Dict['VARDEF_DB_VALUE_UINT16']
    DbVardefValueUint16 = DbItemList(2, RawDataList=VardefValueUint16)
    InitValueUint8 = Dict['INIT_DB_VALUE_UINT8']
    DbInitValueUint8 = DbComItemList(1, RawDataList=InitValueUint8)
    VardefValueUint8 = Dict['VARDEF_DB_VALUE_UINT8']
    DbVardefValueUint8 = DbItemList(1, RawDataList=VardefValueUint8)
    InitValueBoolean = Dict['INIT_DB_VALUE_BOOLEAN']
    DbInitValueBoolean = DbComItemList(1, RawDataList=InitValueBoolean)
    VardefValueBoolean = Dict['VARDEF_DB_VALUE_BOOLEAN']
    DbVardefValueBoolean = DbItemList(1, RawDataList=VardefValueBoolean)
    SkuidValue = Dict['SKUID_VALUE']
    DbSkuidValue = DbItemList(8, RawDataList=SkuidValue)
    UnInitValueUint64 = Dict['UNINIT_GUID_DECL_UINT64']
    DbUnInitValueUint64 = DbItemList(8, RawDataList=UnInitValueUint64)
    UnInitValueUint32 = Dict['UNINIT_GUID_DECL_UINT32']
    DbUnInitValueUint32 = DbItemList(4, RawDataList=UnInitValueUint32)
    UnInitValueUint16 = Dict['UNINIT_GUID_DECL_UINT16']
    DbUnInitValueUint16 = DbItemList(2, RawDataList=UnInitValueUint16)
    UnInitValueUint8 = Dict['UNINIT_GUID_DECL_UINT8']
    DbUnInitValueUint8 = DbItemList(1, RawDataList=UnInitValueUint8)
    UnInitValueBoolean = Dict['UNINIT_GUID_DECL_BOOLEAN']
    DbUnInitValueBoolean = DbItemList(1, RawDataList=UnInitValueBoolean)
    PcdTokenNumberMap = Dict['PCD_ORDER_TOKEN_NUMBER_MAP']
    DbNameTotle = ['SkuidValue', 'InitValueUint64', 'VardefValueUint64', 'InitValueUint32', 'VardefValueUint32', 'VpdHeadValue', 'ExMapTable', 'LocalTokenNumberTable', 'GuidTable', 'StringHeadValue', 'PcdNameOffsetTable', 'VariableTable', 'StringTableLen', 'PcdTokenTable', 'PcdCNameTable', 'SizeTableValue', 'InitValueUint16', 'VardefValueUint16', 'InitValueUint8', 'VardefValueUint8', 'InitValueBoolean', 'VardefValueBoolean', 'UnInitValueUint64', 'UnInitValueUint32', 'UnInitValueUint16', 'UnInitValueUint8', 'UnInitValueBoolean']
    DbTotal = [SkuidValue, InitValueUint64, VardefValueUint64, InitValueUint32, VardefValueUint32, VpdHeadValue, ExMapTable, LocalTokenNumberTable, GuidTable, StringHeadValue, PcdNameOffsetTable, VariableTable, StringTableLen, PcdTokenTable, PcdCNameTable, SizeTableValue, InitValueUint16, VardefValueUint16, InitValueUint8, VardefValueUint8, InitValueBoolean, VardefValueBoolean, UnInitValueUint64, UnInitValueUint32, UnInitValueUint16, UnInitValueUint8, UnInitValueBoolean]
    DbItemTotal = [DbSkuidValue, DbInitValueUint64, DbVardefValueUint64, DbInitValueUint32, DbVardefValueUint32, DbVpdHeadValue, DbExMapTable, DbLocalTokenNumberTable, DbGuidTable, DbStringHeadValue, DbPcdNameOffsetTable, DbVariableTable, DbStringTableLen, DbPcdTokenTable, DbPcdCNameTable, DbSizeTableValue, DbInitValueUint16, DbVardefValueUint16, DbInitValueUint8, DbVardefValueUint8, DbInitValueBoolean, DbVardefValueBoolean, DbUnInitValueUint64, DbUnInitValueUint32, DbUnInitValueUint16, DbUnInitValueUint8, DbUnInitValueBoolean]
    InitTableNum = DbNameTotle.index('VardefValueBoolean') + 1
    FixedHeaderLen = 80
    SkuIdTableOffset = FixedHeaderLen
    for DbIndex in range(len(DbTotal)):
        if DbTotal[DbIndex] is SkuidValue:
            break
        SkuIdTableOffset += DbItemTotal[DbIndex].GetListSize()
    for (LocalTokenNumberTableIndex, (Offset, Table)) in enumerate(LocalTokenNumberTable):
        DbIndex = 0
        DbOffset = FixedHeaderLen
        for DbIndex in range(len(DbTotal)):
            if DbTotal[DbIndex] is Table:
                DbOffset += DbItemTotal[DbIndex].GetInterOffset(Offset)
                break
            DbOffset += DbItemTotal[DbIndex].GetListSize()
            if DbIndex + 1 == InitTableNum:
                if DbOffset % 8:
                    DbOffset += 8 - DbOffset % 8
        else:
            assert False
        TokenTypeValue = Dict['TOKEN_TYPE'][LocalTokenNumberTableIndex]
        TokenTypeValue = GetTokenTypeValue(TokenTypeValue)
        LocalTokenNumberTable[LocalTokenNumberTableIndex] = DbOffset | int(TokenTypeValue)
    for VariableEntries in VariableTable:
        skuindex = 0
        for VariableEntryPerSku in VariableEntries:
            (VariableHeadGuidIndex, VariableHeadStringIndex, SKUVariableOffset, VariableOffset, VariableRefTable, VariableAttribute) = VariableEntryPerSku[:]
            DbIndex = 0
            DbOffset = FixedHeaderLen
            for DbIndex in range(len(DbTotal)):
                if DbTotal[DbIndex] is VariableRefTable:
                    DbOffset += DbItemTotal[DbIndex].GetInterOffset(VariableOffset)
                    break
                DbOffset += DbItemTotal[DbIndex].GetListSize()
                if DbIndex + 1 == InitTableNum:
                    if DbOffset % 8:
                        DbOffset += 8 - DbOffset % 8
            else:
                assert False
            if isinstance(VariableRefTable[0], list):
                DbOffset += skuindex * 4
            skuindex += 1
            if DbIndex >= InitTableNum:
                assert False
            (VarAttr, VarProp) = VariableAttributes.GetVarAttributes(VariableAttribute)
            VariableEntryPerSku[:] = (VariableHeadStringIndex, DbOffset, VariableHeadGuidIndex, SKUVariableOffset, VarAttr, VarProp)
    DbTotalLength = FixedHeaderLen
    for DbIndex in range(len(DbItemTotal)):
        if DbItemTotal[DbIndex] is DbLocalTokenNumberTable:
            LocalTokenNumberTableOffset = DbTotalLength
        elif DbItemTotal[DbIndex] is DbExMapTable:
            ExMapTableOffset = DbTotalLength
        elif DbItemTotal[DbIndex] is DbGuidTable:
            GuidTableOffset = DbTotalLength
        elif DbItemTotal[DbIndex] is DbStringTableLen:
            StringTableOffset = DbTotalLength
        elif DbItemTotal[DbIndex] is DbSizeTableValue:
            SizeTableOffset = DbTotalLength
        elif DbItemTotal[DbIndex] is DbSkuidValue:
            SkuIdTableOffset = DbTotalLength
        elif DbItemTotal[DbIndex] is DbPcdNameOffsetTable:
            DbPcdNameOffset = DbTotalLength
        DbTotalLength += DbItemTotal[DbIndex].GetListSize()
    if not Dict['PCD_INFO_FLAG']:
        DbPcdNameOffset = 0
    LocalTokenCount = GetIntegerValue(Dict['LOCAL_TOKEN_NUMBER'])
    ExTokenCount = GetIntegerValue(Dict['EX_TOKEN_NUMBER'])
    GuidTableCount = GetIntegerValue(Dict['GUID_TABLE_SIZE'])
    SystemSkuId = GetIntegerValue(Dict['SYSTEM_SKU_ID_VALUE'])
    Pad = 218
    UninitDataBaseSize = 0
    for Item in (DbUnInitValueUint64, DbUnInitValueUint32, DbUnInitValueUint16, DbUnInitValueUint8, DbUnInitValueBoolean):
        UninitDataBaseSize += Item.GetListSize()
    if (DbTotalLength - UninitDataBaseSize) % 8:
        DbTotalLength += 8 - (DbTotalLength - UninitDataBaseSize) % 8
    Guid = '{0x3c7d193c, 0x682c, 0x4c14, 0xa6, 0x8f, 0x55, 0x2d, 0xea, 0x4f, 0x43, 0x7e}'
    Guid = StringArrayToList(Guid)
    Buffer = PackByteFormatGUID(Guid)
    b = pack('=L', DATABASE_VERSION)
    Buffer += b
    b = pack('=L', DbTotalLength - UninitDataBaseSize)
    Buffer += b
    b = pack('=Q', SystemSkuId)
    Buffer += b
    b = pack('=L', 0)
    Buffer += b
    b = pack('=L', UninitDataBaseSize)
    Buffer += b
    b = pack('=L', LocalTokenNumberTableOffset)
    Buffer += b
    b = pack('=L', ExMapTableOffset)
    Buffer += b
    b = pack('=L', GuidTableOffset)
    Buffer += b
    b = pack('=L', StringTableOffset)
    Buffer += b
    b = pack('=L', SizeTableOffset)
    Buffer += b
    b = pack('=L', SkuIdTableOffset)
    Buffer += b
    b = pack('=L', DbPcdNameOffset)
    Buffer += b
    b = pack('=H', LocalTokenCount)
    Buffer += b
    b = pack('=H', ExTokenCount)
    Buffer += b
    b = pack('=H', GuidTableCount)
    Buffer += b
    b = pack('=B', Pad)
    Buffer += b
    Buffer += b
    Buffer += b
    Buffer += b
    Buffer += b
    Buffer += b
    Index = 0
    for Item in DbItemTotal:
        Index += 1
        packdata = Item.PackData()
        for i in range(len(packdata)):
            Buffer += packdata[i:i + 1]
        if Index == InitTableNum:
            if len(Buffer) % 8:
                for num in range(8 - len(Buffer) % 8):
                    b = pack('=B', Pad)
                    Buffer += b
            break
    return Buffer

def CreatePcdDatabaseCode(Info, AutoGenC, AutoGenH):
    if False:
        while True:
            i = 10
    if Info.PcdIsDriver == '':
        return
    if Info.PcdIsDriver not in gPcdPhaseMap:
        EdkLogger.error('build', AUTOGEN_ERROR, 'Not supported PcdIsDriver type:%s' % Info.PcdIsDriver, ExtraData='[%s]' % str(Info))
    (AdditionalAutoGenH, AdditionalAutoGenC, PcdDbBuffer) = NewCreatePcdDatabasePhaseSpecificAutoGen(Info.PlatformInfo, 'PEI')
    AutoGenH.Append(AdditionalAutoGenH.String)
    Phase = gPcdPhaseMap[Info.PcdIsDriver]
    if Phase == 'PEI':
        AutoGenC.Append(AdditionalAutoGenC.String)
    if Phase == 'DXE':
        (AdditionalAutoGenH, AdditionalAutoGenC, PcdDbBuffer) = NewCreatePcdDatabasePhaseSpecificAutoGen(Info.PlatformInfo, Phase)
        AutoGenH.Append(AdditionalAutoGenH.String)
        AutoGenC.Append(AdditionalAutoGenC.String)
    if Info.IsBinaryModule:
        DbFileName = os.path.join(Info.PlatformInfo.BuildDir, TAB_FV_DIRECTORY, Phase + 'PcdDataBase.raw')
    else:
        DbFileName = os.path.join(Info.OutputDir, Phase + 'PcdDataBase.raw')
    DbFile = BytesIO()
    DbFile.write(PcdDbBuffer)
    Changed = SaveFileOnChange(DbFileName, DbFile.getvalue(), True)

def CreatePcdDataBase(PcdDBData):
    if False:
        i = 10
        return i + 15
    delta = {}
    for (skuname, skuid) in PcdDBData:
        if len(PcdDBData[skuname, skuid][1]) != len(PcdDBData[TAB_DEFAULT, '0'][1]):
            EdkLogger.error('build', AUTOGEN_ERROR, 'The size of each sku in one pcd are not same')
    for (skuname, skuid) in PcdDBData:
        if skuname == TAB_DEFAULT:
            continue
        delta[skuname, skuid] = [(index, data, hex(data)) for (index, data) in enumerate(PcdDBData[skuname, skuid][1]) if PcdDBData[skuname, skuid][1][index] != PcdDBData[TAB_DEFAULT, '0'][1][index]]
    databasebuff = PcdDBData[TAB_DEFAULT, '0'][0]
    for (skuname, skuid) in delta:
        if len(databasebuff) % 8 > 0:
            for i in range(8 - len(databasebuff) % 8):
                databasebuff += pack('=B', 0)
        databasebuff += pack('=Q', int(skuid))
        databasebuff += pack('=Q', 0)
        databasebuff += pack('=L', 8 + 8 + 4 + 4 * len(delta[skuname, skuid]))
        for item in delta[skuname, skuid]:
            databasebuff += pack('=L', item[0])
            databasebuff = databasebuff[:-1] + pack('=B', item[1])
    totallen = len(databasebuff)
    totallenbuff = pack('=L', totallen)
    newbuffer = databasebuff[:32]
    for i in range(4):
        newbuffer += totallenbuff[i:i + 1]
    for i in range(36, totallen):
        newbuffer += databasebuff[i:i + 1]
    return newbuffer

def CreateVarCheckBin(VarCheckTab):
    if False:
        while True:
            i = 10
    return VarCheckTab[TAB_DEFAULT, '0']

def CreateAutoGen(PcdDriverAutoGenData):
    if False:
        return 10
    autogenC = TemplateString()
    for (skuname, skuid) in PcdDriverAutoGenData:
        autogenC.Append('//SKUID: %s' % skuname)
        autogenC.Append(PcdDriverAutoGenData[skuname, skuid][1].String)
    return (PcdDriverAutoGenData[skuname, skuid][0], autogenC)

def NewCreatePcdDatabasePhaseSpecificAutoGen(Platform, Phase):
    if False:
        i = 10
        return i + 15

    def prune_sku(pcd, skuname):
        if False:
            while True:
                i = 10
        new_pcd = copy.deepcopy(pcd)
        new_pcd.SkuInfoList = {skuname: pcd.SkuInfoList[skuname]}
        new_pcd.isinit = 'INIT'
        if new_pcd.DatumType in TAB_PCD_NUMERIC_TYPES:
            for skuobj in pcd.SkuInfoList.values():
                if skuobj.DefaultValue:
                    defaultvalue = int(skuobj.DefaultValue, 16) if skuobj.DefaultValue.upper().startswith('0X') else int(skuobj.DefaultValue, 10)
                    if defaultvalue != 0:
                        new_pcd.isinit = 'INIT'
                        break
                elif skuobj.VariableName:
                    new_pcd.isinit = 'INIT'
                    break
            else:
                new_pcd.isinit = 'UNINIT'
        return new_pcd
    DynamicPcds = Platform.DynamicPcdList
    DynamicPcdSet_Sku = {(SkuName, skuobj.SkuId): [] for pcd in DynamicPcds for (SkuName, skuobj) in pcd.SkuInfoList.items()}
    for (skuname, skuid) in DynamicPcdSet_Sku:
        DynamicPcdSet_Sku[skuname, skuid] = [prune_sku(pcd, skuname) for pcd in DynamicPcds]
    PcdDBData = {}
    PcdDriverAutoGenData = {}
    VarCheckTableData = {}
    if DynamicPcdSet_Sku:
        for (skuname, skuid) in DynamicPcdSet_Sku:
            (AdditionalAutoGenH, AdditionalAutoGenC, PcdDbBuffer, VarCheckTab) = CreatePcdDatabasePhaseSpecificAutoGen(Platform, DynamicPcdSet_Sku[skuname, skuid], Phase)
            final_data = ()
            for item in range(len(PcdDbBuffer)):
                final_data += unpack('B', PcdDbBuffer[item:item + 1])
            PcdDBData[skuname, skuid] = (PcdDbBuffer, final_data)
            PcdDriverAutoGenData[skuname, skuid] = (AdditionalAutoGenH, AdditionalAutoGenC)
            VarCheckTableData[skuname, skuid] = VarCheckTab
        if Platform.Platform.VarCheckFlag:
            dest = os.path.join(Platform.BuildDir, TAB_FV_DIRECTORY)
            VarCheckTable = CreateVarCheckBin(VarCheckTableData)
            VarCheckTable.dump(dest, Phase)
        (AdditionalAutoGenH, AdditionalAutoGenC) = CreateAutoGen(PcdDriverAutoGenData)
    else:
        (AdditionalAutoGenH, AdditionalAutoGenC, PcdDbBuffer, VarCheckTab) = CreatePcdDatabasePhaseSpecificAutoGen(Platform, {}, Phase)
        final_data = ()
        for item in range(len(PcdDbBuffer)):
            final_data += unpack('B', PcdDbBuffer[item:item + 1])
        PcdDBData[TAB_DEFAULT, '0'] = (PcdDbBuffer, final_data)
    return (AdditionalAutoGenH, AdditionalAutoGenC, CreatePcdDataBase(PcdDBData))

def CreatePcdDatabasePhaseSpecificAutoGen(Platform, DynamicPcdList, Phase):
    if False:
        return 10
    AutoGenC = TemplateString()
    AutoGenH = TemplateString()
    Dict = {'PHASE': Phase, 'SERVICE_DRIVER_VERSION': DATABASE_VERSION, 'GUID_TABLE_SIZE': '1U', 'STRING_TABLE_SIZE': '1U', 'SKUID_TABLE_SIZE': '1U', 'LOCAL_TOKEN_NUMBER_TABLE_SIZE': '0U', 'LOCAL_TOKEN_NUMBER': '0U', 'EXMAPPING_TABLE_SIZE': '1U', 'EX_TOKEN_NUMBER': '0U', 'SIZE_TABLE_SIZE': '2U', 'SKU_HEAD_SIZE': '1U', 'GUID_TABLE_EMPTY': 'TRUE', 'STRING_TABLE_EMPTY': 'TRUE', 'SKUID_TABLE_EMPTY': 'TRUE', 'DATABASE_EMPTY': 'TRUE', 'EXMAP_TABLE_EMPTY': 'TRUE', 'PCD_DATABASE_UNINIT_EMPTY': '  UINT8  dummy; /* PCD_DATABASE_UNINIT is empty */', 'SYSTEM_SKU_ID': '  SKU_ID             SystemSkuId;', 'SYSTEM_SKU_ID_VALUE': '0U'}
    SkuObj = Platform.Platform.SkuIdMgr
    Dict['SYSTEM_SKU_ID_VALUE'] = 0 if SkuObj.SkuUsageType == SkuObj.SINGLE else Platform.Platform.SkuIds[SkuObj.SystemSkuId][0]
    Dict['PCD_INFO_FLAG'] = Platform.Platform.PcdInfoFlag
    for DatumType in TAB_PCD_NUMERIC_TYPES_VOID:
        Dict['VARDEF_CNAME_' + DatumType] = []
        Dict['VARDEF_GUID_' + DatumType] = []
        Dict['VARDEF_SKUID_' + DatumType] = []
        Dict['VARDEF_VALUE_' + DatumType] = []
        Dict['VARDEF_DB_VALUE_' + DatumType] = []
        for Init in ['INIT', 'UNINIT']:
            Dict[Init + '_CNAME_DECL_' + DatumType] = []
            Dict[Init + '_GUID_DECL_' + DatumType] = []
            Dict[Init + '_NUMSKUS_DECL_' + DatumType] = []
            Dict[Init + '_VALUE_' + DatumType] = []
            Dict[Init + '_DB_VALUE_' + DatumType] = []
    for Type in ['STRING_HEAD', 'VPD_HEAD', 'VARIABLE_HEAD']:
        Dict[Type + '_CNAME_DECL'] = []
        Dict[Type + '_GUID_DECL'] = []
        Dict[Type + '_NUMSKUS_DECL'] = []
        Dict[Type + '_VALUE'] = []
    Dict['STRING_DB_VALUE'] = []
    Dict['VPD_DB_VALUE'] = []
    Dict['VARIABLE_DB_VALUE'] = []
    Dict['STRING_TABLE_INDEX'] = []
    Dict['STRING_TABLE_LENGTH'] = []
    Dict['STRING_TABLE_CNAME'] = []
    Dict['STRING_TABLE_GUID'] = []
    Dict['STRING_TABLE_VALUE'] = []
    Dict['STRING_TABLE_DB_VALUE'] = []
    Dict['SIZE_TABLE_CNAME'] = []
    Dict['SIZE_TABLE_GUID'] = []
    Dict['SIZE_TABLE_CURRENT_LENGTH'] = []
    Dict['SIZE_TABLE_MAXIMUM_LENGTH'] = []
    Dict['EXMAPPING_TABLE_EXTOKEN'] = []
    Dict['EXMAPPING_TABLE_LOCAL_TOKEN'] = []
    Dict['EXMAPPING_TABLE_GUID_INDEX'] = []
    Dict['GUID_STRUCTURE'] = []
    Dict['SKUID_VALUE'] = [0]
    Dict['VARDEF_HEADER'] = []
    Dict['LOCAL_TOKEN_NUMBER_DB_VALUE'] = []
    Dict['VARIABLE_DB_VALUE'] = []
    Dict['PCD_TOKENSPACE'] = []
    Dict['PCD_CNAME'] = []
    Dict['PCD_TOKENSPACE_LENGTH'] = []
    Dict['PCD_CNAME_LENGTH'] = []
    Dict['PCD_TOKENSPACE_OFFSET'] = []
    Dict['PCD_CNAME_OFFSET'] = []
    Dict['PCD_TOKENSPACE_MAP'] = []
    Dict['PCD_NAME_OFFSET'] = []
    Dict['PCD_ORDER_TOKEN_NUMBER_MAP'] = {}
    PCD_STRING_INDEX_MAP = {}
    StringTableIndex = 0
    StringTableSize = 0
    NumberOfLocalTokens = 0
    NumberOfPeiLocalTokens = 0
    NumberOfDxeLocalTokens = 0
    NumberOfExTokens = 0
    NumberOfSizeItems = 0
    NumberOfSkuEnabledPcd = 0
    GuidList = []
    VarCheckTab = VAR_CHECK_PCD_VARIABLE_TAB_CONTAINER()
    i = 0
    ReorderedDynPcdList = GetOrderedDynamicPcdList(DynamicPcdList, Platform.PcdTokenNumber)
    for item in ReorderedDynPcdList:
        if item.DatumType not in [TAB_UINT8, TAB_UINT16, TAB_UINT32, TAB_UINT64, TAB_VOID, 'BOOLEAN']:
            item.DatumType = TAB_VOID
    for Pcd in ReorderedDynPcdList:
        VoidStarTypeCurrSize = []
        i += 1
        CName = Pcd.TokenCName
        TokenSpaceGuidCName = Pcd.TokenSpaceGuidCName
        for PcdItem in GlobalData.MixedPcd:
            if (Pcd.TokenCName, Pcd.TokenSpaceGuidCName) in GlobalData.MixedPcd[PcdItem]:
                CName = PcdItem[0]
        EdkLogger.debug(EdkLogger.DEBUG_3, 'PCD: %s %s (%s : %s)' % (CName, TokenSpaceGuidCName, Pcd.Phase, Phase))
        if Pcd.Phase == 'PEI':
            NumberOfPeiLocalTokens += 1
        if Pcd.Phase == 'DXE':
            NumberOfDxeLocalTokens += 1
        if Pcd.Phase != Phase:
            continue
        TokenSpaceGuidStructure = Pcd.TokenSpaceGuidValue
        TokenSpaceGuid = GuidStructureStringToGuidValueName(TokenSpaceGuidStructure)
        if Pcd.Type in PCD_DYNAMIC_EX_TYPE_SET:
            if TokenSpaceGuid not in GuidList:
                GuidList.append(TokenSpaceGuid)
                Dict['GUID_STRUCTURE'].append(TokenSpaceGuidStructure)
            NumberOfExTokens += 1
        ValueList = []
        DbValueList = []
        StringHeadOffsetList = []
        StringDbOffsetList = []
        VpdHeadOffsetList = []
        VpdDbOffsetList = []
        VariableHeadValueList = []
        VariableDbValueList = []
        Pcd.InitString = 'UNINIT'
        if Pcd.DatumType == TAB_VOID:
            if Pcd.Type not in [TAB_PCDS_DYNAMIC_VPD, TAB_PCDS_DYNAMIC_EX_VPD]:
                Pcd.TokenTypeList = ['PCD_TYPE_STRING']
            else:
                Pcd.TokenTypeList = []
        elif Pcd.DatumType == 'BOOLEAN':
            Pcd.TokenTypeList = ['PCD_DATUM_TYPE_UINT8_BOOLEAN']
        else:
            Pcd.TokenTypeList = ['PCD_DATUM_TYPE_' + Pcd.DatumType]
        if len(Pcd.SkuInfoList) > 1:
            NumberOfSkuEnabledPcd += 1
        SkuIdIndex = 1
        VariableHeadList = []
        for SkuName in Pcd.SkuInfoList:
            Sku = Pcd.SkuInfoList[SkuName]
            SkuId = Sku.SkuId
            if SkuId is None or SkuId == '':
                continue
            SkuIdIndex += 1
            if len(Sku.VariableName) > 0:
                VariableGuidStructure = Sku.VariableGuidValue
                VariableGuid = GuidStructureStringToGuidValueName(VariableGuidStructure)
                if Platform.Platform.VarCheckFlag:
                    var_check_obj = VAR_CHECK_PCD_VARIABLE_TAB(VariableGuidStructure, StringToArray(Sku.VariableName))
                    try:
                        var_check_obj.push_back(GetValidationObject(Pcd, Sku.VariableOffset))
                        (VarAttr, _) = VariableAttributes.GetVarAttributes(Sku.VariableAttribute)
                        var_check_obj.SetAttributes(VarAttr)
                        var_check_obj.UpdateSize()
                        VarCheckTab.push_back(var_check_obj)
                    except Exception:
                        ValidInfo = ''
                        if Pcd.validateranges:
                            ValidInfo = Pcd.validateranges[0]
                        if Pcd.validlists:
                            ValidInfo = Pcd.validlists[0]
                        if ValidInfo:
                            EdkLogger.error('build', PCD_VALIDATION_INFO_ERROR, "The PCD '%s.%s' Validation information defined in DEC file has incorrect format." % (Pcd.TokenSpaceGuidCName, Pcd.TokenCName), ExtraData='[%s]' % str(ValidInfo))
                        else:
                            EdkLogger.error('build', PCD_VALIDATION_INFO_ERROR, "The PCD '%s.%s' Validation information defined in DEC file has incorrect format." % (Pcd.TokenSpaceGuidCName, Pcd.TokenCName))
                Pcd.TokenTypeList.append('PCD_TYPE_HII')
                Pcd.InitString = 'INIT'
                VariableNameStructure = StringToArray(Sku.VariableName)
                VariableNameStructureBytes = VariableNameStructure.lstrip('{').rstrip('}').split(',')
                if len(VariableNameStructureBytes) % 2:
                    VariableNameStructure = '{%s,0x00}' % ','.join(VariableNameStructureBytes)
                if VariableNameStructure not in Dict['STRING_TABLE_VALUE']:
                    Dict['STRING_TABLE_CNAME'].append(CName)
                    Dict['STRING_TABLE_GUID'].append(TokenSpaceGuid)
                    if StringTableIndex == 0:
                        Dict['STRING_TABLE_INDEX'].append('')
                    else:
                        Dict['STRING_TABLE_INDEX'].append('_%d' % StringTableIndex)
                    VarNameSize = len(VariableNameStructure.replace(',', ' ').split())
                    Dict['STRING_TABLE_LENGTH'].append(VarNameSize)
                    Dict['STRING_TABLE_VALUE'].append(VariableNameStructure)
                    StringHeadOffsetList.append(str(StringTableSize) + 'U')
                    VarStringDbOffsetList = []
                    VarStringDbOffsetList.append(StringTableSize)
                    Dict['STRING_DB_VALUE'].append(VarStringDbOffsetList)
                    StringTableIndex += 1
                    StringTableSize += len(VariableNameStructure.replace(',', ' ').split())
                VariableHeadStringIndex = 0
                for Index in range(Dict['STRING_TABLE_VALUE'].index(VariableNameStructure)):
                    VariableHeadStringIndex += Dict['STRING_TABLE_LENGTH'][Index]
                VariableHeadList.append(VariableHeadStringIndex)
                VariableHeadStringIndex = VariableHeadList[SkuIdIndex - 2]
                if VariableGuid not in GuidList:
                    GuidList.append(VariableGuid)
                    Dict['GUID_STRUCTURE'].append(VariableGuidStructure)
                VariableHeadGuidIndex = GuidList.index(VariableGuid)
                if 'PCD_TYPE_STRING' in Pcd.TokenTypeList:
                    VariableHeadValueList.append('%dU, offsetof(%s_PCD_DATABASE, Init.%s_%s), %dU, %sU' % (VariableHeadStringIndex, Phase, CName, TokenSpaceGuid, VariableHeadGuidIndex, Sku.VariableOffset))
                else:
                    VariableHeadValueList.append('%dU, offsetof(%s_PCD_DATABASE, Init.%s_%s_VariableDefault_%s), %dU, %sU' % (VariableHeadStringIndex, Phase, CName, TokenSpaceGuid, SkuIdIndex, VariableHeadGuidIndex, Sku.VariableOffset))
                Dict['VARDEF_CNAME_' + Pcd.DatumType].append(CName)
                Dict['VARDEF_GUID_' + Pcd.DatumType].append(TokenSpaceGuid)
                Dict['VARDEF_SKUID_' + Pcd.DatumType].append(SkuIdIndex)
                if 'PCD_TYPE_STRING' in Pcd.TokenTypeList:
                    Dict['VARDEF_VALUE_' + Pcd.DatumType].append('%s_%s[%d]' % (Pcd.TokenCName, TokenSpaceGuid, SkuIdIndex))
                else:
                    Dict['VARDEF_DB_VALUE_' + Pcd.DatumType].append(Sku.HiiDefaultValue)
                    if Pcd.DatumType == TAB_UINT64:
                        Dict['VARDEF_VALUE_' + Pcd.DatumType].append(Sku.HiiDefaultValue + 'ULL')
                    elif Pcd.DatumType in (TAB_UINT32, TAB_UINT16, TAB_UINT8):
                        Dict['VARDEF_VALUE_' + Pcd.DatumType].append(Sku.HiiDefaultValue + 'U')
                    elif Pcd.DatumType == 'BOOLEAN':
                        if eval(Sku.HiiDefaultValue) in [1, 0]:
                            Dict['VARDEF_VALUE_' + Pcd.DatumType].append(str(eval(Sku.HiiDefaultValue)) + 'U')
                    else:
                        Dict['VARDEF_VALUE_' + Pcd.DatumType].append(Sku.HiiDefaultValue)
                if 'PCD_TYPE_STRING' in Pcd.TokenTypeList:
                    VariableHeadValueList.append('%dU, %dU, %sU, offsetof(%s_PCD_DATABASE, Init.%s_%s)' % (VariableHeadGuidIndex, VariableHeadStringIndex, Sku.VariableOffset, Phase, CName, TokenSpaceGuid))
                    VariableOffset = len(Dict['STRING_DB_VALUE'])
                    VariableRefTable = Dict['STRING_DB_VALUE']
                else:
                    VariableHeadValueList.append('%dU, %dU, %sU, offsetof(%s_PCD_DATABASE, Init.%s_%s_VariableDefault_%s)' % (VariableHeadGuidIndex, VariableHeadStringIndex, Sku.VariableOffset, Phase, CName, TokenSpaceGuid, SkuIdIndex))
                    VariableOffset = len(Dict['VARDEF_DB_VALUE_' + Pcd.DatumType]) - 1
                    VariableRefTable = Dict['VARDEF_DB_VALUE_' + Pcd.DatumType]
                VariableDbValueList.append([VariableHeadGuidIndex, VariableHeadStringIndex, Sku.VariableOffset, VariableOffset, VariableRefTable, Sku.VariableAttribute])
            elif Sku.VpdOffset != '':
                Pcd.TokenTypeList.append('PCD_TYPE_VPD')
                Pcd.InitString = 'INIT'
                VpdHeadOffsetList.append(str(Sku.VpdOffset) + 'U')
                VpdDbOffsetList.append(Sku.VpdOffset)
                if Pcd.DatumType == TAB_VOID:
                    NumberOfSizeItems += 1
                    VoidStarTypeCurrSize = [str(Pcd.MaxDatumSize) + 'U']
                continue
            if Pcd.DatumType == TAB_VOID:
                Pcd.TokenTypeList.append('PCD_TYPE_STRING')
                Pcd.InitString = 'INIT'
                if Sku.HiiDefaultValue != '' and Sku.DefaultValue == '':
                    Sku.DefaultValue = Sku.HiiDefaultValue
                if Sku.DefaultValue != '':
                    NumberOfSizeItems += 1
                    Dict['STRING_TABLE_CNAME'].append(CName)
                    Dict['STRING_TABLE_GUID'].append(TokenSpaceGuid)
                    if StringTableIndex == 0:
                        Dict['STRING_TABLE_INDEX'].append('')
                    else:
                        Dict['STRING_TABLE_INDEX'].append('_%d' % StringTableIndex)
                    if Sku.DefaultValue[0] == 'L':
                        DefaultValueBinStructure = StringToArray(Sku.DefaultValue)
                        Size = len(DefaultValueBinStructure.replace(',', ' ').split())
                        Dict['STRING_TABLE_VALUE'].append(DefaultValueBinStructure)
                    elif Sku.DefaultValue[0] == '"':
                        DefaultValueBinStructure = StringToArray(Sku.DefaultValue)
                        Size = len(Sku.DefaultValue) - 2 + 1
                        Dict['STRING_TABLE_VALUE'].append(DefaultValueBinStructure)
                    elif Sku.DefaultValue[0] == '{':
                        DefaultValueBinStructure = StringToArray(Sku.DefaultValue)
                        Size = len(Sku.DefaultValue.split(','))
                        Dict['STRING_TABLE_VALUE'].append(DefaultValueBinStructure)
                    StringHeadOffsetList.append(str(StringTableSize) + 'U')
                    StringDbOffsetList.append(StringTableSize)
                    if Pcd.MaxDatumSize != '':
                        MaxDatumSize = int(Pcd.MaxDatumSize, 0)
                        if MaxDatumSize < Size:
                            if Pcd.MaxSizeUserSet:
                                EdkLogger.error('build', AUTOGEN_ERROR, "The maximum size of VOID* type PCD '%s.%s' is less than its actual size occupied." % (Pcd.TokenSpaceGuidCName, Pcd.TokenCName), ExtraData='[%s]' % str(Platform))
                            else:
                                MaxDatumSize = Size
                    else:
                        MaxDatumSize = Size
                    StringTabLen = MaxDatumSize
                    if StringTabLen % 2:
                        StringTabLen += 1
                    if Sku.VpdOffset == '':
                        VoidStarTypeCurrSize.append(str(Size) + 'U')
                    Dict['STRING_TABLE_LENGTH'].append(StringTabLen)
                    StringTableIndex += 1
                    StringTableSize += StringTabLen
            else:
                if 'PCD_TYPE_HII' not in Pcd.TokenTypeList:
                    Pcd.TokenTypeList.append('PCD_TYPE_DATA')
                    if Sku.DefaultValue == 'TRUE':
                        Pcd.InitString = 'INIT'
                    else:
                        Pcd.InitString = Pcd.isinit
                if Pcd.DatumType == TAB_UINT64:
                    ValueList.append(Sku.DefaultValue + 'ULL')
                elif Pcd.DatumType in (TAB_UINT32, TAB_UINT16, TAB_UINT8):
                    ValueList.append(Sku.DefaultValue + 'U')
                elif Pcd.DatumType == 'BOOLEAN':
                    if Sku.DefaultValue in ['1', '0']:
                        ValueList.append(Sku.DefaultValue + 'U')
                else:
                    ValueList.append(Sku.DefaultValue)
                DbValueList.append(Sku.DefaultValue)
        Pcd.TokenTypeList = list(set(Pcd.TokenTypeList))
        if Pcd.DatumType == TAB_VOID:
            Dict['SIZE_TABLE_CNAME'].append(CName)
            Dict['SIZE_TABLE_GUID'].append(TokenSpaceGuid)
            Dict['SIZE_TABLE_MAXIMUM_LENGTH'].append(str(Pcd.MaxDatumSize) + 'U')
            Dict['SIZE_TABLE_CURRENT_LENGTH'].append(VoidStarTypeCurrSize)
        if 'PCD_TYPE_HII' in Pcd.TokenTypeList:
            Dict['VARIABLE_HEAD_CNAME_DECL'].append(CName)
            Dict['VARIABLE_HEAD_GUID_DECL'].append(TokenSpaceGuid)
            Dict['VARIABLE_HEAD_NUMSKUS_DECL'].append(len(Pcd.SkuInfoList))
            Dict['VARIABLE_HEAD_VALUE'].append('{ %s }\n' % ' },\n    { '.join(VariableHeadValueList))
            Dict['VARDEF_HEADER'].append('_Variable_Header')
            Dict['VARIABLE_DB_VALUE'].append(VariableDbValueList)
        else:
            Dict['VARDEF_HEADER'].append('')
        if 'PCD_TYPE_VPD' in Pcd.TokenTypeList:
            Dict['VPD_HEAD_CNAME_DECL'].append(CName)
            Dict['VPD_HEAD_GUID_DECL'].append(TokenSpaceGuid)
            Dict['VPD_HEAD_NUMSKUS_DECL'].append(len(Pcd.SkuInfoList))
            Dict['VPD_HEAD_VALUE'].append('{ %s }' % ' }, { '.join(VpdHeadOffsetList))
            Dict['VPD_DB_VALUE'].append(VpdDbOffsetList)
        if 'PCD_TYPE_STRING' in Pcd.TokenTypeList:
            Dict['STRING_HEAD_CNAME_DECL'].append(CName)
            Dict['STRING_HEAD_GUID_DECL'].append(TokenSpaceGuid)
            Dict['STRING_HEAD_NUMSKUS_DECL'].append(len(Pcd.SkuInfoList))
            Dict['STRING_HEAD_VALUE'].append(', '.join(StringHeadOffsetList))
            Dict['STRING_DB_VALUE'].append(StringDbOffsetList)
            PCD_STRING_INDEX_MAP[len(Dict['STRING_HEAD_CNAME_DECL']) - 1] = len(Dict['STRING_DB_VALUE']) - 1
        if 'PCD_TYPE_DATA' in Pcd.TokenTypeList:
            Dict[Pcd.InitString + '_CNAME_DECL_' + Pcd.DatumType].append(CName)
            Dict[Pcd.InitString + '_GUID_DECL_' + Pcd.DatumType].append(TokenSpaceGuid)
            Dict[Pcd.InitString + '_NUMSKUS_DECL_' + Pcd.DatumType].append(len(Pcd.SkuInfoList))
            if Pcd.InitString == 'UNINIT':
                Dict['PCD_DATABASE_UNINIT_EMPTY'] = ''
            else:
                Dict[Pcd.InitString + '_VALUE_' + Pcd.DatumType].append(', '.join(ValueList))
                Dict[Pcd.InitString + '_DB_VALUE_' + Pcd.DatumType].append(DbValueList)
    if Phase == 'PEI':
        NumberOfLocalTokens = NumberOfPeiLocalTokens
    if Phase == 'DXE':
        NumberOfLocalTokens = NumberOfDxeLocalTokens
    Dict['TOKEN_INIT'] = ['' for x in range(NumberOfLocalTokens)]
    Dict['TOKEN_CNAME'] = ['' for x in range(NumberOfLocalTokens)]
    Dict['TOKEN_GUID'] = ['' for x in range(NumberOfLocalTokens)]
    Dict['TOKEN_TYPE'] = ['' for x in range(NumberOfLocalTokens)]
    Dict['LOCAL_TOKEN_NUMBER_DB_VALUE'] = ['' for x in range(NumberOfLocalTokens)]
    Dict['PCD_CNAME'] = ['' for x in range(NumberOfLocalTokens)]
    Dict['PCD_TOKENSPACE_MAP'] = ['' for x in range(NumberOfLocalTokens)]
    Dict['PCD_CNAME_LENGTH'] = [0 for x in range(NumberOfLocalTokens)]
    SkuEnablePcdIndex = 0
    for Pcd in ReorderedDynPcdList:
        CName = Pcd.TokenCName
        TokenSpaceGuidCName = Pcd.TokenSpaceGuidCName
        if Pcd.Phase != Phase:
            continue
        TokenSpaceGuid = GuidStructureStringToGuidValueName(Pcd.TokenSpaceGuidValue)
        GeneratedTokenNumber = Platform.PcdTokenNumber[CName, TokenSpaceGuidCName] - 1
        if Phase == 'DXE':
            GeneratedTokenNumber -= NumberOfPeiLocalTokens
        if len(Pcd.SkuInfoList) > 1:
            Dict['PCD_ORDER_TOKEN_NUMBER_MAP'][GeneratedTokenNumber] = SkuEnablePcdIndex
            SkuEnablePcdIndex += 1
        for PcdItem in GlobalData.MixedPcd:
            if (Pcd.TokenCName, Pcd.TokenSpaceGuidCName) in GlobalData.MixedPcd[PcdItem]:
                CName = PcdItem[0]
        EdkLogger.debug(EdkLogger.DEBUG_1, 'PCD = %s.%s' % (CName, TokenSpaceGuidCName))
        EdkLogger.debug(EdkLogger.DEBUG_1, 'phase = %s' % Phase)
        EdkLogger.debug(EdkLogger.DEBUG_1, 'GeneratedTokenNumber = %s' % str(GeneratedTokenNumber))
        Dict['TOKEN_INIT'][GeneratedTokenNumber] = 'Init'
        if Pcd.InitString == 'UNINIT':
            Dict['TOKEN_INIT'][GeneratedTokenNumber] = 'Uninit'
        Dict['TOKEN_CNAME'][GeneratedTokenNumber] = CName
        Dict['TOKEN_GUID'][GeneratedTokenNumber] = TokenSpaceGuid
        Dict['TOKEN_TYPE'][GeneratedTokenNumber] = ' | '.join(Pcd.TokenTypeList)
        if Platform.Platform.PcdInfoFlag:
            TokenSpaceGuidCNameArray = StringToArray('"' + TokenSpaceGuidCName + '"')
            if TokenSpaceGuidCNameArray not in Dict['PCD_TOKENSPACE']:
                Dict['PCD_TOKENSPACE'].append(TokenSpaceGuidCNameArray)
                Dict['PCD_TOKENSPACE_LENGTH'].append(len(TokenSpaceGuidCNameArray.split(',')))
            Dict['PCD_TOKENSPACE_MAP'][GeneratedTokenNumber] = Dict['PCD_TOKENSPACE'].index(TokenSpaceGuidCNameArray)
            CNameBinArray = StringToArray('"' + CName + '"')
            Dict['PCD_CNAME'][GeneratedTokenNumber] = CNameBinArray
            Dict['PCD_CNAME_LENGTH'][GeneratedTokenNumber] = len(CNameBinArray.split(','))
        Pcd.TokenTypeList = list(set(Pcd.TokenTypeList))
        if 'PCD_TYPE_HII' in Pcd.TokenTypeList:
            Offset = GetMatchedIndex(CName, Dict['VARIABLE_HEAD_CNAME_DECL'], TokenSpaceGuid, Dict['VARIABLE_HEAD_GUID_DECL'])
            assert Offset != -1
            Table = Dict['VARIABLE_DB_VALUE']
        if 'PCD_TYPE_VPD' in Pcd.TokenTypeList:
            Offset = GetMatchedIndex(CName, Dict['VPD_HEAD_CNAME_DECL'], TokenSpaceGuid, Dict['VPD_HEAD_GUID_DECL'])
            assert Offset != -1
            Table = Dict['VPD_DB_VALUE']
        if 'PCD_TYPE_STRING' in Pcd.TokenTypeList and 'PCD_TYPE_HII' not in Pcd.TokenTypeList:
            Offset = GetMatchedIndex(CName, Dict['STRING_HEAD_CNAME_DECL'], TokenSpaceGuid, Dict['STRING_HEAD_GUID_DECL'])
            Offset = PCD_STRING_INDEX_MAP[Offset]
            assert Offset != -1
            Table = Dict['STRING_DB_VALUE']
        if 'PCD_TYPE_DATA' in Pcd.TokenTypeList:
            Offset = GetMatchedIndex(CName, Dict[Pcd.InitString + '_CNAME_DECL_' + Pcd.DatumType], TokenSpaceGuid, Dict[Pcd.InitString + '_GUID_DECL_' + Pcd.DatumType])
            assert Offset != -1
            if Pcd.InitString == 'UNINIT':
                Table = Dict[Pcd.InitString + '_GUID_DECL_' + Pcd.DatumType]
            else:
                Table = Dict[Pcd.InitString + '_DB_VALUE_' + Pcd.DatumType]
        Dict['LOCAL_TOKEN_NUMBER_DB_VALUE'][GeneratedTokenNumber] = (Offset, Table)
        if 'PCD_TYPE_HII' in Pcd.TokenTypeList:
            Dict['VARDEF_HEADER'][GeneratedTokenNumber] = '_Variable_Header'
        else:
            Dict['VARDEF_HEADER'][GeneratedTokenNumber] = ''
        if Pcd.Type in PCD_DYNAMIC_EX_TYPE_SET:
            if Phase == 'DXE':
                GeneratedTokenNumber += NumberOfPeiLocalTokens
            Dict['EXMAPPING_TABLE_EXTOKEN'].append(str(Pcd.TokenValue) + 'U')
            Dict['EXMAPPING_TABLE_LOCAL_TOKEN'].append(str(GeneratedTokenNumber + 1) + 'U')
            Dict['EXMAPPING_TABLE_GUID_INDEX'].append(str(GuidList.index(TokenSpaceGuid)) + 'U')
    if Platform.Platform.PcdInfoFlag:
        for index in range(len(Dict['PCD_TOKENSPACE_MAP'])):
            TokenSpaceIndex = StringTableSize
            for i in range(Dict['PCD_TOKENSPACE_MAP'][index]):
                TokenSpaceIndex += Dict['PCD_TOKENSPACE_LENGTH'][i]
            Dict['PCD_TOKENSPACE_OFFSET'].append(TokenSpaceIndex)
        for index in range(len(Dict['PCD_TOKENSPACE'])):
            StringTableSize += Dict['PCD_TOKENSPACE_LENGTH'][index]
            StringTableIndex += 1
        for index in range(len(Dict['PCD_CNAME'])):
            Dict['PCD_CNAME_OFFSET'].append(StringTableSize)
            Dict['PCD_NAME_OFFSET'].append(Dict['PCD_TOKENSPACE_OFFSET'][index])
            Dict['PCD_NAME_OFFSET'].append(StringTableSize)
            StringTableSize += Dict['PCD_CNAME_LENGTH'][index]
            StringTableIndex += 1
    if GuidList != []:
        Dict['GUID_TABLE_EMPTY'] = 'FALSE'
        Dict['GUID_TABLE_SIZE'] = str(len(GuidList)) + 'U'
    else:
        Dict['GUID_STRUCTURE'] = [GuidStringToGuidStructureString('00000000-0000-0000-0000-000000000000')]
    if StringTableIndex == 0:
        Dict['STRING_TABLE_INDEX'].append('')
        Dict['STRING_TABLE_LENGTH'].append(1)
        Dict['STRING_TABLE_CNAME'].append('')
        Dict['STRING_TABLE_GUID'].append('')
        Dict['STRING_TABLE_VALUE'].append('{ 0 }')
    else:
        Dict['STRING_TABLE_EMPTY'] = 'FALSE'
        Dict['STRING_TABLE_SIZE'] = str(StringTableSize) + 'U'
    if Dict['SIZE_TABLE_CNAME'] == []:
        Dict['SIZE_TABLE_CNAME'].append('')
        Dict['SIZE_TABLE_GUID'].append('')
        Dict['SIZE_TABLE_CURRENT_LENGTH'].append(['0U'])
        Dict['SIZE_TABLE_MAXIMUM_LENGTH'].append('0U')
    if NumberOfLocalTokens != 0:
        Dict['DATABASE_EMPTY'] = 'FALSE'
        Dict['LOCAL_TOKEN_NUMBER_TABLE_SIZE'] = NumberOfLocalTokens
        Dict['LOCAL_TOKEN_NUMBER'] = NumberOfLocalTokens
    if NumberOfExTokens != 0:
        Dict['EXMAP_TABLE_EMPTY'] = 'FALSE'
        Dict['EXMAPPING_TABLE_SIZE'] = str(NumberOfExTokens) + 'U'
        Dict['EX_TOKEN_NUMBER'] = str(NumberOfExTokens) + 'U'
    else:
        Dict['EXMAPPING_TABLE_EXTOKEN'].append('0U')
        Dict['EXMAPPING_TABLE_LOCAL_TOKEN'].append('0U')
        Dict['EXMAPPING_TABLE_GUID_INDEX'].append('0U')
    if NumberOfSizeItems != 0:
        Dict['SIZE_TABLE_SIZE'] = str(NumberOfSizeItems * 2) + 'U'
    if NumberOfSkuEnabledPcd != 0:
        Dict['SKU_HEAD_SIZE'] = str(NumberOfSkuEnabledPcd) + 'U'
    for AvailableSkuNumber in SkuObj.SkuIdNumberSet:
        if AvailableSkuNumber not in Dict['SKUID_VALUE']:
            Dict['SKUID_VALUE'].append(AvailableSkuNumber)
    Dict['SKUID_VALUE'][0] = len(Dict['SKUID_VALUE']) - 1
    AutoGenH.Append(gPcdDatabaseAutoGenH.Replace(Dict))
    if NumberOfLocalTokens == 0:
        AutoGenC.Append(gEmptyPcdDatabaseAutoGenC.Replace(Dict))
    else:
        SizeCNameTempList = []
        SizeGuidTempList = []
        SizeCurLenTempList = []
        SizeMaxLenTempList = []
        ReOrderFlag = True
        if len(Dict['SIZE_TABLE_CNAME']) == 1:
            if not (Dict['SIZE_TABLE_CNAME'][0] and Dict['SIZE_TABLE_GUID'][0]):
                ReOrderFlag = False
        if ReOrderFlag:
            for Count in range(len(Dict['TOKEN_CNAME'])):
                for Count1 in range(len(Dict['SIZE_TABLE_CNAME'])):
                    if Dict['TOKEN_CNAME'][Count] == Dict['SIZE_TABLE_CNAME'][Count1] and Dict['TOKEN_GUID'][Count] == Dict['SIZE_TABLE_GUID'][Count1]:
                        SizeCNameTempList.append(Dict['SIZE_TABLE_CNAME'][Count1])
                        SizeGuidTempList.append(Dict['SIZE_TABLE_GUID'][Count1])
                        SizeCurLenTempList.append(Dict['SIZE_TABLE_CURRENT_LENGTH'][Count1])
                        SizeMaxLenTempList.append(Dict['SIZE_TABLE_MAXIMUM_LENGTH'][Count1])
            for Count in range(len(Dict['SIZE_TABLE_CNAME'])):
                Dict['SIZE_TABLE_CNAME'][Count] = SizeCNameTempList[Count]
                Dict['SIZE_TABLE_GUID'][Count] = SizeGuidTempList[Count]
                Dict['SIZE_TABLE_CURRENT_LENGTH'][Count] = SizeCurLenTempList[Count]
                Dict['SIZE_TABLE_MAXIMUM_LENGTH'][Count] = SizeMaxLenTempList[Count]
        AutoGenC.Append(gPcdDatabaseAutoGenC.Replace(Dict))
    Buffer = BuildExDataBase(Dict)
    return (AutoGenH, AutoGenC, Buffer, VarCheckTab)

def GetOrderedDynamicPcdList(DynamicPcdList, PcdTokenNumberList):
    if False:
        while True:
            i = 10
    ReorderedDyPcdList = [None for i in range(len(DynamicPcdList))]
    for Pcd in DynamicPcdList:
        if (Pcd.TokenCName, Pcd.TokenSpaceGuidCName) in PcdTokenNumberList:
            ReorderedDyPcdList[PcdTokenNumberList[Pcd.TokenCName, Pcd.TokenSpaceGuidCName] - 1] = Pcd
    return ReorderedDyPcdList