"""
InfPcdObject
"""
import os
import re
from Logger import StringTable as ST
from Logger import ToolError
import Logger.Log as Logger
from Library import GlobalData
from Library import DataType as DT
from Library.Misc import Sdict
from Library.Misc import GetHelpStringByRemoveHashKey
from Library.ParserValidate import IsValidPcdType
from Library.ParserValidate import IsValidCVariableName
from Library.ParserValidate import IsValidPcdValue
from Library.ParserValidate import IsValidArch
from Library.CommentParsing import ParseComment
from Library.StringUtils import GetSplitValueList
from Library.StringUtils import IsHexDigitUINT32
from Library.ExpressionValidate import IsValidFeatureFlagExp
from Parser.InfAsBuiltProcess import GetPackageListInfo
from Parser.DecParser import Dec
from Object.Parser.InfPackagesObject import InfPackageItem

def ValidateArch(ArchItem, PcdTypeItem1, LineNo, SupArchDict, SupArchList):
    if False:
        print('Hello World!')
    if ArchItem == '' or ArchItem is None:
        ArchItem = 'COMMON'
    if PcdTypeItem1.upper != DT.TAB_INF_FEATURE_PCD.upper():
        ArchList = GetSplitValueList(ArchItem, ' ')
        for ArchItemNew in ArchList:
            if not IsValidArch(ArchItemNew):
                Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % ArchItemNew, File=GlobalData.gINF_MODULE_NAME, Line=LineNo, ExtraData=ArchItemNew)
        SupArchDict[PcdTypeItem1] = ArchList
    else:
        SupArchList.append(ArchItem)
    return (SupArchList, SupArchDict)

def ParsePcdComment(CommentList, PcdTypeItem, PcdItemObj):
    if False:
        i = 10
        return i + 15
    CommentInsList = []
    PreUsage = None
    PreHelpText = ''
    BlockFlag = -1
    FFEHelpText = ''
    CommentItemHelpText = ''
    Count = 0
    for CommentItem in CommentList:
        Count = Count + 1
        (CommentItemUsage, CommentType, CommentString, CommentItemHelpText) = ParseComment(CommentItem, DT.ALL_USAGE_TOKENS, {}, [], False)
        if CommentType and CommentString:
            pass
        if PcdTypeItem == 'FeaturePcd':
            CommentItemUsage = DT.USAGE_ITEM_CONSUMES
            if CommentItemHelpText is None:
                CommentItemHelpText = ''
            if Count == 1:
                FFEHelpText = CommentItemHelpText
            else:
                FFEHelpText = FFEHelpText + DT.END_OF_LINE + CommentItemHelpText
            if Count == len(CommentList):
                CommentItemHelpText = FFEHelpText
                BlockFlag = 4
            else:
                continue
        if CommentItemHelpText is None:
            CommentItemHelpText = ''
            if Count == len(CommentList) and CommentItemUsage == DT.ITEM_UNDEFINED:
                CommentItemHelpText = DT.END_OF_LINE
        if Count == len(CommentList) and (BlockFlag == 1 or BlockFlag == 2):
            if CommentItemUsage == DT.ITEM_UNDEFINED:
                BlockFlag = 4
            else:
                BlockFlag = 3
        elif BlockFlag == -1 and Count == len(CommentList):
            BlockFlag = 4
        if BlockFlag == -1 or BlockFlag == 1 or BlockFlag == 2:
            if CommentItemUsage == DT.ITEM_UNDEFINED:
                if BlockFlag == -1:
                    BlockFlag = 1
                elif BlockFlag == 1:
                    BlockFlag = 2
            elif BlockFlag == 1 or BlockFlag == 2:
                BlockFlag = 3
            elif BlockFlag == -1:
                BlockFlag = 4
        if CommentItemUsage == PreUsage == DT.ITEM_UNDEFINED:
            CommentItemHelpText = PreHelpText + DT.END_OF_LINE + CommentItemHelpText
            PreHelpText = CommentItemHelpText
        if BlockFlag == 4:
            CommentItemIns = InfPcdItemCommentContent()
            CommentItemIns.SetUsageItem(CommentItemUsage)
            CommentItemIns.SetHelpStringItem(CommentItemHelpText)
            CommentInsList.append(CommentItemIns)
            BlockFlag = -1
            PreUsage = None
            PreHelpText = ''
        elif BlockFlag == 3:
            CommentItemIns = InfPcdItemCommentContent()
            CommentItemIns.SetUsageItem(DT.ITEM_UNDEFINED)
            if PreHelpText == '' or PreHelpText.endswith(DT.END_OF_LINE):
                PreHelpText += DT.END_OF_LINE
            CommentItemIns.SetHelpStringItem(PreHelpText)
            CommentInsList.append(CommentItemIns)
            CommentItemIns = InfPcdItemCommentContent()
            CommentItemIns.SetUsageItem(CommentItemUsage)
            CommentItemIns.SetHelpStringItem(CommentItemHelpText)
            CommentInsList.append(CommentItemIns)
            BlockFlag = -1
            PreUsage = None
            PreHelpText = ''
        else:
            PreUsage = CommentItemUsage
            PreHelpText = CommentItemHelpText
    PcdItemObj.SetHelpStringList(CommentInsList)
    return PcdItemObj

class InfPcdItemCommentContent:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.UsageItem = ''
        self.HelpStringItem = ''

    def SetUsageItem(self, UsageItem):
        if False:
            for i in range(10):
                print('nop')
        self.UsageItem = UsageItem

    def GetUsageItem(self):
        if False:
            while True:
                i = 10
        return self.UsageItem

    def SetHelpStringItem(self, HelpStringItem):
        if False:
            print('Hello World!')
        self.HelpStringItem = HelpStringItem

    def GetHelpStringItem(self):
        if False:
            i = 10
            return i + 15
        return self.HelpStringItem

class InfPcdItem:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.CName = ''
        self.Token = ''
        self.TokenSpaceGuidCName = ''
        self.TokenSpaceGuidValue = ''
        self.DatumType = ''
        self.MaxDatumSize = ''
        self.DefaultValue = ''
        self.Offset = ''
        self.ValidUsage = ''
        self.ItemType = ''
        self.SupModuleList = []
        self.HelpStringList = []
        self.FeatureFlagExp = ''
        self.SupArchList = []
        self.PcdErrorsList = []

    def SetCName(self, CName):
        if False:
            return 10
        self.CName = CName

    def GetCName(self):
        if False:
            return 10
        return self.CName

    def SetToken(self, Token):
        if False:
            while True:
                i = 10
        self.Token = Token

    def GetToken(self):
        if False:
            return 10
        return self.Token

    def SetTokenSpaceGuidCName(self, TokenSpaceGuidCName):
        if False:
            while True:
                i = 10
        self.TokenSpaceGuidCName = TokenSpaceGuidCName

    def GetTokenSpaceGuidCName(self):
        if False:
            print('Hello World!')
        return self.TokenSpaceGuidCName

    def SetTokenSpaceGuidValue(self, TokenSpaceGuidValue):
        if False:
            return 10
        self.TokenSpaceGuidValue = TokenSpaceGuidValue

    def GetTokenSpaceGuidValue(self):
        if False:
            return 10
        return self.TokenSpaceGuidValue

    def SetDatumType(self, DatumType):
        if False:
            for i in range(10):
                print('nop')
        self.DatumType = DatumType

    def GetDatumType(self):
        if False:
            while True:
                i = 10
        return self.DatumType

    def SetMaxDatumSize(self, MaxDatumSize):
        if False:
            while True:
                i = 10
        self.MaxDatumSize = MaxDatumSize

    def GetMaxDatumSize(self):
        if False:
            return 10
        return self.MaxDatumSize

    def SetDefaultValue(self, DefaultValue):
        if False:
            while True:
                i = 10
        self.DefaultValue = DefaultValue

    def GetDefaultValue(self):
        if False:
            while True:
                i = 10
        return self.DefaultValue

    def SetPcdErrorsList(self, PcdErrorsList):
        if False:
            while True:
                i = 10
        self.PcdErrorsList = PcdErrorsList

    def GetPcdErrorsList(self):
        if False:
            print('Hello World!')
        return self.PcdErrorsList

    def SetItemType(self, ItemType):
        if False:
            for i in range(10):
                print('nop')
        self.ItemType = ItemType

    def GetItemType(self):
        if False:
            return 10
        return self.ItemType

    def SetSupModuleList(self, SupModuleList):
        if False:
            print('Hello World!')
        self.SupModuleList = SupModuleList

    def GetSupModuleList(self):
        if False:
            print('Hello World!')
        return self.SupModuleList

    def SetHelpStringList(self, HelpStringList):
        if False:
            i = 10
            return i + 15
        self.HelpStringList = HelpStringList

    def GetHelpStringList(self):
        if False:
            return 10
        return self.HelpStringList

    def SetFeatureFlagExp(self, FeatureFlagExp):
        if False:
            while True:
                i = 10
        self.FeatureFlagExp = FeatureFlagExp

    def GetFeatureFlagExp(self):
        if False:
            for i in range(10):
                print('nop')
        return self.FeatureFlagExp

    def SetSupportArchList(self, ArchList):
        if False:
            while True:
                i = 10
        self.SupArchList = ArchList

    def GetSupportArchList(self):
        if False:
            while True:
                i = 10
        return self.SupArchList

    def SetOffset(self, Offset):
        if False:
            i = 10
            return i + 15
        self.Offset = Offset

    def GetOffset(self):
        if False:
            i = 10
            return i + 15
        return self.Offset

    def SetValidUsage(self, ValidUsage):
        if False:
            while True:
                i = 10
        self.ValidUsage = ValidUsage

    def GetValidUsage(self):
        if False:
            for i in range(10):
                print('nop')
        return self.ValidUsage

class InfPcdObject:

    def __init__(self, FileName):
        if False:
            i = 10
            return i + 15
        self.Pcds = Sdict()
        self.FileName = FileName

    def SetPcds(self, PcdContent, KeysList=None, PackageInfo=None):
        if False:
            i = 10
            return i + 15
        if GlobalData.gIS_BINARY_INF:
            self.SetAsBuildPcds(PcdContent, KeysList, PackageInfo)
            return True
        SupArchList = []
        SupArchDict = {}
        PcdTypeItem = ''
        for (PcdTypeItem1, ArchItem, LineNo) in KeysList:
            (SupArchList, SupArchDict) = ValidateArch(ArchItem, PcdTypeItem1, LineNo, SupArchDict, SupArchList)
            if PcdTypeItem1 == '' or PcdTypeItem1 is None:
                return False
            elif not IsValidPcdType(PcdTypeItem1):
                Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_PCD_SECTION_TYPE_ERROR % DT.PCD_USAGE_TYPE_LIST_OF_MODULE, File=GlobalData.gINF_MODULE_NAME, Line=LineNo, ExtraData=PcdTypeItem1)
                return False
            PcdTypeItem = PcdTypeItem1
            for PcdItem in PcdContent:
                PcdItemObj = InfPcdItem()
                CommentList = PcdItem[1]
                CurrentLineOfPcdItem = PcdItem[2]
                PcdItem = PcdItem[0]
                if CommentList is not None and len(CommentList) != 0:
                    PcdItemObj = ParsePcdComment(CommentList, PcdTypeItem, PcdItemObj)
                else:
                    CommentItemIns = InfPcdItemCommentContent()
                    CommentItemIns.SetUsageItem(DT.ITEM_UNDEFINED)
                    PcdItemObj.SetHelpStringList([CommentItemIns])
                if len(PcdItem) >= 1 and len(PcdItem) <= 3:
                    PcdItemObj = SetPcdName(PcdItem, CurrentLineOfPcdItem, PcdItemObj)
                if len(PcdItem) >= 2 and len(PcdItem) <= 3:
                    if IsValidPcdValue(PcdItem[1]) or PcdItem[1].strip() == '':
                        PcdItemObj.SetDefaultValue(PcdItem[1])
                    else:
                        Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_PCD_VALUE_INVALID, File=CurrentLineOfPcdItem[2], Line=CurrentLineOfPcdItem[1], ExtraData=PcdItem[1])
                if len(PcdItem) == 3:
                    if PcdItem[2].strip() == '':
                        Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_FEATURE_FLAG_EXP_MISSING, File=CurrentLineOfPcdItem[2], Line=CurrentLineOfPcdItem[1], ExtraData=CurrentLineOfPcdItem[0])
                    FeatureFlagRtv = IsValidFeatureFlagExp(PcdItem[2].strip())
                    if not FeatureFlagRtv[0]:
                        Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_FEATURE_FLAG_EXP_SYNTAX_INVLID % FeatureFlagRtv[1], File=CurrentLineOfPcdItem[2], Line=CurrentLineOfPcdItem[1], ExtraData=CurrentLineOfPcdItem[0])
                    PcdItemObj.SetFeatureFlagExp(PcdItem[2])
                if len(PcdItem) < 1 or len(PcdItem) > 3:
                    Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_PCD_SECTION_CONTENT_ERROR, File=CurrentLineOfPcdItem[2], Line=CurrentLineOfPcdItem[1], ExtraData=CurrentLineOfPcdItem[0])
                    return False
                if PcdTypeItem.upper != DT.TAB_INF_FEATURE_PCD.upper():
                    PcdItemObj.SetSupportArchList(SupArchDict[PcdTypeItem])
                else:
                    PcdItemObj.SetSupportArchList(SupArchList)
                if (PcdTypeItem, PcdItemObj) in self.Pcds:
                    PcdsList = self.Pcds[PcdTypeItem, PcdItemObj]
                    PcdsList.append(PcdItemObj)
                    self.Pcds[PcdTypeItem, PcdItemObj] = PcdsList
                else:
                    PcdsList = []
                    PcdsList.append(PcdItemObj)
                    self.Pcds[PcdTypeItem, PcdItemObj] = PcdsList
        return True

    def SetAsBuildPcds(self, PcdContent, KeysList=None, PackageInfo=None):
        if False:
            while True:
                i = 10
        for PcdItem in PcdContent:
            PcdItemObj = InfPcdItem()
            CommentList = PcdItem[1]
            CurrentLineOfPcdItem = PcdItem[2]
            PcdItem = PcdItem[0]
            CommentString = ''
            for CommentLine in CommentList:
                CommentString = GetHelpStringByRemoveHashKey(CommentLine)
                CommentItemIns = InfPcdItemCommentContent()
                CommentItemIns.SetHelpStringItem(CommentString)
                CommentItemIns.SetUsageItem(CommentString)
                PcdItemObj.SetHelpStringList(PcdItemObj.GetHelpStringList() + [CommentItemIns])
                if PcdItemObj.GetValidUsage():
                    PcdItemObj.SetValidUsage(PcdItemObj.GetValidUsage() + DT.TAB_VALUE_SPLIT + CommentString)
                else:
                    PcdItemObj.SetValidUsage(CommentString)
            PcdItemObj.SetItemType(KeysList[0][0])
            PcdItemObj = SetPcdName(PcdItem, CurrentLineOfPcdItem, PcdItemObj)
            PcdItemObj = SetValueDatumTypeMaxSizeToken(PcdItem, CurrentLineOfPcdItem, PcdItemObj, KeysList[0][1], PackageInfo)
            PcdTypeItem = KeysList[0][0]
            if (PcdTypeItem, PcdItemObj) in self.Pcds:
                PcdsList = self.Pcds[PcdTypeItem, PcdItemObj]
                PcdsList.append(PcdItemObj)
                self.Pcds[PcdTypeItem, PcdItemObj] = PcdsList
            else:
                PcdsList = []
                PcdsList.append(PcdItemObj)
                self.Pcds[PcdTypeItem, PcdItemObj] = PcdsList

    def GetPcds(self):
        if False:
            while True:
                i = 10
        return self.Pcds

def ParserPcdInfoInDec(String):
    if False:
        for i in range(10):
            print('nop')
    ValueList = GetSplitValueList(String, DT.TAB_VALUE_SPLIT, 3)
    return (ValueList[2], ValueList[3])

def SetValueDatumTypeMaxSizeToken(PcdItem, CurrentLineOfPcdItem, PcdItemObj, Arch, PackageInfo=None):
    if False:
        while True:
            i = 10
    if not PackageInfo:
        PackageInfo = []
        InfFileName = CurrentLineOfPcdItem[2]
        PackageInfoList = GetPackageListInfo(InfFileName, GlobalData.gWORKSPACE, -1)
        for PackageInfoListItem in PackageInfoList:
            PackageInfoIns = InfPackageItem()
            PackageInfoIns.SetPackageName(PackageInfoListItem)
            PackageInfo.append(PackageInfoIns)
    PcdInfoInDecHasFound = False
    for PackageItem in PackageInfo:
        if PcdInfoInDecHasFound:
            break
        PackageName = PackageItem.PackageName
        FullFileName = os.path.normpath(os.path.realpath(os.path.join(GlobalData.gWORKSPACE, PackageName)))
        DecParser = None
        if FullFileName not in GlobalData.gPackageDict:
            DecParser = Dec(FullFileName)
            GlobalData.gPackageDict[FullFileName] = DecParser
        else:
            DecParser = GlobalData.gPackageDict[FullFileName]
        DecPcdsDict = DecParser.GetPcdSectionObject().ValueDict
        for Key in DecPcdsDict.keys():
            if (Key[0] == 'PCDSDYNAMICEX' and PcdItemObj.GetItemType() == 'PcdEx') and (Key[1] == 'COMMON' or Key[1] == Arch):
                for PcdInDec in DecPcdsDict[Key]:
                    if PcdInDec.TokenCName == PcdItemObj.CName and PcdInDec.TokenSpaceGuidCName == PcdItemObj.TokenSpaceGuidCName:
                        PcdItemObj.SetToken(PcdInDec.TokenValue)
                        PcdItemObj.SetDatumType(PcdInDec.DatumType)
                        PcdItemObj.SetSupportArchList([Arch])
                        PcdItemObj.SetDefaultValue(PcdInDec.DefaultValue)
            if (Key[0] == 'PCDSPATCHABLEINMODULE' and PcdItemObj.GetItemType() == 'PatchPcd') and (Key[1] == 'COMMON' or Key[1] == Arch):
                for PcdInDec in DecPcdsDict[Key]:
                    if PcdInDec.TokenCName == PcdItemObj.CName and PcdInDec.TokenSpaceGuidCName == PcdItemObj.TokenSpaceGuidCName:
                        PcdItemObj.SetToken(PcdInDec.TokenValue)
                        PcdItemObj.SetDatumType(PcdInDec.DatumType)
                        PcdItemObj.SetSupportArchList([Arch])
        if PcdItemObj.GetDatumType() == 'VOID*':
            if len(PcdItem) > 1:
                PcdItemObj.SetMaxDatumSize('%s' % len(GetSplitValueList(PcdItem[1], DT.TAB_COMMA_SPLIT)))
        DecGuidsDict = DecParser.GetGuidSectionObject().ValueDict
        for Key in DecGuidsDict.keys():
            if Key == 'COMMON' or Key == Arch:
                for GuidInDec in DecGuidsDict[Key]:
                    if GuidInDec.GuidCName == PcdItemObj.TokenSpaceGuidCName:
                        PcdItemObj.SetTokenSpaceGuidValue(GuidInDec.GuidString)
    if PcdItemObj.GetItemType().upper() == DT.TAB_INF_PATCH_PCD.upper():
        if PcdItem[1] == '0':
            DatumType = PcdItemObj.GetDatumType()
            if DatumType == 'UINT8':
                PcdItem[1] = '0x00'
            if DatumType == 'UINT16':
                PcdItem[1] = '0x0000'
            if DatumType == 'UINT32':
                PcdItem[1] = '0x00000000'
            if DatumType == 'UINT64':
                PcdItem[1] = '0x0000000000000000'
        if ValidatePcdValueOnDatumType(PcdItem[1], PcdItemObj.GetDatumType()):
            PcdItemObj.SetDefaultValue(PcdItem[1])
        else:
            Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_ASBUILD_PCD_VALUE_INVALID % ('"' + PcdItem[1] + '"', '"' + PcdItemObj.GetDatumType() + '"'), File=CurrentLineOfPcdItem[2], Line=CurrentLineOfPcdItem[1], ExtraData=CurrentLineOfPcdItem[0])
        if PcdItemObj.GetItemType().upper() == DT.TAB_INF_PATCH_PCD.upper():
            if not IsHexDigitUINT32(PcdItem[2]):
                Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_ASBUILD_PCD_OFFSET_FORMAT_INVALID % ('"' + PcdItem[2] + '"'), File=CurrentLineOfPcdItem[2], Line=CurrentLineOfPcdItem[1], ExtraData=CurrentLineOfPcdItem[0])
            PcdItemObj.SetOffset(PcdItem[2])
    if PcdItemObj.GetToken() == '' or PcdItemObj.GetDatumType() == '':
        Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_ASBUILD_PCD_DECLARITION_MISS % ('"' + PcdItem[0] + '"'), File=CurrentLineOfPcdItem[2], Line=CurrentLineOfPcdItem[1], ExtraData=CurrentLineOfPcdItem[0])
    return PcdItemObj

def ValidatePcdValueOnDatumType(Value, Type):
    if False:
        print('Hello World!')
    Value = Value.strip()
    if Type == 'BOOLEAN':
        if not (Value == '0x00' or Value == '0x01'):
            return False
    elif Type == 'VOID*':
        if not Value.startswith('{'):
            return False
        if not Value.endswith('}'):
            return False
        Value = Value[1:-1]
        ValueList = GetSplitValueList(Value, DT.TAB_COMMA_SPLIT)
        ReIsValidHexByte = re.compile('^0x[0-9a-f]{1,2}$', re.IGNORECASE)
        for ValueItem in ValueList:
            if not ReIsValidHexByte.match(ValueItem):
                return False
    elif Type == 'UINT8' or Type == 'UINT16' or Type == 'UINT32' or (Type == 'UINT64'):
        ReIsValidUint8z = re.compile('^0[x|X][a-fA-F0-9]{2}$')
        ReIsValidUint16z = re.compile('^0[x|X][a-fA-F0-9]{4}$')
        ReIsValidUint32z = re.compile('^0[x|X][a-fA-F0-9]{8}$')
        ReIsValidUint64z = re.compile('^0[x|X][a-fA-F0-9]{16}$')
        if not ReIsValidUint8z.match(Value) and Type == 'UINT8':
            return False
        elif not ReIsValidUint16z.match(Value) and Type == 'UINT16':
            return False
        elif not ReIsValidUint32z.match(Value) and Type == 'UINT32':
            return False
        elif not ReIsValidUint64z.match(Value) and Type == 'UINT64':
            return False
    else:
        pass
    return True

def SetPcdName(PcdItem, CurrentLineOfPcdItem, PcdItemObj):
    if False:
        print('Hello World!')
    PcdId = GetSplitValueList(PcdItem[0], DT.TAB_SPLIT)
    if len(PcdId) != 2:
        Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_PCD_NAME_FORMAT_ERROR, File=CurrentLineOfPcdItem[2], Line=CurrentLineOfPcdItem[1], ExtraData=CurrentLineOfPcdItem[0])
    else:
        if not IsValidCVariableName(PcdId[0]):
            Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_PCD_CVAR_GUID, File=CurrentLineOfPcdItem[2], Line=CurrentLineOfPcdItem[1], ExtraData=PcdId[0])
        if not IsValidCVariableName(PcdId[1]):
            Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_PCD_CVAR_PCDCNAME, File=CurrentLineOfPcdItem[2], Line=CurrentLineOfPcdItem[1], ExtraData=PcdId[1])
        PcdItemObj.SetTokenSpaceGuidCName(PcdId[0])
        PcdItemObj.SetCName(PcdId[1])
    return PcdItemObj