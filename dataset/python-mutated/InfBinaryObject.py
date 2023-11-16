"""
InfBinaryObject
"""
import os
from copy import deepcopy
from Library import DataType as DT
from Library import GlobalData
import Logger.Log as Logger
from Logger import ToolError
from Logger import StringTable as ST
from Library.Misc import Sdict
from Object.Parser.InfCommonObject import InfSectionCommonDef
from Object.Parser.InfCommonObject import CurrentLine
from Library.Misc import ConvPathFromAbsToRel
from Library.ExpressionValidate import IsValidFeatureFlagExp
from Library.Misc import ValidFile
from Library.ParserValidate import IsValidPath

class InfBianryItem:

    def __init__(self):
        if False:
            return 10
        self.FileName = ''
        self.Target = ''
        self.FeatureFlagExp = ''
        self.HelpString = ''
        self.Type = ''
        self.SupArchList = []

    def SetFileName(self, FileName):
        if False:
            return 10
        self.FileName = FileName

    def GetFileName(self):
        if False:
            for i in range(10):
                print('nop')
        return self.FileName

    def SetTarget(self, Target):
        if False:
            while True:
                i = 10
        self.Target = Target

    def GetTarget(self):
        if False:
            return 10
        return self.Target

    def SetFeatureFlagExp(self, FeatureFlagExp):
        if False:
            i = 10
            return i + 15
        self.FeatureFlagExp = FeatureFlagExp

    def GetFeatureFlagExp(self):
        if False:
            while True:
                i = 10
        return self.FeatureFlagExp

    def SetHelpString(self, HelpString):
        if False:
            return 10
        self.HelpString = HelpString

    def GetHelpString(self):
        if False:
            for i in range(10):
                print('nop')
        return self.HelpString

    def SetType(self, Type):
        if False:
            while True:
                i = 10
        self.Type = Type

    def GetType(self):
        if False:
            return 10
        return self.Type

    def SetSupArchList(self, SupArchList):
        if False:
            i = 10
            return i + 15
        self.SupArchList = SupArchList

    def GetSupArchList(self):
        if False:
            print('Hello World!')
        return self.SupArchList

class InfBianryVerItem(InfBianryItem, CurrentLine):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        InfBianryItem.__init__(self)
        CurrentLine.__init__(self)
        self.VerTypeName = ''

    def SetVerTypeName(self, VerTypeName):
        if False:
            while True:
                i = 10
        self.VerTypeName = VerTypeName

    def GetVerTypeName(self):
        if False:
            i = 10
            return i + 15
        return self.VerTypeName

class InfBianryUiItem(InfBianryItem, CurrentLine):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        InfBianryItem.__init__(self)
        CurrentLine.__init__(self)
        self.UiTypeName = ''

    def SetUiTypeName(self, UiTypeName):
        if False:
            while True:
                i = 10
        self.UiTypeName = UiTypeName

    def GetVerTypeName(self):
        if False:
            i = 10
            return i + 15
        return self.UiTypeName

class InfBianryCommonItem(InfBianryItem, CurrentLine):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.CommonType = ''
        self.TagName = ''
        self.Family = ''
        self.GuidValue = ''
        InfBianryItem.__init__(self)
        CurrentLine.__init__(self)

    def SetCommonType(self, CommonType):
        if False:
            while True:
                i = 10
        self.CommonType = CommonType

    def GetCommonType(self):
        if False:
            return 10
        return self.CommonType

    def SetTagName(self, TagName):
        if False:
            for i in range(10):
                print('nop')
        self.TagName = TagName

    def GetTagName(self):
        if False:
            i = 10
            return i + 15
        return self.TagName

    def SetFamily(self, Family):
        if False:
            return 10
        self.Family = Family

    def GetFamily(self):
        if False:
            for i in range(10):
                print('nop')
        return self.Family

    def SetGuidValue(self, GuidValue):
        if False:
            while True:
                i = 10
        self.GuidValue = GuidValue

    def GetGuidValue(self):
        if False:
            return 10
        return self.GuidValue

class InfBinariesObject(InfSectionCommonDef):

    def __init__(self):
        if False:
            return 10
        self.Binaries = Sdict()
        self.Macros = {}
        InfSectionCommonDef.__init__(self)

    def CheckVer(self, Ver, __SupArchList):
        if False:
            i = 10
            return i + 15
        for VerItem in Ver:
            IsValidFileFlag = False
            VerContent = VerItem[0]
            VerComment = VerItem[1]
            VerCurrentLine = VerItem[2]
            GlobalData.gINF_CURRENT_LINE = VerCurrentLine
            InfBianryVerItemObj = None
            if len(VerContent) < 2:
                Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_BINARY_ITEM_FORMAT_INVALID % (VerContent[0], 2), File=VerCurrentLine.GetFileName(), Line=VerCurrentLine.GetLineNo(), ExtraData=VerCurrentLine.GetLineString())
                return False
            if len(VerContent) > 4:
                Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_BINARY_ITEM_FORMAT_INVALID_MAX % (VerContent[0], 4), File=VerCurrentLine.GetFileName(), Line=VerCurrentLine.GetLineNo(), ExtraData=VerCurrentLine.GetLineString())
                return False
            if len(VerContent) >= 2:
                InfBianryVerItemObj = InfBianryVerItem()
                if VerContent[0] != DT.BINARY_FILE_TYPE_VER:
                    Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_BINARY_VER_TYPE % DT.BINARY_FILE_TYPE_VER, File=VerCurrentLine.GetFileName(), Line=VerCurrentLine.GetLineNo(), ExtraData=VerCurrentLine.GetLineString())
                InfBianryVerItemObj.SetVerTypeName(VerContent[0])
                InfBianryVerItemObj.SetType(VerContent[0])
                FullFileName = os.path.normpath(os.path.realpath(os.path.join(GlobalData.gINF_MODULE_DIR, VerContent[1])))
                if not (ValidFile(FullFileName) or ValidFile(VerContent[1])):
                    Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_BINARY_ITEM_FILE_NOT_EXIST % VerContent[1], File=VerCurrentLine.GetFileName(), Line=VerCurrentLine.GetLineNo(), ExtraData=VerCurrentLine.GetLineString())
                if IsValidPath(VerContent[1], GlobalData.gINF_MODULE_DIR):
                    IsValidFileFlag = True
                else:
                    Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_FILE_NOT_EXIST_OR_NAME_INVALID % VerContent[1], File=VerCurrentLine.GetFileName(), Line=VerCurrentLine.GetLineNo(), ExtraData=VerCurrentLine.GetLineString())
                    return False
                if IsValidFileFlag:
                    VerContent[0] = ConvPathFromAbsToRel(VerContent[0], GlobalData.gINF_MODULE_DIR)
                    InfBianryVerItemObj.SetFileName(VerContent[1])
            if len(VerContent) >= 3:
                InfBianryVerItemObj.SetTarget(VerContent[2])
            if len(VerContent) == 4:
                if VerContent[3].strip() == '':
                    Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_FEATURE_FLAG_EXP_MISSING, File=VerCurrentLine.GetFileName(), Line=VerCurrentLine.GetLineNo(), ExtraData=VerCurrentLine.GetLineString())
                FeatureFlagRtv = IsValidFeatureFlagExp(VerContent[3].strip())
                if not FeatureFlagRtv[0]:
                    Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_FEATURE_FLAG_EXP_SYNTAX_INVLID % FeatureFlagRtv[1], File=VerCurrentLine.GetFileName(), Line=VerCurrentLine.GetLineNo(), ExtraData=VerCurrentLine.GetLineString())
                InfBianryVerItemObj.SetFeatureFlagExp(VerContent[3])
            InfBianryVerItemObj.SetSupArchList(__SupArchList)
            for Item in self.Binaries:
                if Item.GetFileName() == InfBianryVerItemObj.GetFileName():
                    ItemSupArchList = Item.GetSupArchList()
                    for ItemArch in ItemSupArchList:
                        for VerItemObjArch in __SupArchList:
                            if ItemArch == VerItemObjArch:
                                pass
                            if ItemArch.upper() == 'COMMON' or VerItemObjArch.upper() == 'COMMON':
                                pass
            if InfBianryVerItemObj is not None:
                if InfBianryVerItemObj in self.Binaries:
                    BinariesList = self.Binaries[InfBianryVerItemObj]
                    BinariesList.append((InfBianryVerItemObj, VerComment))
                    self.Binaries[InfBianryVerItemObj] = BinariesList
                else:
                    BinariesList = []
                    BinariesList.append((InfBianryVerItemObj, VerComment))
                    self.Binaries[InfBianryVerItemObj] = BinariesList

    def ParseCommonBinary(self, CommonBinary, __SupArchList):
        if False:
            return 10
        for Item in CommonBinary:
            IsValidFileFlag = False
            ItemContent = Item[0]
            ItemComment = Item[1]
            CurrentLineOfItem = Item[2]
            GlobalData.gINF_CURRENT_LINE = CurrentLineOfItem
            InfBianryCommonItemObj = None
            if ItemContent[0] == 'SUBTYPE_GUID':
                if len(ItemContent) < 3:
                    Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_BINARY_ITEM_FORMAT_INVALID % (ItemContent[0], 3), File=CurrentLineOfItem.GetFileName(), Line=CurrentLineOfItem.GetLineNo(), ExtraData=CurrentLineOfItem.GetLineString())
                    return False
            elif len(ItemContent) < 2:
                Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_BINARY_ITEM_FORMAT_INVALID % (ItemContent[0], 2), File=CurrentLineOfItem.GetFileName(), Line=CurrentLineOfItem.GetLineNo(), ExtraData=CurrentLineOfItem.GetLineString())
                return False
            if len(ItemContent) > 7:
                Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_BINARY_ITEM_FORMAT_INVALID_MAX % (ItemContent[0], 7), File=CurrentLineOfItem.GetFileName(), Line=CurrentLineOfItem.GetLineNo(), ExtraData=CurrentLineOfItem.GetLineString())
                return False
            if len(ItemContent) >= 2:
                InfBianryCommonItemObj = InfBianryCommonItem()
                BinaryFileType = ItemContent[0].strip()
                if BinaryFileType == 'RAW' or BinaryFileType == 'ACPI' or BinaryFileType == 'ASL':
                    BinaryFileType = 'BIN'
                if BinaryFileType not in DT.BINARY_FILE_TYPE_LIST:
                    Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_BINARY_ITEM_INVALID_FILETYPE % DT.BINARY_FILE_TYPE_LIST.__str__(), File=CurrentLineOfItem.GetFileName(), Line=CurrentLineOfItem.GetLineNo(), ExtraData=CurrentLineOfItem.GetLineString())
                if BinaryFileType == 'SUBTYPE_GUID':
                    BinaryFileType = 'FREEFORM'
                if BinaryFileType == 'LIB' or BinaryFileType == 'UEFI_APP':
                    Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_BINARY_ITEM_INVALID_FILETYPE % DT.BINARY_FILE_TYPE_LIST.__str__(), File=CurrentLineOfItem.GetFileName(), Line=CurrentLineOfItem.GetLineNo(), ExtraData=CurrentLineOfItem.GetLineString())
                InfBianryCommonItemObj.SetType(BinaryFileType)
                InfBianryCommonItemObj.SetCommonType(ItemContent[0])
                FileName = ''
                if BinaryFileType == 'FREEFORM':
                    InfBianryCommonItemObj.SetGuidValue(ItemContent[1])
                    if len(ItemContent) >= 3:
                        FileName = ItemContent[2]
                    else:
                        Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_BINARY_ITEM_FILENAME_NOT_EXIST, File=CurrentLineOfItem.GetFileName(), Line=CurrentLineOfItem.GetLineNo(), ExtraData=CurrentLineOfItem.GetLineString())
                else:
                    FileName = ItemContent[1]
                FullFileName = os.path.normpath(os.path.realpath(os.path.join(GlobalData.gINF_MODULE_DIR, FileName)))
                if not (ValidFile(FullFileName) or ValidFile(FileName)):
                    Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_BINARY_ITEM_FILE_NOT_EXIST % FileName, File=CurrentLineOfItem.GetFileName(), Line=CurrentLineOfItem.GetLineNo(), ExtraData=CurrentLineOfItem.GetLineString())
                if IsValidPath(FileName, GlobalData.gINF_MODULE_DIR):
                    IsValidFileFlag = True
                else:
                    Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_FILE_NOT_EXIST_OR_NAME_INVALID % FileName, File=CurrentLineOfItem.GetFileName(), Line=CurrentLineOfItem.GetLineNo(), ExtraData=CurrentLineOfItem.GetLineString())
                    return False
                if IsValidFileFlag:
                    ItemContent[0] = ConvPathFromAbsToRel(ItemContent[0], GlobalData.gINF_MODULE_DIR)
                    InfBianryCommonItemObj.SetFileName(FileName)
            if len(ItemContent) >= 3:
                if BinaryFileType != 'FREEFORM':
                    InfBianryCommonItemObj.SetTarget(ItemContent[2])
            if len(ItemContent) >= 4:
                if BinaryFileType != 'FREEFORM':
                    InfBianryCommonItemObj.SetFamily(ItemContent[3])
                else:
                    InfBianryCommonItemObj.SetTarget(ItemContent[3])
            if len(ItemContent) >= 5:
                if BinaryFileType != 'FREEFORM':
                    if ItemContent[4].strip() != '':
                        Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_TAGNAME_NOT_PERMITTED % ItemContent[4], File=CurrentLineOfItem.GetFileName(), Line=CurrentLineOfItem.GetLineNo(), ExtraData=CurrentLineOfItem.GetLineString())
                else:
                    InfBianryCommonItemObj.SetFamily(ItemContent[4])
            if len(ItemContent) >= 6:
                if BinaryFileType != 'FREEFORM':
                    if ItemContent[5].strip() == '':
                        Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_FEATURE_FLAG_EXP_MISSING, File=CurrentLineOfItem.GetFileName(), Line=CurrentLineOfItem.GetLineNo(), ExtraData=CurrentLineOfItem.GetLineString())
                    FeatureFlagRtv = IsValidFeatureFlagExp(ItemContent[5].strip())
                    if not FeatureFlagRtv[0]:
                        Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_FEATURE_FLAG_EXP_SYNTAX_INVLID % FeatureFlagRtv[1], File=CurrentLineOfItem.GetFileName(), Line=CurrentLineOfItem.GetLineNo(), ExtraData=CurrentLineOfItem.GetLineString())
                    InfBianryCommonItemObj.SetFeatureFlagExp(ItemContent[5])
                elif ItemContent[5].strip() != '':
                    Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_TAGNAME_NOT_PERMITTED % ItemContent[5], File=CurrentLineOfItem.GetFileName(), Line=CurrentLineOfItem.GetLineNo(), ExtraData=CurrentLineOfItem.GetLineString())
            if len(ItemContent) == 7:
                if ItemContent[6].strip() == '':
                    Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_FEATURE_FLAG_EXP_MISSING, File=CurrentLineOfItem.GetFileName(), Line=CurrentLineOfItem.GetLineNo(), ExtraData=CurrentLineOfItem.GetLineString())
                FeatureFlagRtv = IsValidFeatureFlagExp(ItemContent[6].strip())
                if not FeatureFlagRtv[0]:
                    Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_FEATURE_FLAG_EXP_SYNTAX_INVLID % FeatureFlagRtv[1], File=CurrentLineOfItem.GetFileName(), Line=CurrentLineOfItem.GetLineNo(), ExtraData=CurrentLineOfItem.GetLineString())
                InfBianryCommonItemObj.SetFeatureFlagExp(ItemContent[6])
            InfBianryCommonItemObj.SetSupArchList(__SupArchList)
            if InfBianryCommonItemObj is not None:
                if InfBianryCommonItemObj in self.Binaries:
                    BinariesList = self.Binaries[InfBianryCommonItemObj]
                    BinariesList.append((InfBianryCommonItemObj, ItemComment))
                    self.Binaries[InfBianryCommonItemObj] = BinariesList
                else:
                    BinariesList = []
                    BinariesList.append((InfBianryCommonItemObj, ItemComment))
                    self.Binaries[InfBianryCommonItemObj] = BinariesList

    def SetBinary(self, UiInf=None, Ver=None, CommonBinary=None, ArchList=None):
        if False:
            while True:
                i = 10
        __SupArchList = []
        for ArchItem in ArchList:
            if ArchItem == '' or ArchItem is None:
                ArchItem = 'COMMON'
            __SupArchList.append(ArchItem)
        if UiInf is not None:
            if len(UiInf) > 0:
                for UiItem in UiInf:
                    IsValidFileFlag = False
                    InfBianryUiItemObj = None
                    UiContent = UiItem[0]
                    UiComment = UiItem[1]
                    UiCurrentLine = UiItem[2]
                    GlobalData.gINF_CURRENT_LINE = deepcopy(UiItem[2])
                    if len(UiContent) < 2:
                        Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_BINARY_ITEM_FORMAT_INVALID % (UiContent[0], 2), File=UiCurrentLine.GetFileName(), Line=UiCurrentLine.GetLineNo(), ExtraData=UiCurrentLine.GetLineString())
                        return False
                    if len(UiContent) > 4:
                        Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_BINARY_ITEM_FORMAT_INVALID_MAX % (UiContent[0], 4), File=UiCurrentLine.GetFileName(), Line=UiCurrentLine.GetLineNo(), ExtraData=UiCurrentLine.GetLineString())
                        return False
                    if len(UiContent) >= 2:
                        InfBianryUiItemObj = InfBianryUiItem()
                        if UiContent[0] != 'UI':
                            Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_BINARY_VER_TYPE % 'UI', File=UiCurrentLine.GetFileName(), Line=UiCurrentLine.GetLineNo(), ExtraData=UiCurrentLine.GetLineString())
                        InfBianryUiItemObj.SetUiTypeName(UiContent[0])
                        InfBianryUiItemObj.SetType(UiContent[0])
                        FullFileName = os.path.normpath(os.path.realpath(os.path.join(GlobalData.gINF_MODULE_DIR, UiContent[1])))
                        if not (ValidFile(FullFileName) or ValidFile(UiContent[1])):
                            Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_BINARY_ITEM_FILE_NOT_EXIST % UiContent[1], File=UiCurrentLine.GetFileName(), Line=UiCurrentLine.GetLineNo(), ExtraData=UiCurrentLine.GetLineString())
                        if IsValidPath(UiContent[1], GlobalData.gINF_MODULE_DIR):
                            IsValidFileFlag = True
                        else:
                            Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_FILE_NOT_EXIST_OR_NAME_INVALID % UiContent[1], File=UiCurrentLine.GetFileName(), Line=UiCurrentLine.GetLineNo(), ExtraData=UiCurrentLine.GetLineString())
                            return False
                        if IsValidFileFlag:
                            UiContent[0] = ConvPathFromAbsToRel(UiContent[0], GlobalData.gINF_MODULE_DIR)
                            InfBianryUiItemObj.SetFileName(UiContent[1])
                    if len(UiContent) >= 3:
                        InfBianryUiItemObj.SetTarget(UiContent[2])
                    if len(UiContent) == 4:
                        if UiContent[3].strip() == '':
                            Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_FEATURE_FLAG_EXP_MISSING, File=UiCurrentLine.GetFileName(), Line=UiCurrentLine.GetLineNo(), ExtraData=UiCurrentLine.GetLineString())
                        FeatureFlagRtv = IsValidFeatureFlagExp(UiContent[3].strip())
                        if not FeatureFlagRtv[0]:
                            Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_FEATURE_FLAG_EXP_SYNTAX_INVLID % FeatureFlagRtv[1], File=UiCurrentLine.GetFileName(), Line=UiCurrentLine.GetLineNo(), ExtraData=UiCurrentLine.GetLineString())
                        InfBianryUiItemObj.SetFeatureFlagExp(UiContent[3])
                    InfBianryUiItemObj.SetSupArchList(__SupArchList)
                    if InfBianryUiItemObj is not None:
                        if InfBianryUiItemObj in self.Binaries:
                            BinariesList = self.Binaries[InfBianryUiItemObj]
                            BinariesList.append((InfBianryUiItemObj, UiComment))
                            self.Binaries[InfBianryUiItemObj] = BinariesList
                        else:
                            BinariesList = []
                            BinariesList.append((InfBianryUiItemObj, UiComment))
                            self.Binaries[InfBianryUiItemObj] = BinariesList
        if Ver is not None and len(Ver) > 0:
            self.CheckVer(Ver, __SupArchList)
        if CommonBinary and len(CommonBinary) > 0:
            self.ParseCommonBinary(CommonBinary, __SupArchList)
        return True

    def GetBinary(self):
        if False:
            for i in range(10):
                print('nop')
        return self.Binaries