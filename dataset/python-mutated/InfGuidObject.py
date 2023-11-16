"""
InfGuidObject
"""
from Library.ParserValidate import IsValidCVariableName
from Library.CommentParsing import ParseComment
from Library.ExpressionValidate import IsValidFeatureFlagExp
from Library.Misc import Sdict
from Library import DataType as DT
import Logger.Log as Logger
from Logger import ToolError
from Logger import StringTable as ST

class InfGuidItemCommentContent:

    def __init__(self):
        if False:
            print('Hello World!')
        self.UsageItem = ''
        self.GuidTypeItem = ''
        self.VariableNameItem = ''
        self.HelpStringItem = ''

    def SetUsageItem(self, UsageItem):
        if False:
            while True:
                i = 10
        self.UsageItem = UsageItem

    def GetUsageItem(self):
        if False:
            print('Hello World!')
        return self.UsageItem

    def SetGuidTypeItem(self, GuidTypeItem):
        if False:
            print('Hello World!')
        self.GuidTypeItem = GuidTypeItem

    def GetGuidTypeItem(self):
        if False:
            print('Hello World!')
        return self.GuidTypeItem

    def SetVariableNameItem(self, VariableNameItem):
        if False:
            while True:
                i = 10
        self.VariableNameItem = VariableNameItem

    def GetVariableNameItem(self):
        if False:
            print('Hello World!')
        return self.VariableNameItem

    def SetHelpStringItem(self, HelpStringItem):
        if False:
            return 10
        self.HelpStringItem = HelpStringItem

    def GetHelpStringItem(self):
        if False:
            return 10
        return self.HelpStringItem

class InfGuidItem:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.Name = ''
        self.FeatureFlagExp = ''
        self.CommentList = []
        self.SupArchList = []

    def SetName(self, Name):
        if False:
            while True:
                i = 10
        self.Name = Name

    def GetName(self):
        if False:
            i = 10
            return i + 15
        return self.Name

    def SetFeatureFlagExp(self, FeatureFlagExp):
        if False:
            print('Hello World!')
        self.FeatureFlagExp = FeatureFlagExp

    def GetFeatureFlagExp(self):
        if False:
            print('Hello World!')
        return self.FeatureFlagExp

    def SetCommentList(self, CommentList):
        if False:
            while True:
                i = 10
        self.CommentList = CommentList

    def GetCommentList(self):
        if False:
            while True:
                i = 10
        return self.CommentList

    def SetSupArchList(self, SupArchList):
        if False:
            print('Hello World!')
        self.SupArchList = SupArchList

    def GetSupArchList(self):
        if False:
            return 10
        return self.SupArchList

def ParseGuidComment(CommentsList, InfGuidItemObj):
    if False:
        while True:
            i = 10
    if CommentsList is not None and len(CommentsList) != 0:
        CommentInsList = []
        PreUsage = None
        PreGuidType = None
        PreHelpText = ''
        BlockFlag = -1
        Count = 0
        for CommentItem in CommentsList:
            Count = Count + 1
            (CommentItemUsage, CommentItemGuidType, CommentItemVarString, CommentItemHelpText) = ParseComment(CommentItem, DT.ALL_USAGE_TOKENS, DT.GUID_TYPE_TOKENS, [], True)
            if CommentItemHelpText is None:
                CommentItemHelpText = ''
                if Count == len(CommentsList) and CommentItemUsage == CommentItemGuidType == DT.ITEM_UNDEFINED:
                    CommentItemHelpText = DT.END_OF_LINE
            if Count == len(CommentsList):
                if BlockFlag == 1 or BlockFlag == 2:
                    if CommentItemUsage == CommentItemGuidType == DT.ITEM_UNDEFINED:
                        BlockFlag = 4
                    else:
                        BlockFlag = 3
                if BlockFlag == -1:
                    BlockFlag = 4
            if BlockFlag == -1 or BlockFlag == 1 or BlockFlag == 2:
                if CommentItemUsage == CommentItemGuidType == DT.ITEM_UNDEFINED:
                    if BlockFlag == -1:
                        BlockFlag = 1
                    elif BlockFlag == 1:
                        BlockFlag = 2
                elif BlockFlag == 1 or BlockFlag == 2:
                    BlockFlag = 3
                elif BlockFlag == -1:
                    BlockFlag = 4
            if CommentItemUsage == CommentItemGuidType == PreUsage == PreGuidType == DT.ITEM_UNDEFINED:
                CommentItemHelpText = PreHelpText + DT.END_OF_LINE + CommentItemHelpText
                PreHelpText = CommentItemHelpText
            if BlockFlag == 4:
                CommentItemIns = InfGuidItemCommentContent()
                CommentItemIns.SetUsageItem(CommentItemUsage)
                CommentItemIns.SetGuidTypeItem(CommentItemGuidType)
                CommentItemIns.SetVariableNameItem(CommentItemVarString)
                if CommentItemHelpText == '' or CommentItemHelpText.endswith(DT.END_OF_LINE):
                    CommentItemHelpText = CommentItemHelpText.strip(DT.END_OF_LINE)
                CommentItemIns.SetHelpStringItem(CommentItemHelpText)
                CommentInsList.append(CommentItemIns)
                BlockFlag = -1
                PreUsage = None
                PreGuidType = None
                PreHelpText = ''
            elif BlockFlag == 3:
                CommentItemIns = InfGuidItemCommentContent()
                CommentItemIns.SetUsageItem(DT.ITEM_UNDEFINED)
                CommentItemIns.SetGuidTypeItem(DT.ITEM_UNDEFINED)
                if PreHelpText == '' or PreHelpText.endswith(DT.END_OF_LINE):
                    PreHelpText = PreHelpText.strip(DT.END_OF_LINE)
                CommentItemIns.SetHelpStringItem(PreHelpText)
                CommentInsList.append(CommentItemIns)
                CommentItemIns = InfGuidItemCommentContent()
                CommentItemIns.SetUsageItem(CommentItemUsage)
                CommentItemIns.SetGuidTypeItem(CommentItemGuidType)
                CommentItemIns.SetVariableNameItem(CommentItemVarString)
                if CommentItemHelpText == '' or CommentItemHelpText.endswith(DT.END_OF_LINE):
                    CommentItemHelpText = CommentItemHelpText.strip(DT.END_OF_LINE)
                CommentItemIns.SetHelpStringItem(CommentItemHelpText)
                CommentInsList.append(CommentItemIns)
                BlockFlag = -1
                PreUsage = None
                PreGuidType = None
                PreHelpText = ''
            else:
                PreUsage = CommentItemUsage
                PreGuidType = CommentItemGuidType
                PreHelpText = CommentItemHelpText
        InfGuidItemObj.SetCommentList(CommentInsList)
    else:
        CommentItemIns = InfGuidItemCommentContent()
        CommentItemIns.SetUsageItem(DT.ITEM_UNDEFINED)
        CommentItemIns.SetGuidTypeItem(DT.ITEM_UNDEFINED)
        InfGuidItemObj.SetCommentList([CommentItemIns])
    return InfGuidItemObj

class InfGuidObject:

    def __init__(self):
        if False:
            return 10
        self.Guids = Sdict()
        self.Macros = {}

    def SetGuid(self, GuidList, Arch=None):
        if False:
            i = 10
            return i + 15
        __SupportArchList = []
        for ArchItem in Arch:
            if ArchItem == '' or ArchItem is None:
                ArchItem = 'COMMON'
            __SupportArchList.append(ArchItem)
        for Item in GuidList:
            CommentsList = None
            if len(Item) == 3:
                CommentsList = Item[1]
            CurrentLineOfItem = Item[2]
            Item = Item[0]
            InfGuidItemObj = InfGuidItem()
            if len(Item) >= 1 and len(Item) <= 2:
                if not IsValidCVariableName(Item[0]):
                    Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_INVALID_CNAME % Item[0], File=CurrentLineOfItem[2], Line=CurrentLineOfItem[1], ExtraData=CurrentLineOfItem[0])
                if Item[0] != '':
                    InfGuidItemObj.SetName(Item[0])
                else:
                    Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_CNAME_MISSING, File=CurrentLineOfItem[2], Line=CurrentLineOfItem[1], ExtraData=CurrentLineOfItem[0])
            if len(Item) == 2:
                if Item[1].strip() == '':
                    Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_FEATURE_FLAG_EXP_MISSING, File=CurrentLineOfItem[2], Line=CurrentLineOfItem[1], ExtraData=CurrentLineOfItem[0])
                FeatureFlagRtv = IsValidFeatureFlagExp(Item[1].strip())
                if not FeatureFlagRtv[0]:
                    Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_FEATURE_FLAG_EXP_SYNTAX_INVLID % FeatureFlagRtv[1], File=CurrentLineOfItem[2], Line=CurrentLineOfItem[1], ExtraData=CurrentLineOfItem[0])
                InfGuidItemObj.SetFeatureFlagExp(Item[1])
            if len(Item) != 1 and len(Item) != 2:
                Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_GUID_PPI_PROTOCOL_SECTION_CONTENT_ERROR, File=CurrentLineOfItem[2], Line=CurrentLineOfItem[1], ExtraData=CurrentLineOfItem[0])
            InfGuidItemObj = ParseGuidComment(CommentsList, InfGuidItemObj)
            InfGuidItemObj.SetSupArchList(__SupportArchList)
            for Item in self.Guids:
                if Item.GetName() == InfGuidItemObj.GetName():
                    ItemSupArchList = Item.GetSupArchList()
                    for ItemArch in ItemSupArchList:
                        for GuidItemObjArch in __SupportArchList:
                            if ItemArch == GuidItemObjArch:
                                pass
                            if ItemArch.upper() == 'COMMON' or GuidItemObjArch.upper() == 'COMMON':
                                pass
            if InfGuidItemObj in self.Guids:
                GuidList = self.Guids[InfGuidItemObj]
                GuidList.append(InfGuidItemObj)
                self.Guids[InfGuidItemObj] = GuidList
            else:
                GuidList = []
                GuidList.append(InfGuidItemObj)
                self.Guids[InfGuidItemObj] = GuidList
        return True

    def GetGuid(self):
        if False:
            while True:
                i = 10
        return self.Guids