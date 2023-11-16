"""
InfPpiObject
"""
from Library.ParserValidate import IsValidCVariableName
from Library.CommentParsing import ParseComment
from Library.ExpressionValidate import IsValidFeatureFlagExp
from Library.Misc import Sdict
from Library import DataType as DT
import Logger.Log as Logger
from Logger import ToolError
from Logger import StringTable as ST

def ParsePpiComment(CommentsList, InfPpiItemObj):
    if False:
        print('Hello World!')
    PreNotify = None
    PreUsage = None
    PreHelpText = ''
    BlockFlag = -1
    CommentInsList = []
    Count = 0
    for CommentItem in CommentsList:
        Count = Count + 1
        (CommentItemUsage, CommentItemNotify, CommentItemString, CommentItemHelpText) = ParseComment(CommentItem, DT.ALL_USAGE_TOKENS, DT.PPI_NOTIFY_TOKENS, ['PPI'], False)
        if CommentItemString:
            pass
        if CommentItemHelpText is None:
            CommentItemHelpText = ''
            if Count == len(CommentsList) and CommentItemUsage == CommentItemNotify == DT.ITEM_UNDEFINED:
                CommentItemHelpText = DT.END_OF_LINE
        if Count == len(CommentsList):
            if BlockFlag == 1 or BlockFlag == 2:
                if CommentItemUsage == CommentItemNotify == DT.ITEM_UNDEFINED:
                    BlockFlag = 4
                else:
                    BlockFlag = 3
            elif BlockFlag == -1:
                BlockFlag = 4
        if BlockFlag == -1 or BlockFlag == 1 or BlockFlag == 2:
            if CommentItemUsage == CommentItemNotify == DT.ITEM_UNDEFINED:
                if BlockFlag == -1:
                    BlockFlag = 1
                elif BlockFlag == 1:
                    BlockFlag = 2
            elif BlockFlag == 1 or BlockFlag == 2:
                BlockFlag = 3
            elif BlockFlag == -1:
                BlockFlag = 4
        if CommentItemUsage == CommentItemNotify == PreUsage == PreNotify == DT.ITEM_UNDEFINED:
            CommentItemHelpText = PreHelpText + DT.END_OF_LINE + CommentItemHelpText
            PreHelpText = CommentItemHelpText
        if BlockFlag == 4:
            CommentItemIns = InfPpiItemCommentContent()
            CommentItemIns.SetUsage(CommentItemUsage)
            CommentItemIns.SetNotify(CommentItemNotify)
            CommentItemIns.SetHelpStringItem(CommentItemHelpText)
            CommentInsList.append(CommentItemIns)
            BlockFlag = -1
            PreUsage = None
            PreNotify = None
            PreHelpText = ''
        elif BlockFlag == 3:
            CommentItemIns = InfPpiItemCommentContent()
            CommentItemIns.SetUsage(DT.ITEM_UNDEFINED)
            CommentItemIns.SetNotify(DT.ITEM_UNDEFINED)
            if PreHelpText == '' or PreHelpText.endswith(DT.END_OF_LINE):
                PreHelpText += DT.END_OF_LINE
            CommentItemIns.SetHelpStringItem(PreHelpText)
            CommentInsList.append(CommentItemIns)
            CommentItemIns = InfPpiItemCommentContent()
            CommentItemIns.SetUsage(CommentItemUsage)
            CommentItemIns.SetNotify(CommentItemNotify)
            CommentItemIns.SetHelpStringItem(CommentItemHelpText)
            CommentInsList.append(CommentItemIns)
            BlockFlag = -1
            PreUsage = None
            PreNotify = None
            PreHelpText = ''
        else:
            PreUsage = CommentItemUsage
            PreNotify = CommentItemNotify
            PreHelpText = CommentItemHelpText
    InfPpiItemObj.SetCommentList(CommentInsList)
    return InfPpiItemObj

class InfPpiItemCommentContent:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.UsageItem = ''
        self.HelpStringItem = ''
        self.Notify = ''
        self.CommentList = []

    def SetUsage(self, UsageItem):
        if False:
            i = 10
            return i + 15
        self.UsageItem = UsageItem

    def GetUsage(self):
        if False:
            i = 10
            return i + 15
        return self.UsageItem

    def SetNotify(self, Notify):
        if False:
            while True:
                i = 10
        if Notify != DT.ITEM_UNDEFINED:
            self.Notify = 'true'

    def GetNotify(self):
        if False:
            return 10
        return self.Notify

    def SetHelpStringItem(self, HelpStringItem):
        if False:
            i = 10
            return i + 15
        self.HelpStringItem = HelpStringItem

    def GetHelpStringItem(self):
        if False:
            for i in range(10):
                print('nop')
        return self.HelpStringItem

class InfPpiItem:

    def __init__(self):
        if False:
            print('Hello World!')
        self.Name = ''
        self.FeatureFlagExp = ''
        self.SupArchList = []
        self.CommentList = []

    def SetName(self, Name):
        if False:
            return 10
        self.Name = Name

    def GetName(self):
        if False:
            print('Hello World!')
        return self.Name

    def SetSupArchList(self, SupArchList):
        if False:
            while True:
                i = 10
        self.SupArchList = SupArchList

    def GetSupArchList(self):
        if False:
            return 10
        return self.SupArchList

    def SetCommentList(self, CommentList):
        if False:
            for i in range(10):
                print('nop')
        self.CommentList = CommentList

    def GetCommentList(self):
        if False:
            i = 10
            return i + 15
        return self.CommentList

    def SetFeatureFlagExp(self, FeatureFlagExp):
        if False:
            return 10
        self.FeatureFlagExp = FeatureFlagExp

    def GetFeatureFlagExp(self):
        if False:
            i = 10
            return i + 15
        return self.FeatureFlagExp

class InfPpiObject:

    def __init__(self):
        if False:
            return 10
        self.Ppis = Sdict()
        self.Macros = {}

    def SetPpi(self, PpiList, Arch=None):
        if False:
            i = 10
            return i + 15
        __SupArchList = []
        for ArchItem in Arch:
            if ArchItem == '' or ArchItem is None:
                ArchItem = 'COMMON'
            __SupArchList.append(ArchItem)
        for Item in PpiList:
            CommentsList = None
            if len(Item) == 3:
                CommentsList = Item[1]
            CurrentLineOfItem = Item[2]
            Item = Item[0]
            InfPpiItemObj = InfPpiItem()
            if len(Item) >= 1 and len(Item) <= 2:
                if not IsValidCVariableName(Item[0]):
                    Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_INVALID_CNAME % Item[0], File=CurrentLineOfItem[2], Line=CurrentLineOfItem[1], ExtraData=CurrentLineOfItem[0])
                if Item[0] != '':
                    InfPpiItemObj.SetName(Item[0])
                else:
                    Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_CNAME_MISSING, File=CurrentLineOfItem[2], Line=CurrentLineOfItem[1], ExtraData=CurrentLineOfItem[0])
            if len(Item) == 2:
                if Item[1].strip() == '':
                    Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_FEATURE_FLAG_EXP_MISSING, File=CurrentLineOfItem[2], Line=CurrentLineOfItem[1], ExtraData=CurrentLineOfItem[0])
                FeatureFlagRtv = IsValidFeatureFlagExp(Item[1].strip())
                if not FeatureFlagRtv[0]:
                    Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_FEATURE_FLAG_EXP_SYNTAX_INVLID % FeatureFlagRtv[1], File=CurrentLineOfItem[2], Line=CurrentLineOfItem[1], ExtraData=CurrentLineOfItem[0])
                InfPpiItemObj.SetFeatureFlagExp(Item[1])
            if len(Item) != 1 and len(Item) != 2:
                Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_GUID_PPI_PROTOCOL_SECTION_CONTENT_ERROR, File=CurrentLineOfItem[2], Line=CurrentLineOfItem[1], ExtraData=CurrentLineOfItem[0])
            if CommentsList is not None and len(CommentsList) != 0:
                InfPpiItemObj = ParsePpiComment(CommentsList, InfPpiItemObj)
            else:
                CommentItemIns = InfPpiItemCommentContent()
                CommentItemIns.SetUsage(DT.ITEM_UNDEFINED)
                CommentItemIns.SetNotify(DT.ITEM_UNDEFINED)
                InfPpiItemObj.SetCommentList([CommentItemIns])
            InfPpiItemObj.SetSupArchList(__SupArchList)
            for Item in self.Ppis:
                if Item.GetName() == InfPpiItemObj.GetName():
                    ItemSupArchList = Item.GetSupArchList()
                    for ItemArch in ItemSupArchList:
                        for PpiItemObjArch in __SupArchList:
                            if ItemArch == PpiItemObjArch:
                                pass
                            if ItemArch.upper() == 'COMMON' or PpiItemObjArch.upper() == 'COMMON':
                                pass
            if InfPpiItemObj in self.Ppis:
                PpiList = self.Ppis[InfPpiItemObj]
                PpiList.append(InfPpiItemObj)
                self.Ppis[InfPpiItemObj] = PpiList
            else:
                PpiList = []
                PpiList.append(InfPpiItemObj)
                self.Ppis[InfPpiItemObj] = PpiList
        return True

    def GetPpi(self):
        if False:
            print('Hello World!')
        return self.Ppis