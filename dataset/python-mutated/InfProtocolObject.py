"""
InfProtocolObject
"""
from Library.ParserValidate import IsValidCVariableName
from Library.CommentParsing import ParseComment
from Library.ExpressionValidate import IsValidFeatureFlagExp
from Library.Misc import Sdict
from Object.Parser.InfMisc import ErrorInInf
from Library import DataType as DT
from Logger import StringTable as ST

def ParseProtocolComment(CommentsList, InfProtocolItemObj):
    if False:
        for i in range(10):
            print('nop')
    CommentInsList = []
    PreUsage = None
    PreNotify = None
    PreHelpText = ''
    BlockFlag = -1
    Count = 0
    for CommentItem in CommentsList:
        Count = Count + 1
        (CommentItemUsage, CommentItemNotify, CommentItemString, CommentItemHelpText) = ParseComment(CommentItem, DT.PROTOCOL_USAGE_TOKENS, DT.PROTOCOL_NOTIFY_TOKENS, ['PROTOCOL'], False)
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
            CommentItemIns = InfProtocolItemCommentContent()
            CommentItemIns.SetUsageItem(CommentItemUsage)
            CommentItemIns.SetNotify(CommentItemNotify)
            CommentItemIns.SetHelpStringItem(CommentItemHelpText)
            CommentInsList.append(CommentItemIns)
            BlockFlag = -1
            PreUsage = None
            PreNotify = None
            PreHelpText = ''
        elif BlockFlag == 3:
            CommentItemIns = InfProtocolItemCommentContent()
            CommentItemIns.SetUsageItem(DT.ITEM_UNDEFINED)
            CommentItemIns.SetNotify(DT.ITEM_UNDEFINED)
            if PreHelpText == '' or PreHelpText.endswith(DT.END_OF_LINE):
                PreHelpText += DT.END_OF_LINE
            CommentItemIns.SetHelpStringItem(PreHelpText)
            CommentInsList.append(CommentItemIns)
            CommentItemIns = InfProtocolItemCommentContent()
            CommentItemIns.SetUsageItem(CommentItemUsage)
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
    InfProtocolItemObj.SetCommentList(CommentInsList)
    return InfProtocolItemObj

class InfProtocolItemCommentContent:

    def __init__(self):
        if False:
            print('Hello World!')
        self.UsageItem = ''
        self.HelpStringItem = ''
        self.Notify = ''
        self.CommentList = []

    def SetUsageItem(self, UsageItem):
        if False:
            while True:
                i = 10
        self.UsageItem = UsageItem

    def GetUsageItem(self):
        if False:
            while True:
                i = 10
        return self.UsageItem

    def SetNotify(self, Notify):
        if False:
            print('Hello World!')
        if Notify != DT.ITEM_UNDEFINED:
            self.Notify = 'true'

    def GetNotify(self):
        if False:
            i = 10
            return i + 15
        return self.Notify

    def SetHelpStringItem(self, HelpStringItem):
        if False:
            for i in range(10):
                print('nop')
        self.HelpStringItem = HelpStringItem

    def GetHelpStringItem(self):
        if False:
            print('Hello World!')
        return self.HelpStringItem

class InfProtocolItem:

    def __init__(self):
        if False:
            print('Hello World!')
        self.Name = ''
        self.FeatureFlagExp = ''
        self.SupArchList = []
        self.CommentList = []

    def SetName(self, Name):
        if False:
            print('Hello World!')
        self.Name = Name

    def GetName(self):
        if False:
            i = 10
            return i + 15
        return self.Name

    def SetFeatureFlagExp(self, FeatureFlagExp):
        if False:
            for i in range(10):
                print('nop')
        self.FeatureFlagExp = FeatureFlagExp

    def GetFeatureFlagExp(self):
        if False:
            return 10
        return self.FeatureFlagExp

    def SetSupArchList(self, SupArchList):
        if False:
            print('Hello World!')
        self.SupArchList = SupArchList

    def GetSupArchList(self):
        if False:
            while True:
                i = 10
        return self.SupArchList

    def SetCommentList(self, CommentList):
        if False:
            print('Hello World!')
        self.CommentList = CommentList

    def GetCommentList(self):
        if False:
            return 10
        return self.CommentList

class InfProtocolObject:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.Protocols = Sdict()
        self.Macros = {}

    def SetProtocol(self, ProtocolContent, Arch=None):
        if False:
            while True:
                i = 10
        __SupArchList = []
        for ArchItem in Arch:
            if ArchItem == '' or ArchItem is None:
                ArchItem = 'COMMON'
            __SupArchList.append(ArchItem)
        for Item in ProtocolContent:
            CommentsList = None
            if len(Item) == 3:
                CommentsList = Item[1]
            CurrentLineOfItem = Item[2]
            LineInfo = (CurrentLineOfItem[2], CurrentLineOfItem[1], CurrentLineOfItem[0])
            Item = Item[0]
            InfProtocolItemObj = InfProtocolItem()
            if len(Item) >= 1 and len(Item) <= 2:
                if not IsValidCVariableName(Item[0]):
                    ErrorInInf(ST.ERR_INF_PARSER_INVALID_CNAME % Item[0], LineInfo=LineInfo)
                if Item[0] != '':
                    InfProtocolItemObj.SetName(Item[0])
                else:
                    ErrorInInf(ST.ERR_INF_PARSER_CNAME_MISSING, LineInfo=LineInfo)
            if len(Item) == 2:
                if Item[1].strip() == '':
                    ErrorInInf(ST.ERR_INF_PARSER_FEATURE_FLAG_EXP_MISSING, LineInfo=LineInfo)
                FeatureFlagRtv = IsValidFeatureFlagExp(Item[1].strip())
                if not FeatureFlagRtv[0]:
                    ErrorInInf(ST.ERR_INF_PARSER_FEATURE_FLAG_EXP_SYNTAX_INVLID % FeatureFlagRtv[1], LineInfo=LineInfo)
                InfProtocolItemObj.SetFeatureFlagExp(Item[1])
            if len(Item) < 1 or len(Item) > 2:
                ErrorInInf(ST.ERR_INF_PARSER_GUID_PPI_PROTOCOL_SECTION_CONTENT_ERROR, LineInfo=LineInfo)
            if CommentsList is not None and len(CommentsList) != 0:
                InfProtocolItemObj = ParseProtocolComment(CommentsList, InfProtocolItemObj)
            else:
                CommentItemIns = InfProtocolItemCommentContent()
                CommentItemIns.SetUsageItem(DT.ITEM_UNDEFINED)
                CommentItemIns.SetNotify(DT.ITEM_UNDEFINED)
                InfProtocolItemObj.SetCommentList([CommentItemIns])
            InfProtocolItemObj.SetSupArchList(__SupArchList)
            for Item in self.Protocols:
                if Item.GetName() == InfProtocolItemObj.GetName():
                    ItemSupArchList = Item.GetSupArchList()
                    for ItemArch in ItemSupArchList:
                        for ProtocolItemObjArch in __SupArchList:
                            if ItemArch == ProtocolItemObjArch:
                                pass
                            if ItemArch.upper() == 'COMMON' or ProtocolItemObjArch.upper() == 'COMMON':
                                pass
            if InfProtocolItemObj in self.Protocols:
                ProcotolList = self.Protocols[InfProtocolItemObj]
                ProcotolList.append(InfProtocolItemObj)
                self.Protocols[InfProtocolItemObj] = ProcotolList
            else:
                ProcotolList = []
                ProcotolList.append(InfProtocolItemObj)
                self.Protocols[InfProtocolItemObj] = ProcotolList
        return True

    def GetProtocol(self):
        if False:
            print('Hello World!')
        return self.Protocols