"""
InfSourcesObject
"""
import os
from Logger import StringTable as ST
from Logger import ToolError
import Logger.Log as Logger
from Library import GlobalData
from Library.Misc import Sdict
from Library.ExpressionValidate import IsValidFeatureFlagExp
from Object.Parser.InfCommonObject import InfSectionCommonDef
from Library.Misc import ValidFile
from Library.ParserValidate import IsValidFamily
from Library.ParserValidate import IsValidPath

def GenSourceInstance(Item, CurrentLineOfItem, ItemObj):
    if False:
        return 10
    IsValidFileFlag = False
    if len(Item) < 6 and len(Item) >= 1:
        if len(Item) == 5:
            if Item[4].strip() == '':
                Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_FEATURE_FLAG_EXP_MISSING, File=CurrentLineOfItem[2], Line=CurrentLineOfItem[1], ExtraData=CurrentLineOfItem[0])
            FeatureFlagRtv = IsValidFeatureFlagExp(Item[4].strip())
            if not FeatureFlagRtv[0]:
                Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_FEATURE_FLAG_EXP_SYNTAX_INVLID % FeatureFlagRtv[1], File=CurrentLineOfItem[2], Line=CurrentLineOfItem[1], ExtraData=CurrentLineOfItem[0])
            ItemObj.SetFeatureFlagExp(Item[4])
        if len(Item) >= 4:
            if Item[3].strip() == '':
                ItemObj.SetToolCode(Item[3])
            else:
                Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_TOOLCODE_NOT_PERMITTED % Item[2], File=CurrentLineOfItem[2], Line=CurrentLineOfItem[1], ExtraData=CurrentLineOfItem[0])
        if len(Item) >= 3:
            if Item[2].strip() == '':
                ItemObj.SetTagName(Item[2])
            else:
                Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_TAGNAME_NOT_PERMITTED % Item[2], File=CurrentLineOfItem[2], Line=CurrentLineOfItem[1], ExtraData=CurrentLineOfItem[0])
        if len(Item) >= 2:
            if IsValidFamily(Item[1].strip()):
                if Item[1].strip() == '*':
                    Item[1] = ''
                ItemObj.SetFamily(Item[1])
            else:
                Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_SOURCE_SECTION_FAMILY_INVALID % Item[1], File=CurrentLineOfItem[2], Line=CurrentLineOfItem[1], ExtraData=CurrentLineOfItem[0])
        if len(Item) >= 1:
            FullFileName = os.path.normpath(os.path.realpath(os.path.join(GlobalData.gINF_MODULE_DIR, Item[0])))
            if not (ValidFile(FullFileName) or ValidFile(Item[0])):
                Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_FILELIST_EXIST % Item[0], File=CurrentLineOfItem[2], Line=CurrentLineOfItem[1], ExtraData=CurrentLineOfItem[0])
            if IsValidPath(Item[0], GlobalData.gINF_MODULE_DIR):
                IsValidFileFlag = True
            else:
                Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_FILE_NOT_EXIST_OR_NAME_INVALID % Item[0], File=CurrentLineOfItem[2], Line=CurrentLineOfItem[1], ExtraData=CurrentLineOfItem[0])
                return False
            if IsValidFileFlag:
                ItemObj.SetSourceFileName(Item[0])
    else:
        Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_SOURCES_SECTION_CONTENT_ERROR, File=CurrentLineOfItem[2], Line=CurrentLineOfItem[1], ExtraData=CurrentLineOfItem[0])
    return ItemObj

class InfSourcesItemObject:

    def __init__(self, SourceFileName='', Family='', TagName='', ToolCode='', FeatureFlagExp=''):
        if False:
            while True:
                i = 10
        self.SourceFileName = SourceFileName
        self.Family = Family
        self.TagName = TagName
        self.ToolCode = ToolCode
        self.FeatureFlagExp = FeatureFlagExp
        self.HeaderString = ''
        self.TailString = ''
        self.SupArchList = []

    def SetSourceFileName(self, SourceFilename):
        if False:
            while True:
                i = 10
        self.SourceFileName = SourceFilename

    def GetSourceFileName(self):
        if False:
            return 10
        return self.SourceFileName

    def SetFamily(self, Family):
        if False:
            for i in range(10):
                print('nop')
        self.Family = Family

    def GetFamily(self):
        if False:
            for i in range(10):
                print('nop')
        return self.Family

    def SetTagName(self, TagName):
        if False:
            while True:
                i = 10
        self.TagName = TagName

    def GetTagName(self):
        if False:
            i = 10
            return i + 15
        return self.TagName

    def SetToolCode(self, ToolCode):
        if False:
            for i in range(10):
                print('nop')
        self.ToolCode = ToolCode

    def GetToolCode(self):
        if False:
            print('Hello World!')
        return self.ToolCode

    def SetFeatureFlagExp(self, FeatureFlagExp):
        if False:
            while True:
                i = 10
        self.FeatureFlagExp = FeatureFlagExp

    def GetFeatureFlagExp(self):
        if False:
            while True:
                i = 10
        return self.FeatureFlagExp

    def SetHeaderString(self, HeaderString):
        if False:
            return 10
        self.HeaderString = HeaderString

    def GetHeaderString(self):
        if False:
            while True:
                i = 10
        return self.HeaderString

    def SetTailString(self, TailString):
        if False:
            i = 10
            return i + 15
        self.TailString = TailString

    def GetTailString(self):
        if False:
            for i in range(10):
                print('nop')
        return self.TailString

    def SetSupArchList(self, SupArchList):
        if False:
            return 10
        self.SupArchList = SupArchList

    def GetSupArchList(self):
        if False:
            print('Hello World!')
        return self.SupArchList

class InfSourcesObject(InfSectionCommonDef):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.Sources = Sdict()
        InfSectionCommonDef.__init__(self)

    def SetSources(self, SourceList, Arch=None):
        if False:
            print('Hello World!')
        __SupArchList = []
        for ArchItem in Arch:
            if ArchItem == '' or ArchItem is None:
                ArchItem = 'COMMON'
            __SupArchList.append(ArchItem)
        for Item in SourceList:
            ItemObj = InfSourcesItemObject()
            CurrentLineOfItem = Item[2]
            Item = Item[0]
            ItemObj = GenSourceInstance(Item, CurrentLineOfItem, ItemObj)
            ItemObj.SetSupArchList(__SupArchList)
            if ItemObj in self.Sources:
                SourceContent = self.Sources[ItemObj]
                SourceContent.append(ItemObj)
                self.Sources[ItemObj] = SourceContent
            else:
                SourceContent = []
                SourceContent.append(ItemObj)
                self.Sources[ItemObj] = SourceContent
        return True

    def GetSources(self):
        if False:
            print('Hello World!')
        return self.Sources