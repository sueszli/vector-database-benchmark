"""
InfDepexObject
"""
from Library import DataType as DT
from Library import GlobalData
import Logger.Log as Logger
from Logger import ToolError
from Logger import StringTable as ST
from Object.Parser.InfCommonObject import InfSectionCommonDef
from Library.ParserValidate import IsValidArch

class InfDepexContentItem:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.SectionType = ''
        self.SectionString = ''

    def SetSectionType(self, SectionType):
        if False:
            for i in range(10):
                print('nop')
        self.SectionType = SectionType

    def GetSectionType(self):
        if False:
            while True:
                i = 10
        return self.SectionType

    def SetSectionString(self, SectionString):
        if False:
            i = 10
            return i + 15
        self.SectionString = SectionString

    def GetSectionString(self):
        if False:
            print('Hello World!')
        return self.SectionString

class InfDepexItem:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.DepexContent = ''
        self.ModuleType = ''
        self.SupArch = ''
        self.HelpString = ''
        self.FeatureFlagExp = ''
        self.InfDepexContentItemList = []

    def SetFeatureFlagExp(self, FeatureFlagExp):
        if False:
            while True:
                i = 10
        self.FeatureFlagExp = FeatureFlagExp

    def GetFeatureFlagExp(self):
        if False:
            print('Hello World!')
        return self.FeatureFlagExp

    def SetSupArch(self, Arch):
        if False:
            print('Hello World!')
        self.SupArch = Arch

    def GetSupArch(self):
        if False:
            return 10
        return self.SupArch

    def SetHelpString(self, HelpString):
        if False:
            return 10
        self.HelpString = HelpString

    def GetHelpString(self):
        if False:
            print('Hello World!')
        return self.HelpString

    def SetModuleType(self, Type):
        if False:
            print('Hello World!')
        self.ModuleType = Type

    def GetModuleType(self):
        if False:
            i = 10
            return i + 15
        return self.ModuleType

    def SetDepexConent(self, Content):
        if False:
            return 10
        self.DepexContent = Content

    def GetDepexContent(self):
        if False:
            for i in range(10):
                print('nop')
        return self.DepexContent

    def SetInfDepexContentItemList(self, InfDepexContentItemList):
        if False:
            return 10
        self.InfDepexContentItemList = InfDepexContentItemList

    def GetInfDepexContentItemList(self):
        if False:
            for i in range(10):
                print('nop')
        return self.InfDepexContentItemList

class InfDepexObject(InfSectionCommonDef):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.Depex = []
        self.AllContent = ''
        self.SectionContent = ''
        InfSectionCommonDef.__init__(self)

    def SetDepex(self, DepexContent, KeyList=None, CommentList=None):
        if False:
            print('Hello World!')
        for KeyItem in KeyList:
            Arch = KeyItem[0]
            ModuleType = KeyItem[1]
            InfDepexItemIns = InfDepexItem()
            if IsValidArch(Arch.strip().upper()):
                InfDepexItemIns.SetSupArch(Arch)
            else:
                Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_DEFINE_NAME_INVALID % Arch, File=GlobalData.gINF_MODULE_NAME, Line=KeyItem[2])
            if ModuleType and ModuleType != 'COMMON':
                if ModuleType in DT.VALID_DEPEX_MODULE_TYPE_LIST:
                    InfDepexItemIns.SetModuleType(ModuleType)
                else:
                    Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_DEPEX_SECTION_MODULE_TYPE_ERROR % ModuleType, File=GlobalData.gINF_MODULE_NAME, Line=KeyItem[2])
            DepexString = ''
            HelpString = ''
            for Line in DepexContent:
                LineContent = Line[0].strip()
                if LineContent.find(DT.TAB_COMMENT_SPLIT) > -1:
                    LineContent = LineContent[:LineContent.find(DT.TAB_COMMENT_SPLIT)]
                if LineContent:
                    DepexString = DepexString + LineContent + DT.END_OF_LINE
                continue
            if DepexString.endswith(DT.END_OF_LINE):
                DepexString = DepexString[:-1]
            if not DepexString.strip():
                continue
            for HelpLine in CommentList:
                HelpString = HelpString + HelpLine + DT.END_OF_LINE
            if HelpString.endswith(DT.END_OF_LINE):
                HelpString = HelpString[:-1]
            InfDepexItemIns.SetDepexConent(DepexString)
            InfDepexItemIns.SetHelpString(HelpString)
            self.Depex.append(InfDepexItemIns)
        return True

    def GetDepex(self):
        if False:
            print('Hello World!')
        return self.Depex

    def GetAllContent(self):
        if False:
            while True:
                i = 10
        return self.AllContent