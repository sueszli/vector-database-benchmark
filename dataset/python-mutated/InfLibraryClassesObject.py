"""
InfLibraryClassesObject
"""
from Logger import StringTable as ST
from Logger import ToolError
import Logger.Log as Logger
from Library import GlobalData
from Library.Misc import Sdict
from Object.Parser.InfCommonObject import CurrentLine
from Library.ExpressionValidate import IsValidFeatureFlagExp
from Library.ParserValidate import IsValidLibName

def GetArchModuleType(KeyList):
    if False:
        for i in range(10):
            print('nop')
    __SupArchList = []
    __SupModuleList = []
    for (ArchItem, ModuleItem) in KeyList:
        if ArchItem == '' or ArchItem is None:
            ArchItem = 'COMMON'
        if ModuleItem == '' or ModuleItem is None:
            ModuleItem = 'COMMON'
        if ArchItem not in __SupArchList:
            __SupArchList.append(ArchItem)
        List = ModuleItem.split('|')
        for Entry in List:
            if Entry not in __SupModuleList:
                __SupModuleList.append(Entry)
    return (__SupArchList, __SupModuleList)

class InfLibraryClassItem:

    def __init__(self, LibName='', FeatureFlagExp='', HelpString=None):
        if False:
            print('Hello World!')
        self.LibName = LibName
        self.FeatureFlagExp = FeatureFlagExp
        self.HelpString = HelpString
        self.CurrentLine = CurrentLine()
        self.SupArchList = []
        self.SupModuleList = []
        self.FileGuid = ''
        self.Version = ''

    def SetLibName(self, LibName):
        if False:
            i = 10
            return i + 15
        self.LibName = LibName

    def GetLibName(self):
        if False:
            return 10
        return self.LibName

    def SetHelpString(self, HelpString):
        if False:
            return 10
        self.HelpString = HelpString

    def GetHelpString(self):
        if False:
            print('Hello World!')
        return self.HelpString

    def SetFeatureFlagExp(self, FeatureFlagExp):
        if False:
            while True:
                i = 10
        self.FeatureFlagExp = FeatureFlagExp

    def GetFeatureFlagExp(self):
        if False:
            i = 10
            return i + 15
        return self.FeatureFlagExp

    def SetSupArchList(self, SupArchList):
        if False:
            print('Hello World!')
        self.SupArchList = SupArchList

    def GetSupArchList(self):
        if False:
            i = 10
            return i + 15
        return self.SupArchList

    def SetSupModuleList(self, SupModuleList):
        if False:
            return 10
        self.SupModuleList = SupModuleList

    def GetSupModuleList(self):
        if False:
            i = 10
            return i + 15
        return self.SupModuleList

    def SetFileGuid(self, FileGuid):
        if False:
            for i in range(10):
                print('nop')
        self.FileGuid = FileGuid

    def GetFileGuid(self):
        if False:
            i = 10
            return i + 15
        return self.FileGuid

    def SetVersion(self, Version):
        if False:
            print('Hello World!')
        self.Version = Version

    def GetVersion(self):
        if False:
            for i in range(10):
                print('nop')
        return self.Version

class InfLibraryClassObject:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.LibraryClasses = Sdict()
        self.Macros = {}

    def SetLibraryClasses(self, LibContent, KeyList=None):
        if False:
            print('Hello World!')
        (__SupArchList, __SupModuleList) = GetArchModuleType(KeyList)
        for LibItem in LibContent:
            LibItemObj = InfLibraryClassItem()
            if not GlobalData.gIS_BINARY_INF:
                HelpStringObj = LibItem[1]
                LibItemObj.CurrentLine.SetFileName(LibItem[2][2])
                LibItemObj.CurrentLine.SetLineNo(LibItem[2][1])
                LibItemObj.CurrentLine.SetLineString(LibItem[2][0])
                LibItem = LibItem[0]
                if HelpStringObj is not None:
                    LibItemObj.SetHelpString(HelpStringObj)
                if len(LibItem) >= 1:
                    if LibItem[0].strip() != '':
                        if IsValidLibName(LibItem[0].strip()):
                            if LibItem[0].strip() != 'NULL':
                                LibItemObj.SetLibName(LibItem[0])
                            else:
                                Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_DEFINE_LIB_NAME_INVALID, File=GlobalData.gINF_MODULE_NAME, Line=LibItemObj.CurrentLine.GetLineNo(), ExtraData=LibItemObj.CurrentLine.GetLineString())
                        else:
                            Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % LibItem[0], File=GlobalData.gINF_MODULE_NAME, Line=LibItemObj.CurrentLine.GetLineNo(), ExtraData=LibItemObj.CurrentLine.GetLineString())
                    else:
                        Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_LIBRARY_SECTION_LIBNAME_MISSING, File=GlobalData.gINF_MODULE_NAME, Line=LibItemObj.CurrentLine.GetLineNo(), ExtraData=LibItemObj.CurrentLine.GetLineString())
                if len(LibItem) == 2:
                    if LibItem[1].strip() == '':
                        Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_FEATURE_FLAG_EXP_MISSING, File=GlobalData.gINF_MODULE_NAME, Line=LibItemObj.CurrentLine.GetLineNo(), ExtraData=LibItemObj.CurrentLine.GetLineString())
                    FeatureFlagRtv = IsValidFeatureFlagExp(LibItem[1].strip())
                    if not FeatureFlagRtv[0]:
                        Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_FEATURE_FLAG_EXP_SYNTAX_INVLID % FeatureFlagRtv[1], File=GlobalData.gINF_MODULE_NAME, Line=LibItemObj.CurrentLine.GetLineNo(), ExtraData=LibItemObj.CurrentLine.GetLineString())
                    LibItemObj.SetFeatureFlagExp(LibItem[1].strip())
                if len(LibItem) < 1 or len(LibItem) > 2:
                    Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_LIBRARY_SECTION_CONTENT_ERROR, File=GlobalData.gINF_MODULE_NAME, Line=LibItemObj.CurrentLine.GetLineNo(), ExtraData=LibItemObj.CurrentLine.GetLineString())
                LibItemObj.SetSupArchList(__SupArchList)
                LibItemObj.SetSupModuleList(__SupModuleList)
            else:
                LibItemObj.SetFileGuid(LibItem[0])
                LibItemObj.SetVersion(LibItem[1])
                LibItemObj.SetSupArchList(__SupArchList)
            if LibItemObj in self.LibraryClasses:
                LibraryList = self.LibraryClasses[LibItemObj]
                LibraryList.append(LibItemObj)
                self.LibraryClasses[LibItemObj] = LibraryList
            else:
                LibraryList = []
                LibraryList.append(LibItemObj)
                self.LibraryClasses[LibItemObj] = LibraryList
        return True

    def GetLibraryClasses(self):
        if False:
            print('Hello World!')
        return self.LibraryClasses