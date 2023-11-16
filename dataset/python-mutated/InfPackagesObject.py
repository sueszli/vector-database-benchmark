"""
InfPackageObject
"""
from Logger import StringTable as ST
from Logger import ToolError
import Logger.Log as Logger
from Library import GlobalData
from Library.Misc import Sdict
from Library.ParserValidate import IsValidPath
from Library.ExpressionValidate import IsValidFeatureFlagExp

class InfPackageItem:

    def __init__(self, PackageName='', FeatureFlagExp='', HelpString=''):
        if False:
            print('Hello World!')
        self.PackageName = PackageName
        self.FeatureFlagExp = FeatureFlagExp
        self.HelpString = HelpString
        self.SupArchList = []

    def SetPackageName(self, PackageName):
        if False:
            for i in range(10):
                print('nop')
        self.PackageName = PackageName

    def GetPackageName(self):
        if False:
            for i in range(10):
                print('nop')
        return self.PackageName

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

    def SetHelpString(self, HelpString):
        if False:
            print('Hello World!')
        self.HelpString = HelpString

    def GetHelpString(self):
        if False:
            i = 10
            return i + 15
        return self.HelpString

    def SetSupArchList(self, SupArchList):
        if False:
            return 10
        self.SupArchList = SupArchList

    def GetSupArchList(self):
        if False:
            print('Hello World!')
        return self.SupArchList

class InfPackageObject:

    def __init__(self):
        if False:
            print('Hello World!')
        self.Packages = Sdict()
        self.Macros = {}

    def SetPackages(self, PackageData, Arch=None):
        if False:
            while True:
                i = 10
        IsValidFileFlag = False
        SupArchList = []
        for ArchItem in Arch:
            if ArchItem == '' or ArchItem is None:
                ArchItem = 'COMMON'
            SupArchList.append(ArchItem)
        for PackageItem in PackageData:
            PackageItemObj = InfPackageItem()
            HelpStringObj = PackageItem[1]
            CurrentLineOfPackItem = PackageItem[2]
            PackageItem = PackageItem[0]
            if HelpStringObj is not None:
                HelpString = HelpStringObj.HeaderComments + HelpStringObj.TailComments
                PackageItemObj.SetHelpString(HelpString)
            if len(PackageItem) >= 1:
                if IsValidPath(PackageItem[0], ''):
                    IsValidFileFlag = True
                elif IsValidPath(PackageItem[0], GlobalData.gINF_MODULE_DIR):
                    IsValidFileFlag = True
                elif IsValidPath(PackageItem[0], GlobalData.gWORKSPACE):
                    IsValidFileFlag = True
                else:
                    Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_FILE_NOT_EXIST_OR_NAME_INVALID % PackageItem[0], File=CurrentLineOfPackItem[2], Line=CurrentLineOfPackItem[1], ExtraData=CurrentLineOfPackItem[0])
                    return False
                if IsValidFileFlag:
                    PackageItemObj.SetPackageName(PackageItem[0])
            if len(PackageItem) == 2:
                if PackageItem[1].strip() == '':
                    Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_FEATURE_FLAG_EXP_MISSING, File=CurrentLineOfPackItem[2], Line=CurrentLineOfPackItem[1], ExtraData=CurrentLineOfPackItem[0])
                FeatureFlagRtv = IsValidFeatureFlagExp(PackageItem[1].strip())
                if not FeatureFlagRtv[0]:
                    Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_FEATURE_FLAG_EXP_SYNTAX_INVLID % FeatureFlagRtv[1], File=CurrentLineOfPackItem[2], Line=CurrentLineOfPackItem[1], ExtraData=CurrentLineOfPackItem[0])
                PackageItemObj.SetFeatureFlagExp(PackageItem[1].strip())
            if len(PackageItem) > 2:
                Logger.Error('InfParser', ToolError.FORMAT_INVALID, ST.ERR_INF_PARSER_PACKAGE_SECTION_CONTENT_ERROR, File=CurrentLineOfPackItem[2], Line=CurrentLineOfPackItem[1], ExtraData=CurrentLineOfPackItem[0])
            PackageItemObj.SetSupArchList(SupArchList)
            for Item in self.Packages:
                if Item.GetPackageName() == PackageItemObj.GetPackageName():
                    ItemSupArchList = Item.GetSupArchList()
                    for ItemArch in ItemSupArchList:
                        for PackageItemObjArch in SupArchList:
                            if ItemArch == PackageItemObjArch:
                                pass
                            if ItemArch.upper() == 'COMMON' or PackageItemObjArch.upper() == 'COMMON':
                                pass
            if PackageItemObj in self.Packages:
                PackageList = self.Packages[PackageItemObj]
                PackageList.append(PackageItemObj)
                self.Packages[PackageItemObj] = PackageList
            else:
                PackageList = []
                PackageList.append(PackageItemObj)
                self.Packages[PackageItemObj] = PackageList
        return True

    def GetPackages(self, Arch=None):
        if False:
            while True:
                i = 10
        if Arch is None:
            return self.Packages