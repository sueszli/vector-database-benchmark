"""
PackageObject
"""
from Object.POM.CommonObject import CommonPropertiesObject
from Object.POM.CommonObject import IdentificationObject
from Object.POM.CommonObject import CommonHeaderObject
from Object.POM.CommonObject import BinaryHeaderObject
from Library.Misc import Sdict

class StandardIncludeFileObject(CommonPropertiesObject):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        CommonPropertiesObject.__init__(self)
        self.IncludeFile = ''

    def SetIncludeFile(self, IncludeFile):
        if False:
            return 10
        self.IncludeFile = IncludeFile

    def GetIncludeFile(self):
        if False:
            while True:
                i = 10
        return self.IncludeFile

class PackageIncludeFileObject(StandardIncludeFileObject):
    pass

class PackageObject(IdentificationObject, CommonHeaderObject, BinaryHeaderObject):

    def __init__(self):
        if False:
            return 10
        IdentificationObject.__init__(self)
        CommonHeaderObject.__init__(self)
        BinaryHeaderObject.__init__(self)
        self.LibraryClassList = []
        self.IncludePathList = []
        self.StandardIncludeFileList = []
        self.PackageIncludeFileList = []
        self.IncludeArchList = []
        self.ProtocolList = []
        self.PpiList = []
        self.GuidList = []
        self.PcdList = []
        self.PcdErrorCommentDict = {}
        self.UserExtensionList = []
        self.MiscFileList = []
        self.ModuleDict = Sdict()
        self.ClonedFromList = []
        self.ModuleFileList = []
        self.PcdChecks = []
        self.UNIFlag = False

    def SetLibraryClassList(self, LibraryClassList):
        if False:
            i = 10
            return i + 15
        self.LibraryClassList = LibraryClassList

    def GetLibraryClassList(self):
        if False:
            i = 10
            return i + 15
        return self.LibraryClassList

    def SetIncludePathList(self, IncludePathList):
        if False:
            return 10
        self.IncludePathList = IncludePathList

    def GetIncludePathList(self):
        if False:
            return 10
        return self.IncludePathList

    def SetIncludeArchList(self, IncludeArchList):
        if False:
            for i in range(10):
                print('nop')
        self.IncludeArchList = IncludeArchList

    def GetIncludeArchList(self):
        if False:
            for i in range(10):
                print('nop')
        return self.IncludeArchList

    def SetStandardIncludeFileList(self, StandardIncludeFileList):
        if False:
            i = 10
            return i + 15
        self.StandardIncludeFileList = StandardIncludeFileList

    def GetStandardIncludeFileList(self):
        if False:
            while True:
                i = 10
        return self.StandardIncludeFileList

    def SetPackageIncludeFileList(self, PackageIncludeFileList):
        if False:
            while True:
                i = 10
        self.PackageIncludeFileList = PackageIncludeFileList

    def GetPackageIncludeFileList(self):
        if False:
            while True:
                i = 10
        return self.PackageIncludeFileList

    def SetProtocolList(self, ProtocolList):
        if False:
            while True:
                i = 10
        self.ProtocolList = ProtocolList

    def GetProtocolList(self):
        if False:
            i = 10
            return i + 15
        return self.ProtocolList

    def SetPpiList(self, PpiList):
        if False:
            while True:
                i = 10
        self.PpiList = PpiList

    def GetPpiList(self):
        if False:
            print('Hello World!')
        return self.PpiList

    def SetGuidList(self, GuidList):
        if False:
            for i in range(10):
                print('nop')
        self.GuidList = GuidList

    def GetGuidList(self):
        if False:
            i = 10
            return i + 15
        return self.GuidList

    def SetPcdList(self, PcdList):
        if False:
            for i in range(10):
                print('nop')
        self.PcdList = PcdList

    def GetPcdList(self):
        if False:
            i = 10
            return i + 15
        return self.PcdList

    def SetUserExtensionList(self, UserExtensionList):
        if False:
            while True:
                i = 10
        self.UserExtensionList = UserExtensionList

    def GetUserExtensionList(self):
        if False:
            print('Hello World!')
        return self.UserExtensionList

    def SetMiscFileList(self, MiscFileList):
        if False:
            for i in range(10):
                print('nop')
        self.MiscFileList = MiscFileList

    def GetMiscFileList(self):
        if False:
            return 10
        return self.MiscFileList

    def SetModuleDict(self, ModuleDict):
        if False:
            i = 10
            return i + 15
        self.ModuleDict = ModuleDict

    def GetModuleDict(self):
        if False:
            for i in range(10):
                print('nop')
        return self.ModuleDict

    def SetClonedFromList(self, ClonedFromList):
        if False:
            return 10
        self.ClonedFromList = ClonedFromList

    def GetClonedFromList(self):
        if False:
            while True:
                i = 10
        return self.ClonedFromList

    def SetModuleFileList(self, ModuleFileList):
        if False:
            return 10
        self.ModuleFileList = ModuleFileList

    def GetModuleFileList(self):
        if False:
            i = 10
            return i + 15
        return self.ModuleFileList