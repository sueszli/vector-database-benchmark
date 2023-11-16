"""
ModuleObject
"""
from Object.POM.CommonObject import CommonPropertiesObject
from Object.POM.CommonObject import IdentificationObject
from Object.POM.CommonObject import CommonHeaderObject
from Object.POM.CommonObject import BinaryHeaderObject
from Object.POM.CommonObject import HelpTextListObject
from Object.POM.CommonObject import GuidVersionObject

class BootModeObject(CommonPropertiesObject, HelpTextListObject):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.SupportedBootModes = ''
        CommonPropertiesObject.__init__(self)
        HelpTextListObject.__init__(self)

    def SetSupportedBootModes(self, SupportedBootModes):
        if False:
            while True:
                i = 10
        self.SupportedBootModes = SupportedBootModes

    def GetSupportedBootModes(self):
        if False:
            return 10
        return self.SupportedBootModes

class EventObject(CommonPropertiesObject, HelpTextListObject):

    def __init__(self):
        if False:
            return 10
        self.EventType = ''
        CommonPropertiesObject.__init__(self)
        HelpTextListObject.__init__(self)

    def SetEventType(self, EventType):
        if False:
            for i in range(10):
                print('nop')
        self.EventType = EventType

    def GetEventType(self):
        if False:
            i = 10
            return i + 15
        return self.EventType

class HobObject(CommonPropertiesObject, HelpTextListObject):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.HobType = ''
        CommonPropertiesObject.__init__(self)
        HelpTextListObject.__init__(self)

    def SetHobType(self, HobType):
        if False:
            return 10
        self.HobType = HobType

    def GetHobType(self):
        if False:
            while True:
                i = 10
        return self.HobType

class SpecObject(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.Spec = ''
        self.Version = ''

    def SetSpec(self, Spec):
        if False:
            i = 10
            return i + 15
        self.Spec = Spec

    def GetSpec(self):
        if False:
            print('Hello World!')
        return self.Spec

    def SetVersion(self, Version):
        if False:
            for i in range(10):
                print('nop')
        self.Version = Version

    def GetVersion(self):
        if False:
            while True:
                i = 10
        return self.Version

class ModuleHeaderObject(IdentificationObject, CommonHeaderObject, BinaryHeaderObject):

    def __init__(self):
        if False:
            return 10
        self.IsLibrary = False
        self.IsLibraryModList = []
        self.ModuleType = ''
        self.BinaryModule = False
        self.PcdIsDriver = ''
        self.PiSpecificationVersion = ''
        self.UefiSpecificationVersion = ''
        self.UNIFlag = False
        self.ModuleUniFile = ''
        self.SpecList = []
        self.BootModeList = []
        self.EventList = []
        self.HobList = []
        self.LibraryClassList = []
        self.SupArchList = []
        IdentificationObject.__init__(self)
        CommonHeaderObject.__init__(self)
        BinaryHeaderObject.__init__(self)

    def SetIsLibrary(self, IsLibrary):
        if False:
            for i in range(10):
                print('nop')
        self.IsLibrary = IsLibrary

    def GetIsLibrary(self):
        if False:
            print('Hello World!')
        return self.IsLibrary

    def SetIsLibraryModList(self, IsLibraryModList):
        if False:
            for i in range(10):
                print('nop')
        self.IsLibraryModList = IsLibraryModList

    def GetIsLibraryModList(self):
        if False:
            print('Hello World!')
        return self.IsLibraryModList

    def SetModuleType(self, ModuleType):
        if False:
            for i in range(10):
                print('nop')
        self.ModuleType = ModuleType

    def GetModuleType(self):
        if False:
            return 10
        return self.ModuleType

    def SetBinaryModule(self, BinaryModule):
        if False:
            i = 10
            return i + 15
        self.BinaryModule = BinaryModule

    def GetBinaryModule(self):
        if False:
            for i in range(10):
                print('nop')
        return self.BinaryModule

    def SetPcdIsDriver(self, PcdIsDriver):
        if False:
            return 10
        self.PcdIsDriver = PcdIsDriver

    def GetPcdIsDriver(self):
        if False:
            print('Hello World!')
        return self.PcdIsDriver

    def SetPiSpecificationVersion(self, PiSpecificationVersion):
        if False:
            return 10
        self.PiSpecificationVersion = PiSpecificationVersion

    def GetPiSpecificationVersion(self):
        if False:
            i = 10
            return i + 15
        return self.PiSpecificationVersion

    def SetUefiSpecificationVersion(self, UefiSpecificationVersion):
        if False:
            return 10
        self.UefiSpecificationVersion = UefiSpecificationVersion

    def GetUefiSpecificationVersion(self):
        if False:
            while True:
                i = 10
        return self.UefiSpecificationVersion

    def SetSpecList(self, SpecList):
        if False:
            while True:
                i = 10
        self.SpecList = SpecList

    def GetSpecList(self):
        if False:
            return 10
        return self.SpecList

    def SetBootModeList(self, BootModeList):
        if False:
            return 10
        self.BootModeList = BootModeList

    def GetBootModeList(self):
        if False:
            print('Hello World!')
        return self.BootModeList

    def SetEventList(self, EventList):
        if False:
            for i in range(10):
                print('nop')
        self.EventList = EventList

    def GetEventList(self):
        if False:
            i = 10
            return i + 15
        return self.EventList

    def SetHobList(self, HobList):
        if False:
            print('Hello World!')
        self.HobList = HobList

    def GetHobList(self):
        if False:
            return 10
        return self.HobList

    def SetLibraryClassList(self, LibraryClassList):
        if False:
            for i in range(10):
                print('nop')
        self.LibraryClassList = LibraryClassList

    def GetLibraryClassList(self):
        if False:
            while True:
                i = 10
        return self.LibraryClassList

    def SetSupArchList(self, SupArchList):
        if False:
            i = 10
            return i + 15
        self.SupArchList = SupArchList

    def GetSupArchList(self):
        if False:
            print('Hello World!')
        return self.SupArchList

    def SetModuleUniFile(self, ModuleUniFile):
        if False:
            print('Hello World!')
        self.ModuleUniFile = ModuleUniFile

    def GetModuleUniFile(self):
        if False:
            print('Hello World!')
        return self.ModuleUniFile

class SourceFileObject(CommonPropertiesObject):

    def __init__(self):
        if False:
            print('Hello World!')
        CommonPropertiesObject.__init__(self)
        self.SourceFile = ''
        self.TagName = ''
        self.ToolCode = ''
        self.Family = ''
        self.FileType = ''

    def SetSourceFile(self, SourceFile):
        if False:
            return 10
        self.SourceFile = SourceFile

    def GetSourceFile(self):
        if False:
            while True:
                i = 10
        return self.SourceFile

    def SetTagName(self, TagName):
        if False:
            while True:
                i = 10
        self.TagName = TagName

    def GetTagName(self):
        if False:
            for i in range(10):
                print('nop')
        return self.TagName

    def SetToolCode(self, ToolCode):
        if False:
            print('Hello World!')
        self.ToolCode = ToolCode

    def GetToolCode(self):
        if False:
            for i in range(10):
                print('nop')
        return self.ToolCode

    def SetFamily(self, Family):
        if False:
            return 10
        self.Family = Family

    def GetFamily(self):
        if False:
            while True:
                i = 10
        return self.Family

    def SetFileType(self, FileType):
        if False:
            return 10
        self.FileType = FileType

    def GetFileType(self):
        if False:
            i = 10
            return i + 15
        return self.FileType

class BinaryFileObject(CommonPropertiesObject):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.FileNamList = []
        self.AsBuiltList = []
        CommonPropertiesObject.__init__(self)

    def SetFileNameList(self, FileNamList):
        if False:
            for i in range(10):
                print('nop')
        self.FileNamList = FileNamList

    def GetFileNameList(self):
        if False:
            i = 10
            return i + 15
        return self.FileNamList

    def SetAsBuiltList(self, AsBuiltList):
        if False:
            while True:
                i = 10
        self.AsBuiltList = AsBuiltList

    def GetAsBuiltList(self):
        if False:
            return 10
        return self.AsBuiltList

class AsBuildLibraryClassObject(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.LibGuid = ''
        self.LibVersion = ''
        self.SupArchList = []

    def SetLibGuid(self, LibGuid):
        if False:
            print('Hello World!')
        self.LibGuid = LibGuid

    def GetLibGuid(self):
        if False:
            i = 10
            return i + 15
        return self.LibGuid

    def SetLibVersion(self, LibVersion):
        if False:
            while True:
                i = 10
        self.LibVersion = LibVersion

    def GetLibVersion(self):
        if False:
            for i in range(10):
                print('nop')
        return self.LibVersion

    def SetSupArchList(self, SupArchList):
        if False:
            print('Hello World!')
        self.SupArchList = SupArchList

    def GetSupArchList(self):
        if False:
            i = 10
            return i + 15
        return self.SupArchList

class AsBuiltObject(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.PatchPcdList = []
        self.PcdExValueList = []
        self.LibraryInstancesList = []
        self.BinaryBuildFlagList = []

    def SetPatchPcdList(self, PatchPcdList):
        if False:
            return 10
        self.PatchPcdList = PatchPcdList

    def GetPatchPcdList(self):
        if False:
            i = 10
            return i + 15
        return self.PatchPcdList

    def SetPcdExList(self, PcdExValueList):
        if False:
            while True:
                i = 10
        self.PcdExValueList = PcdExValueList

    def GetPcdExList(self):
        if False:
            return 10
        return self.PcdExValueList

    def SetLibraryInstancesList(self, LibraryInstancesList):
        if False:
            while True:
                i = 10
        self.LibraryInstancesList = LibraryInstancesList

    def GetLibraryInstancesList(self):
        if False:
            while True:
                i = 10
        return self.LibraryInstancesList

    def SetBuildFlagsList(self, BinaryBuildFlagList):
        if False:
            while True:
                i = 10
        self.BinaryBuildFlagList = BinaryBuildFlagList

    def GetBuildFlagsList(self):
        if False:
            i = 10
            return i + 15
        return self.BinaryBuildFlagList

class BinaryBuildFlagObject(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.Target = ''
        self.TagName = ''
        self.Family = ''
        self.AsBuiltOptionFlags = ''

    def SetTarget(self, Target):
        if False:
            i = 10
            return i + 15
        self.Target = Target

    def GetTarget(self):
        if False:
            while True:
                i = 10
        return self.Target

    def SetTagName(self, TagName):
        if False:
            while True:
                i = 10
        self.TagName = TagName

    def GetTagName(self):
        if False:
            while True:
                i = 10
        return self.TagName

    def SetFamily(self, Family):
        if False:
            for i in range(10):
                print('nop')
        self.Family = Family

    def GetFamily(self):
        if False:
            while True:
                i = 10
        return self.Family

    def SetAsBuiltOptionFlags(self, AsBuiltOptionFlags):
        if False:
            return 10
        self.AsBuiltOptionFlags = AsBuiltOptionFlags

    def GetAsBuiltOptionFlags(self):
        if False:
            return 10
        return self.AsBuiltOptionFlags

class ExternObject(CommonPropertiesObject):

    def __init__(self):
        if False:
            return 10
        self.EntryPoint = ''
        self.UnloadImage = ''
        self.Constructor = ''
        self.Destructor = ''
        self.SupModList = []
        CommonPropertiesObject.__init__(self)

    def SetEntryPoint(self, EntryPoint):
        if False:
            i = 10
            return i + 15
        self.EntryPoint = EntryPoint

    def GetEntryPoint(self):
        if False:
            while True:
                i = 10
        return self.EntryPoint

    def SetUnloadImage(self, UnloadImage):
        if False:
            for i in range(10):
                print('nop')
        self.UnloadImage = UnloadImage

    def GetUnloadImage(self):
        if False:
            for i in range(10):
                print('nop')
        return self.UnloadImage

    def SetConstructor(self, Constructor):
        if False:
            print('Hello World!')
        self.Constructor = Constructor

    def GetConstructor(self):
        if False:
            for i in range(10):
                print('nop')
        return self.Constructor

    def SetDestructor(self, Destructor):
        if False:
            i = 10
            return i + 15
        self.Destructor = Destructor

    def GetDestructor(self):
        if False:
            i = 10
            return i + 15
        return self.Destructor

    def SetSupModList(self, SupModList):
        if False:
            print('Hello World!')
        self.SupModList = SupModList

    def GetSupModList(self):
        if False:
            while True:
                i = 10
        return self.SupModList

class DepexObject(CommonPropertiesObject):

    def __init__(self):
        if False:
            return 10
        self.Depex = ''
        self.ModuelType = ''
        CommonPropertiesObject.__init__(self)

    def SetDepex(self, Depex):
        if False:
            return 10
        self.Depex = Depex

    def GetDepex(self):
        if False:
            print('Hello World!')
        return self.Depex

    def SetModuleType(self, ModuleType):
        if False:
            for i in range(10):
                print('nop')
        self.ModuelType = ModuleType

    def GetModuleType(self):
        if False:
            i = 10
            return i + 15
        return self.ModuelType

class PackageDependencyObject(GuidVersionObject, CommonPropertiesObject):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.Package = ''
        self.PackageFilePath = ''
        GuidVersionObject.__init__(self)
        CommonPropertiesObject.__init__(self)

    def SetPackageFilePath(self, PackageFilePath):
        if False:
            i = 10
            return i + 15
        self.PackageFilePath = PackageFilePath

    def GetPackageFilePath(self):
        if False:
            while True:
                i = 10
        return self.PackageFilePath

    def SetPackage(self, Package):
        if False:
            print('Hello World!')
        self.Package = Package

    def GetPackage(self):
        if False:
            for i in range(10):
                print('nop')
        return self.Package

class BuildOptionObject(CommonPropertiesObject):

    def __init__(self):
        if False:
            print('Hello World!')
        CommonPropertiesObject.__init__(self)
        self.BuildOption = ''

    def SetBuildOption(self, BuildOption):
        if False:
            print('Hello World!')
        self.BuildOption = BuildOption

    def GetBuildOption(self):
        if False:
            i = 10
            return i + 15
        return self.BuildOption

class ModuleObject(ModuleHeaderObject):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.HeaderDict = {}
        self.LibraryClassList = []
        self.SourceFileList = []
        self.BinaryFileList = []
        self.PackageDependencyList = []
        self.PeiDepex = []
        self.DxeDepex = []
        self.SmmDepex = []
        self.ProtocolList = []
        self.PpiList = []
        self.GuidList = []
        self.PcdList = []
        self.ExternList = []
        self.BuildOptionList = []
        self.UserExtensionList = []
        self.MiscFileList = []
        self.ClonedFrom = None
        ModuleHeaderObject.__init__(self)

    def SetHeaderDict(self, HeaderDict):
        if False:
            return 10
        self.HeaderDict = HeaderDict

    def GetHeaderDict(self):
        if False:
            print('Hello World!')
        return self.HeaderDict

    def SetLibraryClassList(self, LibraryClassList):
        if False:
            return 10
        self.LibraryClassList = LibraryClassList

    def GetLibraryClassList(self):
        if False:
            return 10
        return self.LibraryClassList

    def SetSourceFileList(self, SourceFileList):
        if False:
            while True:
                i = 10
        self.SourceFileList = SourceFileList

    def GetSourceFileList(self):
        if False:
            i = 10
            return i + 15
        return self.SourceFileList

    def SetBinaryFileList(self, BinaryFileList):
        if False:
            print('Hello World!')
        self.BinaryFileList = BinaryFileList

    def GetBinaryFileList(self):
        if False:
            return 10
        return self.BinaryFileList

    def SetPackageDependencyList(self, PackageDependencyList):
        if False:
            while True:
                i = 10
        self.PackageDependencyList = PackageDependencyList

    def GetPackageDependencyList(self):
        if False:
            print('Hello World!')
        return self.PackageDependencyList

    def SetPeiDepex(self, PeiDepex):
        if False:
            i = 10
            return i + 15
        self.PeiDepex = PeiDepex

    def GetPeiDepex(self):
        if False:
            i = 10
            return i + 15
        return self.PeiDepex

    def SetDxeDepex(self, DxeDepex):
        if False:
            while True:
                i = 10
        self.DxeDepex = DxeDepex

    def GetDxeDepex(self):
        if False:
            for i in range(10):
                print('nop')
        return self.DxeDepex

    def SetSmmDepex(self, SmmDepex):
        if False:
            for i in range(10):
                print('nop')
        self.SmmDepex = SmmDepex

    def GetSmmDepex(self):
        if False:
            for i in range(10):
                print('nop')
        return self.SmmDepex

    def SetPpiList(self, PpiList):
        if False:
            i = 10
            return i + 15
        self.PpiList = PpiList

    def GetPpiList(self):
        if False:
            for i in range(10):
                print('nop')
        return self.PpiList

    def SetProtocolList(self, ProtocolList):
        if False:
            i = 10
            return i + 15
        self.ProtocolList = ProtocolList

    def GetProtocolList(self):
        if False:
            print('Hello World!')
        return self.ProtocolList

    def SetPcdList(self, PcdList):
        if False:
            print('Hello World!')
        self.PcdList = PcdList

    def GetPcdList(self):
        if False:
            i = 10
            return i + 15
        return self.PcdList

    def SetGuidList(self, GuidList):
        if False:
            print('Hello World!')
        self.GuidList = GuidList

    def GetGuidList(self):
        if False:
            i = 10
            return i + 15
        return self.GuidList

    def SetExternList(self, ExternList):
        if False:
            i = 10
            return i + 15
        self.ExternList = ExternList

    def GetExternList(self):
        if False:
            while True:
                i = 10
        return self.ExternList

    def SetBuildOptionList(self, BuildOptionList):
        if False:
            i = 10
            return i + 15
        self.BuildOptionList = BuildOptionList

    def GetBuildOptionList(self):
        if False:
            i = 10
            return i + 15
        return self.BuildOptionList

    def SetUserExtensionList(self, UserExtensionList):
        if False:
            print('Hello World!')
        self.UserExtensionList = UserExtensionList

    def GetUserExtensionList(self):
        if False:
            i = 10
            return i + 15
        return self.UserExtensionList

    def SetMiscFileList(self, MiscFileList):
        if False:
            print('Hello World!')
        self.MiscFileList = MiscFileList

    def GetMiscFileList(self):
        if False:
            return 10
        return self.MiscFileList

    def SetClonedFrom(self, ClonedFrom):
        if False:
            for i in range(10):
                print('nop')
        self.ClonedFrom = ClonedFrom

    def GetClonedFrom(self):
        if False:
            for i in range(10):
                print('nop')
        return self.ClonedFrom