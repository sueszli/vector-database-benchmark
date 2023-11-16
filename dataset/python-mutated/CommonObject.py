"""
Common Object
"""
from Library.DataType import TAB_LANGUAGE_EN_US

class HelpTextObject(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.HelpText = TextObject()

    def SetHelpText(self, HelpText):
        if False:
            print('Hello World!')
        self.HelpText = HelpText

    def GetHelpText(self):
        if False:
            i = 10
            return i + 15
        return self.HelpText

class HelpTextListObject(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.HelpTextList = []

    def SetHelpTextList(self, HelpTextList):
        if False:
            for i in range(10):
                print('nop')
        self.HelpTextList = HelpTextList

    def GetHelpTextList(self):
        if False:
            return 10
        return self.HelpTextList

class PromptListObject(object):

    def __init__(self):
        if False:
            return 10
        self.PromptList = []

    def SetPromptList(self, PromptList):
        if False:
            i = 10
            return i + 15
        self.PromptList = PromptList

    def GetPromptList(self):
        if False:
            i = 10
            return i + 15
        return self.PromptList

class CommonPropertiesObject(HelpTextObject, HelpTextListObject):

    def __init__(self):
        if False:
            return 10
        self.Usage = []
        self.FeatureFlag = ''
        self.SupArchList = []
        self.GuidValue = ''
        HelpTextObject.__init__(self)
        HelpTextListObject.__init__(self)

    def SetUsage(self, Usage):
        if False:
            print('Hello World!')
        self.Usage = Usage

    def GetUsage(self):
        if False:
            while True:
                i = 10
        return self.Usage

    def SetFeatureFlag(self, FeatureFlag):
        if False:
            while True:
                i = 10
        self.FeatureFlag = FeatureFlag

    def GetFeatureFlag(self):
        if False:
            for i in range(10):
                print('nop')
        return self.FeatureFlag

    def SetSupArchList(self, SupArchList):
        if False:
            print('Hello World!')
        self.SupArchList = SupArchList

    def GetSupArchList(self):
        if False:
            print('Hello World!')
        return self.SupArchList

    def SetGuidValue(self, GuidValue):
        if False:
            for i in range(10):
                print('nop')
        self.GuidValue = GuidValue

    def GetGuidValue(self):
        if False:
            return 10
        return self.GuidValue

class CommonHeaderObject(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.AbstractList = []
        self.DescriptionList = []
        self.CopyrightList = []
        self.LicenseList = []

    def SetAbstract(self, Abstract):
        if False:
            while True:
                i = 10
        if isinstance(Abstract, list):
            self.AbstractList = Abstract
        else:
            self.AbstractList.append(Abstract)

    def GetAbstract(self):
        if False:
            print('Hello World!')
        return self.AbstractList

    def SetDescription(self, Description):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(Description, list):
            self.DescriptionList = Description
        else:
            self.DescriptionList.append(Description)

    def GetDescription(self):
        if False:
            while True:
                i = 10
        return self.DescriptionList

    def SetCopyright(self, Copyright):
        if False:
            while True:
                i = 10
        if isinstance(Copyright, list):
            self.CopyrightList = Copyright
        else:
            self.CopyrightList.append(Copyright)

    def GetCopyright(self):
        if False:
            i = 10
            return i + 15
        return self.CopyrightList

    def SetLicense(self, License):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(License, list):
            self.LicenseList = License
        else:
            self.LicenseList.append(License)

    def GetLicense(self):
        if False:
            i = 10
            return i + 15
        return self.LicenseList

class BinaryHeaderObject(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.BinaryHeaderAbstractList = []
        self.BinaryHeaderDescriptionList = []
        self.BinaryHeaderCopyrightList = []
        self.BinaryHeaderLicenseList = []

    def SetBinaryHeaderAbstract(self, Abstract):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(Abstract, list) and Abstract:
            self.BinaryHeaderAbstractList = Abstract
        elif isinstance(Abstract, tuple) and Abstract[1]:
            self.BinaryHeaderAbstractList.append(Abstract)

    def GetBinaryHeaderAbstract(self):
        if False:
            for i in range(10):
                print('nop')
        return self.BinaryHeaderAbstractList

    def SetBinaryHeaderDescription(self, Description):
        if False:
            print('Hello World!')
        if isinstance(Description, list) and Description:
            self.BinaryHeaderDescriptionList = Description
        elif isinstance(Description, tuple) and Description[1]:
            self.BinaryHeaderDescriptionList.append(Description)

    def GetBinaryHeaderDescription(self):
        if False:
            while True:
                i = 10
        return self.BinaryHeaderDescriptionList

    def SetBinaryHeaderCopyright(self, Copyright):
        if False:
            return 10
        if isinstance(Copyright, list) and Copyright:
            self.BinaryHeaderCopyrightList = Copyright
        elif isinstance(Copyright, tuple) and Copyright[1]:
            self.BinaryHeaderCopyrightList.append(Copyright)

    def GetBinaryHeaderCopyright(self):
        if False:
            while True:
                i = 10
        return self.BinaryHeaderCopyrightList

    def SetBinaryHeaderLicense(self, License):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(License, list) and License:
            self.BinaryHeaderLicenseList = License
        elif isinstance(License, tuple) and License[1]:
            self.BinaryHeaderLicenseList.append(License)

    def GetBinaryHeaderLicense(self):
        if False:
            while True:
                i = 10
        return self.BinaryHeaderLicenseList

class ClonedRecordObject(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.IdNum = 0
        self.FarGuid = ''
        self.PackageGuid = ''
        self.PackageVersion = ''
        self.ModuleGuid = ''
        self.ModuleVersion = ''

    def SetId(self, IdNo):
        if False:
            return 10
        self.IdNum = IdNo

    def GetId(self):
        if False:
            for i in range(10):
                print('nop')
        return self.IdNum

    def SetFarGuid(self, FarGuid):
        if False:
            return 10
        self.FarGuid = FarGuid

    def GetFarGuid(self):
        if False:
            return 10
        return self.FarGuid

    def SetPackageGuid(self, PackageGuid):
        if False:
            return 10
        self.PackageGuid = PackageGuid

    def GetPackageGuid(self):
        if False:
            i = 10
            return i + 15
        return self.PackageGuid

    def SetPackageVersion(self, PackageVersion):
        if False:
            for i in range(10):
                print('nop')
        self.PackageVersion = PackageVersion

    def GetPackageVersion(self):
        if False:
            while True:
                i = 10
        return self.PackageVersion

    def SetModuleGuid(self, ModuleGuid):
        if False:
            while True:
                i = 10
        self.ModuleGuid = ModuleGuid

    def GetModuleGuid(self):
        if False:
            return 10
        return self.ModuleGuid

    def SetModuleVersion(self, ModuleVersion):
        if False:
            return 10
        self.ModuleVersion = ModuleVersion

    def GetModuleVersion(self):
        if False:
            for i in range(10):
                print('nop')
        return self.ModuleVersion

class TextObject(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.Lang = TAB_LANGUAGE_EN_US
        self.String = ''

    def SetLang(self, Lang):
        if False:
            print('Hello World!')
        self.Lang = Lang

    def GetLang(self):
        if False:
            i = 10
            return i + 15
        return self.Lang

    def SetString(self, String):
        if False:
            print('Hello World!')
        self.String = String

    def GetString(self):
        if False:
            i = 10
            return i + 15
        return self.String

class FileNameObject(CommonPropertiesObject):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.FileType = ''
        self.Filename = ''
        CommonPropertiesObject.__init__(self)

    def SetFileType(self, FileType):
        if False:
            i = 10
            return i + 15
        self.FileType = FileType

    def GetFileType(self):
        if False:
            return 10
        return self.FileType

    def SetFilename(self, Filename):
        if False:
            i = 10
            return i + 15
        self.Filename = Filename

    def GetFilename(self):
        if False:
            while True:
                i = 10
        return self.Filename

class FileObject(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.Executable = ''
        self.Uri = ''
        self.OsType = ''

    def SetExecutable(self, Executable):
        if False:
            for i in range(10):
                print('nop')
        self.Executable = Executable

    def GetExecutable(self):
        if False:
            print('Hello World!')
        return self.Executable

    def SetURI(self, URI):
        if False:
            i = 10
            return i + 15
        self.Uri = URI

    def GetURI(self):
        if False:
            i = 10
            return i + 15
        return self.Uri

    def SetOS(self, OsType):
        if False:
            return 10
        self.OsType = OsType

    def GetOS(self):
        if False:
            i = 10
            return i + 15
        return self.OsType

class MiscFileObject(CommonHeaderObject):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.Name = ''
        self.FileList = []
        CommonHeaderObject.__init__(self)

    def SetName(self, Name):
        if False:
            return 10
        self.Name = Name

    def GetName(self):
        if False:
            i = 10
            return i + 15
        return self.Name

    def SetFileList(self, FileList):
        if False:
            i = 10
            return i + 15
        self.FileList = FileList

    def GetFileList(self):
        if False:
            return 10
        return self.FileList

class ToolsObject(MiscFileObject):
    pass

class GuidVersionObject(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.Guid = ''
        self.Version = ''

    def SetGuid(self, Guid):
        if False:
            for i in range(10):
                print('nop')
        self.Guid = Guid

    def GetGuid(self):
        if False:
            i = 10
            return i + 15
        return self.Guid

    def SetVersion(self, Version):
        if False:
            print('Hello World!')
        self.Version = Version

    def GetVersion(self):
        if False:
            return 10
        return self.Version

class IdentificationObject(GuidVersionObject):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.Name = ''
        self.BaseName = ''
        self.FileName = ''
        self.FullPath = ''
        self.RelaPath = ''
        self.PackagePath = ''
        self.ModulePath = ''
        self.CombinePath = ''
        GuidVersionObject.__init__(self)

    def SetName(self, Name):
        if False:
            return 10
        self.Name = Name

    def GetName(self):
        if False:
            return 10
        return self.Name

    def SetBaseName(self, BaseName):
        if False:
            print('Hello World!')
        self.BaseName = BaseName

    def GetBaseName(self):
        if False:
            for i in range(10):
                print('nop')
        return self.BaseName

    def SetFileName(self, FileName):
        if False:
            print('Hello World!')
        self.FileName = FileName

    def GetFileName(self):
        if False:
            while True:
                i = 10
        return self.FileName

    def SetFullPath(self, FullPath):
        if False:
            return 10
        self.FullPath = FullPath

    def GetFullPath(self):
        if False:
            return 10
        return self.FullPath

    def SetRelaPath(self, RelaPath):
        if False:
            i = 10
            return i + 15
        self.RelaPath = RelaPath

    def GetRelaPath(self):
        if False:
            for i in range(10):
                print('nop')
        return self.RelaPath

    def SetPackagePath(self, PackagePath):
        if False:
            i = 10
            return i + 15
        self.PackagePath = PackagePath

    def GetPackagePath(self):
        if False:
            for i in range(10):
                print('nop')
        return self.PackagePath

    def SetModulePath(self, ModulePath):
        if False:
            return 10
        self.ModulePath = ModulePath

    def GetModulePath(self):
        if False:
            for i in range(10):
                print('nop')
        return self.ModulePath

    def SetCombinePath(self, CombinePath):
        if False:
            for i in range(10):
                print('nop')
        self.CombinePath = CombinePath

    def GetCombinePath(self):
        if False:
            while True:
                i = 10
        return self.CombinePath

class GuidProtocolPpiCommonObject(CommonPropertiesObject):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.Name = ''
        self.CName = ''
        self.Guid = ''
        self.SupModuleList = []
        CommonPropertiesObject.__init__(self)

    def SetName(self, Name):
        if False:
            i = 10
            return i + 15
        self.Name = Name

    def GetName(self):
        if False:
            while True:
                i = 10
        return self.Name

    def SetCName(self, CName):
        if False:
            return 10
        self.CName = CName

    def GetCName(self):
        if False:
            print('Hello World!')
        return self.CName

    def SetGuid(self, Guid):
        if False:
            print('Hello World!')
        self.Guid = Guid

    def GetGuid(self):
        if False:
            while True:
                i = 10
        return self.Guid

    def SetSupModuleList(self, SupModuleList):
        if False:
            print('Hello World!')
        self.SupModuleList = SupModuleList

    def GetSupModuleList(self):
        if False:
            return 10
        return self.SupModuleList

class GuidObject(GuidProtocolPpiCommonObject):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.VariableName = ''
        self.GuidTypeList = []
        GuidProtocolPpiCommonObject.__init__(self)

    def SetVariableName(self, VariableName):
        if False:
            return 10
        self.VariableName = VariableName

    def GetVariableName(self):
        if False:
            for i in range(10):
                print('nop')
        return self.VariableName

    def SetGuidTypeList(self, GuidTypeList):
        if False:
            return 10
        self.GuidTypeList = GuidTypeList

    def GetGuidTypeList(self):
        if False:
            print('Hello World!')
        return self.GuidTypeList

class ProtocolObject(GuidProtocolPpiCommonObject):

    def __init__(self):
        if False:
            return 10
        self.Notify = False
        GuidProtocolPpiCommonObject.__init__(self)

    def SetNotify(self, Notify):
        if False:
            i = 10
            return i + 15
        self.Notify = Notify

    def GetNotify(self):
        if False:
            i = 10
            return i + 15
        return self.Notify

class PpiObject(GuidProtocolPpiCommonObject):

    def __init__(self):
        if False:
            print('Hello World!')
        self.Notify = False
        GuidProtocolPpiCommonObject.__init__(self)

    def SetNotify(self, Notify):
        if False:
            print('Hello World!')
        self.Notify = Notify

    def GetNotify(self):
        if False:
            return 10
        return self.Notify

class DefineClass(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.Define = {}

class UserExtensionObject(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.UserID = ''
        self.Identifier = ''
        self.BinaryAbstractList = []
        self.BinaryDescriptionList = []
        self.BinaryCopyrightList = []
        self.BinaryLicenseList = []
        self.UniLangDefsList = []
        self.DefinesDict = {}
        self.BuildOptionDict = {}
        self.IncludesDict = {}
        self.SourcesDict = {}
        self.BinariesDict = {}
        self.Statement = ''
        self.SupArchList = []

    def SetStatement(self, Statement):
        if False:
            print('Hello World!')
        self.Statement = Statement

    def GetStatement(self):
        if False:
            for i in range(10):
                print('nop')
        return self.Statement

    def SetSupArchList(self, ArchList):
        if False:
            for i in range(10):
                print('nop')
        self.SupArchList = ArchList

    def GetSupArchList(self):
        if False:
            i = 10
            return i + 15
        return self.SupArchList

    def SetUserID(self, UserID):
        if False:
            while True:
                i = 10
        self.UserID = UserID

    def GetUserID(self):
        if False:
            return 10
        return self.UserID

    def SetIdentifier(self, Identifier):
        if False:
            print('Hello World!')
        self.Identifier = Identifier

    def GetIdentifier(self):
        if False:
            print('Hello World!')
        return self.Identifier

    def SetUniLangDefsList(self, UniLangDefsList):
        if False:
            for i in range(10):
                print('nop')
        self.UniLangDefsList = UniLangDefsList

    def GetUniLangDefsList(self):
        if False:
            print('Hello World!')
        return self.UniLangDefsList

    def SetBinaryAbstract(self, BinaryAbstractList):
        if False:
            print('Hello World!')
        self.BinaryAbstractList = BinaryAbstractList

    def GetBinaryAbstract(self, Lang=None):
        if False:
            i = 10
            return i + 15
        if Lang:
            for (Key, Value) in self.BinaryAbstractList:
                if Key == Lang:
                    return Value
            return None
        else:
            return self.BinaryAbstractList

    def SetBinaryDescription(self, BinaryDescriptionList):
        if False:
            i = 10
            return i + 15
        self.BinaryDescriptionList = BinaryDescriptionList

    def GetBinaryDescription(self, Lang=None):
        if False:
            i = 10
            return i + 15
        if Lang:
            for (Key, Value) in self.BinaryDescriptionList:
                if Key == Lang:
                    return Value
            return None
        else:
            return self.BinaryDescriptionList

    def SetBinaryCopyright(self, BinaryCopyrightList):
        if False:
            print('Hello World!')
        self.BinaryCopyrightList = BinaryCopyrightList

    def GetBinaryCopyright(self, Lang=None):
        if False:
            for i in range(10):
                print('nop')
        if Lang:
            for (Key, Value) in self.BinaryCopyrightList:
                if Key == Lang:
                    return Value
            return None
        else:
            return self.BinaryCopyrightList

    def SetBinaryLicense(self, BinaryLicenseList):
        if False:
            return 10
        self.BinaryLicenseList = BinaryLicenseList

    def GetBinaryLicense(self, Lang=None):
        if False:
            i = 10
            return i + 15
        if Lang:
            for (Key, Value) in self.BinaryLicenseList:
                if Key == Lang:
                    return Value
            return None
        else:
            return self.BinaryLicenseList

    def SetDefinesDict(self, DefinesDict):
        if False:
            return 10
        self.DefinesDict = DefinesDict

    def GetDefinesDict(self):
        if False:
            return 10
        return self.DefinesDict

    def SetBuildOptionDict(self, BuildOptionDict):
        if False:
            return 10
        self.BuildOptionDict = BuildOptionDict

    def GetBuildOptionDict(self):
        if False:
            print('Hello World!')
        return self.BuildOptionDict

    def SetIncludesDict(self, IncludesDict):
        if False:
            print('Hello World!')
        self.IncludesDict = IncludesDict

    def GetIncludesDict(self):
        if False:
            i = 10
            return i + 15
        return self.IncludesDict

    def SetSourcesDict(self, SourcesDict):
        if False:
            return 10
        self.SourcesDict = SourcesDict

    def GetSourcesDict(self):
        if False:
            i = 10
            return i + 15
        return self.SourcesDict

    def SetBinariesDict(self, BinariesDict):
        if False:
            while True:
                i = 10
        self.BinariesDict = BinariesDict

    def GetBinariesDict(self):
        if False:
            while True:
                i = 10
        return self.BinariesDict

class LibraryClassObject(CommonPropertiesObject):

    def __init__(self):
        if False:
            return 10
        self.LibraryClass = ''
        self.IncludeHeader = ''
        self.SupModuleList = []
        self.RecommendedInstance = GuidVersionObject()
        CommonPropertiesObject.__init__(self)

    def SetLibraryClass(self, LibraryClass):
        if False:
            return 10
        self.LibraryClass = LibraryClass

    def GetLibraryClass(self):
        if False:
            print('Hello World!')
        return self.LibraryClass

    def SetSupModuleList(self, SupModuleList):
        if False:
            i = 10
            return i + 15
        self.SupModuleList = SupModuleList

    def GetSupModuleList(self):
        if False:
            i = 10
            return i + 15
        return self.SupModuleList

    def SetIncludeHeader(self, IncludeHeader):
        if False:
            print('Hello World!')
        self.IncludeHeader = IncludeHeader

    def GetIncludeHeader(self):
        if False:
            print('Hello World!')
        return self.IncludeHeader

    def SetRecommendedInstance(self, RecommendedInstance):
        if False:
            i = 10
            return i + 15
        self.RecommendedInstance = RecommendedInstance

    def GetRecommendedInstance(self):
        if False:
            i = 10
            return i + 15
        return self.RecommendedInstance

class PcdErrorObject(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.ValidValue = ''
        self.ValidValueLang = ''
        self.ValidValueRange = ''
        self.Expression = ''
        self.ErrorNumber = ''
        self.ErrorMessageList = []
        self.TokenSpaceGuidCName = ''
        self.CName = ''
        self.FileLine = ''
        self.LineNum = 0

    def SetValidValue(self, ValidValue):
        if False:
            i = 10
            return i + 15
        self.ValidValue = ValidValue

    def GetValidValue(self):
        if False:
            while True:
                i = 10
        return self.ValidValue

    def SetValidValueLang(self, ValidValueLang):
        if False:
            print('Hello World!')
        self.ValidValueLang = ValidValueLang

    def GetValidValueLang(self):
        if False:
            for i in range(10):
                print('nop')
        return self.ValidValueLang

    def SetValidValueRange(self, ValidValueRange):
        if False:
            i = 10
            return i + 15
        self.ValidValueRange = ValidValueRange

    def GetValidValueRange(self):
        if False:
            return 10
        return self.ValidValueRange

    def SetExpression(self, Expression):
        if False:
            return 10
        self.Expression = Expression

    def GetExpression(self):
        if False:
            i = 10
            return i + 15
        return self.Expression

    def SetErrorNumber(self, ErrorNumber):
        if False:
            while True:
                i = 10
        self.ErrorNumber = ErrorNumber

    def GetErrorNumber(self):
        if False:
            i = 10
            return i + 15
        return self.ErrorNumber

    def SetErrorMessageList(self, ErrorMessageList):
        if False:
            for i in range(10):
                print('nop')
        self.ErrorMessageList = ErrorMessageList

    def GetErrorMessageList(self):
        if False:
            return 10
        return self.ErrorMessageList

    def SetTokenSpaceGuidCName(self, TokenSpaceGuidCName):
        if False:
            while True:
                i = 10
        self.TokenSpaceGuidCName = TokenSpaceGuidCName

    def GetTokenSpaceGuidCName(self):
        if False:
            return 10
        return self.TokenSpaceGuidCName

    def SetCName(self, CName):
        if False:
            while True:
                i = 10
        self.CName = CName

    def GetCName(self):
        if False:
            for i in range(10):
                print('nop')
        return self.CName

    def SetFileLine(self, FileLine):
        if False:
            while True:
                i = 10
        self.FileLine = FileLine

    def GetFileLine(self):
        if False:
            i = 10
            return i + 15
        return self.FileLine

    def SetLineNum(self, LineNum):
        if False:
            return 10
        self.LineNum = LineNum

    def GetLineNum(self):
        if False:
            return 10
        return self.LineNum

class IncludeObject(CommonPropertiesObject):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.FilePath = ''
        self.ModuleType = ''
        self.SupModuleList = []
        self.Comment = ''
        CommonPropertiesObject.__init__(self)

    def SetFilePath(self, FilePath):
        if False:
            print('Hello World!')
        self.FilePath = FilePath

    def GetFilePath(self):
        if False:
            while True:
                i = 10
        return self.FilePath

    def SetModuleType(self, ModuleType):
        if False:
            for i in range(10):
                print('nop')
        self.ModuleType = ModuleType

    def GetModuleType(self):
        if False:
            for i in range(10):
                print('nop')
        return self.ModuleType

    def SetSupModuleList(self, SupModuleList):
        if False:
            print('Hello World!')
        self.SupModuleList = SupModuleList

    def GetSupModuleList(self):
        if False:
            return 10
        return self.SupModuleList

    def SetComment(self, Comment):
        if False:
            for i in range(10):
                print('nop')
        self.Comment = Comment

    def GetComment(self):
        if False:
            print('Hello World!')
        return self.Comment

class PcdObject(CommonPropertiesObject, HelpTextListObject, PromptListObject):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.PcdCName = ''
        self.CName = ''
        self.Token = ''
        self.TokenSpaceGuidCName = ''
        self.TokenSpaceGuidValue = ''
        self.DatumType = ''
        self.MaxDatumSize = ''
        self.DefaultValue = ''
        self.Offset = ''
        self.ValidUsage = ''
        self.ItemType = ''
        self.PcdErrorsList = []
        self.SupModuleList = []
        CommonPropertiesObject.__init__(self)
        HelpTextListObject.__init__(self)
        PromptListObject.__init__(self)

    def SetPcdCName(self, PcdCName):
        if False:
            i = 10
            return i + 15
        self.PcdCName = PcdCName

    def GetPcdCName(self):
        if False:
            return 10
        return self.PcdCName

    def SetCName(self, CName):
        if False:
            print('Hello World!')
        self.CName = CName

    def GetCName(self):
        if False:
            i = 10
            return i + 15
        return self.CName

    def SetToken(self, Token):
        if False:
            return 10
        self.Token = Token

    def GetOffset(self):
        if False:
            while True:
                i = 10
        return self.Offset

    def SetOffset(self, Offset):
        if False:
            while True:
                i = 10
        self.Offset = Offset

    def GetToken(self):
        if False:
            print('Hello World!')
        return self.Token

    def SetTokenSpaceGuidCName(self, TokenSpaceGuidCName):
        if False:
            print('Hello World!')
        self.TokenSpaceGuidCName = TokenSpaceGuidCName

    def GetTokenSpaceGuidCName(self):
        if False:
            i = 10
            return i + 15
        return self.TokenSpaceGuidCName

    def SetTokenSpaceGuidValue(self, TokenSpaceGuidValue):
        if False:
            i = 10
            return i + 15
        self.TokenSpaceGuidValue = TokenSpaceGuidValue

    def GetTokenSpaceGuidValue(self):
        if False:
            while True:
                i = 10
        return self.TokenSpaceGuidValue

    def SetDatumType(self, DatumType):
        if False:
            return 10
        self.DatumType = DatumType

    def GetDatumType(self):
        if False:
            while True:
                i = 10
        return self.DatumType

    def SetMaxDatumSize(self, MaxDatumSize):
        if False:
            return 10
        self.MaxDatumSize = MaxDatumSize

    def GetMaxDatumSize(self):
        if False:
            for i in range(10):
                print('nop')
        return self.MaxDatumSize

    def SetDefaultValue(self, DefaultValue):
        if False:
            for i in range(10):
                print('nop')
        self.DefaultValue = DefaultValue

    def GetDefaultValue(self):
        if False:
            return 10
        return self.DefaultValue

    def SetValidUsage(self, ValidUsage):
        if False:
            for i in range(10):
                print('nop')
        self.ValidUsage = ValidUsage

    def GetValidUsage(self):
        if False:
            print('Hello World!')
        return self.ValidUsage

    def SetPcdErrorsList(self, PcdErrorsList):
        if False:
            while True:
                i = 10
        self.PcdErrorsList = PcdErrorsList

    def GetPcdErrorsList(self):
        if False:
            i = 10
            return i + 15
        return self.PcdErrorsList

    def SetItemType(self, ItemType):
        if False:
            return 10
        self.ItemType = ItemType

    def GetItemType(self):
        if False:
            print('Hello World!')
        return self.ItemType

    def SetSupModuleList(self, SupModuleList):
        if False:
            i = 10
            return i + 15
        self.SupModuleList = SupModuleList

    def GetSupModuleList(self):
        if False:
            i = 10
            return i + 15
        return self.SupModuleList