"""
InfDefineObject
"""
import os
import re
from Logger import StringTable as ST
from Logger import ToolError
from Library import GlobalData
from Library import DataType as DT
from Library.StringUtils import GetSplitValueList
from Library.Misc import CheckGuidRegFormat
from Library.Misc import Sdict
from Library.Misc import ConvPathFromAbsToRel
from Library.Misc import ValidateUNIFilePath
from Library.ExpressionValidate import IsValidFeatureFlagExp
from Library.ParserValidate import IsValidWord
from Library.ParserValidate import IsValidInfMoudleType
from Library.ParserValidate import IsValidHex
from Library.ParserValidate import IsValidHexVersion
from Library.ParserValidate import IsValidDecVersion
from Library.ParserValidate import IsValidCVariableName
from Library.ParserValidate import IsValidBoolType
from Library.ParserValidate import IsValidPath
from Library.ParserValidate import IsValidFamily
from Library.ParserValidate import IsValidIdentifier
from Library.ParserValidate import IsValidDecVersionVal
from Object.Parser.InfCommonObject import InfLineCommentObject
from Object.Parser.InfCommonObject import CurrentLine
from Object.Parser.InfCommonObject import InfSectionCommonDef
from Object.Parser.InfMisc import ErrorInInf
from Object.Parser.InfDefineCommonObject import InfDefineLibraryItem
from Object.Parser.InfDefineCommonObject import InfDefineEntryPointItem
from Object.Parser.InfDefineCommonObject import InfDefineUnloadImageItem
from Object.Parser.InfDefineCommonObject import InfDefineConstructorItem
from Object.Parser.InfDefineCommonObject import InfDefineDestructorItem

class InfDefSectionOptionRomInfo:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.PciVendorId = None
        self.PciDeviceId = None
        self.PciClassCode = None
        self.PciRevision = None
        self.PciCompress = None
        self.CurrentLine = ['', -1, '']

    def SetPciVendorId(self, PciVendorId, Comments):
        if False:
            i = 10
            return i + 15
        if self.PciVendorId is not None:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_ITEM_MORE_THAN_ONE_FOUND % DT.TAB_INF_DEFINES_PCI_VENDOR_ID, LineInfo=self.CurrentLine)
            return False
        if IsValidHex(PciVendorId):
            self.PciVendorId = InfDefMember()
            self.PciVendorId.SetValue(PciVendorId)
            self.PciVendorId.Comments = Comments
            return True
        else:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % PciVendorId, LineInfo=self.CurrentLine)
            return False

    def GetPciVendorId(self):
        if False:
            return 10
        return self.PciVendorId

    def SetPciDeviceId(self, PciDeviceId, Comments):
        if False:
            print('Hello World!')
        if self.PciDeviceId is not None:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_ITEM_MORE_THAN_ONE_FOUND % DT.TAB_INF_DEFINES_PCI_DEVICE_ID, LineInfo=self.CurrentLine)
            return False
        if IsValidHex(PciDeviceId):
            self.PciDeviceId = InfDefMember()
            self.PciDeviceId.SetValue(PciDeviceId)
            self.PciDeviceId.Comments = Comments
            return True
        else:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % PciDeviceId, LineInfo=self.CurrentLine)
            return False

    def GetPciDeviceId(self):
        if False:
            i = 10
            return i + 15
        return self.PciDeviceId

    def SetPciClassCode(self, PciClassCode, Comments):
        if False:
            return 10
        if self.PciClassCode is not None:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_ITEM_MORE_THAN_ONE_FOUND % DT.TAB_INF_DEFINES_PCI_CLASS_CODE, LineInfo=self.CurrentLine)
            return False
        if IsValidHex(PciClassCode):
            self.PciClassCode = InfDefMember()
            self.PciClassCode.SetValue(PciClassCode)
            self.PciClassCode.Comments = Comments
            return True
        else:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % PciClassCode, LineInfo=self.CurrentLine)
            return False

    def GetPciClassCode(self):
        if False:
            print('Hello World!')
        return self.PciClassCode

    def SetPciRevision(self, PciRevision, Comments):
        if False:
            return 10
        if self.PciRevision is not None:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_ITEM_MORE_THAN_ONE_FOUND % DT.TAB_INF_DEFINES_PCI_REVISION, LineInfo=self.CurrentLine)
            return False
        if IsValidHex(PciRevision):
            self.PciRevision = InfDefMember()
            self.PciRevision.SetValue(PciRevision)
            self.PciRevision.Comments = Comments
            return True
        else:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % PciRevision, LineInfo=self.CurrentLine)
            return False

    def GetPciRevision(self):
        if False:
            for i in range(10):
                print('nop')
        return self.PciRevision

    def SetPciCompress(self, PciCompress, Comments):
        if False:
            print('Hello World!')
        if self.PciCompress is not None:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_ITEM_MORE_THAN_ONE_FOUND % DT.TAB_INF_DEFINES_PCI_COMPRESS, LineInfo=self.CurrentLine)
            return False
        if PciCompress == 'TRUE' or PciCompress == 'FALSE':
            self.PciCompress = InfDefMember()
            self.PciCompress.SetValue(PciCompress)
            self.PciCompress.Comments = Comments
            return True
        else:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % PciCompress, LineInfo=self.CurrentLine)
            return False

    def GetPciCompress(self):
        if False:
            for i in range(10):
                print('nop')
        return self.PciCompress

class InfDefSection(InfDefSectionOptionRomInfo):

    def __init__(self):
        if False:
            print('Hello World!')
        self.BaseName = None
        self.FileGuid = None
        self.ModuleType = None
        self.ModuleUniFileName = None
        self.InfVersion = None
        self.EdkReleaseVersion = None
        self.UefiSpecificationVersion = None
        self.PiSpecificationVersion = None
        self.LibraryClass = []
        self.Package = None
        self.VersionString = None
        self.PcdIsDriver = None
        self.EntryPoint = []
        self.UnloadImages = []
        self.Constructor = []
        self.Destructor = []
        self.Shadow = None
        self.CustomMakefile = []
        self.Specification = []
        self.UefiHiiResourceSection = None
        self.DpxSource = []
        self.CurrentLine = ['', -1, '']
        InfDefSectionOptionRomInfo.__init__(self)

    def SetBaseName(self, BaseName, Comments):
        if False:
            print('Hello World!')
        if self.BaseName is not None:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_ITEM_MORE_THAN_ONE_FOUND % DT.TAB_INF_DEFINES_BASE_NAME, LineInfo=self.CurrentLine)
            return False
        if not (BaseName == '' or BaseName is None):
            if IsValidWord(BaseName) and (not BaseName.startswith('_')):
                self.BaseName = InfDefMember()
                self.BaseName.SetValue(BaseName)
                self.BaseName.Comments = Comments
                return True
            else:
                ErrorInInf(ST.ERR_INF_PARSER_DEFINE_NAME_INVALID % BaseName, LineInfo=self.CurrentLine)
                return False

    def GetBaseName(self):
        if False:
            return 10
        return self.BaseName

    def SetFileGuid(self, FileGuid, Comments):
        if False:
            for i in range(10):
                print('nop')
        if self.FileGuid is not None:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_ITEM_MORE_THAN_ONE_FOUND % DT.TAB_INF_DEFINES_FILE_GUID, LineInfo=self.CurrentLine)
            return False
        if CheckGuidRegFormat(FileGuid):
            self.FileGuid = InfDefMember()
            self.FileGuid.SetValue(FileGuid)
            self.FileGuid.Comments = Comments
            return True
        else:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_GUID_INVALID % FileGuid, LineInfo=self.CurrentLine)
            return False

    def GetFileGuid(self):
        if False:
            return 10
        return self.FileGuid

    def SetModuleType(self, ModuleType, Comments):
        if False:
            i = 10
            return i + 15
        if self.ModuleType is not None:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_ITEM_MORE_THAN_ONE_FOUND % DT.TAB_INF_DEFINES_MODULE_TYPE, LineInfo=self.CurrentLine)
            return False
        if IsValidInfMoudleType(ModuleType):
            self.ModuleType = InfDefMember()
            self.ModuleType.SetValue(ModuleType)
            self.ModuleType.CurrentLine = CurrentLine()
            self.ModuleType.CurrentLine.SetLineNo(self.CurrentLine[1])
            self.ModuleType.CurrentLine.SetLineString(self.CurrentLine[2])
            self.ModuleType.CurrentLine.SetFileName(self.CurrentLine[0])
            self.ModuleType.Comments = Comments
            return True
        else:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_MODULETYPE_INVALID % ModuleType, LineInfo=self.CurrentLine)
            return False

    def GetModuleType(self):
        if False:
            for i in range(10):
                print('nop')
        return self.ModuleType

    def SetModuleUniFileName(self, ModuleUniFileName, Comments):
        if False:
            while True:
                i = 10
        if Comments:
            pass
        if self.ModuleUniFileName is not None:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_ITEM_MORE_THAN_ONE_FOUND % DT.TAB_INF_DEFINES_MODULE_UNI_FILE, LineInfo=self.CurrentLine)
        self.ModuleUniFileName = ModuleUniFileName

    def GetModuleUniFileName(self):
        if False:
            for i in range(10):
                print('nop')
        return self.ModuleUniFileName

    def SetInfVersion(self, InfVersion, Comments):
        if False:
            print('Hello World!')
        if self.InfVersion is not None:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_ITEM_MORE_THAN_ONE_FOUND % DT.TAB_INF_DEFINES_INF_VERSION, LineInfo=self.CurrentLine)
            return False
        if IsValidHex(InfVersion):
            if InfVersion < '0x00010005':
                ErrorInInf(ST.ERR_INF_PARSER_NOT_SUPPORT_EDKI_INF, ErrorCode=ToolError.EDK1_INF_ERROR, LineInfo=self.CurrentLine)
        elif IsValidDecVersionVal(InfVersion):
            if InfVersion < 65541:
                ErrorInInf(ST.ERR_INF_PARSER_NOT_SUPPORT_EDKI_INF, ErrorCode=ToolError.EDK1_INF_ERROR, LineInfo=self.CurrentLine)
        else:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % InfVersion, LineInfo=self.CurrentLine)
            return False
        self.InfVersion = InfDefMember()
        self.InfVersion.SetValue(InfVersion)
        self.InfVersion.Comments = Comments
        return True

    def GetInfVersion(self):
        if False:
            for i in range(10):
                print('nop')
        return self.InfVersion

    def SetEdkReleaseVersion(self, EdkReleaseVersion, Comments):
        if False:
            while True:
                i = 10
        if self.EdkReleaseVersion is not None:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_ITEM_MORE_THAN_ONE_FOUND % DT.TAB_INF_DEFINES_EDK_RELEASE_VERSION, LineInfo=self.CurrentLine)
            return False
        if IsValidHexVersion(EdkReleaseVersion) or IsValidDecVersionVal(EdkReleaseVersion):
            self.EdkReleaseVersion = InfDefMember()
            self.EdkReleaseVersion.SetValue(EdkReleaseVersion)
            self.EdkReleaseVersion.Comments = Comments
            return True
        else:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % EdkReleaseVersion, LineInfo=self.CurrentLine)
            return False

    def GetEdkReleaseVersion(self):
        if False:
            print('Hello World!')
        return self.EdkReleaseVersion

    def SetUefiSpecificationVersion(self, UefiSpecificationVersion, Comments):
        if False:
            i = 10
            return i + 15
        if self.UefiSpecificationVersion is not None:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_ITEM_MORE_THAN_ONE_FOUND % DT.TAB_INF_DEFINES_UEFI_SPECIFICATION_VERSION, LineInfo=self.CurrentLine)
            return False
        if IsValidHexVersion(UefiSpecificationVersion) or IsValidDecVersionVal(UefiSpecificationVersion):
            self.UefiSpecificationVersion = InfDefMember()
            self.UefiSpecificationVersion.SetValue(UefiSpecificationVersion)
            self.UefiSpecificationVersion.Comments = Comments
            return True
        else:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % UefiSpecificationVersion, LineInfo=self.CurrentLine)
            return False

    def GetUefiSpecificationVersion(self):
        if False:
            print('Hello World!')
        return self.UefiSpecificationVersion

    def SetPiSpecificationVersion(self, PiSpecificationVersion, Comments):
        if False:
            i = 10
            return i + 15
        if self.PiSpecificationVersion is not None:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_ITEM_MORE_THAN_ONE_FOUND % DT.TAB_INF_DEFINES_PI_SPECIFICATION_VERSION, LineInfo=self.CurrentLine)
            return False
        if IsValidHexVersion(PiSpecificationVersion) or IsValidDecVersionVal(PiSpecificationVersion):
            self.PiSpecificationVersion = InfDefMember()
            self.PiSpecificationVersion.SetValue(PiSpecificationVersion)
            self.PiSpecificationVersion.Comments = Comments
            return True
        else:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % PiSpecificationVersion, LineInfo=self.CurrentLine)
            return False

    def GetPiSpecificationVersion(self):
        if False:
            while True:
                i = 10
        return self.PiSpecificationVersion

    def SetLibraryClass(self, LibraryClass, Comments):
        if False:
            i = 10
            return i + 15
        ValueList = GetSplitValueList(LibraryClass)
        Name = ValueList[0]
        if IsValidWord(Name):
            InfDefineLibraryItemObj = InfDefineLibraryItem()
            InfDefineLibraryItemObj.SetLibraryName(Name)
            InfDefineLibraryItemObj.Comments = Comments
            if len(ValueList) == 2:
                Type = ValueList[1]
                TypeList = GetSplitValueList(Type, ' ')
                TypeList = [Type for Type in TypeList if Type != '']
                for Item in TypeList:
                    if Item not in DT.MODULE_LIST:
                        ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % Item, LineInfo=self.CurrentLine)
                        return False
                InfDefineLibraryItemObj.SetTypes(TypeList)
            self.LibraryClass.append(InfDefineLibraryItemObj)
        else:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % Name, LineInfo=self.CurrentLine)
            return False
        return True

    def GetLibraryClass(self):
        if False:
            return 10
        return self.LibraryClass

    def SetVersionString(self, VersionString, Comments):
        if False:
            for i in range(10):
                print('nop')
        if self.VersionString is not None:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_ITEM_MORE_THAN_ONE_FOUND % DT.TAB_INF_DEFINES_VERSION_STRING, LineInfo=self.CurrentLine)
            return False
        if not IsValidDecVersion(VersionString):
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % VersionString, LineInfo=self.CurrentLine)
        self.VersionString = InfDefMember()
        self.VersionString.SetValue(VersionString)
        self.VersionString.Comments = Comments
        return True

    def GetVersionString(self):
        if False:
            while True:
                i = 10
        return self.VersionString

    def SetPcdIsDriver(self, PcdIsDriver, Comments):
        if False:
            return 10
        if self.PcdIsDriver is not None:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_ITEM_MORE_THAN_ONE_FOUND % DT.TAB_INF_DEFINES_PCD_IS_DRIVER, LineInfo=self.CurrentLine)
            return False
        if PcdIsDriver == 'PEI_PCD_DRIVER' or PcdIsDriver == 'DXE_PCD_DRIVER':
            self.PcdIsDriver = InfDefMember()
            self.PcdIsDriver.SetValue(PcdIsDriver)
            self.PcdIsDriver.Comments = Comments
            return True
        else:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % PcdIsDriver, LineInfo=self.CurrentLine)
            return False

    def GetPcdIsDriver(self):
        if False:
            for i in range(10):
                print('nop')
        return self.PcdIsDriver

    def SetEntryPoint(self, EntryPoint, Comments):
        if False:
            return 10
        ValueList = []
        TokenList = GetSplitValueList(EntryPoint, DT.TAB_VALUE_SPLIT)
        ValueList[0:len(TokenList)] = TokenList
        InfDefineEntryPointItemObj = InfDefineEntryPointItem()
        if not IsValidCVariableName(ValueList[0]):
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % ValueList[0], LineInfo=self.CurrentLine)
        InfDefineEntryPointItemObj.SetCName(ValueList[0])
        if len(ValueList) == 2:
            if ValueList[1].strip() == '':
                ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % ValueList[1], LineInfo=self.CurrentLine)
            FeatureFlagRtv = IsValidFeatureFlagExp(ValueList[1].strip())
            if not FeatureFlagRtv[0]:
                ErrorInInf(ST.ERR_INF_PARSER_FEATURE_FLAG_EXP_SYNTAX_INVLID % FeatureFlagRtv[1], LineInfo=self.CurrentLine)
            InfDefineEntryPointItemObj.SetFeatureFlagExp(ValueList[1])
        if len(ValueList) > 2:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % EntryPoint, LineInfo=self.CurrentLine)
        InfDefineEntryPointItemObj.Comments = Comments
        self.EntryPoint.append(InfDefineEntryPointItemObj)

    def GetEntryPoint(self):
        if False:
            i = 10
            return i + 15
        return self.EntryPoint

    def SetUnloadImages(self, UnloadImages, Comments):
        if False:
            while True:
                i = 10
        ValueList = []
        TokenList = GetSplitValueList(UnloadImages, DT.TAB_VALUE_SPLIT)
        ValueList[0:len(TokenList)] = TokenList
        InfDefineUnloadImageItemObj = InfDefineUnloadImageItem()
        if not IsValidCVariableName(ValueList[0]):
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % ValueList[0], LineInfo=self.CurrentLine)
        InfDefineUnloadImageItemObj.SetCName(ValueList[0])
        if len(ValueList) == 2:
            if ValueList[1].strip() == '':
                ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % ValueList[1], LineInfo=self.CurrentLine)
            FeatureFlagRtv = IsValidFeatureFlagExp(ValueList[1].strip())
            if not FeatureFlagRtv[0]:
                ErrorInInf(ST.ERR_INF_PARSER_FEATURE_FLAG_EXP_SYNTAX_INVLID % FeatureFlagRtv[1], LineInfo=self.CurrentLine)
            InfDefineUnloadImageItemObj.SetFeatureFlagExp(ValueList[1])
        if len(ValueList) > 2:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % UnloadImages, LineInfo=self.CurrentLine)
        InfDefineUnloadImageItemObj.Comments = Comments
        self.UnloadImages.append(InfDefineUnloadImageItemObj)

    def GetUnloadImages(self):
        if False:
            i = 10
            return i + 15
        return self.UnloadImages

    def SetConstructor(self, Constructor, Comments):
        if False:
            for i in range(10):
                print('nop')
        ValueList = []
        TokenList = GetSplitValueList(Constructor, DT.TAB_VALUE_SPLIT)
        ValueList[0:len(TokenList)] = TokenList
        InfDefineConstructorItemObj = InfDefineConstructorItem()
        if not IsValidCVariableName(ValueList[0]):
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % ValueList[0], LineInfo=self.CurrentLine)
        InfDefineConstructorItemObj.SetCName(ValueList[0])
        if len(ValueList) >= 2:
            ModList = GetSplitValueList(ValueList[1], ' ')
            if ValueList[1].strip() == '':
                ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % ValueList[1], LineInfo=self.CurrentLine)
            for ModItem in ModList:
                if ModItem not in DT.MODULE_LIST:
                    ErrorInInf(ST.ERR_INF_PARSER_DEFINE_MODULETYPE_INVALID % ModItem, LineInfo=self.CurrentLine)
            InfDefineConstructorItemObj.SetSupModList(ModList)
        if len(ValueList) == 3:
            if ValueList[2].strip() == '':
                ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % ValueList[2], LineInfo=self.CurrentLine)
            FeatureFlagRtv = IsValidFeatureFlagExp(ValueList[2].strip())
            if not FeatureFlagRtv[0]:
                ErrorInInf(ST.ERR_INF_PARSER_FEATURE_FLAG_EXP_SYNTAX_INVLID % FeatureFlagRtv[2], LineInfo=self.CurrentLine)
            InfDefineConstructorItemObj.SetFeatureFlagExp(ValueList[2])
        if len(ValueList) > 3:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % Constructor, LineInfo=self.CurrentLine)
        InfDefineConstructorItemObj.Comments = Comments
        self.Constructor.append(InfDefineConstructorItemObj)

    def GetConstructor(self):
        if False:
            i = 10
            return i + 15
        return self.Constructor

    def SetDestructor(self, Destructor, Comments):
        if False:
            print('Hello World!')
        ValueList = []
        TokenList = GetSplitValueList(Destructor, DT.TAB_VALUE_SPLIT)
        ValueList[0:len(TokenList)] = TokenList
        InfDefineDestructorItemObj = InfDefineDestructorItem()
        if not IsValidCVariableName(ValueList[0]):
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % ValueList[0], LineInfo=self.CurrentLine)
        InfDefineDestructorItemObj.SetCName(ValueList[0])
        if len(ValueList) >= 2:
            ModList = GetSplitValueList(ValueList[1].strip(), ' ')
            if ValueList[1].strip() == '':
                ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % ValueList[1], LineInfo=self.CurrentLine)
            for ModItem in ModList:
                if ModItem not in DT.MODULE_LIST:
                    ErrorInInf(ST.ERR_INF_PARSER_DEFINE_MODULETYPE_INVALID % ModItem, LineInfo=self.CurrentLine)
            InfDefineDestructorItemObj.SetSupModList(ModList)
        if len(ValueList) == 3:
            if ValueList[2].strip() == '':
                ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % ValueList[2], LineInfo=self.CurrentLine)
            FeatureFlagRtv = IsValidFeatureFlagExp(ValueList[2].strip())
            if not FeatureFlagRtv[0]:
                ErrorInInf(ST.ERR_INF_PARSER_FEATURE_FLAG_EXP_SYNTAX_INVLID % FeatureFlagRtv[1], LineInfo=self.CurrentLine)
            InfDefineDestructorItemObj.SetFeatureFlagExp(ValueList[2])
        if len(ValueList) > 3:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % Destructor, LineInfo=self.CurrentLine)
        InfDefineDestructorItemObj.Comments = Comments
        self.Destructor.append(InfDefineDestructorItemObj)

    def GetDestructor(self):
        if False:
            print('Hello World!')
        return self.Destructor

    def SetShadow(self, Shadow, Comments):
        if False:
            while True:
                i = 10
        if self.Shadow is not None:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_ITEM_MORE_THAN_ONE_FOUND % DT.TAB_INF_DEFINES_SHADOW, LineInfo=self.CurrentLine)
            return False
        if IsValidBoolType(Shadow):
            self.Shadow = InfDefMember()
            self.Shadow.SetValue(Shadow)
            self.Shadow.Comments = Comments
            return True
        else:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % Shadow, LineInfo=self.CurrentLine)
            return False

    def GetShadow(self):
        if False:
            i = 10
            return i + 15
        return self.Shadow

    def SetCustomMakefile(self, CustomMakefile, Comments):
        if False:
            for i in range(10):
                print('nop')
        if not (CustomMakefile == '' or CustomMakefile is None):
            ValueList = GetSplitValueList(CustomMakefile)
            if len(ValueList) == 1:
                FileName = ValueList[0]
                Family = ''
            else:
                Family = ValueList[0]
                FileName = ValueList[1]
            Family = Family.strip()
            if Family != '':
                if not IsValidFamily(Family):
                    ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % Family, LineInfo=self.CurrentLine)
                    return False
            IsValidFileFlag = False
            ModulePath = os.path.split(self.CurrentLine[0])[0]
            if IsValidPath(FileName, ModulePath):
                IsValidFileFlag = True
            else:
                ErrorInInf(ST.ERR_INF_PARSER_FILE_NOT_EXIST_OR_NAME_INVALID % FileName, LineInfo=self.CurrentLine)
                return False
            if IsValidFileFlag:
                FileName = ConvPathFromAbsToRel(FileName, GlobalData.gINF_MODULE_DIR)
                self.CustomMakefile.append((Family, FileName, Comments))
                IsValidFileFlag = False
            return True
        else:
            return False

    def GetCustomMakefile(self):
        if False:
            while True:
                i = 10
        return self.CustomMakefile

    def SetSpecification(self, Specification, Comments):
        if False:
            while True:
                i = 10
        __ValueList = []
        TokenList = GetSplitValueList(Specification, DT.TAB_EQUAL_SPLIT, 1)
        __ValueList[0:len(TokenList)] = TokenList
        if len(__ValueList) != 2:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_ITEM_NO_NAME + ' Or ' + ST.ERR_INF_PARSER_DEFINE_ITEM_NO_VALUE, LineInfo=self.CurrentLine)
        Name = __ValueList[0].strip()
        Version = __ValueList[1].strip()
        if IsValidIdentifier(Name):
            if IsValidDecVersion(Version):
                self.Specification.append((Name, Version, Comments))
                return True
            else:
                ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % Version, LineInfo=self.CurrentLine)
                return False
        else:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % Name, LineInfo=self.CurrentLine)
            return False
        return True

    def GetSpecification(self):
        if False:
            return 10
        return self.Specification

    def SetUefiHiiResourceSection(self, UefiHiiResourceSection, Comments):
        if False:
            i = 10
            return i + 15
        if self.UefiHiiResourceSection is not None:
            ErrorInInf(ST.ERR_INF_PARSER_DEFINE_ITEM_MORE_THAN_ONE_FOUND % DT.TAB_INF_DEFINES_UEFI_HII_RESOURCE_SECTION, LineInfo=self.CurrentLine)
            return False
        if not (UefiHiiResourceSection == '' or UefiHiiResourceSection is None):
            if IsValidBoolType(UefiHiiResourceSection):
                self.UefiHiiResourceSection = InfDefMember()
                self.UefiHiiResourceSection.SetValue(UefiHiiResourceSection)
                self.UefiHiiResourceSection.Comments = Comments
                return True
            else:
                ErrorInInf(ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % UefiHiiResourceSection, LineInfo=self.CurrentLine)
                return False
        else:
            return False

    def GetUefiHiiResourceSection(self):
        if False:
            print('Hello World!')
        return self.UefiHiiResourceSection

    def SetDpxSource(self, DpxSource, Comments):
        if False:
            return 10
        IsValidFileFlag = False
        ModulePath = os.path.split(self.CurrentLine[0])[0]
        if IsValidPath(DpxSource, ModulePath):
            IsValidFileFlag = True
        else:
            ErrorInInf(ST.ERR_INF_PARSER_FILE_NOT_EXIST_OR_NAME_INVALID % DpxSource, LineInfo=self.CurrentLine)
            return False
        if IsValidFileFlag:
            DpxSource = ConvPathFromAbsToRel(DpxSource, GlobalData.gINF_MODULE_DIR)
            self.DpxSource.append((DpxSource, Comments))
            IsValidFileFlag = False
        return True

    def GetDpxSource(self):
        if False:
            return 10
        return self.DpxSource
gFUNCTION_MAPPING_FOR_DEFINE_SECTION = {DT.TAB_INF_DEFINES_BASE_NAME: InfDefSection.SetBaseName, DT.TAB_INF_DEFINES_FILE_GUID: InfDefSection.SetFileGuid, DT.TAB_INF_DEFINES_MODULE_TYPE: InfDefSection.SetModuleType, DT.TAB_INF_DEFINES_INF_VERSION: InfDefSection.SetInfVersion, DT.TAB_INF_DEFINES_MODULE_UNI_FILE: InfDefSection.SetModuleUniFileName, DT.TAB_INF_DEFINES_EDK_RELEASE_VERSION: InfDefSection.SetEdkReleaseVersion, DT.TAB_INF_DEFINES_UEFI_SPECIFICATION_VERSION: InfDefSection.SetUefiSpecificationVersion, DT.TAB_INF_DEFINES_PI_SPECIFICATION_VERSION: InfDefSection.SetPiSpecificationVersion, DT.TAB_INF_DEFINES_LIBRARY_CLASS: InfDefSection.SetLibraryClass, DT.TAB_INF_DEFINES_VERSION_STRING: InfDefSection.SetVersionString, DT.TAB_INF_DEFINES_PCD_IS_DRIVER: InfDefSection.SetPcdIsDriver, DT.TAB_INF_DEFINES_ENTRY_POINT: InfDefSection.SetEntryPoint, DT.TAB_INF_DEFINES_UNLOAD_IMAGE: InfDefSection.SetUnloadImages, DT.TAB_INF_DEFINES_CONSTRUCTOR: InfDefSection.SetConstructor, DT.TAB_INF_DEFINES_DESTRUCTOR: InfDefSection.SetDestructor, DT.TAB_INF_DEFINES_SHADOW: InfDefSection.SetShadow, DT.TAB_INF_DEFINES_PCI_VENDOR_ID: InfDefSection.SetPciVendorId, DT.TAB_INF_DEFINES_PCI_DEVICE_ID: InfDefSection.SetPciDeviceId, DT.TAB_INF_DEFINES_PCI_CLASS_CODE: InfDefSection.SetPciClassCode, DT.TAB_INF_DEFINES_PCI_REVISION: InfDefSection.SetPciRevision, DT.TAB_INF_DEFINES_PCI_COMPRESS: InfDefSection.SetPciCompress, DT.TAB_INF_DEFINES_CUSTOM_MAKEFILE: InfDefSection.SetCustomMakefile, DT.TAB_INF_DEFINES_SPEC: InfDefSection.SetSpecification, DT.TAB_INF_DEFINES_UEFI_HII_RESOURCE_SECTION: InfDefSection.SetUefiHiiResourceSection, DT.TAB_INF_DEFINES_DPX_SOURCE: InfDefSection.SetDpxSource}

class InfDefMember:

    def __init__(self, Name='', Value=''):
        if False:
            while True:
                i = 10
        self.Comments = InfLineCommentObject()
        self.Name = Name
        self.Value = Value
        self.CurrentLine = CurrentLine()

    def GetName(self):
        if False:
            while True:
                i = 10
        return self.Name

    def SetName(self, Name):
        if False:
            return 10
        self.Name = Name

    def GetValue(self):
        if False:
            return 10
        return self.Value

    def SetValue(self, Value):
        if False:
            i = 10
            return i + 15
        self.Value = Value

class InfDefObject(InfSectionCommonDef):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.Defines = Sdict()
        InfSectionCommonDef.__init__(self)

    def SetDefines(self, DefineContent, Arch=None):
        if False:
            for i in range(10):
                print('nop')
        HasFoundInfVersionFalg = False
        LineInfo = ['', -1, '']
        ArchListString = ' '.join(Arch)
        for InfDefMemberObj in DefineContent:
            ProcessFunc = None
            Name = InfDefMemberObj.GetName()
            Value = InfDefMemberObj.GetValue()
            if Name == DT.TAB_INF_DEFINES_MODULE_UNI_FILE:
                ValidateUNIFilePath(Value)
                Value = os.path.join(os.path.dirname(InfDefMemberObj.CurrentLine.FileName), Value)
                if not os.path.isfile(Value) or not os.path.exists(Value):
                    LineInfo[0] = InfDefMemberObj.CurrentLine.GetFileName()
                    LineInfo[1] = InfDefMemberObj.CurrentLine.GetLineNo()
                    LineInfo[2] = InfDefMemberObj.CurrentLine.GetLineString()
                    ErrorInInf(ST.ERR_INF_PARSER_FILE_NOT_EXIST_OR_NAME_INVALID % Name, LineInfo=LineInfo)
            InfLineCommentObj = InfLineCommentObject()
            InfLineCommentObj.SetHeaderComments(InfDefMemberObj.Comments.GetHeaderComments())
            InfLineCommentObj.SetTailComments(InfDefMemberObj.Comments.GetTailComments())
            if Name == 'COMPONENT_TYPE':
                ErrorInInf(ST.ERR_INF_PARSER_NOT_SUPPORT_EDKI_INF, ErrorCode=ToolError.EDK1_INF_ERROR, RaiseError=True)
            if Name == DT.TAB_INF_DEFINES_INF_VERSION:
                HasFoundInfVersionFalg = True
            if not (Name == '' or Name is None):
                ReName = re.compile('SPEC ', re.DOTALL)
                if ReName.match(Name):
                    SpecValue = Name[Name.find('SPEC') + len('SPEC'):].strip()
                    Name = 'SPEC'
                    Value = SpecValue + ' = ' + Value
                if ArchListString in self.Defines:
                    DefineList = self.Defines[ArchListString]
                    LineInfo[0] = InfDefMemberObj.CurrentLine.GetFileName()
                    LineInfo[1] = InfDefMemberObj.CurrentLine.GetLineNo()
                    LineInfo[2] = InfDefMemberObj.CurrentLine.GetLineString()
                    DefineList.CurrentLine = LineInfo
                    if Name not in gFUNCTION_MAPPING_FOR_DEFINE_SECTION.keys():
                        ErrorInInf(ST.ERR_INF_PARSER_DEFINE_SECTION_KEYWORD_INVALID % Name, LineInfo=LineInfo)
                    else:
                        ProcessFunc = gFUNCTION_MAPPING_FOR_DEFINE_SECTION[Name]
                    if ProcessFunc is not None:
                        ProcessFunc(DefineList, Value, InfLineCommentObj)
                    self.Defines[ArchListString] = DefineList
                else:
                    DefineList = InfDefSection()
                    LineInfo[0] = InfDefMemberObj.CurrentLine.GetFileName()
                    LineInfo[1] = InfDefMemberObj.CurrentLine.GetLineNo()
                    LineInfo[2] = InfDefMemberObj.CurrentLine.GetLineString()
                    DefineList.CurrentLine = LineInfo
                    if Name not in gFUNCTION_MAPPING_FOR_DEFINE_SECTION.keys():
                        ErrorInInf(ST.ERR_INF_PARSER_DEFINE_SECTION_KEYWORD_INVALID % Name, LineInfo=LineInfo)
                    else:
                        ProcessFunc = gFUNCTION_MAPPING_FOR_DEFINE_SECTION[Name]
                    if ProcessFunc is not None:
                        ProcessFunc(DefineList, Value, InfLineCommentObj)
                    self.Defines[ArchListString] = DefineList
        if not HasFoundInfVersionFalg:
            ErrorInInf(ST.ERR_INF_PARSER_NOT_SUPPORT_EDKI_INF, ErrorCode=ToolError.EDK1_INF_ERROR, RaiseError=True)
        return True

    def GetDefines(self):
        if False:
            print('Hello World!')
        return self.Defines