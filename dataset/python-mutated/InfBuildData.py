from __future__ import absolute_import
from Common.DataType import *
from Common.Misc import *
from Common.caching import cached_property, cached_class_function
from types import *
from .MetaFileParser import *
from collections import OrderedDict
from Workspace.BuildClassObject import ModuleBuildClassObject, LibraryClassObject, PcdClassObject
from Common.Expression import ValueExpressionEx, PcdPattern

def _ProtocolValue(CName, PackageList, Inffile=None):
    if False:
        print('Hello World!')
    for P in PackageList:
        ProtocolKeys = list(P.Protocols.keys())
        if Inffile and P._PrivateProtocols:
            if not Inffile.startswith(P.MetaFile.Dir):
                ProtocolKeys = [x for x in P.Protocols if x not in P._PrivateProtocols]
        if CName in ProtocolKeys:
            return P.Protocols[CName]
    return None

def _PpiValue(CName, PackageList, Inffile=None):
    if False:
        return 10
    for P in PackageList:
        PpiKeys = list(P.Ppis.keys())
        if Inffile and P._PrivatePpis:
            if not Inffile.startswith(P.MetaFile.Dir):
                PpiKeys = [x for x in P.Ppis if x not in P._PrivatePpis]
        if CName in PpiKeys:
            return P.Ppis[CName]
    return None

class InfBuildData(ModuleBuildClassObject):
    _PROPERTY_ = {TAB_INF_DEFINES_BASE_NAME: '_BaseName', TAB_INF_DEFINES_FILE_GUID: '_Guid', TAB_INF_DEFINES_MODULE_TYPE: '_ModuleType', TAB_INF_DEFINES_COMPONENT_TYPE: '_ComponentType', TAB_INF_DEFINES_MAKEFILE_NAME: '_MakefileName', TAB_INF_DEFINES_DPX_SOURCE: '_DxsFile', TAB_INF_DEFINES_VERSION_NUMBER: '_Version', TAB_INF_DEFINES_VERSION_STRING: '_Version', TAB_INF_DEFINES_VERSION: '_Version', TAB_INF_DEFINES_PCD_IS_DRIVER: '_PcdIsDriver', TAB_INF_DEFINES_SHADOW: '_Shadow'}
    _NMAKE_FLAG_PATTERN_ = re.compile('(?:EBC_)?([A-Z]+)_(?:STD_|PROJ_|ARCH_)?FLAGS(?:_DLL|_ASL|_EXE)?', re.UNICODE)
    _TOOL_CODE_ = {'C': 'CC', BINARY_FILE_TYPE_LIB: 'SLINK', 'LINK': 'DLINK'}

    def __init__(self, FilePath, RawData, BuildDatabase, Arch=TAB_ARCH_COMMON, Target=None, Toolchain=None):
        if False:
            return 10
        self.MetaFile = FilePath
        self._ModuleDir = FilePath.Dir
        self._RawData = RawData
        self._Bdb = BuildDatabase
        self._Arch = Arch
        self._Target = Target
        self._Toolchain = Toolchain
        self._Platform = TAB_COMMON
        self._TailComments = None
        self._BaseName = None
        self._DxsFile = None
        self._ModuleType = None
        self._ComponentType = None
        self._BuildType = None
        self._Guid = None
        self._Version = None
        self._PcdIsDriver = None
        self._BinaryModule = None
        self._Shadow = None
        self._MakefileName = None
        self._CustomMakefile = None
        self._Specification = None
        self._LibraryClass = None
        self._ModuleEntryPointList = None
        self._ModuleUnloadImageList = None
        self._ConstructorList = None
        self._DestructorList = None
        self._Defs = OrderedDict()
        self._ProtocolComments = None
        self._PpiComments = None
        self._GuidsUsedByPcd = OrderedDict()
        self._GuidComments = None
        self._PcdComments = None
        self._BuildOptions = None
        self._DependencyFileList = None
        self.UpdatePcdTypeDict()
        self.LibInstances = []
        self.ReferenceModules = set()

    def SetReferenceModule(self, Module):
        if False:
            print('Hello World!')
        self.ReferenceModules.add(Module)
        return self

    def __setitem__(self, key, value):
        if False:
            print('Hello World!')
        self.__dict__[self._PROPERTY_[key]] = value

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        return self.__dict__[self._PROPERTY_[key]]

    def __contains__(self, key):
        if False:
            return 10
        return key in self._PROPERTY_

    @cached_property
    def _Macros(self):
        if False:
            return 10
        RetVal = {}
        return RetVal

    @cached_property
    def Arch(self):
        if False:
            print('Hello World!')
        return self._Arch

    @cached_property
    def Platform(self):
        if False:
            return 10
        return self._Platform

    @cached_property
    def HeaderComments(self):
        if False:
            while True:
                i = 10
        return [a[0] for a in self._RawData[MODEL_META_DATA_HEADER_COMMENT]]

    @cached_property
    def TailComments(self):
        if False:
            while True:
                i = 10
        return [a[0] for a in self._RawData[MODEL_META_DATA_TAIL_COMMENT]]

    @cached_class_function
    def _GetHeaderInfo(self):
        if False:
            for i in range(10):
                print('nop')
        RecordList = self._RawData[MODEL_META_DATA_HEADER, self._Arch, self._Platform]
        for Record in RecordList:
            (Name, Value) = (Record[1], ReplaceMacro(Record[2], self._Macros, False))
            if Name in self:
                self[Name] = Value
                self._Defs[Name] = Value
                self._Macros[Name] = Value
            elif Name in ('EFI_SPECIFICATION_VERSION', 'UEFI_SPECIFICATION_VERSION', 'EDK_RELEASE_VERSION', 'PI_SPECIFICATION_VERSION'):
                if Name in ('EFI_SPECIFICATION_VERSION', 'UEFI_SPECIFICATION_VERSION'):
                    Name = 'UEFI_SPECIFICATION_VERSION'
                if self._Specification is None:
                    self._Specification = OrderedDict()
                self._Specification[Name] = GetHexVerValue(Value)
                if self._Specification[Name] is None:
                    EdkLogger.error('build', FORMAT_NOT_SUPPORTED, "'%s' format is not supported for %s" % (Value, Name), File=self.MetaFile, Line=Record[-1])
            elif Name == 'LIBRARY_CLASS':
                if self._LibraryClass is None:
                    self._LibraryClass = []
                ValueList = GetSplitValueList(Value)
                LibraryClass = ValueList[0]
                if len(ValueList) > 1:
                    SupModuleList = GetSplitValueList(ValueList[1], ' ')
                else:
                    SupModuleList = SUP_MODULE_LIST
                self._LibraryClass.append(LibraryClassObject(LibraryClass, SupModuleList))
            elif Name == 'ENTRY_POINT':
                if self._ModuleEntryPointList is None:
                    self._ModuleEntryPointList = []
                self._ModuleEntryPointList.append(Value)
            elif Name == 'UNLOAD_IMAGE':
                if self._ModuleUnloadImageList is None:
                    self._ModuleUnloadImageList = []
                if not Value:
                    continue
                self._ModuleUnloadImageList.append(Value)
            elif Name == 'CONSTRUCTOR':
                if self._ConstructorList is None:
                    self._ConstructorList = []
                if not Value:
                    continue
                self._ConstructorList.append(Value)
            elif Name == 'DESTRUCTOR':
                if self._DestructorList is None:
                    self._DestructorList = []
                if not Value:
                    continue
                self._DestructorList.append(Value)
            elif Name == TAB_INF_DEFINES_CUSTOM_MAKEFILE:
                TokenList = GetSplitValueList(Value)
                if self._CustomMakefile is None:
                    self._CustomMakefile = {}
                if len(TokenList) < 2:
                    self._CustomMakefile[TAB_COMPILER_MSFT] = TokenList[0]
                    self._CustomMakefile['GCC'] = TokenList[0]
                else:
                    if TokenList[0] not in [TAB_COMPILER_MSFT, 'GCC']:
                        EdkLogger.error('build', FORMAT_NOT_SUPPORTED, 'No supported family [%s]' % TokenList[0], File=self.MetaFile, Line=Record[-1])
                    self._CustomMakefile[TokenList[0]] = TokenList[1]
            else:
                self._Defs[Name] = Value
                self._Macros[Name] = Value
        if not self._ModuleType:
            EdkLogger.error('build', ATTRIBUTE_NOT_AVAILABLE, 'MODULE_TYPE is not given', File=self.MetaFile)
        if self._ModuleType not in SUP_MODULE_LIST:
            RecordList = self._RawData[MODEL_META_DATA_HEADER, self._Arch, self._Platform]
            for Record in RecordList:
                Name = Record[1]
                if Name == 'MODULE_TYPE':
                    LineNo = Record[6]
                    break
            EdkLogger.error('build', FORMAT_NOT_SUPPORTED, 'MODULE_TYPE %s is not supported for EDK II, valid values are:\n %s' % (self._ModuleType, ' '.join((l for l in SUP_MODULE_LIST))), File=self.MetaFile, Line=LineNo)
        if self._Specification is None or not 'PI_SPECIFICATION_VERSION' in self._Specification or int(self._Specification['PI_SPECIFICATION_VERSION'], 16) < 65546:
            if self._ModuleType == SUP_MODULE_SMM_CORE:
                EdkLogger.error('build', FORMAT_NOT_SUPPORTED, "SMM_CORE module type can't be used in the module with PI_SPECIFICATION_VERSION less than 0x0001000A", File=self.MetaFile)
        if self._Specification is None or not 'PI_SPECIFICATION_VERSION' in self._Specification or int(self._Specification['PI_SPECIFICATION_VERSION'], 16) < 65586:
            if self._ModuleType == SUP_MODULE_MM_CORE_STANDALONE:
                EdkLogger.error('build', FORMAT_NOT_SUPPORTED, "MM_CORE_STANDALONE module type can't be used in the module with PI_SPECIFICATION_VERSION less than 0x00010032", File=self.MetaFile)
            if self._ModuleType == SUP_MODULE_MM_STANDALONE:
                EdkLogger.error('build', FORMAT_NOT_SUPPORTED, "MM_STANDALONE module type can't be used in the module with PI_SPECIFICATION_VERSION less than 0x00010032", File=self.MetaFile)
        if 'PCI_DEVICE_ID' in self._Defs and 'PCI_VENDOR_ID' in self._Defs and ('PCI_CLASS_CODE' in self._Defs) and ('PCI_REVISION' in self._Defs):
            self._BuildType = 'UEFI_OPTIONROM'
            if 'PCI_COMPRESS' in self._Defs:
                if self._Defs['PCI_COMPRESS'] not in ('TRUE', 'FALSE'):
                    EdkLogger.error('build', FORMAT_INVALID, 'Expected TRUE/FALSE for PCI_COMPRESS: %s' % self.MetaFile)
        elif 'UEFI_HII_RESOURCE_SECTION' in self._Defs and self._Defs['UEFI_HII_RESOURCE_SECTION'] == 'TRUE':
            self._BuildType = 'UEFI_HII'
        else:
            self._BuildType = self._ModuleType.upper()
        if self._DxsFile:
            File = PathClass(NormPath(self._DxsFile), self._ModuleDir, Arch=self._Arch)
            (ErrorCode, ErrorInfo) = File.Validate('.dxs', CaseSensitive=False)
            if ErrorCode != 0:
                EdkLogger.error('build', ErrorCode, ExtraData=ErrorInfo, File=self.MetaFile, Line=LineNo)
            if not self._DependencyFileList:
                self._DependencyFileList = []
            self._DependencyFileList.append(File)

    @cached_property
    def AutoGenVersion(self):
        if False:
            for i in range(10):
                print('nop')
        RetVal = 65536
        RecordList = self._RawData[MODEL_META_DATA_HEADER, self._Arch, self._Platform]
        for Record in RecordList:
            if Record[1] == TAB_INF_DEFINES_INF_VERSION:
                if '.' in Record[2]:
                    ValueList = Record[2].split('.')
                    Major = '%04o' % int(ValueList[0], 0)
                    Minor = '%04o' % int(ValueList[1], 0)
                    RetVal = int('0x' + Major + Minor, 0)
                else:
                    RetVal = int(Record[2], 0)
                break
        return RetVal

    @cached_property
    def BaseName(self):
        if False:
            while True:
                i = 10
        if self._BaseName is None:
            self._GetHeaderInfo()
            if self._BaseName is None:
                EdkLogger.error('build', ATTRIBUTE_NOT_AVAILABLE, 'No BASE_NAME name', File=self.MetaFile)
        return self._BaseName

    @cached_property
    def DxsFile(self):
        if False:
            return 10
        if self._DxsFile is None:
            self._GetHeaderInfo()
            if self._DxsFile is None:
                self._DxsFile = ''
        return self._DxsFile

    @cached_property
    def ModuleType(self):
        if False:
            i = 10
            return i + 15
        if self._ModuleType is None:
            self._GetHeaderInfo()
            if self._ModuleType is None:
                self._ModuleType = SUP_MODULE_BASE
            if self._ModuleType not in SUP_MODULE_LIST:
                self._ModuleType = SUP_MODULE_USER_DEFINED
        return self._ModuleType

    @cached_property
    def ComponentType(self):
        if False:
            while True:
                i = 10
        if self._ComponentType is None:
            self._GetHeaderInfo()
            if self._ComponentType is None:
                self._ComponentType = SUP_MODULE_USER_DEFINED
        return self._ComponentType

    @cached_property
    def BuildType(self):
        if False:
            while True:
                i = 10
        if self._BuildType is None:
            self._GetHeaderInfo()
            if not self._BuildType:
                self._BuildType = SUP_MODULE_BASE
        return self._BuildType

    @cached_property
    def Guid(self):
        if False:
            for i in range(10):
                print('nop')
        if self._Guid is None:
            self._GetHeaderInfo()
            if self._Guid is None:
                self._Guid = '00000000-0000-0000-0000-000000000000'
        return self._Guid

    @cached_property
    def Version(self):
        if False:
            while True:
                i = 10
        if self._Version is None:
            self._GetHeaderInfo()
            if self._Version is None:
                self._Version = '0.0'
        return self._Version

    @cached_property
    def PcdIsDriver(self):
        if False:
            return 10
        if self._PcdIsDriver is None:
            self._GetHeaderInfo()
            if self._PcdIsDriver is None:
                self._PcdIsDriver = ''
        return self._PcdIsDriver

    @cached_property
    def Shadow(self):
        if False:
            for i in range(10):
                print('nop')
        if self._Shadow is None:
            self._GetHeaderInfo()
            if self._Shadow and self._Shadow.upper() == 'TRUE':
                self._Shadow = True
            else:
                self._Shadow = False
        return self._Shadow

    @cached_property
    def CustomMakefile(self):
        if False:
            for i in range(10):
                print('nop')
        if self._CustomMakefile is None:
            self._GetHeaderInfo()
            if self._CustomMakefile is None:
                self._CustomMakefile = {}
        return self._CustomMakefile

    @cached_property
    def Specification(self):
        if False:
            i = 10
            return i + 15
        if self._Specification is None:
            self._GetHeaderInfo()
            if self._Specification is None:
                self._Specification = {}
        return self._Specification

    @cached_property
    def LibraryClass(self):
        if False:
            i = 10
            return i + 15
        if self._LibraryClass is None:
            self._GetHeaderInfo()
            if self._LibraryClass is None:
                self._LibraryClass = []
        return self._LibraryClass

    @cached_property
    def ModuleEntryPointList(self):
        if False:
            i = 10
            return i + 15
        if self._ModuleEntryPointList is None:
            self._GetHeaderInfo()
            if self._ModuleEntryPointList is None:
                self._ModuleEntryPointList = []
        return self._ModuleEntryPointList

    @cached_property
    def ModuleUnloadImageList(self):
        if False:
            print('Hello World!')
        if self._ModuleUnloadImageList is None:
            self._GetHeaderInfo()
            if self._ModuleUnloadImageList is None:
                self._ModuleUnloadImageList = []
        return self._ModuleUnloadImageList

    @cached_property
    def ConstructorList(self):
        if False:
            i = 10
            return i + 15
        if self._ConstructorList is None:
            self._GetHeaderInfo()
            if self._ConstructorList is None:
                self._ConstructorList = []
        return self._ConstructorList

    @cached_property
    def DestructorList(self):
        if False:
            print('Hello World!')
        if self._DestructorList is None:
            self._GetHeaderInfo()
            if self._DestructorList is None:
                self._DestructorList = []
        return self._DestructorList

    @cached_property
    def Defines(self):
        if False:
            return 10
        self._GetHeaderInfo()
        return self._Defs

    @cached_class_function
    def _GetBinaries(self):
        if False:
            print('Hello World!')
        RetVal = []
        RecordList = self._RawData[MODEL_EFI_BINARY_FILE, self._Arch, self._Platform]
        Macros = self._Macros
        Macros['PROCESSOR'] = self._Arch
        for Record in RecordList:
            FileType = Record[0]
            LineNo = Record[-1]
            Target = TAB_COMMON
            FeatureFlag = []
            if Record[2]:
                TokenList = GetSplitValueList(Record[2], TAB_VALUE_SPLIT)
                if TokenList:
                    Target = TokenList[0]
                if len(TokenList) > 1:
                    FeatureFlag = Record[1:]
            File = PathClass(NormPath(Record[1], Macros), self._ModuleDir, '', FileType, True, self._Arch, '', Target)
            (ErrorCode, ErrorInfo) = File.Validate()
            if ErrorCode != 0:
                EdkLogger.error('build', ErrorCode, ExtraData=ErrorInfo, File=self.MetaFile, Line=LineNo)
            RetVal.append(File)
        return RetVal

    @cached_property
    def Binaries(self):
        if False:
            print('Hello World!')
        RetVal = self._GetBinaries()
        if GlobalData.gIgnoreSource and (not RetVal):
            ErrorInfo = 'The INF file does not contain any RetVal to use in creating the image\n'
            EdkLogger.error('build', RESOURCE_NOT_AVAILABLE, ExtraData=ErrorInfo, File=self.MetaFile)
        return RetVal

    @cached_property
    def Sources(self):
        if False:
            while True:
                i = 10
        self._GetHeaderInfo()
        if GlobalData.gIgnoreSource:
            return []
        RetVal = []
        RecordList = self._RawData[MODEL_EFI_SOURCE_FILE, self._Arch, self._Platform]
        Macros = self._Macros
        for Record in RecordList:
            LineNo = Record[-1]
            ToolChainFamily = Record[1]
            OptionsList = ['', '', '']
            TokenList = GetSplitValueList(Record[2], TAB_VALUE_SPLIT)
            for Index in range(len(TokenList)):
                OptionsList[Index] = TokenList[Index]
            if OptionsList[2]:
                FeaturePcdExpression = self.CheckFeatureFlagPcd(OptionsList[2])
                if not FeaturePcdExpression:
                    continue
            File = PathClass(NormPath(Record[0], Macros), self._ModuleDir, '', '', False, self._Arch, ToolChainFamily, '', OptionsList[0], OptionsList[1])
            (ErrorCode, ErrorInfo) = File.Validate()
            if ErrorCode != 0:
                EdkLogger.error('build', ErrorCode, ExtraData=ErrorInfo, File=self.MetaFile, Line=LineNo)
            RetVal.append(File)
        if self._DependencyFileList:
            RetVal.extend(self._DependencyFileList)
        return RetVal

    @cached_property
    def LibraryClasses(self):
        if False:
            return 10
        RetVal = OrderedDict()
        RecordList = self._RawData[MODEL_EFI_LIBRARY_CLASS, self._Arch, self._Platform]
        for Record in RecordList:
            Lib = Record[0]
            Instance = Record[1]
            if Instance:
                Instance = NormPath(Instance, self._Macros)
                RetVal[Lib] = Instance
            else:
                RetVal[Lib] = None
        return RetVal

    @cached_property
    def Libraries(self):
        if False:
            return 10
        RetVal = []
        RecordList = self._RawData[MODEL_EFI_LIBRARY_INSTANCE, self._Arch, self._Platform]
        for Record in RecordList:
            LibraryName = ReplaceMacro(Record[0], self._Macros, False)
            LibraryName = os.path.splitext(LibraryName)[0]
            if LibraryName not in RetVal:
                RetVal.append(LibraryName)
        return RetVal

    @cached_property
    def ProtocolComments(self):
        if False:
            for i in range(10):
                print('nop')
        self.Protocols
        return self._ProtocolComments

    @cached_property
    def Protocols(self):
        if False:
            for i in range(10):
                print('nop')
        RetVal = OrderedDict()
        self._ProtocolComments = OrderedDict()
        RecordList = self._RawData[MODEL_EFI_PROTOCOL, self._Arch, self._Platform]
        for Record in RecordList:
            CName = Record[0]
            Value = _ProtocolValue(CName, self.Packages, self.MetaFile.Path)
            if Value is None:
                PackageList = '\n\t'.join((str(P) for P in self.Packages))
                EdkLogger.error('build', RESOURCE_NOT_AVAILABLE, 'Value of Protocol [%s] is not found under [Protocols] section in' % CName, ExtraData=PackageList, File=self.MetaFile, Line=Record[-1])
            RetVal[CName] = Value
            CommentRecords = self._RawData[MODEL_META_DATA_COMMENT, self._Arch, self._Platform, Record[5]]
            self._ProtocolComments[CName] = [a[0] for a in CommentRecords]
        return RetVal

    @cached_property
    def PpiComments(self):
        if False:
            for i in range(10):
                print('nop')
        self.Ppis
        return self._PpiComments

    @cached_property
    def Ppis(self):
        if False:
            return 10
        RetVal = OrderedDict()
        self._PpiComments = OrderedDict()
        RecordList = self._RawData[MODEL_EFI_PPI, self._Arch, self._Platform]
        for Record in RecordList:
            CName = Record[0]
            Value = _PpiValue(CName, self.Packages, self.MetaFile.Path)
            if Value is None:
                PackageList = '\n\t'.join((str(P) for P in self.Packages))
                EdkLogger.error('build', RESOURCE_NOT_AVAILABLE, 'Value of PPI [%s] is not found under [Ppis] section in ' % CName, ExtraData=PackageList, File=self.MetaFile, Line=Record[-1])
            RetVal[CName] = Value
            CommentRecords = self._RawData[MODEL_META_DATA_COMMENT, self._Arch, self._Platform, Record[5]]
            self._PpiComments[CName] = [a[0] for a in CommentRecords]
        return RetVal

    @cached_property
    def GuidComments(self):
        if False:
            return 10
        self.Guids
        return self._GuidComments

    @cached_property
    def Guids(self):
        if False:
            for i in range(10):
                print('nop')
        RetVal = OrderedDict()
        self._GuidComments = OrderedDict()
        RecordList = self._RawData[MODEL_EFI_GUID, self._Arch, self._Platform]
        for Record in RecordList:
            CName = Record[0]
            Value = GuidValue(CName, self.Packages, self.MetaFile.Path)
            if Value is None:
                PackageList = '\n\t'.join((str(P) for P in self.Packages))
                EdkLogger.error('build', RESOURCE_NOT_AVAILABLE, 'Value of Guid [%s] is not found under [Guids] section in' % CName, ExtraData=PackageList, File=self.MetaFile, Line=Record[-1])
            RetVal[CName] = Value
            CommentRecords = self._RawData[MODEL_META_DATA_COMMENT, self._Arch, self._Platform, Record[5]]
            self._GuidComments[CName] = [a[0] for a in CommentRecords]
        for Type in [MODEL_PCD_FIXED_AT_BUILD, MODEL_PCD_PATCHABLE_IN_MODULE, MODEL_PCD_FEATURE_FLAG, MODEL_PCD_DYNAMIC, MODEL_PCD_DYNAMIC_EX]:
            RecordList = self._RawData[Type, self._Arch, self._Platform]
            for (TokenSpaceGuid, _, _, _, _, _, LineNo) in RecordList:
                if TokenSpaceGuid not in RetVal:
                    Value = GuidValue(TokenSpaceGuid, self.Packages, self.MetaFile.Path)
                    if Value is None:
                        PackageList = '\n\t'.join((str(P) for P in self.Packages))
                        EdkLogger.error('build', RESOURCE_NOT_AVAILABLE, 'Value of Guid [%s] is not found under [Guids] section in' % TokenSpaceGuid, ExtraData=PackageList, File=self.MetaFile, Line=LineNo)
                    RetVal[TokenSpaceGuid] = Value
                    self._GuidsUsedByPcd[TokenSpaceGuid] = Value
        return RetVal

    @cached_property
    def Includes(self):
        if False:
            return 10
        RetVal = []
        Macros = self._Macros
        Macros['PROCESSOR'] = GlobalData.gEdkGlobal.get('PROCESSOR', self._Arch)
        RecordList = self._RawData[MODEL_EFI_INCLUDE, self._Arch, self._Platform]
        for Record in RecordList:
            File = NormPath(Record[0], Macros)
            if File[0] == '.':
                File = os.path.join(self._ModuleDir, File)
            else:
                File = mws.join(GlobalData.gWorkspace, File)
            File = RealPath(os.path.normpath(File))
            if File:
                RetVal.append(File)
        return RetVal

    @cached_property
    def Packages(self):
        if False:
            while True:
                i = 10
        RetVal = []
        RecordList = self._RawData[MODEL_META_DATA_PACKAGE, self._Arch, self._Platform]
        Macros = self._Macros
        for Record in RecordList:
            File = PathClass(NormPath(Record[0], Macros), GlobalData.gWorkspace, Arch=self._Arch)
            (ErrorCode, ErrorInfo) = File.Validate('.dec')
            if ErrorCode != 0:
                LineNo = Record[-1]
                EdkLogger.error('build', ErrorCode, ExtraData=ErrorInfo, File=self.MetaFile, Line=LineNo)
            RetVal.append(self._Bdb[File, self._Arch, self._Target, self._Toolchain])
        return RetVal

    @cached_property
    def PcdComments(self):
        if False:
            return 10
        self.Pcds
        return self._PcdComments

    @cached_property
    def Pcds(self):
        if False:
            i = 10
            return i + 15
        self._PcdComments = OrderedDict()
        RetVal = OrderedDict()
        RetVal.update(self._GetPcd(MODEL_PCD_FIXED_AT_BUILD))
        RetVal.update(self._GetPcd(MODEL_PCD_PATCHABLE_IN_MODULE))
        RetVal.update(self._GetPcd(MODEL_PCD_FEATURE_FLAG))
        RetVal.update(self._GetPcd(MODEL_PCD_DYNAMIC))
        RetVal.update(self._GetPcd(MODEL_PCD_DYNAMIC_EX))
        return RetVal

    @cached_property
    def ModulePcdList(self):
        if False:
            while True:
                i = 10
        RetVal = self.Pcds
        return RetVal

    @cached_property
    def LibraryPcdList(self):
        if False:
            while True:
                i = 10
        if bool(self.LibraryClass):
            return []
        RetVal = {}
        Pcds = set()
        for Library in self.LibInstances:
            PcdsInLibrary = OrderedDict()
            for Key in Library.Pcds:
                if Key in self.Pcds or Key in Pcds:
                    continue
                Pcds.add(Key)
                PcdsInLibrary[Key] = copy.copy(Library.Pcds[Key])
            RetVal[Library] = PcdsInLibrary
        return RetVal

    @cached_property
    def PcdsName(self):
        if False:
            while True:
                i = 10
        PcdsName = set()
        for Type in (MODEL_PCD_FIXED_AT_BUILD, MODEL_PCD_PATCHABLE_IN_MODULE, MODEL_PCD_FEATURE_FLAG, MODEL_PCD_DYNAMIC, MODEL_PCD_DYNAMIC_EX):
            RecordList = self._RawData[Type, self._Arch, self._Platform]
            for (TokenSpaceGuid, PcdCName, _, _, _, _, _) in RecordList:
                PcdsName.add((PcdCName, TokenSpaceGuid))
        return PcdsName

    @cached_property
    def BuildOptions(self):
        if False:
            return 10
        if self._BuildOptions is None:
            self._BuildOptions = OrderedDict()
            RecordList = self._RawData[MODEL_META_DATA_BUILD_OPTION, self._Arch, self._Platform]
            for Record in RecordList:
                ToolChainFamily = Record[0]
                ToolChain = Record[1]
                Option = Record[2]
                if (ToolChainFamily, ToolChain) not in self._BuildOptions or Option.startswith('='):
                    self._BuildOptions[ToolChainFamily, ToolChain] = Option
                else:
                    OptionString = self._BuildOptions[ToolChainFamily, ToolChain]
                    self._BuildOptions[ToolChainFamily, ToolChain] = OptionString + ' ' + Option
        return self._BuildOptions

    @cached_property
    def Depex(self):
        if False:
            return 10
        RetVal = tdict(False, 2)
        if not self.Sources and self.Binaries:
            return RetVal
        RecordList = self._RawData[MODEL_EFI_DEPEX, self._Arch]
        if len(self.LibraryClass) == 0 and len(RecordList) == 0:
            if self.ModuleType == SUP_MODULE_DXE_DRIVER or self.ModuleType == SUP_MODULE_PEIM or self.ModuleType == SUP_MODULE_DXE_SMM_DRIVER or (self.ModuleType == SUP_MODULE_DXE_SAL_DRIVER) or (self.ModuleType == SUP_MODULE_DXE_RUNTIME_DRIVER):
                EdkLogger.error('build', RESOURCE_NOT_AVAILABLE, 'No [Depex] section or no valid expression in [Depex] section for [%s] module' % self.ModuleType, File=self.MetaFile)
        if len(RecordList) != 0 and (self.ModuleType == SUP_MODULE_USER_DEFINED or self.ModuleType == SUP_MODULE_HOST_APPLICATION):
            for Record in RecordList:
                if Record[4] not in [SUP_MODULE_PEIM, SUP_MODULE_DXE_DRIVER, SUP_MODULE_DXE_SMM_DRIVER]:
                    EdkLogger.error('build', FORMAT_INVALID, "'%s' module must specify the type of [Depex] section" % self.ModuleType, File=self.MetaFile)
        TemporaryDictionary = OrderedDict()
        for Record in RecordList:
            DepexStr = ReplaceMacro(Record[0], self._Macros, False)
            Arch = Record[3]
            ModuleType = Record[4]
            TokenList = DepexStr.split()
            if (Arch, ModuleType) not in TemporaryDictionary:
                TemporaryDictionary[Arch, ModuleType] = []
            DepexList = TemporaryDictionary[Arch, ModuleType]
            for Token in TokenList:
                if Token in DEPEX_SUPPORTED_OPCODE_SET:
                    DepexList.append(Token)
                elif Token.endswith('.inf'):
                    ModuleFile = os.path.normpath(Token)
                    Module = self.BuildDatabase[ModuleFile]
                    if Module is None:
                        EdkLogger.error('build', RESOURCE_NOT_AVAILABLE, 'Module is not found in active platform', ExtraData=Token, File=self.MetaFile, Line=Record[-1])
                    DepexList.append(Module.Guid)
                else:
                    if '.' in Token:
                        if tuple(Token.split('.')[::-1]) not in self.Pcds:
                            EdkLogger.error('build', RESOURCE_NOT_AVAILABLE, 'PCD [{}] used in [Depex] section should be listed in module PCD section'.format(Token), File=self.MetaFile, Line=Record[-1])
                        elif self.Pcds[tuple(Token.split('.')[::-1])].DatumType != TAB_VOID:
                            EdkLogger.error('build', FORMAT_INVALID, 'PCD [{}] used in [Depex] section should be VOID* datum type'.format(Token), File=self.MetaFile, Line=Record[-1])
                        Value = Token
                    else:
                        Value = _ProtocolValue(Token, self.Packages, self.MetaFile.Path)
                        if Value is None:
                            Value = _PpiValue(Token, self.Packages, self.MetaFile.Path)
                            if Value is None:
                                Value = GuidValue(Token, self.Packages, self.MetaFile.Path)
                    if Value is None:
                        PackageList = '\n\t'.join((str(P) for P in self.Packages))
                        EdkLogger.error('build', RESOURCE_NOT_AVAILABLE, 'Value of [%s] is not found in' % Token, ExtraData=PackageList, File=self.MetaFile, Line=Record[-1])
                    DepexList.append(Value)
        for (Arch, ModuleType) in TemporaryDictionary:
            RetVal[Arch, ModuleType] = TemporaryDictionary[Arch, ModuleType]
        return RetVal

    @cached_property
    def DepexExpression(self):
        if False:
            i = 10
            return i + 15
        RetVal = tdict(False, 2)
        RecordList = self._RawData[MODEL_EFI_DEPEX, self._Arch]
        TemporaryDictionary = OrderedDict()
        for Record in RecordList:
            DepexStr = ReplaceMacro(Record[0], self._Macros, False)
            Arch = Record[3]
            ModuleType = Record[4]
            TokenList = DepexStr.split()
            if (Arch, ModuleType) not in TemporaryDictionary:
                TemporaryDictionary[Arch, ModuleType] = ''
            for Token in TokenList:
                TemporaryDictionary[Arch, ModuleType] = TemporaryDictionary[Arch, ModuleType] + Token.strip() + ' '
        for (Arch, ModuleType) in TemporaryDictionary:
            RetVal[Arch, ModuleType] = TemporaryDictionary[Arch, ModuleType]
        return RetVal

    def LocalPkg(self):
        if False:
            for i in range(10):
                print('nop')
        module_path = self.MetaFile.File
        subdir = os.path.split(module_path)[0]
        TopDir = ''
        while subdir:
            (subdir, TopDir) = os.path.split(subdir)
        for file_name in os.listdir(os.path.join(self.MetaFile.Root, TopDir)):
            if file_name.upper().endswith('DEC'):
                pkg = os.path.join(TopDir, file_name)
        return pkg

    @cached_class_function
    def GetGuidsUsedByPcd(self):
        if False:
            while True:
                i = 10
        self.Guid
        return self._GuidsUsedByPcd

    def _GetPcd(self, Type):
        if False:
            i = 10
            return i + 15
        Pcds = OrderedDict()
        PcdDict = tdict(True, 4)
        PcdList = []
        RecordList = self._RawData[Type, self._Arch, self._Platform]
        for (TokenSpaceGuid, PcdCName, Setting, Arch, Platform, Id, LineNo) in RecordList:
            PcdDict[Arch, Platform, PcdCName, TokenSpaceGuid] = (Setting, LineNo)
            PcdList.append((PcdCName, TokenSpaceGuid))
            CommentRecords = self._RawData[MODEL_META_DATA_COMMENT, self._Arch, self._Platform, Id]
            Comments = []
            for CmtRec in CommentRecords:
                Comments.append(CmtRec[0])
            self._PcdComments[TokenSpaceGuid, PcdCName] = Comments
        _GuidDict = self.Guids.copy()
        for (PcdCName, TokenSpaceGuid) in PcdList:
            PcdRealName = PcdCName
            (Setting, LineNo) = PcdDict[self._Arch, self.Platform, PcdCName, TokenSpaceGuid]
            if Setting is None:
                continue
            ValueList = AnalyzePcdData(Setting)
            DefaultValue = ValueList[0]
            Pcd = PcdClassObject(PcdCName, TokenSpaceGuid, '', '', DefaultValue, '', '', {}, False, self.Guids[TokenSpaceGuid])
            if Type == MODEL_PCD_PATCHABLE_IN_MODULE and ValueList[1]:
                Pcd.Offset = ValueList[1]
            if (PcdRealName, TokenSpaceGuid) in GlobalData.MixedPcd:
                for Package in self.Packages:
                    for key in Package.Pcds:
                        if (Package.Pcds[key].TokenCName, Package.Pcds[key].TokenSpaceGuidCName) == (PcdRealName, TokenSpaceGuid):
                            for item in GlobalData.MixedPcd[PcdRealName, TokenSpaceGuid]:
                                Pcd_Type = item[0].split('_')[-1]
                                if Pcd_Type == Package.Pcds[key].Type:
                                    Value = Package.Pcds[key]
                                    Value.TokenCName = Package.Pcds[key].TokenCName + '_' + Pcd_Type
                                    if len(key) == 2:
                                        newkey = (Value.TokenCName, key[1])
                                    elif len(key) == 3:
                                        newkey = (Value.TokenCName, key[1], key[2])
                                    del Package.Pcds[key]
                                    Package.Pcds[newkey] = Value
                                    break
                                else:
                                    pass
                        else:
                            pass
            for Package in self.Packages:
                _GuidDict.update(Package.Guids)
                PcdType = self._PCD_TYPE_STRING_[Type]
                if Type == MODEL_PCD_DYNAMIC:
                    Pcd.Pending = True
                    for T in PCD_TYPE_LIST:
                        if (PcdRealName, TokenSpaceGuid) in GlobalData.MixedPcd:
                            for item in GlobalData.MixedPcd[PcdRealName, TokenSpaceGuid]:
                                if str(item[0]).endswith(T) and (item[0], item[1], T) in Package.Pcds:
                                    PcdType = T
                                    PcdCName = item[0]
                                    break
                                else:
                                    pass
                            break
                        elif (PcdRealName, TokenSpaceGuid, T) in Package.Pcds:
                            PcdType = T
                            break
                else:
                    Pcd.Pending = False
                    if (PcdRealName, TokenSpaceGuid) in GlobalData.MixedPcd:
                        for item in GlobalData.MixedPcd[PcdRealName, TokenSpaceGuid]:
                            Pcd_Type = item[0].split('_')[-1]
                            if Pcd_Type == PcdType:
                                PcdCName = item[0]
                                break
                            else:
                                pass
                    else:
                        pass
                if (PcdCName, TokenSpaceGuid, PcdType) in Package.Pcds:
                    PcdInPackage = Package.Pcds[PcdCName, TokenSpaceGuid, PcdType]
                    Pcd.Type = PcdType
                    Pcd.TokenValue = PcdInPackage.TokenValue
                    if Pcd.TokenValue is None or Pcd.TokenValue == '':
                        EdkLogger.error('build', FORMAT_INVALID, 'No TokenValue for PCD [%s.%s] in [%s]!' % (TokenSpaceGuid, PcdRealName, str(Package)), File=self.MetaFile, Line=LineNo, ExtraData=None)
                    ReIsValidPcdTokenValue = re.compile('^[0][x|X][0]*[0-9a-fA-F]{1,8}$', re.DOTALL)
                    if Pcd.TokenValue.startswith('0x') or Pcd.TokenValue.startswith('0X'):
                        if ReIsValidPcdTokenValue.match(Pcd.TokenValue) is None:
                            EdkLogger.error('build', FORMAT_INVALID, 'The format of TokenValue [%s] of PCD [%s.%s] in [%s] is invalid:' % (Pcd.TokenValue, TokenSpaceGuid, PcdRealName, str(Package)), File=self.MetaFile, Line=LineNo, ExtraData=None)
                    else:
                        try:
                            TokenValueInt = int(Pcd.TokenValue, 10)
                            if TokenValueInt < 0 or TokenValueInt > 4294967295:
                                EdkLogger.error('build', FORMAT_INVALID, 'The format of TokenValue [%s] of PCD [%s.%s] in [%s] is invalid, as a decimal it should between: 0 - 4294967295!' % (Pcd.TokenValue, TokenSpaceGuid, PcdRealName, str(Package)), File=self.MetaFile, Line=LineNo, ExtraData=None)
                        except:
                            EdkLogger.error('build', FORMAT_INVALID, 'The format of TokenValue [%s] of PCD [%s.%s] in [%s] is invalid, it should be hexadecimal or decimal!' % (Pcd.TokenValue, TokenSpaceGuid, PcdRealName, str(Package)), File=self.MetaFile, Line=LineNo, ExtraData=None)
                    Pcd.DatumType = PcdInPackage.DatumType
                    Pcd.MaxDatumSize = PcdInPackage.MaxDatumSize
                    Pcd.InfDefaultValue = Pcd.DefaultValue
                    if not Pcd.DefaultValue:
                        Pcd.DefaultValue = PcdInPackage.DefaultValue
                    else:
                        try:
                            Pcd.DefaultValue = ValueExpressionEx(Pcd.DefaultValue, Pcd.DatumType, _GuidDict)(True)
                        except BadExpression as Value:
                            EdkLogger.error('Parser', FORMAT_INVALID, 'PCD [%s.%s] Value "%s", %s' % (TokenSpaceGuid, PcdRealName, Pcd.DefaultValue, Value), File=self.MetaFile, Line=LineNo)
                    break
            else:
                EdkLogger.error('build', FORMAT_INVALID, 'PCD [%s.%s] in [%s] is not found in dependent packages:' % (TokenSpaceGuid, PcdRealName, self.MetaFile), File=self.MetaFile, Line=LineNo, ExtraData='\t%s' % '\n\t'.join((str(P) for P in self.Packages)))
            Pcds[PcdCName, TokenSpaceGuid] = Pcd
        return Pcds

    @property
    def IsBinaryModule(self):
        if False:
            i = 10
            return i + 15
        if self.Binaries and (not self.Sources) or GlobalData.gIgnoreSource:
            return True
        return False

    def CheckFeatureFlagPcd(self, Instance):
        if False:
            for i in range(10):
                print('nop')
        Pcds = GlobalData.gPlatformFinalPcds.copy()
        if PcdPattern.search(Instance):
            PcdTuple = tuple(Instance.split('.')[::-1])
            if PcdTuple in self.Pcds:
                if not (self.Pcds[PcdTuple].Type == 'FeatureFlag' or self.Pcds[PcdTuple].Type == 'FixedAtBuild'):
                    EdkLogger.error('build', FORMAT_INVALID, '\nFeatureFlagPcd must be defined in a [PcdsFeatureFlag] or [PcdsFixedAtBuild] section of Dsc or Dec file', File=str(self), ExtraData=Instance)
                if not Instance in Pcds:
                    Pcds[Instance] = self.Pcds[PcdTuple].DefaultValue
            else:
                EdkLogger.error('build', FORMAT_INVALID, '\nFeatureFlagPcd must be defined in [FeaturePcd] or [FixedPcd] of Inf file', File=str(self), ExtraData=Instance)
            if Instance in Pcds:
                if Pcds[Instance] == '0':
                    return False
                elif Pcds[Instance] == '1':
                    return True
            try:
                Value = ValueExpression(Instance, Pcds)()
                if Value == True:
                    return True
                return False
            except:
                EdkLogger.warn('build', FORMAT_INVALID, 'The FeatureFlagExpression cannot be evaluated', File=str(self), ExtraData=Instance)
                return False
        else:
            for (Name, Guid) in self.Pcds:
                if self.Pcds[Name, Guid].Type == 'FeatureFlag' or self.Pcds[Name, Guid].Type == 'FixedAtBuild':
                    PcdFullName = '%s.%s' % (Guid, Name)
                    if not PcdFullName in Pcds:
                        Pcds[PcdFullName] = self.Pcds[Name, Guid].DefaultValue
            try:
                Value = ValueExpression(Instance, Pcds)()
                if Value == True:
                    return True
                return False
            except:
                EdkLogger.warn('build', FORMAT_INVALID, 'The FeatureFlagExpression cannot be evaluated', File=str(self), ExtraData=Instance)
                return False

def ExtendCopyDictionaryLists(CopyToDict, CopyFromDict):
    if False:
        i = 10
        return i + 15
    for Key in CopyFromDict:
        CopyToDict[Key].extend(CopyFromDict[Key])