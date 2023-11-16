from __future__ import absolute_import
from AutoGen.AutoGen import AutoGen
from Common.LongFilePathSupport import LongFilePath, CopyLongFilePath
from Common.BuildToolError import *
from Common.DataType import *
from Common.Misc import *
from Common.StringUtils import NormPath, GetSplitList
from collections import defaultdict
from Workspace.WorkspaceCommon import OrderedListDict
import os.path as path
import copy
import hashlib
from . import InfSectionParser
from . import GenC
from . import GenMake
from . import GenDepex
from io import BytesIO
from GenPatchPcdTable.GenPatchPcdTable import parsePcdInfoFromMapFile
from Workspace.MetaFileCommentParser import UsageList
from .GenPcdDb import CreatePcdDatabaseCode
from Common.caching import cached_class_function
from AutoGen.ModuleAutoGenHelper import PlatformInfo, WorkSpaceInfo
import json
import tempfile
gMakeTypeMap = {TAB_COMPILER_MSFT: 'nmake', 'GCC': 'gmake'}
gBuildOptIncludePatternMsft = re.compile('(?:.*?)/I[ \\t]*([^ ]*)', re.MULTILINE | re.DOTALL)
gBuildOptIncludePatternOther = re.compile('(?:.*?)-I[ \\t]*([^ ]*)', re.MULTILINE | re.DOTALL)
gAutoGenCodeFileName = 'AutoGen.c'
gAutoGenHeaderFileName = 'AutoGen.h'
gAutoGenStringFileName = '%(module_name)sStrDefs.h'
gAutoGenStringFormFileName = '%(module_name)sStrDefs.hpk'
gAutoGenDepexFileName = '%(module_name)s.depex'
gAutoGenImageDefFileName = '%(module_name)sImgDefs.h'
gAutoGenIdfFileName = '%(module_name)sIdf.hpk'
gInfSpecVersion = '0x00010017'
gEfiVarStoreNamePattern = re.compile('\\s*name\\s*=\\s*(\\w+)')
gEfiVarStoreGuidPattern = re.compile('\\s*guid\\s*=\\s*({.*?{.*?}\\s*})')
gAsBuiltInfHeaderString = TemplateString('${header_comments}\n\n# DO NOT EDIT\n# FILE auto-generated\n\n[Defines]\n  INF_VERSION                = ${module_inf_version}\n  BASE_NAME                  = ${module_name}\n  FILE_GUID                  = ${module_guid}\n  MODULE_TYPE                = ${module_module_type}${BEGIN}\n  VERSION_STRING             = ${module_version_string}${END}${BEGIN}\n  PCD_IS_DRIVER              = ${pcd_is_driver_string}${END}${BEGIN}\n  UEFI_SPECIFICATION_VERSION = ${module_uefi_specification_version}${END}${BEGIN}\n  PI_SPECIFICATION_VERSION   = ${module_pi_specification_version}${END}${BEGIN}\n  ENTRY_POINT                = ${module_entry_point}${END}${BEGIN}\n  UNLOAD_IMAGE               = ${module_unload_image}${END}${BEGIN}\n  CONSTRUCTOR                = ${module_constructor}${END}${BEGIN}\n  DESTRUCTOR                 = ${module_destructor}${END}${BEGIN}\n  SHADOW                     = ${module_shadow}${END}${BEGIN}\n  PCI_VENDOR_ID              = ${module_pci_vendor_id}${END}${BEGIN}\n  PCI_DEVICE_ID              = ${module_pci_device_id}${END}${BEGIN}\n  PCI_CLASS_CODE             = ${module_pci_class_code}${END}${BEGIN}\n  PCI_REVISION               = ${module_pci_revision}${END}${BEGIN}\n  BUILD_NUMBER               = ${module_build_number}${END}${BEGIN}\n  SPEC                       = ${module_spec}${END}${BEGIN}\n  UEFI_HII_RESOURCE_SECTION  = ${module_uefi_hii_resource_section}${END}${BEGIN}\n  MODULE_UNI_FILE            = ${module_uni_file}${END}\n\n[Packages.${module_arch}]${BEGIN}\n  ${package_item}${END}\n\n[Binaries.${module_arch}]${BEGIN}\n  ${binary_item}${END}\n\n[PatchPcd.${module_arch}]${BEGIN}\n  ${patchablepcd_item}\n${END}\n\n[Protocols.${module_arch}]${BEGIN}\n  ${protocol_item}\n${END}\n\n[Ppis.${module_arch}]${BEGIN}\n  ${ppi_item}\n${END}\n\n[Guids.${module_arch}]${BEGIN}\n  ${guid_item}\n${END}\n\n[PcdEx.${module_arch}]${BEGIN}\n  ${pcd_item}\n${END}\n\n[LibraryClasses.${module_arch}]\n## @LIB_INSTANCES${BEGIN}\n#  ${libraryclasses_item}${END}\n\n${depexsection_item}\n\n${userextension_tianocore_item}\n\n${tail_comments}\n\n[BuildOptions.${module_arch}]\n## @AsBuilt${BEGIN}\n##   ${flags_item}${END}\n')

def ExtendCopyDictionaryLists(CopyToDict, CopyFromDict):
    if False:
        print('Hello World!')
    for Key in CopyFromDict:
        CopyToDict[Key].extend(CopyFromDict[Key])

def _MakeDir(PathList):
    if False:
        for i in range(10):
            print('nop')
    RetVal = path.join(*PathList)
    CreateDirectory(RetVal)
    return RetVal

def _ConvertStringToByteArray(Value):
    if False:
        while True:
            i = 10
    Value = Value.strip()
    if not Value:
        return None
    if Value[0] == '{':
        if not Value.endswith('}'):
            return None
        Value = Value.replace(' ', '').replace('{', '').replace('}', '')
        ValFields = Value.split(',')
        try:
            for Index in range(len(ValFields)):
                ValFields[Index] = str(int(ValFields[Index], 0))
        except ValueError:
            return None
        Value = '{' + ','.join(ValFields) + '}'
        return Value
    Unicode = False
    if Value.startswith('L"'):
        if not Value.endswith('"'):
            return None
        Value = Value[1:]
        Unicode = True
    elif not Value.startswith('"') or not Value.endswith('"'):
        return None
    Value = eval(Value)
    NewValue = '{'
    for Index in range(0, len(Value)):
        if Unicode:
            NewValue = NewValue + str(ord(Value[Index]) % 65536) + ','
        else:
            NewValue = NewValue + str(ord(Value[Index]) % 256) + ','
    Value = NewValue + '0}'
    return Value

class ModuleAutoGen(AutoGen):

    def __init__(self, Workspace, MetaFile, Target, Toolchain, Arch, *args, **kwargs):
        if False:
            return 10
        if not hasattr(self, '_Init'):
            self._InitWorker(Workspace, MetaFile, Target, Toolchain, Arch, *args)
            self._Init = True
    TimeDict = {}

    def __new__(cls, Workspace, MetaFile, Target, Toolchain, Arch, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if not PlatformInfo(Workspace, args[0], Target, Toolchain, Arch, args[-1]).ValidModule(MetaFile):
            EdkLogger.verbose('Module [%s] for [%s] is not employed by active platform\n' % (MetaFile, Arch))
            return None
        return super(ModuleAutoGen, cls).__new__(cls, Workspace, MetaFile, Target, Toolchain, Arch, *args, **kwargs)

    def _InitWorker(self, Workspace, ModuleFile, Target, Toolchain, Arch, PlatformFile, DataPipe):
        if False:
            for i in range(10):
                print('nop')
        EdkLogger.debug(EdkLogger.DEBUG_9, 'AutoGen module [%s] [%s]' % (ModuleFile, Arch))
        GlobalData.gProcessingFile = '%s [%s, %s, %s]' % (ModuleFile, Arch, Toolchain, Target)
        self.Workspace = Workspace
        self.WorkspaceDir = ''
        self.PlatformInfo = None
        self.DataPipe = DataPipe
        self.__init_platform_info__()
        self.MetaFile = ModuleFile
        self.SourceDir = self.MetaFile.SubDir
        self.SourceDir = mws.relpath(self.SourceDir, self.WorkspaceDir)
        self.ToolChain = Toolchain
        self.BuildTarget = Target
        self.Arch = Arch
        self.ToolChainFamily = self.PlatformInfo.ToolChainFamily
        self.BuildRuleFamily = self.PlatformInfo.BuildRuleFamily
        self.IsCodeFileCreated = False
        self.IsAsBuiltInfCreated = False
        self.DepexGenerated = False
        self.BuildDatabase = self.Workspace.BuildDatabase
        self.BuildRuleOrder = None
        self.BuildTime = 0
        self._GuidComments = OrderedListDict()
        self._ProtocolComments = OrderedListDict()
        self._PpiComments = OrderedListDict()
        self._BuildTargets = None
        self._IntroBuildTargetList = None
        self._FinalBuildTargetList = None
        self._FileTypes = None
        self.AutoGenDepSet = set()
        self.ReferenceModules = []
        self.ConstPcd = {}
        self.FileDependCache = {}

    def __init_platform_info__(self):
        if False:
            i = 10
            return i + 15
        pinfo = self.DataPipe.Get('P_Info')
        self.WorkspaceDir = pinfo.get('WorkspaceDir')
        self.PlatformInfo = PlatformInfo(self.Workspace, pinfo.get('ActivePlatform'), pinfo.get('Target'), pinfo.get('ToolChain'), pinfo.get('Arch'), self.DataPipe)

    @cached_class_function
    def __hash__(self):
        if False:
            return 10
        return hash((self.MetaFile, self.Arch, self.ToolChain, self.BuildTarget))

    def __repr__(self):
        if False:
            print('Hello World!')
        return '%s [%s]' % (self.MetaFile, self.Arch)

    @cached_property
    def FixedAtBuildPcds(self):
        if False:
            i = 10
            return i + 15
        RetVal = []
        for Pcd in self.ModulePcdList:
            if Pcd.Type != TAB_PCDS_FIXED_AT_BUILD:
                continue
            if Pcd not in RetVal:
                RetVal.append(Pcd)
        return RetVal

    @cached_property
    def FixedVoidTypePcds(self):
        if False:
            i = 10
            return i + 15
        RetVal = {}
        for Pcd in self.FixedAtBuildPcds:
            if Pcd.DatumType == TAB_VOID:
                if '.'.join((Pcd.TokenSpaceGuidCName, Pcd.TokenCName)) not in RetVal:
                    RetVal['.'.join((Pcd.TokenSpaceGuidCName, Pcd.TokenCName))] = Pcd.DefaultValue
        return RetVal

    @property
    def UniqueBaseName(self):
        if False:
            return 10
        ModuleNames = self.DataPipe.Get('M_Name')
        if not ModuleNames:
            return self.Name
        return ModuleNames.get((self.Name, self.MetaFile), self.Name)

    @cached_property
    def Macros(self):
        if False:
            i = 10
            return i + 15
        return OrderedDict((('WORKSPACE', self.WorkspaceDir), ('MODULE_NAME', self.Name), ('MODULE_NAME_GUID', self.UniqueBaseName), ('MODULE_GUID', self.Guid), ('MODULE_VERSION', self.Version), ('MODULE_TYPE', self.ModuleType), ('MODULE_FILE', str(self.MetaFile)), ('MODULE_FILE_BASE_NAME', self.MetaFile.BaseName), ('MODULE_RELATIVE_DIR', self.SourceDir), ('MODULE_DIR', self.SourceDir), ('BASE_NAME', self.Name), ('ARCH', self.Arch), ('TOOLCHAIN', self.ToolChain), ('TOOLCHAIN_TAG', self.ToolChain), ('TOOL_CHAIN_TAG', self.ToolChain), ('TARGET', self.BuildTarget), ('BUILD_DIR', self.PlatformInfo.BuildDir), ('BIN_DIR', os.path.join(self.PlatformInfo.BuildDir, self.Arch)), ('LIB_DIR', os.path.join(self.PlatformInfo.BuildDir, self.Arch)), ('MODULE_BUILD_DIR', self.BuildDir), ('OUTPUT_DIR', self.OutputDir), ('DEBUG_DIR', self.DebugDir), ('DEST_DIR_OUTPUT', self.OutputDir), ('DEST_DIR_DEBUG', self.DebugDir), ('PLATFORM_NAME', self.PlatformInfo.Name), ('PLATFORM_GUID', self.PlatformInfo.Guid), ('PLATFORM_VERSION', self.PlatformInfo.Version), ('PLATFORM_RELATIVE_DIR', self.PlatformInfo.SourceDir), ('PLATFORM_DIR', mws.join(self.WorkspaceDir, self.PlatformInfo.SourceDir)), ('PLATFORM_OUTPUT_DIR', self.PlatformInfo.OutputDir), ('FFS_OUTPUT_DIR', self.FfsOutputDir)))

    @cached_property
    def Module(self):
        if False:
            print('Hello World!')
        return self.BuildDatabase[self.MetaFile, self.Arch, self.BuildTarget, self.ToolChain]

    @cached_property
    def Name(self):
        if False:
            while True:
                i = 10
        return self.Module.BaseName

    @cached_property
    def DxsFile(self):
        if False:
            i = 10
            return i + 15
        return self.Module.DxsFile

    @cached_property
    def Guid(self):
        if False:
            print('Hello World!')
        if os.path.basename(self.MetaFile.File) != os.path.basename(self.MetaFile.Path):
            return os.path.basename(self.MetaFile.Path)[:36]
        return self.Module.Guid

    @cached_property
    def Version(self):
        if False:
            print('Hello World!')
        return self.Module.Version

    @cached_property
    def ModuleType(self):
        if False:
            while True:
                i = 10
        return self.Module.ModuleType

    @cached_property
    def ComponentType(self):
        if False:
            for i in range(10):
                print('nop')
        return self.Module.ComponentType

    @cached_property
    def BuildType(self):
        if False:
            for i in range(10):
                print('nop')
        return self.Module.BuildType

    @cached_property
    def PcdIsDriver(self):
        if False:
            print('Hello World!')
        return self.Module.PcdIsDriver

    @cached_property
    def AutoGenVersion(self):
        if False:
            return 10
        return self.Module.AutoGenVersion

    @cached_property
    def IsLibrary(self):
        if False:
            for i in range(10):
                print('nop')
        return bool(self.Module.LibraryClass)

    @cached_property
    def IsBinaryModule(self):
        if False:
            for i in range(10):
                print('nop')
        return self.Module.IsBinaryModule

    @cached_property
    def BuildDir(self):
        if False:
            for i in range(10):
                print('nop')
        return _MakeDir((self.PlatformInfo.BuildDir, self.Arch, self.SourceDir, self.MetaFile.BaseName))

    @cached_property
    def OutputDir(self):
        if False:
            return 10
        return _MakeDir((self.BuildDir, 'OUTPUT'))

    @cached_property
    def FfsOutputDir(self):
        if False:
            return 10
        if GlobalData.gFdfParser:
            return path.join(self.PlatformInfo.BuildDir, TAB_FV_DIRECTORY, 'Ffs', self.Guid + self.Name)
        return ''

    @cached_property
    def DebugDir(self):
        if False:
            i = 10
            return i + 15
        return _MakeDir((self.BuildDir, 'DEBUG'))

    @cached_property
    def CustomMakefile(self):
        if False:
            i = 10
            return i + 15
        RetVal = {}
        for Type in self.Module.CustomMakefile:
            MakeType = gMakeTypeMap[Type] if Type in gMakeTypeMap else 'nmake'
            File = os.path.join(self.SourceDir, self.Module.CustomMakefile[Type])
            RetVal[MakeType] = File
        return RetVal

    @cached_property
    def MakeFileDir(self):
        if False:
            return 10
        return self.BuildDir

    @cached_property
    def BuildCommand(self):
        if False:
            for i in range(10):
                print('nop')
        return self.PlatformInfo.BuildCommand

    @cached_property
    def PackageList(self):
        if False:
            i = 10
            return i + 15
        PkagList = []
        if self.Module.Packages:
            PkagList.extend(self.Module.Packages)
        Platform = self.BuildDatabase[self.PlatformInfo.MetaFile, self.Arch, self.BuildTarget, self.ToolChain]
        for Package in Platform.Packages:
            if Package in PkagList:
                continue
            PkagList.append(Package)
        return PkagList

    @cached_property
    def DerivedPackageList(self):
        if False:
            i = 10
            return i + 15
        PackageList = []
        PackageList.extend(self.PackageList)
        for M in self.DependentLibraryList:
            for Package in M.Packages:
                if Package in PackageList:
                    continue
                PackageList.append(Package)
        return PackageList

    def _GetDepexExpresionString(self):
        if False:
            i = 10
            return i + 15
        DepexStr = ''
        DepexList = []
        if self.Module.DxsFile:
            return DepexStr
        for M in [self.Module] + self.DependentLibraryList:
            Filename = M.MetaFile.Path
            InfObj = InfSectionParser.InfSectionParser(Filename)
            DepexExpressionList = InfObj.GetDepexExpresionList()
            for DepexExpression in DepexExpressionList:
                for key in DepexExpression:
                    (Arch, ModuleType) = key
                    DepexExpr = [x for x in DepexExpression[key] if not str(x).startswith('#')]
                    if self.ModuleType.upper() == SUP_MODULE_USER_DEFINED or self.ModuleType.upper() == SUP_MODULE_HOST_APPLICATION:
                        if Arch.upper() == self.Arch.upper() and ModuleType.upper() != TAB_ARCH_COMMON:
                            DepexList.append({(Arch, ModuleType): DepexExpr})
                    elif Arch.upper() == TAB_ARCH_COMMON or (Arch.upper() == self.Arch.upper() and ModuleType.upper() in [TAB_ARCH_COMMON, self.ModuleType.upper()]):
                        DepexList.append({(Arch, ModuleType): DepexExpr})
        if self.ModuleType.upper() == SUP_MODULE_USER_DEFINED or self.ModuleType.upper() == SUP_MODULE_HOST_APPLICATION:
            for Depex in DepexList:
                for key in Depex:
                    DepexStr += '[Depex.%s.%s]\n' % key
                    DepexStr += '\n'.join(('# ' + val for val in Depex[key]))
                    DepexStr += '\n\n'
            if not DepexStr:
                return '[Depex.%s]\n' % self.Arch
            return DepexStr
        Count = 0
        for Depex in DepexList:
            Count += 1
            if DepexStr != '':
                DepexStr += ' AND '
            DepexStr += '('
            for D in Depex.values():
                DepexStr += ' '.join((val for val in D))
            Index = DepexStr.find('END')
            if Index > -1 and Index == len(DepexStr) - 3:
                DepexStr = DepexStr[:-3]
            DepexStr = DepexStr.strip()
            DepexStr += ')'
        if Count == 1:
            DepexStr = DepexStr.lstrip('(').rstrip(')').strip()
        if not DepexStr:
            return '[Depex.%s]\n' % self.Arch
        return '[Depex.%s]\n#  ' % self.Arch + DepexStr

    @cached_property
    def DepexList(self):
        if False:
            i = 10
            return i + 15
        if self.DxsFile or self.IsLibrary or TAB_DEPENDENCY_EXPRESSION_FILE in self.FileTypes:
            return {}
        DepexList = []
        FixedVoidTypePcds = {}
        for M in [self] + self.LibraryAutoGenList:
            FixedVoidTypePcds.update(M.FixedVoidTypePcds)
        for M in [self] + self.LibraryAutoGenList:
            Inherited = False
            for D in M.Module.Depex[self.Arch, self.ModuleType]:
                if DepexList != []:
                    DepexList.append('AND')
                DepexList.append('(')
                NewList = []
                for item in D:
                    if '.' not in item:
                        NewList.append(item)
                    else:
                        try:
                            Value = FixedVoidTypePcds[item]
                            if len(Value.split(',')) != 16:
                                EdkLogger.error('build', FORMAT_INVALID, '{} used in [Depex] section should be used as FixedAtBuild type and VOID* datum type and 16 bytes in the module.'.format(item))
                            NewList.append(Value)
                        except:
                            EdkLogger.error('build', FORMAT_INVALID, '{} used in [Depex] section should be used as FixedAtBuild type and VOID* datum type in the module.'.format(item))
                DepexList.extend(NewList)
                if DepexList[-1] == 'END':
                    DepexList.pop()
                DepexList.append(')')
                Inherited = True
            if Inherited:
                EdkLogger.verbose('DEPEX[%s] (+%s) = %s' % (self.Name, M.Module.BaseName, DepexList))
            if 'BEFORE' in DepexList or 'AFTER' in DepexList:
                break
            if len(DepexList) > 0:
                EdkLogger.verbose('')
        return {self.ModuleType: DepexList}

    @cached_property
    def DepexExpressionDict(self):
        if False:
            i = 10
            return i + 15
        if self.DxsFile or self.IsLibrary or TAB_DEPENDENCY_EXPRESSION_FILE in self.FileTypes:
            return {}
        DepexExpressionString = ''
        for M in [self.Module] + self.DependentLibraryList:
            Inherited = False
            for D in M.DepexExpression[self.Arch, self.ModuleType]:
                if DepexExpressionString != '':
                    DepexExpressionString += ' AND '
                DepexExpressionString += '('
                DepexExpressionString += D
                DepexExpressionString = DepexExpressionString.rstrip('END').strip()
                DepexExpressionString += ')'
                Inherited = True
            if Inherited:
                EdkLogger.verbose('DEPEX[%s] (+%s) = %s' % (self.Name, M.BaseName, DepexExpressionString))
            if 'BEFORE' in DepexExpressionString or 'AFTER' in DepexExpressionString:
                break
        if len(DepexExpressionString) > 0:
            EdkLogger.verbose('')
        return {self.ModuleType: DepexExpressionString}

    def _GetTianoCoreUserExtensionList(self):
        if False:
            i = 10
            return i + 15
        TianoCoreUserExtentionList = []
        for M in [self.Module] + self.DependentLibraryList:
            Filename = M.MetaFile.Path
            InfObj = InfSectionParser.InfSectionParser(Filename)
            TianoCoreUserExtenList = InfObj.GetUserExtensionTianoCore()
            for TianoCoreUserExtent in TianoCoreUserExtenList:
                for Section in TianoCoreUserExtent:
                    ItemList = Section.split(TAB_SPLIT)
                    Arch = self.Arch
                    if len(ItemList) == 4:
                        Arch = ItemList[3]
                    if Arch.upper() == TAB_ARCH_COMMON or Arch.upper() == self.Arch.upper():
                        TianoCoreList = []
                        TianoCoreList.extend([TAB_SECTION_START + Section + TAB_SECTION_END])
                        TianoCoreList.extend(TianoCoreUserExtent[Section][:])
                        TianoCoreList.append('\n')
                        TianoCoreUserExtentionList.append(TianoCoreList)
        return TianoCoreUserExtentionList

    @cached_property
    def Specification(self):
        if False:
            for i in range(10):
                print('nop')
        return self.Module.Specification

    @cached_property
    def BuildOption(self):
        if False:
            i = 10
            return i + 15
        (RetVal, self.BuildRuleOrder) = self.PlatformInfo.ApplyBuildOption(self.Module)
        if self.BuildRuleOrder:
            self.BuildRuleOrder = ['.%s' % Ext for Ext in self.BuildRuleOrder.split()]
        return RetVal

    @cached_property
    def BuildOptionIncPathList(self):
        if False:
            for i in range(10):
                print('nop')
        if self.PlatformInfo.ToolChainFamily in TAB_COMPILER_MSFT:
            BuildOptIncludeRegEx = gBuildOptIncludePatternMsft
        elif self.PlatformInfo.ToolChainFamily in ('INTEL', 'GCC'):
            BuildOptIncludeRegEx = gBuildOptIncludePatternOther
        else:
            return []
        RetVal = []
        for Tool in ('CC', 'PP', 'VFRPP', 'ASLPP', 'ASLCC', 'APP', 'ASM'):
            try:
                FlagOption = self.BuildOption[Tool]['FLAGS']
            except KeyError:
                FlagOption = ''
            IncPathList = [NormPath(Path, self.Macros) for Path in BuildOptIncludeRegEx.findall(FlagOption)]
            if GlobalData.gDisableIncludePathCheck == False:
                for Path in IncPathList:
                    if Path not in self.IncludePathList and CommonPath([Path, self.MetaFile.Dir]) != self.MetaFile.Dir:
                        ErrMsg = "The include directory for the EDK II module in this line is invalid %s specified in %s FLAGS '%s'" % (Path, Tool, FlagOption)
                        EdkLogger.error('build', PARAMETER_INVALID, ExtraData=ErrMsg, File=str(self.MetaFile))
            RetVal += IncPathList
        return RetVal

    @cached_property
    def SourceFileList(self):
        if False:
            return 10
        RetVal = []
        ToolChainTagSet = {'', TAB_STAR, self.ToolChain}
        ToolChainFamilySet = {'', TAB_STAR, self.ToolChainFamily, self.BuildRuleFamily}
        for F in self.Module.Sources:
            if F.TagName not in ToolChainTagSet:
                EdkLogger.debug(EdkLogger.DEBUG_9, 'The toolchain [%s] for processing file [%s] is found, but [%s] is currently used' % (F.TagName, str(F), self.ToolChain))
                continue
            if F.ToolChainFamily not in ToolChainFamilySet:
                EdkLogger.debug(EdkLogger.DEBUG_0, 'The file [%s] must be built by tools of [%s], but current toolchain family is [%s], buildrule family is [%s]' % (str(F), F.ToolChainFamily, self.ToolChainFamily, self.BuildRuleFamily))
                continue
            if F.Dir not in self.IncludePathList:
                self.IncludePathList.insert(0, F.Dir)
            RetVal.append(F)
        self._MatchBuildRuleOrder(RetVal)
        for F in RetVal:
            self._ApplyBuildRule(F, TAB_UNKNOWN_FILE)
        return RetVal

    def _MatchBuildRuleOrder(self, FileList):
        if False:
            for i in range(10):
                print('nop')
        Order_Dict = {}
        self.BuildOption
        for SingleFile in FileList:
            if self.BuildRuleOrder and SingleFile.Ext in self.BuildRuleOrder and (SingleFile.Ext in self.BuildRules):
                key = SingleFile.Path.rsplit(SingleFile.Ext, 1)[0]
                if key in Order_Dict:
                    Order_Dict[key].append(SingleFile.Ext)
                else:
                    Order_Dict[key] = [SingleFile.Ext]
        RemoveList = []
        for F in Order_Dict:
            if len(Order_Dict[F]) > 1:
                Order_Dict[F].sort(key=lambda i: self.BuildRuleOrder.index(i))
                for Ext in Order_Dict[F][1:]:
                    RemoveList.append(F + Ext)
        for item in RemoveList:
            FileList.remove(item)
        return FileList

    @cached_property
    def UnicodeFileList(self):
        if False:
            print('Hello World!')
        return self.FileTypes.get(TAB_UNICODE_FILE, [])

    @cached_property
    def VfrFileList(self):
        if False:
            return 10
        return self.FileTypes.get(TAB_VFR_FILE, [])

    @cached_property
    def IdfFileList(self):
        if False:
            for i in range(10):
                print('nop')
        return self.FileTypes.get(TAB_IMAGE_FILE, [])

    @cached_property
    def BinaryFileList(self):
        if False:
            i = 10
            return i + 15
        RetVal = []
        for F in self.Module.Binaries:
            if F.Target not in [TAB_ARCH_COMMON, TAB_STAR] and F.Target != self.BuildTarget:
                continue
            RetVal.append(F)
            self._ApplyBuildRule(F, F.Type, BinaryFileList=RetVal)
        return RetVal

    @cached_property
    def BuildRules(self):
        if False:
            return 10
        RetVal = {}
        BuildRuleDatabase = self.PlatformInfo.BuildRule
        for Type in BuildRuleDatabase.FileTypeList:
            RuleObject = BuildRuleDatabase[Type, self.BuildType, self.Arch, self.BuildRuleFamily]
            if not RuleObject:
                if self.ModuleType != self.BuildType:
                    RuleObject = BuildRuleDatabase[Type, self.ModuleType, self.Arch, self.BuildRuleFamily]
            if not RuleObject:
                RuleObject = BuildRuleDatabase[Type, self.BuildType, self.Arch, self.ToolChainFamily]
                if not RuleObject:
                    if self.ModuleType != self.BuildType:
                        RuleObject = BuildRuleDatabase[Type, self.ModuleType, self.Arch, self.ToolChainFamily]
            if not RuleObject:
                continue
            RuleObject = RuleObject.Instantiate(self.Macros)
            RetVal[Type] = RuleObject
            for Ext in RuleObject.SourceFileExtList:
                RetVal[Ext] = RuleObject
        return RetVal

    def _ApplyBuildRule(self, File, FileType, BinaryFileList=None):
        if False:
            print('Hello World!')
        if self._BuildTargets is None:
            self._IntroBuildTargetList = set()
            self._FinalBuildTargetList = set()
            self._BuildTargets = defaultdict(set)
            self._FileTypes = defaultdict(set)
        if not BinaryFileList:
            BinaryFileList = self.BinaryFileList
        SubDirectory = os.path.join(self.OutputDir, File.SubDir)
        if not os.path.exists(SubDirectory):
            CreateDirectory(SubDirectory)
        TargetList = set()
        FinalTargetName = set()
        RuleChain = set()
        SourceList = [File]
        Index = 0
        self.BuildOption
        while Index < len(SourceList):
            if Index > 0:
                FileType = TAB_UNKNOWN_FILE
            Source = SourceList[Index]
            Index = Index + 1
            if Source != File:
                CreateDirectory(Source.Dir)
            if File.IsBinary and File == Source and (File in BinaryFileList):
                if not self.IsLibrary:
                    continue
                RuleObject = self.BuildRules[TAB_DEFAULT_BINARY_FILE]
            elif FileType in self.BuildRules:
                RuleObject = self.BuildRules[FileType]
            elif Source.Ext in self.BuildRules:
                RuleObject = self.BuildRules[Source.Ext]
            else:
                FinalTargetName.add(Source)
                continue
            FileType = RuleObject.SourceFileType
            self._FileTypes[FileType].add(Source)
            if self.IsLibrary and FileType == TAB_STATIC_LIBRARY:
                FinalTargetName.add(Source)
                continue
            Target = RuleObject.Apply(Source, self.BuildRuleOrder)
            if not Target:
                FinalTargetName.add(Source)
                continue
            TargetList.add(Target)
            self._BuildTargets[FileType].add(Target)
            if not Source.IsBinary and Source == File:
                self._IntroBuildTargetList.add(Target)
            if FileType in RuleChain:
                EdkLogger.error('build', ERROR_STATEMENT, 'Cyclic dependency detected while generating rule for %s' % str(Source))
            RuleChain.add(FileType)
            SourceList.extend(Target.Outputs)
        for FTargetName in FinalTargetName:
            for Target in TargetList:
                if FTargetName == Target.Target:
                    self._FinalBuildTargetList.add(Target)

    @cached_property
    def Targets(self):
        if False:
            print('Hello World!')
        if self._BuildTargets is None:
            self._IntroBuildTargetList = set()
            self._FinalBuildTargetList = set()
            self._BuildTargets = defaultdict(set)
            self._FileTypes = defaultdict(set)
        self.SourceFileList
        self.BinaryFileList
        return self._BuildTargets

    @cached_property
    def IntroTargetList(self):
        if False:
            for i in range(10):
                print('nop')
        self.Targets
        return self._IntroBuildTargetList

    @cached_property
    def CodaTargetList(self):
        if False:
            for i in range(10):
                print('nop')
        self.Targets
        return self._FinalBuildTargetList

    @cached_property
    def FileTypes(self):
        if False:
            i = 10
            return i + 15
        self.Targets
        return self._FileTypes

    @cached_property
    def DependentPackageList(self):
        if False:
            print('Hello World!')
        return self.PackageList

    @cached_property
    def AutoGenFileList(self):
        if False:
            return 10
        AutoGenUniIdf = self.BuildType != 'UEFI_HII'
        UniStringBinBuffer = BytesIO()
        IdfGenBinBuffer = BytesIO()
        RetVal = {}
        AutoGenC = TemplateString()
        AutoGenH = TemplateString()
        StringH = TemplateString()
        StringIdf = TemplateString()
        GenC.CreateCode(self, AutoGenC, AutoGenH, StringH, AutoGenUniIdf, UniStringBinBuffer, StringIdf, AutoGenUniIdf, IdfGenBinBuffer)
        if str(AutoGenC) != '' and (len(self.Module.LibraryClasses) > 0 or TAB_OBJECT_FILE in self.FileTypes):
            AutoFile = PathClass(gAutoGenCodeFileName, self.DebugDir)
            RetVal[AutoFile] = str(AutoGenC)
            self._ApplyBuildRule(AutoFile, TAB_UNKNOWN_FILE)
        if str(AutoGenH) != '':
            AutoFile = PathClass(gAutoGenHeaderFileName, self.DebugDir)
            RetVal[AutoFile] = str(AutoGenH)
            self._ApplyBuildRule(AutoFile, TAB_UNKNOWN_FILE)
        if str(StringH) != '':
            AutoFile = PathClass(gAutoGenStringFileName % {'module_name': self.Name}, self.DebugDir)
            RetVal[AutoFile] = str(StringH)
            self._ApplyBuildRule(AutoFile, TAB_UNKNOWN_FILE)
        if UniStringBinBuffer is not None and UniStringBinBuffer.getvalue() != b'':
            AutoFile = PathClass(gAutoGenStringFormFileName % {'module_name': self.Name}, self.OutputDir)
            RetVal[AutoFile] = UniStringBinBuffer.getvalue()
            AutoFile.IsBinary = True
            self._ApplyBuildRule(AutoFile, TAB_UNKNOWN_FILE)
        if UniStringBinBuffer is not None:
            UniStringBinBuffer.close()
        if str(StringIdf) != '':
            AutoFile = PathClass(gAutoGenImageDefFileName % {'module_name': self.Name}, self.DebugDir)
            RetVal[AutoFile] = str(StringIdf)
            self._ApplyBuildRule(AutoFile, TAB_UNKNOWN_FILE)
        if IdfGenBinBuffer is not None and IdfGenBinBuffer.getvalue() != b'':
            AutoFile = PathClass(gAutoGenIdfFileName % {'module_name': self.Name}, self.OutputDir)
            RetVal[AutoFile] = IdfGenBinBuffer.getvalue()
            AutoFile.IsBinary = True
            self._ApplyBuildRule(AutoFile, TAB_UNKNOWN_FILE)
        if IdfGenBinBuffer is not None:
            IdfGenBinBuffer.close()
        return RetVal

    @cached_property
    def DependentLibraryList(self):
        if False:
            for i in range(10):
                print('nop')
        if self.IsLibrary:
            return []
        return self.PlatformInfo.ApplyLibraryInstance(self.Module)

    @cached_property
    def ModulePcdList(self):
        if False:
            print('Hello World!')
        RetVal = self.PlatformInfo.ApplyPcdSetting(self, self.Module.Pcds)
        return RetVal

    @cached_property
    def _PcdComments(self):
        if False:
            for i in range(10):
                print('nop')
        ReVal = OrderedListDict()
        ExtendCopyDictionaryLists(ReVal, self.Module.PcdComments)
        if not self.IsLibrary:
            for Library in self.DependentLibraryList:
                ExtendCopyDictionaryLists(ReVal, Library.PcdComments)
        return ReVal

    @cached_property
    def LibraryPcdList(self):
        if False:
            print('Hello World!')
        if self.IsLibrary:
            return []
        RetVal = []
        Pcds = set()
        for Library in self.DependentLibraryList:
            PcdsInLibrary = OrderedDict()
            for Key in Library.Pcds:
                if Key in self.Module.Pcds or Key in Pcds:
                    continue
                Pcds.add(Key)
                PcdsInLibrary[Key] = copy.copy(Library.Pcds[Key])
            RetVal.extend(self.PlatformInfo.ApplyPcdSetting(self, PcdsInLibrary, Library=Library))
        return RetVal

    @cached_property
    def GuidList(self):
        if False:
            return 10
        RetVal = self.Module.Guids
        for Library in self.DependentLibraryList:
            RetVal.update(Library.Guids)
            ExtendCopyDictionaryLists(self._GuidComments, Library.GuidComments)
        ExtendCopyDictionaryLists(self._GuidComments, self.Module.GuidComments)
        return RetVal

    @cached_property
    def GetGuidsUsedByPcd(self):
        if False:
            while True:
                i = 10
        RetVal = OrderedDict(self.Module.GetGuidsUsedByPcd())
        for Library in self.DependentLibraryList:
            RetVal.update(Library.GetGuidsUsedByPcd())
        return RetVal

    @cached_property
    def ProtocolList(self):
        if False:
            i = 10
            return i + 15
        RetVal = OrderedDict(self.Module.Protocols)
        for Library in self.DependentLibraryList:
            RetVal.update(Library.Protocols)
            ExtendCopyDictionaryLists(self._ProtocolComments, Library.ProtocolComments)
        ExtendCopyDictionaryLists(self._ProtocolComments, self.Module.ProtocolComments)
        return RetVal

    @cached_property
    def PpiList(self):
        if False:
            while True:
                i = 10
        RetVal = OrderedDict(self.Module.Ppis)
        for Library in self.DependentLibraryList:
            RetVal.update(Library.Ppis)
            ExtendCopyDictionaryLists(self._PpiComments, Library.PpiComments)
        ExtendCopyDictionaryLists(self._PpiComments, self.Module.PpiComments)
        return RetVal

    @cached_property
    def IncludePathList(self):
        if False:
            return 10
        RetVal = []
        RetVal.append(self.MetaFile.Dir)
        RetVal.append(self.DebugDir)
        for Package in self.PackageList:
            PackageDir = mws.join(self.WorkspaceDir, Package.MetaFile.Dir)
            if PackageDir not in RetVal:
                RetVal.append(PackageDir)
            IncludesList = Package.Includes
            if Package._PrivateIncludes:
                if not self.MetaFile.OriginalPath.Path.startswith(PackageDir):
                    IncludesList = list(set(Package.Includes).difference(set(Package._PrivateIncludes)))
            for Inc in IncludesList:
                if Inc not in RetVal:
                    RetVal.append(str(Inc))
        RetVal.extend(self.IncPathFromBuildOptions)
        return RetVal

    @cached_property
    def IncPathFromBuildOptions(self):
        if False:
            for i in range(10):
                print('nop')
        IncPathList = []
        for tool in self.BuildOption:
            if 'FLAGS' in self.BuildOption[tool]:
                flags = self.BuildOption[tool]['FLAGS']
                whitespace = False
                for flag in flags.split(' '):
                    flag = flag.strip()
                    if flag.startswith(('/I', '-I')):
                        if len(flag) > 2:
                            if os.path.exists(flag[2:]):
                                IncPathList.append(flag[2:])
                        else:
                            whitespace = True
                            continue
                    if whitespace and flag:
                        if os.path.exists(flag):
                            IncPathList.append(flag)
                            whitespace = False
        return IncPathList

    @cached_property
    def IncludePathLength(self):
        if False:
            for i in range(10):
                print('nop')
        return sum((len(inc) + 1 for inc in self.IncludePathList))

    @cached_property
    def PackageIncludePathList(self):
        if False:
            for i in range(10):
                print('nop')
        IncludesList = []
        for Package in self.PackageList:
            PackageDir = mws.join(self.WorkspaceDir, Package.MetaFile.Dir)
            IncludesList = Package.Includes
            if Package._PrivateIncludes:
                if not self.MetaFile.Path.startswith(PackageDir):
                    IncludesList = list(set(Package.Includes).difference(set(Package._PrivateIncludes)))
        return IncludesList

    def _GetPcdsMaybeUsedByVfr(self):
        if False:
            i = 10
            return i + 15
        if not self.SourceFileList:
            return []
        NameGuids = set()
        for SrcFile in self.SourceFileList:
            if SrcFile.Ext.lower() != '.vfr':
                continue
            Vfri = os.path.join(self.OutputDir, SrcFile.BaseName + '.i')
            if not os.path.exists(Vfri):
                continue
            VfriFile = open(Vfri, 'r')
            Content = VfriFile.read()
            VfriFile.close()
            Pos = Content.find('efivarstore')
            while Pos != -1:
                Index = Pos - 1
                while Index >= 0 and Content[Index] in ' \t\r\n':
                    Index -= 1
                if Index >= 0 and Content[Index] != ';':
                    Pos = Content.find('efivarstore', Pos + len('efivarstore'))
                    continue
                Name = gEfiVarStoreNamePattern.search(Content, Pos)
                if not Name:
                    break
                Guid = gEfiVarStoreGuidPattern.search(Content, Pos)
                if not Guid:
                    break
                NameArray = _ConvertStringToByteArray('L"' + Name.group(1) + '"')
                NameGuids.add((NameArray, GuidStructureStringToGuidString(Guid.group(1))))
                Pos = Content.find('efivarstore', Name.end())
        if not NameGuids:
            return []
        HiiExPcds = []
        for Pcd in self.PlatformInfo.Pcds.values():
            if Pcd.Type != TAB_PCDS_DYNAMIC_EX_HII:
                continue
            for SkuInfo in Pcd.SkuInfoList.values():
                Value = GuidValue(SkuInfo.VariableGuid, self.PlatformInfo.PackageList, self.MetaFile.Path)
                if not Value:
                    continue
                Name = _ConvertStringToByteArray(SkuInfo.VariableName)
                Guid = GuidStructureStringToGuidString(Value)
                if (Name, Guid) in NameGuids and Pcd not in HiiExPcds:
                    HiiExPcds.append(Pcd)
                    break
        return HiiExPcds

    def _GenOffsetBin(self):
        if False:
            while True:
                i = 10
        VfrUniBaseName = {}
        for SourceFile in self.Module.Sources:
            if SourceFile.Type.upper() == '.VFR':
                VfrUniBaseName[SourceFile.BaseName] = SourceFile.BaseName + 'Bin'
            elif SourceFile.Type.upper() == '.UNI':
                VfrUniBaseName['UniOffsetName'] = self.Name + 'Strings'
        if not VfrUniBaseName:
            return None
        MapFileName = os.path.join(self.OutputDir, self.Name + '.map')
        EfiFileName = os.path.join(self.OutputDir, self.Name + '.efi')
        VfrUniOffsetList = GetVariableOffset(MapFileName, EfiFileName, list(VfrUniBaseName.values()))
        if not VfrUniOffsetList:
            return None
        OutputName = '%sOffset.bin' % self.Name
        UniVfrOffsetFileName = os.path.join(self.OutputDir, OutputName)
        try:
            fInputfile = open(UniVfrOffsetFileName, 'wb+', 0)
        except:
            EdkLogger.error('build', FILE_OPEN_FAILURE, 'File open failed for %s' % UniVfrOffsetFileName, None)
        fStringIO = BytesIO()
        for Item in VfrUniOffsetList:
            if Item[0].find('Strings') != -1:
                UniGuid = b'\xe0\xc5\x13\x89\xf63\x86M\x9b\xf1C\xef\x89\xfc\x06f'
                fStringIO.write(UniGuid)
                UniValue = pack('Q', int(Item[1], 16))
                fStringIO.write(UniValue)
            else:
                VfrGuid = b'\xb4|\xbc\xd0Gj_I\xaa\x11q\x07F\xda\x06\xa2'
                fStringIO.write(VfrGuid)
                VfrValue = pack('Q', int(Item[1], 16))
                fStringIO.write(VfrValue)
        try:
            fInputfile.write(fStringIO.getvalue())
        except:
            EdkLogger.error('build', FILE_WRITE_FAILURE, 'Write data to file %s failed, please check whether the file been locked or using by other applications.' % UniVfrOffsetFileName, None)
        fStringIO.close()
        fInputfile.close()
        return OutputName

    @cached_property
    def OutputFile(self):
        if False:
            return 10
        retVal = set()
        for (Root, Dirs, Files) in os.walk(self.BuildDir):
            for File in Files:
                if not (File.lower().endswith('.obj') or File.lower().endswith('.debug')):
                    NewFile = path.join(Root, File)
                    retVal.add(NewFile)
        for (Root, Dirs, Files) in os.walk(self.FfsOutputDir):
            for File in Files:
                NewFile = path.join(Root, File)
                retVal.add(NewFile)
        return retVal

    def CreateAsBuiltInf(self):
        if False:
            for i in range(10):
                print('nop')
        if self.IsAsBuiltInfCreated:
            return
        if self.IsLibrary:
            return
        if not self.SourceFileList:
            return
        if self.BinaryFileList:
            return
        Pcds = []
        PatchablePcds = []
        Packages = []
        PcdCheckList = []
        PcdTokenSpaceList = []
        for Pcd in self.ModulePcdList + self.LibraryPcdList:
            if Pcd.Type == TAB_PCDS_PATCHABLE_IN_MODULE:
                PatchablePcds.append(Pcd)
                PcdCheckList.append((Pcd.TokenCName, Pcd.TokenSpaceGuidCName, TAB_PCDS_PATCHABLE_IN_MODULE))
            elif Pcd.Type in PCD_DYNAMIC_EX_TYPE_SET:
                if Pcd not in Pcds:
                    Pcds.append(Pcd)
                    PcdCheckList.append((Pcd.TokenCName, Pcd.TokenSpaceGuidCName, TAB_PCDS_DYNAMIC_EX))
                    PcdCheckList.append((Pcd.TokenCName, Pcd.TokenSpaceGuidCName, TAB_PCDS_DYNAMIC))
                    PcdTokenSpaceList.append(Pcd.TokenSpaceGuidCName)
        GuidList = OrderedDict(self.GuidList)
        for TokenSpace in self.GetGuidsUsedByPcd:
            if TokenSpace not in PcdTokenSpaceList and TokenSpace in GuidList:
                GuidList.pop(TokenSpace)
        CheckList = (GuidList, self.PpiList, self.ProtocolList, PcdCheckList)
        for Package in self.DerivedPackageList:
            if Package in Packages:
                continue
            BeChecked = (Package.Guids, Package.Ppis, Package.Protocols, Package.Pcds)
            Found = False
            for Index in range(len(BeChecked)):
                for Item in CheckList[Index]:
                    if Item in BeChecked[Index]:
                        Packages.append(Package)
                        Found = True
                        break
                if Found:
                    break
        VfrPcds = self._GetPcdsMaybeUsedByVfr()
        for Pkg in self.PlatformInfo.PackageList:
            if Pkg in Packages:
                continue
            for VfrPcd in VfrPcds:
                if (VfrPcd.TokenCName, VfrPcd.TokenSpaceGuidCName, TAB_PCDS_DYNAMIC_EX) in Pkg.Pcds or (VfrPcd.TokenCName, VfrPcd.TokenSpaceGuidCName, TAB_PCDS_DYNAMIC) in Pkg.Pcds:
                    Packages.append(Pkg)
                    break
        ModuleType = SUP_MODULE_DXE_DRIVER if self.ModuleType == SUP_MODULE_UEFI_DRIVER and self.DepexGenerated else self.ModuleType
        DriverType = self.PcdIsDriver if self.PcdIsDriver else ''
        Guid = self.Guid
        MDefs = self.Module.Defines
        AsBuiltInfDict = {'module_name': self.Name, 'module_guid': Guid, 'module_module_type': ModuleType, 'module_version_string': [MDefs['VERSION_STRING']] if 'VERSION_STRING' in MDefs else [], 'pcd_is_driver_string': [], 'module_uefi_specification_version': [], 'module_pi_specification_version': [], 'module_entry_point': self.Module.ModuleEntryPointList, 'module_unload_image': self.Module.ModuleUnloadImageList, 'module_constructor': self.Module.ConstructorList, 'module_destructor': self.Module.DestructorList, 'module_shadow': [MDefs['SHADOW']] if 'SHADOW' in MDefs else [], 'module_pci_vendor_id': [MDefs['PCI_VENDOR_ID']] if 'PCI_VENDOR_ID' in MDefs else [], 'module_pci_device_id': [MDefs['PCI_DEVICE_ID']] if 'PCI_DEVICE_ID' in MDefs else [], 'module_pci_class_code': [MDefs['PCI_CLASS_CODE']] if 'PCI_CLASS_CODE' in MDefs else [], 'module_pci_revision': [MDefs['PCI_REVISION']] if 'PCI_REVISION' in MDefs else [], 'module_build_number': [MDefs['BUILD_NUMBER']] if 'BUILD_NUMBER' in MDefs else [], 'module_spec': [MDefs['SPEC']] if 'SPEC' in MDefs else [], 'module_uefi_hii_resource_section': [MDefs['UEFI_HII_RESOURCE_SECTION']] if 'UEFI_HII_RESOURCE_SECTION' in MDefs else [], 'module_uni_file': [MDefs['MODULE_UNI_FILE']] if 'MODULE_UNI_FILE' in MDefs else [], 'module_arch': self.Arch, 'package_item': [Package.MetaFile.File.replace('\\', '/') for Package in Packages], 'binary_item': [], 'patchablepcd_item': [], 'pcd_item': [], 'protocol_item': [], 'ppi_item': [], 'guid_item': [], 'flags_item': [], 'libraryclasses_item': []}
        if 'MODULE_UNI_FILE' in MDefs:
            UNIFile = os.path.join(self.MetaFile.Dir, MDefs['MODULE_UNI_FILE'])
            if os.path.isfile(UNIFile):
                shutil.copy2(UNIFile, self.OutputDir)
        if self.AutoGenVersion > int(gInfSpecVersion, 0):
            AsBuiltInfDict['module_inf_version'] = '0x%08x' % self.AutoGenVersion
        else:
            AsBuiltInfDict['module_inf_version'] = gInfSpecVersion
        if DriverType:
            AsBuiltInfDict['pcd_is_driver_string'].append(DriverType)
        if 'UEFI_SPECIFICATION_VERSION' in self.Specification:
            AsBuiltInfDict['module_uefi_specification_version'].append(self.Specification['UEFI_SPECIFICATION_VERSION'])
        if 'PI_SPECIFICATION_VERSION' in self.Specification:
            AsBuiltInfDict['module_pi_specification_version'].append(self.Specification['PI_SPECIFICATION_VERSION'])
        OutputDir = self.OutputDir.replace('\\', '/').strip('/')
        DebugDir = self.DebugDir.replace('\\', '/').strip('/')
        for Item in self.CodaTargetList:
            File = Item.Target.Path.replace('\\', '/').strip('/').replace(DebugDir, '').replace(OutputDir, '').strip('/')
            if os.path.isabs(File):
                File = File.replace('\\', '/').strip('/').replace(OutputDir, '').strip('/')
            if Item.Target.Ext.lower() == '.aml':
                AsBuiltInfDict['binary_item'].append('ASL|' + File)
            elif Item.Target.Ext.lower() == '.acpi':
                AsBuiltInfDict['binary_item'].append('ACPI|' + File)
            elif Item.Target.Ext.lower() == '.efi':
                AsBuiltInfDict['binary_item'].append('PE32|' + self.Name + '.efi')
            else:
                AsBuiltInfDict['binary_item'].append('BIN|' + File)
        if not self.DepexGenerated:
            DepexFile = os.path.join(self.OutputDir, self.Name + '.depex')
            if os.path.exists(DepexFile):
                self.DepexGenerated = True
        if self.DepexGenerated:
            if self.ModuleType in [SUP_MODULE_PEIM]:
                AsBuiltInfDict['binary_item'].append('PEI_DEPEX|' + self.Name + '.depex')
            elif self.ModuleType in [SUP_MODULE_DXE_DRIVER, SUP_MODULE_DXE_RUNTIME_DRIVER, SUP_MODULE_DXE_SAL_DRIVER, SUP_MODULE_UEFI_DRIVER]:
                AsBuiltInfDict['binary_item'].append('DXE_DEPEX|' + self.Name + '.depex')
            elif self.ModuleType in [SUP_MODULE_DXE_SMM_DRIVER]:
                AsBuiltInfDict['binary_item'].append('SMM_DEPEX|' + self.Name + '.depex')
        Bin = self._GenOffsetBin()
        if Bin:
            AsBuiltInfDict['binary_item'].append('BIN|%s' % Bin)
        for (Root, Dirs, Files) in os.walk(OutputDir):
            for File in Files:
                if File.lower().endswith('.pdb'):
                    AsBuiltInfDict['binary_item'].append('DISPOSABLE|' + File)
        HeaderComments = self.Module.HeaderComments
        StartPos = 0
        for Index in range(len(HeaderComments)):
            if HeaderComments[Index].find('@BinaryHeader') != -1:
                HeaderComments[Index] = HeaderComments[Index].replace('@BinaryHeader', '@file')
                StartPos = Index
                break
        AsBuiltInfDict['header_comments'] = '\n'.join(HeaderComments[StartPos:]).replace(':#', '://')
        AsBuiltInfDict['tail_comments'] = '\n'.join(self.Module.TailComments)
        GenList = [(self.ProtocolList, self._ProtocolComments, 'protocol_item'), (self.PpiList, self._PpiComments, 'ppi_item'), (GuidList, self._GuidComments, 'guid_item')]
        for Item in GenList:
            for CName in Item[0]:
                Comments = '\n  '.join(Item[1][CName]) if CName in Item[1] else ''
                Entry = Comments + '\n  ' + CName if Comments else CName
                AsBuiltInfDict[Item[2]].append(Entry)
        PatchList = parsePcdInfoFromMapFile(os.path.join(self.OutputDir, self.Name + '.map'), os.path.join(self.OutputDir, self.Name + '.efi'))
        if PatchList:
            for Pcd in PatchablePcds:
                TokenCName = Pcd.TokenCName
                for PcdItem in GlobalData.MixedPcd:
                    if (Pcd.TokenCName, Pcd.TokenSpaceGuidCName) in GlobalData.MixedPcd[PcdItem]:
                        TokenCName = PcdItem[0]
                        break
                for PatchPcd in PatchList:
                    if TokenCName == PatchPcd[0]:
                        break
                else:
                    continue
                PcdValue = ''
                if Pcd.DatumType == 'BOOLEAN':
                    BoolValue = Pcd.DefaultValue.upper()
                    if BoolValue == 'TRUE':
                        Pcd.DefaultValue = '1'
                    elif BoolValue == 'FALSE':
                        Pcd.DefaultValue = '0'
                if Pcd.DatumType in TAB_PCD_NUMERIC_TYPES:
                    HexFormat = '0x%02x'
                    if Pcd.DatumType == TAB_UINT16:
                        HexFormat = '0x%04x'
                    elif Pcd.DatumType == TAB_UINT32:
                        HexFormat = '0x%08x'
                    elif Pcd.DatumType == TAB_UINT64:
                        HexFormat = '0x%016x'
                    PcdValue = HexFormat % int(Pcd.DefaultValue, 0)
                else:
                    if Pcd.MaxDatumSize is None or Pcd.MaxDatumSize == '':
                        EdkLogger.error('build', AUTOGEN_ERROR, 'Unknown [MaxDatumSize] of PCD [%s.%s]' % (Pcd.TokenSpaceGuidCName, TokenCName))
                    ArraySize = int(Pcd.MaxDatumSize, 0)
                    PcdValue = Pcd.DefaultValue
                    if PcdValue[0] != '{':
                        Unicode = False
                        if PcdValue[0] == 'L':
                            Unicode = True
                        PcdValue = PcdValue.lstrip('L')
                        PcdValue = eval(PcdValue)
                        NewValue = '{'
                        for Index in range(0, len(PcdValue)):
                            if Unicode:
                                CharVal = ord(PcdValue[Index])
                                NewValue = NewValue + '0x%02x' % (CharVal & 255) + ', ' + '0x%02x' % (CharVal >> 8) + ', '
                            else:
                                NewValue = NewValue + '0x%02x' % (ord(PcdValue[Index]) % 256) + ', '
                        Padding = '0x00, '
                        if Unicode:
                            Padding = Padding * 2
                            ArraySize = ArraySize // 2
                        if ArraySize < len(PcdValue) + 1:
                            if Pcd.MaxSizeUserSet:
                                EdkLogger.error('build', AUTOGEN_ERROR, "The maximum size of VOID* type PCD '%s.%s' is less than its actual size occupied." % (Pcd.TokenSpaceGuidCName, TokenCName))
                            else:
                                ArraySize = len(PcdValue) + 1
                        if ArraySize > len(PcdValue) + 1:
                            NewValue = NewValue + Padding * (ArraySize - len(PcdValue) - 1)
                        PcdValue = NewValue + Padding.strip().rstrip(',') + '}'
                    elif len(PcdValue.split(',')) <= ArraySize:
                        PcdValue = PcdValue.rstrip('}') + ', 0x00' * (ArraySize - len(PcdValue.split(',')))
                        PcdValue += '}'
                    elif Pcd.MaxSizeUserSet:
                        EdkLogger.error('build', AUTOGEN_ERROR, "The maximum size of VOID* type PCD '%s.%s' is less than its actual size occupied." % (Pcd.TokenSpaceGuidCName, TokenCName))
                    else:
                        ArraySize = len(PcdValue) + 1
                PcdItem = '%s.%s|%s|0x%X' % (Pcd.TokenSpaceGuidCName, TokenCName, PcdValue, PatchPcd[1])
                PcdComments = ''
                if (Pcd.TokenSpaceGuidCName, Pcd.TokenCName) in self._PcdComments:
                    PcdComments = '\n  '.join(self._PcdComments[Pcd.TokenSpaceGuidCName, Pcd.TokenCName])
                if PcdComments:
                    PcdItem = PcdComments + '\n  ' + PcdItem
                AsBuiltInfDict['patchablepcd_item'].append(PcdItem)
        for Pcd in Pcds + VfrPcds:
            PcdCommentList = []
            HiiInfo = ''
            TokenCName = Pcd.TokenCName
            for PcdItem in GlobalData.MixedPcd:
                if (Pcd.TokenCName, Pcd.TokenSpaceGuidCName) in GlobalData.MixedPcd[PcdItem]:
                    TokenCName = PcdItem[0]
                    break
            if Pcd.Type == TAB_PCDS_DYNAMIC_EX_HII:
                for SkuName in Pcd.SkuInfoList:
                    SkuInfo = Pcd.SkuInfoList[SkuName]
                    HiiInfo = '## %s|%s|%s' % (SkuInfo.VariableName, SkuInfo.VariableGuid, SkuInfo.VariableOffset)
                    break
            if (Pcd.TokenSpaceGuidCName, Pcd.TokenCName) in self._PcdComments:
                PcdCommentList = self._PcdComments[Pcd.TokenSpaceGuidCName, Pcd.TokenCName][:]
            if HiiInfo:
                UsageIndex = -1
                UsageStr = ''
                for (Index, Comment) in enumerate(PcdCommentList):
                    for Usage in UsageList:
                        if Comment.find(Usage) != -1:
                            UsageStr = Usage
                            UsageIndex = Index
                            break
                if UsageIndex != -1:
                    PcdCommentList[UsageIndex] = '## %s %s %s' % (UsageStr, HiiInfo, PcdCommentList[UsageIndex].replace(UsageStr, ''))
                else:
                    PcdCommentList.append('## UNDEFINED ' + HiiInfo)
            PcdComments = '\n  '.join(PcdCommentList)
            PcdEntry = Pcd.TokenSpaceGuidCName + '.' + TokenCName
            if PcdComments:
                PcdEntry = PcdComments + '\n  ' + PcdEntry
            AsBuiltInfDict['pcd_item'].append(PcdEntry)
        for Item in self.BuildOption:
            if 'FLAGS' in self.BuildOption[Item]:
                AsBuiltInfDict['flags_item'].append('%s:%s_%s_%s_%s_FLAGS = %s' % (self.ToolChainFamily, self.BuildTarget, self.ToolChain, self.Arch, Item, self.BuildOption[Item]['FLAGS'].strip()))
        for Library in self.LibraryAutoGenList:
            AsBuiltInfDict['libraryclasses_item'].append(Library.MetaFile.File.replace('\\', '/'))
        UserExtStr = ''
        for TianoCore in self._GetTianoCoreUserExtensionList():
            UserExtStr += '\n'.join(TianoCore)
            ExtensionFile = os.path.join(self.MetaFile.Dir, TianoCore[1])
            if os.path.isfile(ExtensionFile):
                shutil.copy2(ExtensionFile, self.OutputDir)
        AsBuiltInfDict['userextension_tianocore_item'] = UserExtStr
        DepexExpression = self._GetDepexExpresionString()
        AsBuiltInfDict['depexsection_item'] = DepexExpression if DepexExpression else ''
        AsBuiltInf = TemplateString()
        AsBuiltInf.Append(gAsBuiltInfHeaderString.Replace(AsBuiltInfDict))
        SaveFileOnChange(os.path.join(self.OutputDir, self.Name + '.inf'), str(AsBuiltInf), False)
        self.IsAsBuiltInfCreated = True

    def CacheCopyFile(self, DestDir, SourceDir, File):
        if False:
            for i in range(10):
                print('nop')
        if os.path.isdir(File):
            return
        sub_dir = os.path.relpath(File, SourceDir)
        destination_file = os.path.join(DestDir, sub_dir)
        destination_dir = os.path.dirname(destination_file)
        CreateDirectory(destination_dir)
        try:
            CopyFileOnChange(File, destination_dir)
        except:
            EdkLogger.quiet('[cache warning]: fail to copy file:%s to folder:%s' % (File, destination_dir))
            return

    def CopyModuleToCache(self):
        if False:
            while True:
                i = 10
        MakeHashStr = None
        PreMakeHashStr = None
        MakeTimeStamp = 0
        PreMakeTimeStamp = 0
        Files = [f for f in os.listdir(LongFilePath(self.BuildDir)) if path.isfile(LongFilePath(path.join(self.BuildDir, f)))]
        for File in Files:
            if '.MakeHashFileList.' in File:
                FileTimeStamp = os.stat(LongFilePath(path.join(self.BuildDir, File)))[8]
                if FileTimeStamp > MakeTimeStamp:
                    MakeTimeStamp = FileTimeStamp
                    MakeHashStr = File.split('.')[-1]
                    if len(MakeHashStr) != 32:
                        EdkLogger.quiet('[cache error]: wrong MakeHashFileList file:%s' % File)
            if '.PreMakeHashFileList.' in File:
                FileTimeStamp = os.stat(LongFilePath(path.join(self.BuildDir, File)))[8]
                if FileTimeStamp > PreMakeTimeStamp:
                    PreMakeTimeStamp = FileTimeStamp
                    PreMakeHashStr = File.split('.')[-1]
                    if len(PreMakeHashStr) != 32:
                        EdkLogger.quiet('[cache error]: wrong PreMakeHashFileList file:%s' % File)
        if not MakeHashStr:
            EdkLogger.quiet('[cache error]: No MakeHashFileList file for module:%s[%s]' % (self.MetaFile.Path, self.Arch))
            return
        if not PreMakeHashStr:
            EdkLogger.quiet('[cache error]: No PreMakeHashFileList file for module:%s[%s]' % (self.MetaFile.Path, self.Arch))
            return
        FileDir = path.join(GlobalData.gBinCacheDest, self.PlatformInfo.OutputDir, self.BuildTarget + '_' + self.ToolChain, self.Arch, self.SourceDir, self.MetaFile.BaseName)
        FfsDir = path.join(GlobalData.gBinCacheDest, self.PlatformInfo.OutputDir, self.BuildTarget + '_' + self.ToolChain, TAB_FV_DIRECTORY, 'Ffs', self.Guid + self.Name)
        CacheFileDir = path.join(FileDir, MakeHashStr)
        CacheFfsDir = path.join(FfsDir, MakeHashStr)
        CreateDirectory(CacheFileDir)
        CreateDirectory(CacheFfsDir)
        ModuleHashPair = path.join(FileDir, self.Name + '.ModuleHashPair')
        ModuleHashPairList = []
        if os.path.exists(ModuleHashPair):
            with open(ModuleHashPair, 'r') as f:
                ModuleHashPairList = json.load(f)
        if not (PreMakeHashStr, MakeHashStr) in set(map(tuple, ModuleHashPairList)):
            ModuleHashPairList.insert(0, (PreMakeHashStr, MakeHashStr))
            with open(ModuleHashPair, 'w') as f:
                json.dump(ModuleHashPairList, f, indent=2)
        if not self.OutputFile:
            Ma = self.BuildDatabase[self.MetaFile, self.Arch, self.BuildTarget, self.ToolChain]
            self.OutputFile = Ma.Binaries
        for File in self.OutputFile:
            if File.startswith(os.path.abspath(self.FfsOutputDir) + os.sep):
                self.CacheCopyFile(CacheFfsDir, self.FfsOutputDir, File)
            elif self.Name + '.autogen.hash.' in File or self.Name + '.autogen.hashchain.' in File or self.Name + '.hash.' in File or (self.Name + '.hashchain.' in File) or (self.Name + '.PreMakeHashFileList.' in File) or (self.Name + '.MakeHashFileList.' in File):
                self.CacheCopyFile(FileDir, self.BuildDir, File)
            else:
                self.CacheCopyFile(CacheFileDir, self.BuildDir, File)

    @cached_class_function
    def CreateMakeFile(self, CreateLibraryMakeFile=True, GenFfsList=[]):
        if False:
            print('Hello World!')

        def CreateTimeStamp():
            if False:
                for i in range(10):
                    print('nop')
            FileSet = {self.MetaFile.Path}
            for SourceFile in self.Module.Sources:
                FileSet.add(SourceFile.Path)
            for Lib in self.DependentLibraryList:
                FileSet.add(Lib.MetaFile.Path)
            for f in self.AutoGenDepSet:
                FileSet.add(f.Path)
            if os.path.exists(self.TimeStampPath):
                os.remove(self.TimeStampPath)
            SaveFileOnChange(self.TimeStampPath, '\n'.join(FileSet), False)
        if self.IsBinaryModule:
            return
        self.GenFfsList = GenFfsList
        if not self.IsLibrary and CreateLibraryMakeFile:
            for LibraryAutoGen in self.LibraryAutoGenList:
                LibraryAutoGen.CreateMakeFile()
        if self.CanSkip():
            return
        if len(self.CustomMakefile) == 0:
            Makefile = GenMake.ModuleMakefile(self)
        else:
            Makefile = GenMake.CustomMakefile(self)
        if Makefile.Generate():
            EdkLogger.debug(EdkLogger.DEBUG_9, 'Generated makefile for module %s [%s]' % (self.Name, self.Arch))
        else:
            EdkLogger.debug(EdkLogger.DEBUG_9, 'Skipped the generation of makefile for module %s [%s]' % (self.Name, self.Arch))
        CreateTimeStamp()
        MakefileType = Makefile._FileType
        MakefileName = Makefile._FILE_NAME_[MakefileType]
        MakefilePath = os.path.join(self.MakeFileDir, MakefileName)
        FilePath = path.join(self.BuildDir, self.Name + '.makefile')
        SaveFileOnChange(FilePath, MakefilePath, False)

    def CopyBinaryFiles(self):
        if False:
            while True:
                i = 10
        for File in self.Module.Binaries:
            SrcPath = File.Path
            DstPath = os.path.join(self.OutputDir, os.path.basename(SrcPath))
            CopyLongFilePath(SrcPath, DstPath)

    def CreateCodeFile(self, CreateLibraryCodeFile=True):
        if False:
            print('Hello World!')
        if self.IsCodeFileCreated:
            return
        if self.IsBinaryModule and self.PcdIsDriver != '':
            CreatePcdDatabaseCode(self, TemplateString(), TemplateString())
            return
        if self.IsBinaryModule:
            if self.IsLibrary:
                self.CopyBinaryFiles()
            return
        if not self.IsLibrary and CreateLibraryCodeFile:
            for LibraryAutoGen in self.LibraryAutoGenList:
                LibraryAutoGen.CreateCodeFile()
        self.LibraryAutoGenList
        AutoGenList = []
        IgoredAutoGenList = []
        for File in self.AutoGenFileList:
            if GenC.Generate(File.Path, self.AutoGenFileList[File], File.IsBinary):
                AutoGenList.append(str(File))
            else:
                IgoredAutoGenList.append(str(File))
        for ModuleType in self.DepexList:
            if len(self.DepexList[ModuleType]) == 0 or ModuleType == SUP_MODULE_USER_DEFINED or ModuleType == SUP_MODULE_HOST_APPLICATION:
                continue
            Dpx = GenDepex.DependencyExpression(self.DepexList[ModuleType], ModuleType, True)
            DpxFile = gAutoGenDepexFileName % {'module_name': self.Name}
            if len(Dpx.PostfixNotation) != 0:
                self.DepexGenerated = True
            if Dpx.Generate(path.join(self.OutputDir, DpxFile)):
                AutoGenList.append(str(DpxFile))
            else:
                IgoredAutoGenList.append(str(DpxFile))
        if IgoredAutoGenList == []:
            EdkLogger.debug(EdkLogger.DEBUG_9, 'Generated [%s] files for module %s [%s]' % (' '.join(AutoGenList), self.Name, self.Arch))
        elif AutoGenList == []:
            EdkLogger.debug(EdkLogger.DEBUG_9, 'Skipped the generation of [%s] files for module %s [%s]' % (' '.join(IgoredAutoGenList), self.Name, self.Arch))
        else:
            EdkLogger.debug(EdkLogger.DEBUG_9, 'Generated [%s] (skipped %s) files for module %s [%s]' % (' '.join(AutoGenList), ' '.join(IgoredAutoGenList), self.Name, self.Arch))
        self.IsCodeFileCreated = True
        return AutoGenList

    @cached_property
    def LibraryAutoGenList(self):
        if False:
            i = 10
            return i + 15
        RetVal = []
        for Library in self.DependentLibraryList:
            La = ModuleAutoGen(self.Workspace, Library.MetaFile, self.BuildTarget, self.ToolChain, self.Arch, self.PlatformInfo.MetaFile, self.DataPipe)
            La.IsLibrary = True
            if La not in RetVal:
                RetVal.append(La)
                for Lib in La.CodaTargetList:
                    self._ApplyBuildRule(Lib.Target, TAB_UNKNOWN_FILE)
        return RetVal

    def GenCMakeHash(self):
        if False:
            for i in range(10):
                print('nop')
        DependencyFileSet = set()
        if self.AutoGenFileList:
            for File in set(self.AutoGenFileList):
                DependencyFileSet.add(File)
        abspath = path.join(self.BuildDir, self.Name + '.makefile')
        try:
            with open(LongFilePath(abspath), 'r') as fd:
                lines = fd.readlines()
        except Exception as e:
            EdkLogger.error('build', FILE_NOT_FOUND, "%s doesn't exist" % abspath, ExtraData=str(e), RaiseError=False)
        if lines:
            DependencyFileSet.update(lines)
        FileList = []
        m = hashlib.md5()
        for File in sorted(DependencyFileSet, key=lambda x: str(x)):
            if not path.exists(LongFilePath(str(File))):
                EdkLogger.quiet('[cache warning]: header file %s is missing for module: %s[%s]' % (File, self.MetaFile.Path, self.Arch))
                continue
            with open(LongFilePath(str(File)), 'rb') as f:
                Content = f.read()
            m.update(Content)
            FileList.append((str(File), hashlib.md5(Content).hexdigest()))
        HashChainFile = path.join(self.BuildDir, self.Name + '.autogen.hashchain.' + m.hexdigest())
        GlobalData.gCMakeHashFile[self.MetaFile.Path, self.Arch] = HashChainFile
        try:
            with open(LongFilePath(HashChainFile), 'w') as f:
                json.dump(FileList, f, indent=2)
        except:
            EdkLogger.quiet('[cache warning]: fail to save hashchain file:%s' % HashChainFile)
            return False

    def GenModuleHash(self):
        if False:
            i = 10
            return i + 15
        DependencyFileSet = set()
        DependencyFileSet.add(self.MetaFile.Path)
        if self.SourceFileList:
            for File in set(self.SourceFileList):
                DependencyFileSet.add(File.Path)
        abspath = path.join(self.BuildDir, 'deps.txt')
        rt = None
        try:
            with open(LongFilePath(abspath), 'r') as fd:
                lines = fd.readlines()
                if lines:
                    rt = set([item.lstrip().strip('\n') for item in lines if item.strip('\n').endswith('.h')])
        except Exception as e:
            EdkLogger.error('build', FILE_NOT_FOUND, "%s doesn't exist" % abspath, ExtraData=str(e), RaiseError=False)
        if rt:
            DependencyFileSet.update(rt)
        FileList = []
        m = hashlib.md5()
        BuildDirStr = path.abspath(self.BuildDir).lower()
        for File in sorted(DependencyFileSet, key=lambda x: str(x)):
            if BuildDirStr in path.abspath(File).lower():
                continue
            if not path.exists(LongFilePath(File)):
                EdkLogger.quiet('[cache warning]: header file %s is missing for module: %s[%s]' % (File, self.MetaFile.Path, self.Arch))
                continue
            with open(LongFilePath(File), 'rb') as f:
                Content = f.read()
            m.update(Content)
            FileList.append((File, hashlib.md5(Content).hexdigest()))
        HashChainFile = path.join(self.BuildDir, self.Name + '.hashchain.' + m.hexdigest())
        GlobalData.gModuleHashFile[self.MetaFile.Path, self.Arch] = HashChainFile
        try:
            with open(LongFilePath(HashChainFile), 'w') as f:
                json.dump(FileList, f, indent=2)
        except:
            EdkLogger.quiet('[cache warning]: fail to save hashchain file:%s' % HashChainFile)
            return False

    def GenPreMakefileHashList(self):
        if False:
            print('Hello World!')
        if self.IsBinaryModule:
            return
        FileList = []
        m = hashlib.md5()
        HashFile = GlobalData.gPlatformHashFile
        if path.exists(LongFilePath(HashFile)):
            FileList.append(HashFile)
            m.update(HashFile.encode('utf-8'))
        else:
            EdkLogger.quiet('[cache warning]: No Platform HashFile: %s' % HashFile)
        if self.DependentPackageList:
            for Pkg in sorted(self.DependentPackageList, key=lambda x: x.PackageName):
                if not (Pkg.PackageName, Pkg.Arch) in GlobalData.gPackageHashFile:
                    EdkLogger.quiet('[cache warning]:No Package %s for module %s[%s]' % (Pkg.PackageName, self.MetaFile.Path, self.Arch))
                    continue
                HashFile = GlobalData.gPackageHashFile[Pkg.PackageName, Pkg.Arch]
                if path.exists(LongFilePath(HashFile)):
                    FileList.append(HashFile)
                    m.update(HashFile.encode('utf-8'))
                else:
                    EdkLogger.quiet('[cache warning]:No Package HashFile: %s' % HashFile)
        if (self.MetaFile.Path, self.Arch) in GlobalData.gModuleHashFile:
            HashFile = GlobalData.gModuleHashFile[self.MetaFile.Path, self.Arch]
        else:
            EdkLogger.quiet('[cache error]:No ModuleHashFile for module: %s[%s]' % (self.MetaFile.Path, self.Arch))
        if path.exists(LongFilePath(HashFile)):
            FileList.append(HashFile)
            m.update(HashFile.encode('utf-8'))
        else:
            EdkLogger.quiet('[cache warning]:No Module HashFile: %s' % HashFile)
        if self.LibraryAutoGenList:
            for Lib in sorted(self.LibraryAutoGenList, key=lambda x: x.MetaFile.Path):
                if (Lib.MetaFile.Path, Lib.Arch) in GlobalData.gModuleHashFile:
                    HashFile = GlobalData.gModuleHashFile[Lib.MetaFile.Path, Lib.Arch]
                else:
                    EdkLogger.quiet('[cache error]:No ModuleHashFile for lib: %s[%s]' % (Lib.MetaFile.Path, Lib.Arch))
                if path.exists(LongFilePath(HashFile)):
                    FileList.append(HashFile)
                    m.update(HashFile.encode('utf-8'))
                else:
                    EdkLogger.quiet('[cache warning]:No Lib HashFile: %s' % HashFile)
        FilePath = path.join(self.BuildDir, self.Name + '.PreMakeHashFileList.' + m.hexdigest())
        try:
            with open(LongFilePath(FilePath), 'w') as f:
                json.dump(FileList, f, indent=0)
        except:
            EdkLogger.quiet('[cache warning]: fail to save PreMake HashFileList: %s' % FilePath)

    def GenMakefileHashList(self):
        if False:
            i = 10
            return i + 15
        if self.IsBinaryModule:
            return
        FileList = []
        m = hashlib.md5()
        HashFile = GlobalData.gCMakeHashFile[self.MetaFile.Path, self.Arch]
        if path.exists(LongFilePath(HashFile)):
            FileList.append(HashFile)
            m.update(HashFile.encode('utf-8'))
        else:
            EdkLogger.quiet('[cache warning]:No AutoGen HashFile: %s' % HashFile)
        if (self.MetaFile.Path, self.Arch) in GlobalData.gModuleHashFile:
            HashFile = GlobalData.gModuleHashFile[self.MetaFile.Path, self.Arch]
        else:
            EdkLogger.quiet('[cache error]:No ModuleHashFile for module: %s[%s]' % (self.MetaFile.Path, self.Arch))
        if path.exists(LongFilePath(HashFile)):
            FileList.append(HashFile)
            m.update(HashFile.encode('utf-8'))
        else:
            EdkLogger.quiet('[cache warning]:No Module HashFile: %s' % HashFile)
        if self.LibraryAutoGenList:
            for Lib in sorted(self.LibraryAutoGenList, key=lambda x: x.MetaFile.Path):
                if (Lib.MetaFile.Path, Lib.Arch) in GlobalData.gModuleHashFile:
                    HashFile = GlobalData.gModuleHashFile[Lib.MetaFile.Path, Lib.Arch]
                else:
                    EdkLogger.quiet('[cache error]:No ModuleHashFile for lib: %s[%s]' % (Lib.MetaFile.Path, Lib.Arch))
                if path.exists(LongFilePath(HashFile)):
                    FileList.append(HashFile)
                    m.update(HashFile.encode('utf-8'))
                else:
                    EdkLogger.quiet('[cache warning]:No Lib HashFile: %s' % HashFile)
        FilePath = path.join(self.BuildDir, self.Name + '.MakeHashFileList.' + m.hexdigest())
        try:
            with open(LongFilePath(FilePath), 'w') as f:
                json.dump(FileList, f, indent=0)
        except:
            EdkLogger.quiet('[cache warning]: fail to save Make HashFileList: %s' % FilePath)

    def CheckHashChainFile(self, HashChainFile):
        if False:
            print('Hello World!')
        HashStr = HashChainFile.split('.')[-1]
        if len(HashStr) != 32:
            EdkLogger.quiet('[cache error]: wrong format HashChainFile:%s' % File)
            return False
        try:
            with open(LongFilePath(HashChainFile), 'r') as f:
                HashChainList = json.load(f)
        except:
            EdkLogger.quiet('[cache error]: fail to load HashChainFile: %s' % HashChainFile)
            return False
        for (idx, (SrcFile, SrcHash)) in enumerate(HashChainList):
            if SrcFile in GlobalData.gFileHashDict:
                DestHash = GlobalData.gFileHashDict[SrcFile]
            else:
                try:
                    with open(LongFilePath(SrcFile), 'rb') as f:
                        Content = f.read()
                        DestHash = hashlib.md5(Content).hexdigest()
                        GlobalData.gFileHashDict[SrcFile] = DestHash
                except IOError as X:
                    GlobalData.gFileHashDict[SrcFile] = 0
                    EdkLogger.quiet('[cache insight]: first cache miss file in %s is %s' % (HashChainFile, SrcFile))
                    return False
            if SrcHash != DestHash:
                EdkLogger.quiet('[cache insight]: first cache miss file in %s is %s' % (HashChainFile, SrcFile))
                return False
        return True

    def CanSkipbyMakeCache(self):
        if False:
            print('Hello World!')
        if not GlobalData.gBinCacheSource:
            return False
        if (self.MetaFile.Path, self.Arch) in GlobalData.gModuleMakeCacheStatus:
            return GlobalData.gModuleMakeCacheStatus[self.MetaFile.Path, self.Arch]
        if self.IsBinaryModule:
            print('[cache miss]: MakeCache: Skip BinaryModule:', self.MetaFile.Path, self.Arch)
            GlobalData.gModuleMakeCacheStatus[self.MetaFile.Path, self.Arch] = False
            return False
        for f_ext in self.SourceFileList:
            if '.inc' in str(f_ext):
                print("[cache miss]: MakeCache: Skip '.inc' File:", self.MetaFile.Path, self.Arch)
                GlobalData.gModuleMakeCacheStatus[self.MetaFile.Path, self.Arch] = False
                return False
        ModuleCacheDir = path.join(GlobalData.gBinCacheSource, self.PlatformInfo.OutputDir, self.BuildTarget + '_' + self.ToolChain, self.Arch, self.SourceDir, self.MetaFile.BaseName)
        FfsDir = path.join(GlobalData.gBinCacheSource, self.PlatformInfo.OutputDir, self.BuildTarget + '_' + self.ToolChain, TAB_FV_DIRECTORY, 'Ffs', self.Guid + self.Name)
        ModuleHashPairList = []
        ModuleHashPair = path.join(ModuleCacheDir, self.Name + '.ModuleHashPair')
        try:
            with open(LongFilePath(ModuleHashPair), 'r') as f:
                ModuleHashPairList = json.load(f)
        except:
            GlobalData.gModuleMakeCacheStatus[self.MetaFile.Path, self.Arch] = False
            EdkLogger.quiet('[cache warning]: fail to load ModuleHashPair file: %s' % ModuleHashPair)
            print('[cache miss]: MakeCache:', self.MetaFile.Path, self.Arch)
            return False
        for (idx, (PreMakefileHash, MakeHash)) in enumerate(ModuleHashPairList):
            SourceHashDir = path.join(ModuleCacheDir, MakeHash)
            SourceFfsHashDir = path.join(FfsDir, MakeHash)
            PreMakeHashFileList_FilePah = path.join(ModuleCacheDir, self.Name + '.PreMakeHashFileList.' + PreMakefileHash)
            MakeHashFileList_FilePah = path.join(ModuleCacheDir, self.Name + '.MakeHashFileList.' + MakeHash)
            try:
                with open(LongFilePath(MakeHashFileList_FilePah), 'r') as f:
                    MakeHashFileList = json.load(f)
            except:
                EdkLogger.quiet('[cache error]: fail to load MakeHashFileList file: %s' % MakeHashFileList_FilePah)
                continue
            HashMiss = False
            for HashChainFile in MakeHashFileList:
                HashChainStatus = None
                if HashChainFile in GlobalData.gHashChainStatus:
                    HashChainStatus = GlobalData.gHashChainStatus[HashChainFile]
                if HashChainStatus == False:
                    HashMiss = True
                    break
                elif HashChainStatus == True:
                    continue
                RelativePath = os.path.relpath(HashChainFile, self.WorkspaceDir)
                NewFilePath = os.path.join(GlobalData.gBinCacheSource, RelativePath)
                if self.CheckHashChainFile(NewFilePath):
                    GlobalData.gHashChainStatus[HashChainFile] = True
                    if self.Name + '.hashchain.' in HashChainFile:
                        GlobalData.gModuleHashFile[self.MetaFile.Path, self.Arch] = HashChainFile
                else:
                    GlobalData.gHashChainStatus[HashChainFile] = False
                    HashMiss = True
                    break
            if HashMiss:
                continue
            for (root, dir, files) in os.walk(SourceHashDir):
                for f in files:
                    File = path.join(root, f)
                    self.CacheCopyFile(self.BuildDir, SourceHashDir, File)
            if os.path.exists(SourceFfsHashDir):
                for (root, dir, files) in os.walk(SourceFfsHashDir):
                    for f in files:
                        File = path.join(root, f)
                        self.CacheCopyFile(self.FfsOutputDir, SourceFfsHashDir, File)
            if self.Name == 'PcdPeim' or self.Name == 'PcdDxe':
                CreatePcdDatabaseCode(self, TemplateString(), TemplateString())
            print('[cache hit]: MakeCache:', self.MetaFile.Path, self.Arch)
            GlobalData.gModuleMakeCacheStatus[self.MetaFile.Path, self.Arch] = True
            return True
        print('[cache miss]: MakeCache:', self.MetaFile.Path, self.Arch)
        GlobalData.gModuleMakeCacheStatus[self.MetaFile.Path, self.Arch] = False
        return False

    def CanSkipbyPreMakeCache(self):
        if False:
            for i in range(10):
                print('nop')
        if not GlobalData.gUseHashCache or GlobalData.gBinCacheDest:
            return False
        if (self.MetaFile.Path, self.Arch) in GlobalData.gModulePreMakeCacheStatus:
            return GlobalData.gModulePreMakeCacheStatus[self.MetaFile.Path, self.Arch]
        if self.IsBinaryModule:
            print('[cache miss]: PreMakeCache: Skip BinaryModule:', self.MetaFile.Path, self.Arch)
            GlobalData.gModulePreMakeCacheStatus[self.MetaFile.Path, self.Arch] = False
            return False
        for f_ext in self.SourceFileList:
            if '.inc' in str(f_ext):
                print("[cache miss]: PreMakeCache: Skip '.inc' File:", self.MetaFile.Path, self.Arch)
                GlobalData.gModulePreMakeCacheStatus[self.MetaFile.Path, self.Arch] = False
                return False
        if not GlobalData.gBinCacheSource:
            Files = [path.join(self.BuildDir, f) for f in os.listdir(self.BuildDir) if path.isfile(path.join(self.BuildDir, f))]
            PreMakeHashFileList_FilePah = None
            MakeTimeStamp = 0
            for File in Files:
                if '.PreMakeHashFileList.' in File:
                    FileTimeStamp = os.stat(path.join(self.BuildDir, File))[8]
                    if FileTimeStamp > MakeTimeStamp:
                        MakeTimeStamp = FileTimeStamp
                        PreMakeHashFileList_FilePah = File
            if not PreMakeHashFileList_FilePah:
                GlobalData.gModulePreMakeCacheStatus[self.MetaFile.Path, self.Arch] = False
                return False
            try:
                with open(LongFilePath(PreMakeHashFileList_FilePah), 'r') as f:
                    PreMakeHashFileList = json.load(f)
            except:
                EdkLogger.quiet('[cache error]: fail to load PreMakeHashFileList file: %s' % PreMakeHashFileList_FilePah)
                print('[cache miss]: PreMakeCache:', self.MetaFile.Path, self.Arch)
                GlobalData.gModulePreMakeCacheStatus[self.MetaFile.Path, self.Arch] = False
                return False
            HashMiss = False
            for HashChainFile in PreMakeHashFileList:
                HashChainStatus = None
                if HashChainFile in GlobalData.gHashChainStatus:
                    HashChainStatus = GlobalData.gHashChainStatus[HashChainFile]
                if HashChainStatus == False:
                    HashMiss = True
                    break
                elif HashChainStatus == True:
                    continue
                if self.CheckHashChainFile(HashChainFile):
                    GlobalData.gHashChainStatus[HashChainFile] = True
                    if self.Name + '.hashchain.' in HashChainFile:
                        GlobalData.gModuleHashFile[self.MetaFile.Path, self.Arch] = HashChainFile
                else:
                    GlobalData.gHashChainStatus[HashChainFile] = False
                    HashMiss = True
                    break
            if HashMiss:
                print('[cache miss]: PreMakeCache:', self.MetaFile.Path, self.Arch)
                GlobalData.gModulePreMakeCacheStatus[self.MetaFile.Path, self.Arch] = False
                return False
            else:
                print('[cache hit]: PreMakeCache:', self.MetaFile.Path, self.Arch)
                GlobalData.gModulePreMakeCacheStatus[self.MetaFile.Path, self.Arch] = True
                return True
        ModuleCacheDir = path.join(GlobalData.gBinCacheSource, self.PlatformInfo.OutputDir, self.BuildTarget + '_' + self.ToolChain, self.Arch, self.SourceDir, self.MetaFile.BaseName)
        FfsDir = path.join(GlobalData.gBinCacheSource, self.PlatformInfo.OutputDir, self.BuildTarget + '_' + self.ToolChain, TAB_FV_DIRECTORY, 'Ffs', self.Guid + self.Name)
        ModuleHashPairList = []
        ModuleHashPair = path.join(ModuleCacheDir, self.Name + '.ModuleHashPair')
        try:
            with open(LongFilePath(ModuleHashPair), 'r') as f:
                ModuleHashPairList = json.load(f)
        except:
            GlobalData.gModulePreMakeCacheStatus[self.MetaFile.Path, self.Arch] = False
            EdkLogger.quiet('[cache warning]: fail to load ModuleHashPair file: %s' % ModuleHashPair)
            print('[cache miss]: PreMakeCache:', self.MetaFile.Path, self.Arch)
            return False
        for (idx, (PreMakefileHash, MakeHash)) in enumerate(ModuleHashPairList):
            SourceHashDir = path.join(ModuleCacheDir, MakeHash)
            SourceFfsHashDir = path.join(FfsDir, MakeHash)
            PreMakeHashFileList_FilePah = path.join(ModuleCacheDir, self.Name + '.PreMakeHashFileList.' + PreMakefileHash)
            MakeHashFileList_FilePah = path.join(ModuleCacheDir, self.Name + '.MakeHashFileList.' + MakeHash)
            try:
                with open(LongFilePath(PreMakeHashFileList_FilePah), 'r') as f:
                    PreMakeHashFileList = json.load(f)
            except:
                EdkLogger.quiet('[cache error]: fail to load PreMakeHashFileList file: %s' % PreMakeHashFileList_FilePah)
                continue
            HashMiss = False
            for HashChainFile in PreMakeHashFileList:
                HashChainStatus = None
                if HashChainFile in GlobalData.gHashChainStatus:
                    HashChainStatus = GlobalData.gHashChainStatus[HashChainFile]
                if HashChainStatus == False:
                    HashMiss = True
                    break
                elif HashChainStatus == True:
                    continue
                RelativePath = os.path.relpath(HashChainFile, self.WorkspaceDir)
                NewFilePath = os.path.join(GlobalData.gBinCacheSource, RelativePath)
                if self.CheckHashChainFile(NewFilePath):
                    GlobalData.gHashChainStatus[HashChainFile] = True
                else:
                    GlobalData.gHashChainStatus[HashChainFile] = False
                    HashMiss = True
                    break
            if HashMiss:
                continue
            for (root, dir, files) in os.walk(SourceHashDir):
                for f in files:
                    File = path.join(root, f)
                    self.CacheCopyFile(self.BuildDir, SourceHashDir, File)
            if os.path.exists(SourceFfsHashDir):
                for (root, dir, files) in os.walk(SourceFfsHashDir):
                    for f in files:
                        File = path.join(root, f)
                        self.CacheCopyFile(self.FfsOutputDir, SourceFfsHashDir, File)
            if self.Name == 'PcdPeim' or self.Name == 'PcdDxe':
                CreatePcdDatabaseCode(self, TemplateString(), TemplateString())
            print('[cache hit]: PreMakeCache:', self.MetaFile.Path, self.Arch)
            GlobalData.gModulePreMakeCacheStatus[self.MetaFile.Path, self.Arch] = True
            return True
        print('[cache miss]: PreMakeCache:', self.MetaFile.Path, self.Arch)
        GlobalData.gModulePreMakeCacheStatus[self.MetaFile.Path, self.Arch] = False
        return False

    def CanSkipbyCache(self, gHitSet):
        if False:
            for i in range(10):
                print('nop')
        if not GlobalData.gBinCacheSource:
            return False
        if self in gHitSet:
            return True
        return False

    def CanSkip(self):
        if False:
            i = 10
            return i + 15
        if GlobalData.gUseHashCache or GlobalData.gBinCacheDest or GlobalData.gBinCacheSource:
            return False
        if self.MakeFileDir in GlobalData.gSikpAutoGenCache:
            return True
        if not os.path.exists(self.TimeStampPath):
            return False
        DstTimeStamp = os.stat(self.TimeStampPath)[8]
        SrcTimeStamp = self.Workspace._SrcTimeStamp
        if SrcTimeStamp > DstTimeStamp:
            return False
        with open(self.TimeStampPath, 'r') as f:
            for source in f:
                source = source.rstrip('\n')
                if not os.path.exists(source):
                    return False
                if source not in ModuleAutoGen.TimeDict:
                    ModuleAutoGen.TimeDict[source] = os.stat(source)[8]
                if ModuleAutoGen.TimeDict[source] > DstTimeStamp:
                    return False
        GlobalData.gSikpAutoGenCache.add(self.MakeFileDir)
        return True

    @cached_property
    def TimeStampPath(self):
        if False:
            return 10
        return os.path.join(self.MakeFileDir, 'AutoGenTimeStamp')