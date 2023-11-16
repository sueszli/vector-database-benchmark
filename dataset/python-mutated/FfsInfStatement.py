from __future__ import absolute_import
from . import Rule
import Common.LongFilePathOs as os
from io import BytesIO
from struct import *
from .GenFdsGlobalVariable import GenFdsGlobalVariable
from .Ffs import SectionSuffix, FdfFvFileTypeToFileType
import subprocess
import sys
from . import Section
from . import RuleSimpleFile
from . import RuleComplexFile
from CommonDataClass.FdfClass import FfsInfStatementClassObject
from Common.MultipleWorkspace import MultipleWorkspace as mws
from Common.DataType import SUP_MODULE_USER_DEFINED
from Common.DataType import SUP_MODULE_HOST_APPLICATION
from Common.StringUtils import *
from Common.Misc import PathClass
from Common.Misc import GuidStructureByteArrayToGuidString
from Common.Misc import ProcessDuplicatedInf
from Common.Misc import GetVariableOffset
from Common import EdkLogger
from Common.BuildToolError import *
from .GuidSection import GuidSection
from .FvImageSection import FvImageSection
from Common.Misc import PeImageClass
from AutoGen.GenDepex import DependencyExpression
from PatchPcdValue.PatchPcdValue import PatchBinaryFile
from Common.LongFilePathSupport import CopyLongFilePath
from Common.LongFilePathSupport import OpenLongFilePath as open
import Common.GlobalData as GlobalData
from .DepexSection import DepexSection
from Common.Misc import SaveFileOnChange
from Common.Expression import *
from Common.DataType import *

class FfsInfStatement(FfsInfStatementClassObject):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        FfsInfStatementClassObject.__init__(self)
        self.TargetOverrideList = []
        self.ShadowFromInfFile = None
        self.KeepRelocFromRule = None
        self.InDsc = True
        self.OptRomDefs = {}
        self.PiSpecVersion = '0x00000000'
        self.InfModule = None
        self.FinalTargetSuffixMap = {}
        self.CurrentLineNum = None
        self.CurrentLineContent = None
        self.FileName = None
        self.InfFileName = None
        self.OverrideGuid = None
        self.PatchedBinFile = ''
        self.MacroDict = {}
        self.Depex = False

    def GetFinalTargetSuffixMap(self):
        if False:
            i = 10
            return i + 15
        if not self.InfModule or not self.CurrentArch:
            return []
        if not self.FinalTargetSuffixMap:
            FinalBuildTargetList = GenFdsGlobalVariable.GetModuleCodaTargetList(self.InfModule, self.CurrentArch)
            for File in FinalBuildTargetList:
                self.FinalTargetSuffixMap.setdefault(os.path.splitext(File)[1], []).append(File)
            if '.depex' not in self.FinalTargetSuffixMap and self.InfModule.ModuleType != SUP_MODULE_USER_DEFINED and (self.InfModule.ModuleType != SUP_MODULE_HOST_APPLICATION) and (not self.InfModule.DxsFile) and (not self.InfModule.LibraryClass):
                ModuleType = self.InfModule.ModuleType
                PlatformDataBase = GenFdsGlobalVariable.WorkSpace.BuildObject[GenFdsGlobalVariable.ActivePlatform, self.CurrentArch, GenFdsGlobalVariable.TargetName, GenFdsGlobalVariable.ToolChainTag]
                if ModuleType != SUP_MODULE_USER_DEFINED and ModuleType != SUP_MODULE_HOST_APPLICATION:
                    for LibraryClass in PlatformDataBase.LibraryClasses.GetKeys():
                        if LibraryClass.startswith('NULL') and PlatformDataBase.LibraryClasses[LibraryClass, ModuleType]:
                            self.InfModule.LibraryClasses[LibraryClass] = PlatformDataBase.LibraryClasses[LibraryClass, ModuleType]
                StrModule = str(self.InfModule)
                PlatformModule = None
                if StrModule in PlatformDataBase.Modules:
                    PlatformModule = PlatformDataBase.Modules[StrModule]
                    for LibraryClass in PlatformModule.LibraryClasses:
                        if LibraryClass.startswith('NULL'):
                            self.InfModule.LibraryClasses[LibraryClass] = PlatformModule.LibraryClasses[LibraryClass]
                DependencyList = [self.InfModule]
                LibraryInstance = {}
                DepexList = []
                while len(DependencyList) > 0:
                    Module = DependencyList.pop(0)
                    if not Module:
                        continue
                    for Dep in Module.Depex[self.CurrentArch, ModuleType]:
                        if DepexList != []:
                            DepexList.append('AND')
                        DepexList.append('(')
                        DepexList.extend(Dep)
                        if DepexList[-1] == 'END':
                            DepexList.pop()
                        DepexList.append(')')
                    if 'BEFORE' in DepexList or 'AFTER' in DepexList:
                        break
                    for LibName in Module.LibraryClasses:
                        if LibName in LibraryInstance:
                            continue
                        if PlatformModule and LibName in PlatformModule.LibraryClasses:
                            LibraryPath = PlatformModule.LibraryClasses[LibName]
                        else:
                            LibraryPath = PlatformDataBase.LibraryClasses[LibName, ModuleType]
                        if not LibraryPath:
                            LibraryPath = Module.LibraryClasses[LibName]
                        if not LibraryPath:
                            continue
                        LibraryModule = GenFdsGlobalVariable.WorkSpace.BuildObject[LibraryPath, self.CurrentArch, GenFdsGlobalVariable.TargetName, GenFdsGlobalVariable.ToolChainTag]
                        LibraryInstance[LibName] = LibraryModule
                        DependencyList.append(LibraryModule)
                if DepexList:
                    Dpx = DependencyExpression(DepexList, ModuleType, True)
                    if len(Dpx.PostfixNotation) != 0:
                        self.FinalTargetSuffixMap['.depex'] = [os.path.join(self.EfiOutputPath, self.BaseName) + '.depex']
        return self.FinalTargetSuffixMap

    def __InfParse__(self, Dict=None, IsGenFfs=False):
        if False:
            return 10
        GenFdsGlobalVariable.VerboseLogger(' Begine parsing INf file : %s' % self.InfFileName)
        self.InfFileName = self.InfFileName.replace('$(WORKSPACE)', '')
        if len(self.InfFileName) > 1 and self.InfFileName[0] == '\\' and (self.InfFileName[1] == '\\'):
            pass
        elif self.InfFileName[0] == '\\' or self.InfFileName[0] == '/':
            self.InfFileName = self.InfFileName[1:]
        if self.InfFileName.find('$') == -1:
            InfPath = NormPath(self.InfFileName)
            if not os.path.exists(InfPath):
                InfPath = GenFdsGlobalVariable.ReplaceWorkspaceMacro(InfPath)
                if not os.path.exists(InfPath):
                    EdkLogger.error('GenFds', GENFDS_ERROR, 'Non-existant Module %s !' % self.InfFileName)
        self.CurrentArch = self.GetCurrentArch()
        PathClassObj = PathClass(self.InfFileName, GenFdsGlobalVariable.WorkSpaceDir)
        (ErrorCode, ErrorInfo) = PathClassObj.Validate('.inf')
        if ErrorCode != 0:
            EdkLogger.error('GenFds', ErrorCode, ExtraData=ErrorInfo)
        InfLowerPath = str(PathClassObj).lower()
        if self.OverrideGuid:
            PathClassObj = ProcessDuplicatedInf(PathClassObj, self.OverrideGuid, GenFdsGlobalVariable.WorkSpaceDir)
        if self.CurrentArch is not None:
            Inf = GenFdsGlobalVariable.WorkSpace.BuildObject[PathClassObj, self.CurrentArch, GenFdsGlobalVariable.TargetName, GenFdsGlobalVariable.ToolChainTag]
            self.BaseName = Inf.BaseName
            self.ModuleGuid = Inf.Guid
            self.ModuleType = Inf.ModuleType
            if Inf.Specification is not None and 'PI_SPECIFICATION_VERSION' in Inf.Specification:
                self.PiSpecVersion = Inf.Specification['PI_SPECIFICATION_VERSION']
            if Inf.AutoGenVersion < 65541:
                self.ModuleType = Inf.ComponentType
            self.VersionString = Inf.Version
            self.BinFileList = Inf.Binaries
            self.SourceFileList = Inf.Sources
            if self.KeepReloc is None and Inf.Shadow:
                self.ShadowFromInfFile = Inf.Shadow
        else:
            Inf = GenFdsGlobalVariable.WorkSpace.BuildObject[PathClassObj, TAB_COMMON, GenFdsGlobalVariable.TargetName, GenFdsGlobalVariable.ToolChainTag]
            self.BaseName = Inf.BaseName
            self.ModuleGuid = Inf.Guid
            self.ModuleType = Inf.ModuleType
            if Inf.Specification is not None and 'PI_SPECIFICATION_VERSION' in Inf.Specification:
                self.PiSpecVersion = Inf.Specification['PI_SPECIFICATION_VERSION']
            self.VersionString = Inf.Version
            self.BinFileList = Inf.Binaries
            self.SourceFileList = Inf.Sources
            if self.BinFileList == []:
                EdkLogger.error('GenFds', GENFDS_ERROR, 'INF %s specified in FDF could not be found in build ARCH %s!' % (self.InfFileName, GenFdsGlobalVariable.ArchList))
        if self.OverrideGuid:
            self.ModuleGuid = self.OverrideGuid
        if len(self.SourceFileList) != 0 and (not self.InDsc):
            EdkLogger.warn('GenFds', GENFDS_ERROR, 'Module %s NOT found in DSC file; Is it really a binary module?' % self.InfFileName)
        if self.ModuleType == SUP_MODULE_SMM_CORE and int(self.PiSpecVersion, 16) < 65546:
            EdkLogger.error('GenFds', FORMAT_NOT_SUPPORTED, "SMM_CORE module type can't be used in the module with PI_SPECIFICATION_VERSION less than 0x0001000A", File=self.InfFileName)
        if self.ModuleType == SUP_MODULE_MM_CORE_STANDALONE and int(self.PiSpecVersion, 16) < 65586:
            EdkLogger.error('GenFds', FORMAT_NOT_SUPPORTED, "MM_CORE_STANDALONE module type can't be used in the module with PI_SPECIFICATION_VERSION less than 0x00010032", File=self.InfFileName)
        if Inf._Defs is not None and len(Inf._Defs) > 0:
            self.OptRomDefs.update(Inf._Defs)
        self.PatchPcds = []
        InfPcds = Inf.Pcds
        Platform = GenFdsGlobalVariable.WorkSpace.BuildObject[GenFdsGlobalVariable.ActivePlatform, self.CurrentArch, GenFdsGlobalVariable.TargetName, GenFdsGlobalVariable.ToolChainTag]
        FdfPcdDict = GenFdsGlobalVariable.FdfParser.Profile.PcdDict
        PlatformPcds = Platform.Pcds
        DscModules = {}
        for DscModule in Platform.Modules:
            DscModules[str(DscModule).lower()] = Platform.Modules[DscModule]
        for PcdKey in InfPcds:
            Pcd = InfPcds[PcdKey]
            if not hasattr(Pcd, 'Offset'):
                continue
            if Pcd.Type != TAB_PCDS_PATCHABLE_IN_MODULE:
                continue
            PatchPcd = None
            if InfLowerPath in DscModules and PcdKey in DscModules[InfLowerPath].Pcds:
                PatchPcd = DscModules[InfLowerPath].Pcds[PcdKey]
            elif PcdKey in Platform.Pcds:
                PatchPcd = Platform.Pcds[PcdKey]
            DscOverride = False
            if PatchPcd and Pcd.Type == PatchPcd.Type:
                DefaultValue = PatchPcd.DefaultValue
                DscOverride = True
            FdfOverride = False
            if PcdKey in FdfPcdDict:
                DefaultValue = FdfPcdDict[PcdKey]
                FdfOverride = True
            BuildOptionOverride = False
            if GlobalData.BuildOptionPcd:
                for pcd in GlobalData.BuildOptionPcd:
                    if PcdKey == (pcd[1], pcd[0]):
                        if pcd[2]:
                            continue
                        DefaultValue = pcd[3]
                        BuildOptionOverride = True
                        break
            if not DscOverride and (not FdfOverride) and (not BuildOptionOverride):
                continue
            if DefaultValue:
                try:
                    DefaultValue = ValueExpressionEx(DefaultValue, Pcd.DatumType, Platform._GuidDict)(True)
                except BadExpression:
                    EdkLogger.error('GenFds', GENFDS_ERROR, 'PCD [%s.%s] Value "%s"' % (Pcd.TokenSpaceGuidCName, Pcd.TokenCName, DefaultValue), File=self.InfFileName)
            if Pcd.InfDefaultValue:
                try:
                    Pcd.InfDefaultValue = ValueExpressionEx(Pcd.InfDefaultValue, Pcd.DatumType, Platform._GuidDict)(True)
                except BadExpression:
                    EdkLogger.error('GenFds', GENFDS_ERROR, 'PCD [%s.%s] Value "%s"' % (Pcd.TokenSpaceGuidCName, Pcd.TokenCName, Pcd.DefaultValue), File=self.InfFileName)
            if Pcd.DatumType == TAB_VOID:
                if Pcd.InfDefaultValue == DefaultValue or not DefaultValue:
                    continue
                if DefaultValue[0] == 'L':
                    MaxDatumSize = str((len(DefaultValue) - 2) * 2)
                elif DefaultValue[0] == '{':
                    MaxDatumSize = str(len(DefaultValue.split(',')))
                else:
                    MaxDatumSize = str(len(DefaultValue) - 1)
                if DscOverride:
                    Pcd.MaxDatumSize = PatchPcd.MaxDatumSize
                if not Pcd.MaxDatumSize:
                    Pcd.MaxDatumSize = str(len(Pcd.InfDefaultValue.split(',')))
            else:
                Base1 = Base2 = 10
                if Pcd.InfDefaultValue.upper().startswith('0X'):
                    Base1 = 16
                if DefaultValue.upper().startswith('0X'):
                    Base2 = 16
                try:
                    PcdValueInImg = int(Pcd.InfDefaultValue, Base1)
                    PcdValueInDscOrFdf = int(DefaultValue, Base2)
                    if PcdValueInImg == PcdValueInDscOrFdf:
                        continue
                except:
                    continue
            if Pcd.DatumType == TAB_VOID:
                if int(MaxDatumSize) > int(Pcd.MaxDatumSize):
                    EdkLogger.error('GenFds', GENFDS_ERROR, "The size of VOID* type PCD '%s.%s' exceeds its maximum size %d bytes." % (Pcd.TokenSpaceGuidCName, Pcd.TokenCName, int(MaxDatumSize) - int(Pcd.MaxDatumSize)))
            elif PcdValueInDscOrFdf > MAX_VAL_TYPE[Pcd.DatumType] or PcdValueInImg > MAX_VAL_TYPE[Pcd.DatumType]:
                EdkLogger.error('GenFds', GENFDS_ERROR, "The size of %s type PCD '%s.%s' doesn't match its data type." % (Pcd.DatumType, Pcd.TokenSpaceGuidCName, Pcd.TokenCName))
            self.PatchPcds.append((Pcd, DefaultValue))
        self.InfModule = Inf
        self.PcdIsDriver = Inf.PcdIsDriver
        self.IsBinaryModule = Inf.IsBinaryModule
        if len(Inf.Depex.data) > 0 and len(Inf.DepexExpression.data) > 0:
            self.Depex = True
        GenFdsGlobalVariable.VerboseLogger('BaseName : %s' % self.BaseName)
        GenFdsGlobalVariable.VerboseLogger('ModuleGuid : %s' % self.ModuleGuid)
        GenFdsGlobalVariable.VerboseLogger('ModuleType : %s' % self.ModuleType)
        GenFdsGlobalVariable.VerboseLogger('VersionString : %s' % self.VersionString)
        GenFdsGlobalVariable.VerboseLogger('InfFileName :%s' % self.InfFileName)
        if IsGenFfs:
            Rule = self.__GetRule__()
            if GlobalData.gGuidPatternEnd.match(Rule.NameGuid):
                self.ModuleGuid = Rule.NameGuid
        self.OutputPath = os.path.join(GenFdsGlobalVariable.FfsDir, self.ModuleGuid + self.BaseName)
        if not os.path.exists(self.OutputPath):
            os.makedirs(self.OutputPath)
        (self.EfiOutputPath, self.EfiDebugPath) = self.__GetEFIOutPutPath__()
        GenFdsGlobalVariable.VerboseLogger('ModuelEFIPath: ' + self.EfiOutputPath)

    def PatchEfiFile(self, EfiFile, FileType):
        if False:
            for i in range(10):
                print('nop')
        if not self.PatchPcds:
            return EfiFile
        if FileType != BINARY_FILE_TYPE_PE32 and self.ModuleType != SUP_MODULE_USER_DEFINED and (self.ModuleType != SUP_MODULE_HOST_APPLICATION):
            return EfiFile
        Basename = os.path.basename(EfiFile)
        Output = os.path.normpath(os.path.join(self.OutputPath, Basename))
        if self.PatchedBinFile == Output:
            return Output
        if self.PatchedBinFile:
            EdkLogger.error('GenFds', GENFDS_ERROR, 'Only one binary file can be patched:\n  a binary file has been patched: %s\n  current file: %s' % (self.PatchedBinFile, EfiFile), File=self.InfFileName)
        CopyLongFilePath(EfiFile, Output)
        for (Pcd, Value) in self.PatchPcds:
            (RetVal, RetStr) = PatchBinaryFile(Output, int(Pcd.Offset, 0), Pcd.DatumType, Value, Pcd.MaxDatumSize)
            if RetVal:
                EdkLogger.error('GenFds', GENFDS_ERROR, RetStr, File=self.InfFileName)
        self.PatchedBinFile = Output
        return Output

    def GenFfs(self, Dict=None, FvChildAddr=[], FvParentAddr=None, IsMakefile=False, FvName=None):
        if False:
            i = 10
            return i + 15
        if Dict is None:
            Dict = {}
        self.__InfParse__(Dict, IsGenFfs=True)
        Arch = self.GetCurrentArch()
        SrcFile = mws.join(GenFdsGlobalVariable.WorkSpaceDir, self.InfFileName)
        DestFile = os.path.join(self.OutputPath, self.ModuleGuid + '.ffs')
        SrcFileDir = '.'
        SrcPath = os.path.dirname(SrcFile)
        SrcFileName = os.path.basename(SrcFile)
        (SrcFileBase, SrcFileExt) = os.path.splitext(SrcFileName)
        DestPath = os.path.dirname(DestFile)
        DestFileName = os.path.basename(DestFile)
        (DestFileBase, DestFileExt) = os.path.splitext(DestFileName)
        self.MacroDict = {'${src}': SrcFile, '${s_path}': SrcPath, '${s_dir}': SrcFileDir, '${s_name}': SrcFileName, '${s_base}': SrcFileBase, '${s_ext}': SrcFileExt, '${dst}': DestFile, '${d_path}': DestPath, '${d_name}': DestFileName, '${d_base}': DestFileBase, '${d_ext}': DestFileExt}
        if len(self.BinFileList) > 0:
            if self.Rule is None or self.Rule == '':
                self.Rule = 'BINARY'
        if not IsMakefile and GenFdsGlobalVariable.EnableGenfdsMultiThread and (self.Rule != 'BINARY'):
            IsMakefile = True
        Rule = self.__GetRule__()
        GenFdsGlobalVariable.VerboseLogger('Packing binaries from inf file : %s' % self.InfFileName)
        if self.ModuleType == SUP_MODULE_DXE_SMM_DRIVER and int(self.PiSpecVersion, 16) >= 65546:
            if Rule.FvFileType == 'DRIVER':
                Rule.FvFileType = 'SMM'
        if self.ModuleType == SUP_MODULE_DXE_SMM_DRIVER and int(self.PiSpecVersion, 16) < 65546:
            if Rule.FvFileType == 'SMM' or Rule.FvFileType == SUP_MODULE_SMM_CORE:
                EdkLogger.error('GenFds', FORMAT_NOT_SUPPORTED, "Framework SMM module doesn't support SMM or SMM_CORE FV file type", File=self.InfFileName)
        MakefilePath = None
        if self.IsBinaryModule:
            IsMakefile = False
        if IsMakefile:
            PathClassObj = PathClass(self.InfFileName, GenFdsGlobalVariable.WorkSpaceDir)
            if self.OverrideGuid:
                PathClassObj = ProcessDuplicatedInf(PathClassObj, self.OverrideGuid, GenFdsGlobalVariable.WorkSpaceDir)
            MakefilePath = (PathClassObj.Path, Arch)
        if isinstance(Rule, RuleSimpleFile.RuleSimpleFile):
            SectionOutputList = self.__GenSimpleFileSection__(Rule, IsMakefile=IsMakefile)
            FfsOutput = self.__GenSimpleFileFfs__(Rule, SectionOutputList, MakefilePath=MakefilePath)
            return FfsOutput
        elif isinstance(Rule, RuleComplexFile.RuleComplexFile):
            (InputSectList, InputSectAlignments) = self.__GenComplexFileSection__(Rule, FvChildAddr, FvParentAddr, IsMakefile=IsMakefile)
            FfsOutput = self.__GenComplexFileFfs__(Rule, InputSectList, InputSectAlignments, MakefilePath=MakefilePath)
            return FfsOutput

    def __ExtendMacro__(self, String):
        if False:
            for i in range(10):
                print('nop')
        MacroDict = {'$(INF_OUTPUT)': self.EfiOutputPath, '$(MODULE_NAME)': self.BaseName, '$(BUILD_NUMBER)': self.BuildNum, '$(INF_VERSION)': self.VersionString, '$(NAMED_GUID)': self.ModuleGuid}
        String = GenFdsGlobalVariable.MacroExtend(String, MacroDict)
        String = GenFdsGlobalVariable.MacroExtend(String, self.MacroDict)
        return String

    def __GetRule__(self):
        if False:
            for i in range(10):
                print('nop')
        CurrentArchList = []
        if self.CurrentArch is None:
            CurrentArchList = ['common']
        else:
            CurrentArchList.append(self.CurrentArch)
        for CurrentArch in CurrentArchList:
            RuleName = 'RULE' + '.' + CurrentArch.upper() + '.' + self.ModuleType.upper()
            if self.Rule is not None:
                RuleName = RuleName + '.' + self.Rule.upper()
            Rule = GenFdsGlobalVariable.FdfParser.Profile.RuleDict.get(RuleName)
            if Rule is not None:
                GenFdsGlobalVariable.VerboseLogger('Want To Find Rule Name is : ' + RuleName)
                return Rule
        RuleName = 'RULE' + '.' + TAB_COMMON + '.' + self.ModuleType.upper()
        if self.Rule is not None:
            RuleName = RuleName + '.' + self.Rule.upper()
        GenFdsGlobalVariable.VerboseLogger('Trying to apply common rule %s for INF %s' % (RuleName, self.InfFileName))
        Rule = GenFdsGlobalVariable.FdfParser.Profile.RuleDict.get(RuleName)
        if Rule is not None:
            GenFdsGlobalVariable.VerboseLogger('Want To Find Rule Name is : ' + RuleName)
            return Rule
        if Rule is None:
            EdkLogger.error('GenFds', GENFDS_ERROR, "Don't Find common rule %s for INF %s" % (RuleName, self.InfFileName))

    def __GetPlatformArchList__(self):
        if False:
            while True:
                i = 10
        InfFileKey = os.path.normpath(mws.join(GenFdsGlobalVariable.WorkSpaceDir, self.InfFileName))
        DscArchList = []
        for Arch in GenFdsGlobalVariable.ArchList:
            PlatformDataBase = GenFdsGlobalVariable.WorkSpace.BuildObject[GenFdsGlobalVariable.ActivePlatform, Arch, GenFdsGlobalVariable.TargetName, GenFdsGlobalVariable.ToolChainTag]
            if PlatformDataBase is not None:
                if InfFileKey in PlatformDataBase.Modules:
                    DscArchList.append(Arch)
                else:
                    for key in PlatformDataBase.Modules:
                        if InfFileKey == str(PlatformDataBase.Modules[key].MetaFile.Path):
                            DscArchList.append(Arch)
                            break
        return DscArchList

    def GetCurrentArch(self):
        if False:
            return 10
        TargetArchList = GenFdsGlobalVariable.ArchList
        PlatformArchList = self.__GetPlatformArchList__()
        CurArchList = TargetArchList
        if PlatformArchList != []:
            CurArchList = list(set(TargetArchList) & set(PlatformArchList))
        GenFdsGlobalVariable.VerboseLogger('Valid target architecture(s) is : ' + ' '.join(CurArchList))
        ArchList = []
        if self.KeyStringList != []:
            for Key in self.KeyStringList:
                Key = GenFdsGlobalVariable.MacroExtend(Key)
                (Target, Tag, Arch) = Key.split('_')
                if Arch in CurArchList:
                    ArchList.append(Arch)
                if Target not in self.TargetOverrideList:
                    self.TargetOverrideList.append(Target)
        else:
            ArchList = CurArchList
        UseArchList = TargetArchList
        if self.UseArch is not None:
            UseArchList = []
            UseArchList.append(self.UseArch)
            ArchList = list(set(UseArchList) & set(ArchList))
        self.InfFileName = NormPath(self.InfFileName)
        if len(PlatformArchList) == 0:
            self.InDsc = False
            PathClassObj = PathClass(self.InfFileName, GenFdsGlobalVariable.WorkSpaceDir)
            (ErrorCode, ErrorInfo) = PathClassObj.Validate('.inf')
            if ErrorCode != 0:
                EdkLogger.error('GenFds', ErrorCode, ExtraData=ErrorInfo)
        if len(ArchList) == 1:
            Arch = ArchList[0]
            return Arch
        elif len(ArchList) > 1:
            if len(PlatformArchList) == 0:
                EdkLogger.error('GenFds', GENFDS_ERROR, 'GenFds command line option has multiple ARCHs %s. Not able to determine which ARCH is valid for Module %s !' % (str(ArchList), self.InfFileName))
            else:
                EdkLogger.error('GenFds', GENFDS_ERROR, 'Module built under multiple ARCHs %s. Not able to determine which output to put into flash for Module %s !' % (str(ArchList), self.InfFileName))
        else:
            EdkLogger.error('GenFds', GENFDS_ERROR, 'Module %s appears under ARCH %s in platform %s, but current deduced ARCH is %s, so NO build output could be put into flash.' % (self.InfFileName, str(PlatformArchList), GenFdsGlobalVariable.ActivePlatform, str(set(UseArchList) & set(TargetArchList))))

    def __GetEFIOutPutPath__(self):
        if False:
            for i in range(10):
                print('nop')
        Arch = ''
        OutputPath = ''
        DebugPath = ''
        (ModulePath, FileName) = os.path.split(self.InfFileName)
        Index = FileName.rfind('.')
        FileName = FileName[0:Index]
        if self.OverrideGuid:
            FileName = self.OverrideGuid
        Arch = 'NoneArch'
        if self.CurrentArch is not None:
            Arch = self.CurrentArch
        OutputPath = os.path.join(GenFdsGlobalVariable.OutputDirDict[Arch], Arch, ModulePath, FileName, 'OUTPUT')
        DebugPath = os.path.join(GenFdsGlobalVariable.OutputDirDict[Arch], Arch, ModulePath, FileName, 'DEBUG')
        OutputPath = os.path.abspath(OutputPath)
        DebugPath = os.path.abspath(DebugPath)
        return (OutputPath, DebugPath)

    def __GenSimpleFileSection__(self, Rule, IsMakefile=False):
        if False:
            print('Hello World!')
        FileList = []
        OutputFileList = []
        GenSecInputFile = None
        if Rule.FileName is not None:
            GenSecInputFile = self.__ExtendMacro__(Rule.FileName)
            if os.path.isabs(GenSecInputFile):
                GenSecInputFile = os.path.normpath(GenSecInputFile)
            else:
                GenSecInputFile = os.path.normpath(os.path.join(self.EfiOutputPath, GenSecInputFile))
        else:
            (FileList, IsSect) = Section.Section.GetFileList(self, '', Rule.FileExtension)
        Index = 1
        SectionType = Rule.SectionType
        if self.ModuleType == SUP_MODULE_DXE_SMM_DRIVER and int(self.PiSpecVersion, 16) >= 65546:
            if SectionType == BINARY_FILE_TYPE_DXE_DEPEX:
                SectionType = BINARY_FILE_TYPE_SMM_DEPEX
        if self.ModuleType == SUP_MODULE_DXE_SMM_DRIVER and int(self.PiSpecVersion, 16) < 65546:
            if SectionType == BINARY_FILE_TYPE_SMM_DEPEX:
                EdkLogger.error('GenFds', FORMAT_NOT_SUPPORTED, "Framework SMM module doesn't support SMM_DEPEX section type", File=self.InfFileName)
        NoStrip = True
        if self.ModuleType in (SUP_MODULE_SEC, SUP_MODULE_PEI_CORE, SUP_MODULE_PEIM):
            if self.KeepReloc is not None:
                NoStrip = self.KeepReloc
            elif Rule.KeepReloc is not None:
                NoStrip = Rule.KeepReloc
            elif self.ShadowFromInfFile is not None:
                NoStrip = self.ShadowFromInfFile
        if FileList != []:
            for File in FileList:
                SecNum = '%d' % Index
                GenSecOutputFile = self.__ExtendMacro__(Rule.NameGuid) + SectionSuffix[SectionType] + SUP_MODULE_SEC + SecNum
                Index = Index + 1
                OutputFile = os.path.join(self.OutputPath, GenSecOutputFile)
                File = GenFdsGlobalVariable.MacroExtend(File, Dict, self.CurrentArch)
                if self.Alignment == 'Auto' and (SectionType == BINARY_FILE_TYPE_PE32 or SectionType == BINARY_FILE_TYPE_TE):
                    ImageObj = PeImageClass(File)
                    if ImageObj.SectionAlignment < 1024:
                        self.Alignment = str(ImageObj.SectionAlignment)
                    elif ImageObj.SectionAlignment < 1048576:
                        self.Alignment = str(ImageObj.SectionAlignment // 1024) + 'K'
                    else:
                        self.Alignment = str(ImageObj.SectionAlignment // 1048576) + 'M'
                if not NoStrip:
                    FileBeforeStrip = os.path.join(self.OutputPath, ModuleName + '.reloc')
                    if not os.path.exists(FileBeforeStrip) or os.path.getmtime(File) > os.path.getmtime(FileBeforeStrip):
                        CopyLongFilePath(File, FileBeforeStrip)
                    StrippedFile = os.path.join(self.OutputPath, ModuleName + '.stipped')
                    GenFdsGlobalVariable.GenerateFirmwareImage(StrippedFile, [File], Strip=True, IsMakefile=IsMakefile)
                    File = StrippedFile
                if SectionType == BINARY_FILE_TYPE_TE:
                    TeFile = os.path.join(self.OutputPath, self.ModuleGuid + 'Te.raw')
                    GenFdsGlobalVariable.GenerateFirmwareImage(TeFile, [File], Type='te', IsMakefile=IsMakefile)
                    File = TeFile
                GenFdsGlobalVariable.GenerateSection(OutputFile, [File], Section.Section.SectionType[SectionType], IsMakefile=IsMakefile)
                OutputFileList.append(OutputFile)
        else:
            SecNum = '%d' % Index
            GenSecOutputFile = self.__ExtendMacro__(Rule.NameGuid) + SectionSuffix[SectionType] + SUP_MODULE_SEC + SecNum
            OutputFile = os.path.join(self.OutputPath, GenSecOutputFile)
            GenSecInputFile = GenFdsGlobalVariable.MacroExtend(GenSecInputFile, Dict, self.CurrentArch)
            if self.Alignment == 'Auto' and (SectionType == BINARY_FILE_TYPE_PE32 or SectionType == BINARY_FILE_TYPE_TE):
                ImageObj = PeImageClass(GenSecInputFile)
                if ImageObj.SectionAlignment < 1024:
                    self.Alignment = str(ImageObj.SectionAlignment)
                elif ImageObj.SectionAlignment < 1048576:
                    self.Alignment = str(ImageObj.SectionAlignment // 1024) + 'K'
                else:
                    self.Alignment = str(ImageObj.SectionAlignment // 1048576) + 'M'
            if not NoStrip:
                FileBeforeStrip = os.path.join(self.OutputPath, ModuleName + '.reloc')
                if not os.path.exists(FileBeforeStrip) or os.path.getmtime(GenSecInputFile) > os.path.getmtime(FileBeforeStrip):
                    CopyLongFilePath(GenSecInputFile, FileBeforeStrip)
                StrippedFile = os.path.join(self.OutputPath, ModuleName + '.stipped')
                GenFdsGlobalVariable.GenerateFirmwareImage(StrippedFile, [GenSecInputFile], Strip=True, IsMakefile=IsMakefile)
                GenSecInputFile = StrippedFile
            if SectionType == BINARY_FILE_TYPE_TE:
                TeFile = os.path.join(self.OutputPath, self.ModuleGuid + 'Te.raw')
                GenFdsGlobalVariable.GenerateFirmwareImage(TeFile, [GenSecInputFile], Type='te', IsMakefile=IsMakefile)
                GenSecInputFile = TeFile
            GenFdsGlobalVariable.GenerateSection(OutputFile, [GenSecInputFile], Section.Section.SectionType[SectionType], IsMakefile=IsMakefile)
            OutputFileList.append(OutputFile)
        return OutputFileList

    def __GenSimpleFileFfs__(self, Rule, InputFileList, MakefilePath=None):
        if False:
            for i in range(10):
                print('nop')
        FfsOutput = self.OutputPath + os.sep + self.__ExtendMacro__(Rule.NameGuid) + '.ffs'
        GenFdsGlobalVariable.VerboseLogger(self.__ExtendMacro__(Rule.NameGuid))
        InputSection = []
        SectionAlignments = []
        for InputFile in InputFileList:
            InputSection.append(InputFile)
            SectionAlignments.append(Rule.SectAlignment)
        if Rule.NameGuid is not None and Rule.NameGuid.startswith('PCD('):
            PcdValue = GenFdsGlobalVariable.GetPcdValue(Rule.NameGuid)
            if len(PcdValue) == 0:
                EdkLogger.error('GenFds', GENFDS_ERROR, '%s NOT defined.' % Rule.NameGuid)
            if PcdValue.startswith('{'):
                PcdValue = GuidStructureByteArrayToGuidString(PcdValue)
            RegistryGuidStr = PcdValue
            if len(RegistryGuidStr) == 0:
                EdkLogger.error('GenFds', GENFDS_ERROR, 'GUID value for %s in wrong format.' % Rule.NameGuid)
            self.ModuleGuid = RegistryGuidStr
            GenFdsGlobalVariable.GenerateFfs(FfsOutput, InputSection, FdfFvFileTypeToFileType[Rule.FvFileType], self.ModuleGuid, Fixed=Rule.Fixed, CheckSum=Rule.CheckSum, Align=Rule.Alignment, SectionAlign=SectionAlignments, MakefilePath=MakefilePath)
        return FfsOutput

    def __GenComplexFileSection__(self, Rule, FvChildAddr, FvParentAddr, IsMakefile=False):
        if False:
            while True:
                i = 10
        if self.ModuleType in (SUP_MODULE_SEC, SUP_MODULE_PEI_CORE, SUP_MODULE_PEIM, SUP_MODULE_MM_CORE_STANDALONE):
            if Rule.KeepReloc is not None:
                self.KeepRelocFromRule = Rule.KeepReloc
        SectFiles = []
        SectAlignments = []
        Index = 1
        HasGeneratedFlag = False
        if self.PcdIsDriver == 'PEI_PCD_DRIVER':
            if self.IsBinaryModule:
                PcdExDbFileName = os.path.join(GenFdsGlobalVariable.FvDir, 'PEIPcdDataBase.raw')
            else:
                PcdExDbFileName = os.path.join(self.EfiOutputPath, 'PEIPcdDataBase.raw')
            PcdExDbSecName = os.path.join(self.OutputPath, 'PEIPcdDataBaseSec.raw')
            GenFdsGlobalVariable.GenerateSection(PcdExDbSecName, [PcdExDbFileName], 'EFI_SECTION_RAW', IsMakefile=IsMakefile)
            SectFiles.append(PcdExDbSecName)
            SectAlignments.append(None)
        elif self.PcdIsDriver == 'DXE_PCD_DRIVER':
            if self.IsBinaryModule:
                PcdExDbFileName = os.path.join(GenFdsGlobalVariable.FvDir, 'DXEPcdDataBase.raw')
            else:
                PcdExDbFileName = os.path.join(self.EfiOutputPath, 'DXEPcdDataBase.raw')
            PcdExDbSecName = os.path.join(self.OutputPath, 'DXEPcdDataBaseSec.raw')
            GenFdsGlobalVariable.GenerateSection(PcdExDbSecName, [PcdExDbFileName], 'EFI_SECTION_RAW', IsMakefile=IsMakefile)
            SectFiles.append(PcdExDbSecName)
            SectAlignments.append(None)
        for Sect in Rule.SectionList:
            SecIndex = '%d' % Index
            SectList = []
            if self.ModuleType == SUP_MODULE_DXE_SMM_DRIVER and int(self.PiSpecVersion, 16) >= 65546:
                if Sect.SectionType == BINARY_FILE_TYPE_DXE_DEPEX:
                    Sect.SectionType = BINARY_FILE_TYPE_SMM_DEPEX
            if self.ModuleType == SUP_MODULE_DXE_SMM_DRIVER and int(self.PiSpecVersion, 16) < 65546:
                if Sect.SectionType == BINARY_FILE_TYPE_SMM_DEPEX:
                    EdkLogger.error('GenFds', FORMAT_NOT_SUPPORTED, "Framework SMM module doesn't support SMM_DEPEX section type", File=self.InfFileName)
            if FvChildAddr != []:
                if isinstance(Sect, FvImageSection):
                    Sect.FvAddr = FvChildAddr.pop(0)
                elif isinstance(Sect, GuidSection):
                    Sect.FvAddr = FvChildAddr
            if FvParentAddr is not None and isinstance(Sect, GuidSection):
                Sect.FvParentAddr = FvParentAddr
            if Rule.KeyStringList != []:
                (SectList, Align) = Sect.GenSection(self.OutputPath, self.ModuleGuid, SecIndex, Rule.KeyStringList, self, IsMakefile=IsMakefile)
            else:
                (SectList, Align) = Sect.GenSection(self.OutputPath, self.ModuleGuid, SecIndex, self.KeyStringList, self, IsMakefile=IsMakefile)
            if not HasGeneratedFlag:
                UniVfrOffsetFileSection = ''
                ModuleFileName = mws.join(GenFdsGlobalVariable.WorkSpaceDir, self.InfFileName)
                InfData = GenFdsGlobalVariable.WorkSpace.BuildObject[PathClass(ModuleFileName), self.CurrentArch]
                VfrUniBaseName = {}
                VfrUniOffsetList = []
                for SourceFile in InfData.Sources:
                    if SourceFile.Type.upper() == '.VFR':
                        VfrUniBaseName[SourceFile.BaseName] = SourceFile.BaseName + 'Bin'
                    if SourceFile.Type.upper() == '.UNI':
                        VfrUniBaseName['UniOffsetName'] = self.BaseName + 'Strings'
                if len(VfrUniBaseName) > 0:
                    if IsMakefile:
                        if InfData.BuildType != 'UEFI_HII':
                            UniVfrOffsetFileName = os.path.join(self.OutputPath, self.BaseName + '.offset')
                            UniVfrOffsetFileSection = os.path.join(self.OutputPath, self.BaseName + 'Offset' + '.raw')
                            UniVfrOffsetFileNameList = []
                            UniVfrOffsetFileNameList.append(UniVfrOffsetFileName)
                            TrimCmd = 'Trim --Vfr-Uni-Offset -o %s --ModuleName=%s --DebugDir=%s ' % (UniVfrOffsetFileName, self.BaseName, self.EfiDebugPath)
                            GenFdsGlobalVariable.SecCmdList.append(TrimCmd)
                            GenFdsGlobalVariable.GenerateSection(UniVfrOffsetFileSection, [UniVfrOffsetFileName], 'EFI_SECTION_RAW', IsMakefile=True)
                    else:
                        VfrUniOffsetList = self.__GetBuildOutputMapFileVfrUniInfo(VfrUniBaseName)
                        if VfrUniOffsetList:
                            UniVfrOffsetFileName = os.path.join(self.OutputPath, self.BaseName + '.offset')
                            UniVfrOffsetFileSection = os.path.join(self.OutputPath, self.BaseName + 'Offset' + '.raw')
                            FfsInfStatement.__GenUniVfrOffsetFile(VfrUniOffsetList, UniVfrOffsetFileName)
                            UniVfrOffsetFileNameList = []
                            UniVfrOffsetFileNameList.append(UniVfrOffsetFileName)
                            'Call GenSection'
                            GenFdsGlobalVariable.GenerateSection(UniVfrOffsetFileSection, UniVfrOffsetFileNameList, 'EFI_SECTION_RAW')
                    if UniVfrOffsetFileSection:
                        SectList.append(UniVfrOffsetFileSection)
                        HasGeneratedFlag = True
            for SecName in SectList:
                SectFiles.append(SecName)
                SectAlignments.append(Align)
            Index = Index + 1
        return (SectFiles, SectAlignments)

    def __GenComplexFileFfs__(self, Rule, InputFile, Alignments, MakefilePath=None):
        if False:
            while True:
                i = 10
        if Rule.NameGuid is not None and Rule.NameGuid.startswith('PCD('):
            PcdValue = GenFdsGlobalVariable.GetPcdValue(Rule.NameGuid)
            if len(PcdValue) == 0:
                EdkLogger.error('GenFds', GENFDS_ERROR, '%s NOT defined.' % Rule.NameGuid)
            if PcdValue.startswith('{'):
                PcdValue = GuidStructureByteArrayToGuidString(PcdValue)
            RegistryGuidStr = PcdValue
            if len(RegistryGuidStr) == 0:
                EdkLogger.error('GenFds', GENFDS_ERROR, 'GUID value for %s in wrong format.' % Rule.NameGuid)
            self.ModuleGuid = RegistryGuidStr
        FfsOutput = os.path.join(self.OutputPath, self.ModuleGuid + '.ffs')
        GenFdsGlobalVariable.GenerateFfs(FfsOutput, InputFile, FdfFvFileTypeToFileType[Rule.FvFileType], self.ModuleGuid, Fixed=Rule.Fixed, CheckSum=Rule.CheckSum, Align=Rule.Alignment, SectionAlign=Alignments, MakefilePath=MakefilePath)
        return FfsOutput

    def __GetBuildOutputMapFileVfrUniInfo(self, VfrUniBaseName):
        if False:
            for i in range(10):
                print('nop')
        MapFileName = os.path.join(self.EfiOutputPath, self.BaseName + '.map')
        EfiFileName = os.path.join(self.EfiOutputPath, self.BaseName + '.efi')
        return GetVariableOffset(MapFileName, EfiFileName, list(VfrUniBaseName.values()))

    @staticmethod
    def __GenUniVfrOffsetFile(VfrUniOffsetList, UniVfrOffsetFileName):
        if False:
            print('Hello World!')
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
                type(Item[1])
                VfrValue = pack('Q', int(Item[1], 16))
                fStringIO.write(VfrValue)
        try:
            SaveFileOnChange(UniVfrOffsetFileName, fStringIO.getvalue())
        except:
            EdkLogger.error('GenFds', FILE_WRITE_FAILURE, 'Write data to file %s failed, please check whether the file been locked or using by other applications.' % UniVfrOffsetFileName, None)
        fStringIO.close()