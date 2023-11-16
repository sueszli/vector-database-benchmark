from __future__ import absolute_import
from Workspace.WorkspaceDatabase import WorkspaceDatabase, BuildDB
from Common.caching import cached_property
from AutoGen.BuildEngine import BuildRule, AutoGenReqBuildRuleVerNum
from AutoGen.AutoGen import CalculatePriorityValue
from Common.Misc import CheckPcdDatum, GuidValue
from Common.Expression import ValueExpressionEx
from Common.DataType import *
from CommonDataClass.Exceptions import *
from CommonDataClass.CommonClass import SkuInfoClass
import Common.EdkLogger as EdkLogger
from Common.BuildToolError import OPTION_CONFLICT, FORMAT_INVALID, RESOURCE_NOT_AVAILABLE
from Common.MultipleWorkspace import MultipleWorkspace as mws
from collections import defaultdict
from Common.Misc import PathClass
import os
PrioList = {'0x11111': 16, '0x01111': 15, '0x10111': 14, '0x00111': 13, '0x11011': 12, '0x01011': 11, '0x10011': 10, '0x00011': 9, '0x11101': 8, '0x01101': 7, '0x10101': 6, '0x00101': 5, '0x11001': 4, '0x01001': 3, '0x10001': 2, '0x00001': 1}

class AutoGenInfo(object):
    __ObjectCache = {}

    @classmethod
    def GetCache(cls):
        if False:
            print('Hello World!')
        return cls.__ObjectCache

    def __new__(cls, Workspace, MetaFile, Target, Toolchain, Arch, *args, **kwargs):
        if False:
            return 10
        Key = (Target, Toolchain, Arch, MetaFile)
        if Key in cls.__ObjectCache:
            return cls.__ObjectCache[Key]
        RetVal = cls.__ObjectCache[Key] = super(AutoGenInfo, cls).__new__(cls)
        return RetVal

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash(self.MetaFile)

    def __str__(self):
        if False:
            return 10
        return str(self.MetaFile)

    def __eq__(self, Other):
        if False:
            print('Hello World!')
        return Other and self.MetaFile == Other

    def _ExpandBuildOption(self, Options, ModuleStyle=None, ToolDef=None):
        if False:
            for i in range(10):
                print('nop')
        if not ToolDef:
            ToolDef = self.ToolDefinition
        BuildOptions = {}
        FamilyMatch = False
        FamilyIsNull = True
        OverrideList = {}
        for Key in Options:
            if Key[0] == self.BuildRuleFamily and (ModuleStyle is None or len(Key) < 3 or (len(Key) > 2 and Key[2] == ModuleStyle)):
                (Target, ToolChain, Arch, CommandType, Attr) = Key[1].split('_')
                if (Target == self.BuildTarget or Target == TAB_STAR) and (ToolChain == self.ToolChain or ToolChain == TAB_STAR) and (Arch == self.Arch or Arch == TAB_STAR) and Options[Key].startswith('='):
                    if OverrideList.get(Key[1]) is not None:
                        OverrideList.pop(Key[1])
                    OverrideList[Key[1]] = Options[Key]
        if len(OverrideList) >= 2:
            KeyList = list(OverrideList.keys())
            for Index in range(len(KeyList)):
                NowKey = KeyList[Index]
                (Target1, ToolChain1, Arch1, CommandType1, Attr1) = NowKey.split('_')
                for Index1 in range(len(KeyList) - Index - 1):
                    NextKey = KeyList[Index1 + Index + 1]
                    (Target2, ToolChain2, Arch2, CommandType2, Attr2) = NextKey.split('_')
                    if (Target1 == Target2 or Target1 == TAB_STAR or Target2 == TAB_STAR) and (ToolChain1 == ToolChain2 or ToolChain1 == TAB_STAR or ToolChain2 == TAB_STAR) and (Arch1 == Arch2 or Arch1 == TAB_STAR or Arch2 == TAB_STAR) and (CommandType1 == CommandType2 or CommandType1 == TAB_STAR or CommandType2 == TAB_STAR) and (Attr1 == Attr2 or Attr1 == TAB_STAR or Attr2 == TAB_STAR):
                        if CalculatePriorityValue(NowKey) > CalculatePriorityValue(NextKey):
                            if Options.get((self.BuildRuleFamily, NextKey)) is not None:
                                Options.pop((self.BuildRuleFamily, NextKey))
                        elif Options.get((self.BuildRuleFamily, NowKey)) is not None:
                            Options.pop((self.BuildRuleFamily, NowKey))
        for Key in Options:
            if ModuleStyle is not None and len(Key) > 2:
                if ModuleStyle == EDK_NAME and Key[2] != EDK_NAME:
                    continue
                elif ModuleStyle == EDKII_NAME and Key[2] != EDKII_NAME:
                    continue
            Family = Key[0]
            (Target, Tag, Arch, Tool, Attr) = Key[1].split('_')
            if Family != '':
                Found = False
                if Tool in ToolDef:
                    FamilyIsNull = False
                    if TAB_TOD_DEFINES_BUILDRULEFAMILY in ToolDef[Tool]:
                        if Family == ToolDef[Tool][TAB_TOD_DEFINES_BUILDRULEFAMILY]:
                            FamilyMatch = True
                            Found = True
                if TAB_STAR in ToolDef:
                    FamilyIsNull = False
                    if TAB_TOD_DEFINES_BUILDRULEFAMILY in ToolDef[TAB_STAR]:
                        if Family == ToolDef[TAB_STAR][TAB_TOD_DEFINES_BUILDRULEFAMILY]:
                            FamilyMatch = True
                            Found = True
                if not Found:
                    continue
            if Target == TAB_STAR or Target == self.BuildTarget:
                if Tag == TAB_STAR or Tag == self.ToolChain:
                    if Arch == TAB_STAR or Arch == self.Arch:
                        if Tool not in BuildOptions:
                            BuildOptions[Tool] = {}
                        if Attr != 'FLAGS' or Attr not in BuildOptions[Tool] or Options[Key].startswith('='):
                            BuildOptions[Tool][Attr] = Options[Key]
                        elif Attr != 'PATH':
                            BuildOptions[Tool][Attr] += ' ' + Options[Key]
                        else:
                            BuildOptions[Tool][Attr] = Options[Key]
        if FamilyMatch or FamilyIsNull:
            return BuildOptions
        for Key in Options:
            if ModuleStyle is not None and len(Key) > 2:
                if ModuleStyle == EDK_NAME and Key[2] != EDK_NAME:
                    continue
                elif ModuleStyle == EDKII_NAME and Key[2] != EDKII_NAME:
                    continue
            Family = Key[0]
            (Target, Tag, Arch, Tool, Attr) = Key[1].split('_')
            if Family == '':
                continue
            Found = False
            if Tool in ToolDef:
                if TAB_TOD_DEFINES_FAMILY in ToolDef[Tool]:
                    if Family == ToolDef[Tool][TAB_TOD_DEFINES_FAMILY]:
                        Found = True
            if TAB_STAR in ToolDef:
                if TAB_TOD_DEFINES_FAMILY in ToolDef[TAB_STAR]:
                    if Family == ToolDef[TAB_STAR][TAB_TOD_DEFINES_FAMILY]:
                        Found = True
            if not Found:
                continue
            if Target == TAB_STAR or Target == self.BuildTarget:
                if Tag == TAB_STAR or Tag == self.ToolChain:
                    if Arch == TAB_STAR or Arch == self.Arch:
                        if Tool not in BuildOptions:
                            BuildOptions[Tool] = {}
                        if Attr != 'FLAGS' or Attr not in BuildOptions[Tool] or Options[Key].startswith('='):
                            BuildOptions[Tool][Attr] = Options[Key]
                        elif Attr != 'PATH':
                            BuildOptions[Tool][Attr] += ' ' + Options[Key]
                        else:
                            BuildOptions[Tool][Attr] = Options[Key]
        return BuildOptions

class WorkSpaceInfo(AutoGenInfo):

    def __init__(self, Workspace, MetaFile, Target, ToolChain, Arch):
        if False:
            for i in range(10):
                print('nop')
        if not hasattr(self, '_Init'):
            self.do_init(Workspace, MetaFile, Target, ToolChain, Arch)
            self._Init = True

    def do_init(self, Workspace, MetaFile, Target, ToolChain, Arch):
        if False:
            return 10
        self._SrcTimeStamp = 0
        self.Db = BuildDB
        self.BuildDatabase = self.Db.BuildObject
        self.Target = Target
        self.ToolChain = ToolChain
        self.WorkspaceDir = Workspace
        self.ActivePlatform = MetaFile
        self.ArchList = Arch
        self.AutoGenObjectList = []

    @property
    def BuildDir(self):
        if False:
            for i in range(10):
                print('nop')
        return self.AutoGenObjectList[0].BuildDir

    @property
    def Name(self):
        if False:
            while True:
                i = 10
        return self.AutoGenObjectList[0].Platform.PlatformName

    @property
    def FlashDefinition(self):
        if False:
            while True:
                i = 10
        return self.AutoGenObjectList[0].Platform.FlashDefinition

    @property
    def GenFdsCommandDict(self):
        if False:
            i = 10
            return i + 15
        FdsCommandDict = self.AutoGenObjectList[0].DataPipe.Get('FdsCommandDict')
        if FdsCommandDict:
            return FdsCommandDict
        return {}

    @cached_property
    def FvDir(self):
        if False:
            while True:
                i = 10
        return os.path.join(self.BuildDir, TAB_FV_DIRECTORY)

class PlatformInfo(AutoGenInfo):

    def __init__(self, Workspace, MetaFile, Target, ToolChain, Arch, DataPipe):
        if False:
            for i in range(10):
                print('nop')
        if not hasattr(self, '_Init'):
            self.do_init(Workspace, MetaFile, Target, ToolChain, Arch, DataPipe)
            self._Init = True

    def do_init(self, Workspace, MetaFile, Target, ToolChain, Arch, DataPipe):
        if False:
            i = 10
            return i + 15
        self.Wa = Workspace
        self.WorkspaceDir = self.Wa.WorkspaceDir
        self.MetaFile = MetaFile
        self.Arch = Arch
        self.Target = Target
        self.BuildTarget = Target
        self.ToolChain = ToolChain
        self.Platform = self.Wa.BuildDatabase[self.MetaFile, self.Arch, self.Target, self.ToolChain]
        self.SourceDir = MetaFile.SubDir
        self.DataPipe = DataPipe

    @cached_property
    def _AsBuildModuleList(self):
        if False:
            for i in range(10):
                print('nop')
        retVal = self.DataPipe.Get('AsBuildModuleList')
        if retVal is None:
            retVal = {}
        return retVal

    def ValidModule(self, Module):
        if False:
            print('Hello World!')
        return Module in self.Platform.Modules or Module in self.Platform.LibraryInstances or Module in self._AsBuildModuleList

    @cached_property
    def ToolChainFamily(self):
        if False:
            return 10
        retVal = self.DataPipe.Get('ToolChainFamily')
        if retVal is None:
            retVal = {}
        return retVal

    @cached_property
    def BuildRuleFamily(self):
        if False:
            print('Hello World!')
        retVal = self.DataPipe.Get('BuildRuleFamily')
        if retVal is None:
            retVal = {}
        return retVal

    @cached_property
    def _MbList(self):
        if False:
            for i in range(10):
                print('nop')
        return [self.Wa.BuildDatabase[m, self.Arch, self.BuildTarget, self.ToolChain] for m in self.Platform.Modules]

    @cached_property
    def PackageList(self):
        if False:
            return 10
        RetVal = set()
        for (dec_file, Arch) in self.DataPipe.Get('PackageList'):
            RetVal.add(self.Wa.BuildDatabase[dec_file, Arch, self.BuildTarget, self.ToolChain])
        return list(RetVal)

    @cached_property
    def BuildDir(self):
        if False:
            print('Hello World!')
        if os.path.isabs(self.OutputDir):
            RetVal = os.path.join(os.path.abspath(self.OutputDir), self.Target + '_' + self.ToolChain)
        else:
            RetVal = os.path.join(self.WorkspaceDir, self.OutputDir, self.Target + '_' + self.ToolChain)
        return RetVal

    @cached_property
    def OutputDir(self):
        if False:
            for i in range(10):
                print('nop')
        return self.Platform.OutputDirectory

    @cached_property
    def Name(self):
        if False:
            for i in range(10):
                print('nop')
        return self.Platform.PlatformName

    @cached_property
    def Guid(self):
        if False:
            for i in range(10):
                print('nop')
        return self.Platform.Guid

    @cached_property
    def Version(self):
        if False:
            while True:
                i = 10
        return self.Platform.Version

    @cached_property
    def ToolDefinition(self):
        if False:
            print('Hello World!')
        retVal = self.DataPipe.Get('TOOLDEF')
        if retVal is None:
            retVal = {}
        return retVal

    @cached_property
    def BuildCommand(self):
        if False:
            while True:
                i = 10
        retVal = self.DataPipe.Get('BuildCommand')
        if retVal is None:
            retVal = []
        return retVal

    @cached_property
    def PcdTokenNumber(self):
        if False:
            while True:
                i = 10
        retVal = self.DataPipe.Get('PCD_TNUM')
        if retVal is None:
            retVal = {}
        return retVal

    def _OverridePcd(self, ToPcd, FromPcd, Module='', Msg='', Library=''):
        if False:
            print('Hello World!')
        TokenCName = ToPcd.TokenCName
        for PcdItem in self.MixedPcd:
            if (ToPcd.TokenCName, ToPcd.TokenSpaceGuidCName) in self.MixedPcd[PcdItem]:
                TokenCName = PcdItem[0]
                break
        if FromPcd is not None:
            if ToPcd.Pending and FromPcd.Type:
                ToPcd.Type = FromPcd.Type
            elif ToPcd.Type and FromPcd.Type and (ToPcd.Type != FromPcd.Type) and (ToPcd.Type in FromPcd.Type):
                if ToPcd.Type.strip() == TAB_PCDS_DYNAMIC_EX:
                    ToPcd.Type = FromPcd.Type
            elif ToPcd.Type and FromPcd.Type and (ToPcd.Type != FromPcd.Type):
                if Library:
                    Module = str(Module) + " 's library file (" + str(Library) + ')'
                EdkLogger.error('build', OPTION_CONFLICT, 'Mismatched PCD type', ExtraData='%s.%s is used as [%s] in module %s, but as [%s] in %s.' % (ToPcd.TokenSpaceGuidCName, TokenCName, ToPcd.Type, Module, FromPcd.Type, Msg), File=self.MetaFile)
            if FromPcd.MaxDatumSize:
                ToPcd.MaxDatumSize = FromPcd.MaxDatumSize
                ToPcd.MaxSizeUserSet = FromPcd.MaxDatumSize
            if FromPcd.DefaultValue:
                ToPcd.DefaultValue = FromPcd.DefaultValue
            if FromPcd.TokenValue:
                ToPcd.TokenValue = FromPcd.TokenValue
            if FromPcd.DatumType:
                ToPcd.DatumType = FromPcd.DatumType
            if FromPcd.SkuInfoList:
                ToPcd.SkuInfoList = FromPcd.SkuInfoList
            if FromPcd.UserDefinedDefaultStoresFlag:
                ToPcd.UserDefinedDefaultStoresFlag = FromPcd.UserDefinedDefaultStoresFlag
            if ToPcd.DefaultValue:
                try:
                    ToPcd.DefaultValue = ValueExpressionEx(ToPcd.DefaultValue, ToPcd.DatumType, self._GuidDict)(True)
                except BadExpression as Value:
                    EdkLogger.error('Parser', FORMAT_INVALID, 'PCD [%s.%s] Value "%s", %s' % (ToPcd.TokenSpaceGuidCName, ToPcd.TokenCName, ToPcd.DefaultValue, Value), File=self.MetaFile)
            (IsValid, Cause) = CheckPcdDatum(ToPcd.DatumType, ToPcd.DefaultValue)
            if not IsValid:
                EdkLogger.error('build', FORMAT_INVALID, Cause, File=self.MetaFile, ExtraData='%s.%s' % (ToPcd.TokenSpaceGuidCName, TokenCName))
            ToPcd.validateranges = FromPcd.validateranges
            ToPcd.validlists = FromPcd.validlists
            ToPcd.expressions = FromPcd.expressions
            ToPcd.CustomAttribute = FromPcd.CustomAttribute
        if FromPcd is not None and ToPcd.DatumType == TAB_VOID and (not ToPcd.MaxDatumSize):
            EdkLogger.debug(EdkLogger.DEBUG_9, 'No MaxDatumSize specified for PCD %s.%s' % (ToPcd.TokenSpaceGuidCName, TokenCName))
            Value = ToPcd.DefaultValue
            if not Value:
                ToPcd.MaxDatumSize = '1'
            elif Value[0] == 'L':
                ToPcd.MaxDatumSize = str((len(Value) - 2) * 2)
            elif Value[0] == '{':
                ToPcd.MaxDatumSize = str(len(Value.split(',')))
            else:
                ToPcd.MaxDatumSize = str(len(Value) - 1)
        if (ToPcd.Type in PCD_DYNAMIC_TYPE_SET or ToPcd.Type in PCD_DYNAMIC_EX_TYPE_SET) and (not ToPcd.SkuInfoList):
            if self.Platform.SkuName in self.Platform.SkuIds:
                SkuName = self.Platform.SkuName
            else:
                SkuName = TAB_DEFAULT
            ToPcd.SkuInfoList = {SkuName: SkuInfoClass(SkuName, self.Platform.SkuIds[SkuName][0], '', '', '', '', '', ToPcd.DefaultValue)}

    def ApplyPcdSetting(self, Ma, Pcds, Library=''):
        if False:
            while True:
                i = 10
        Module = Ma.Module
        for (Name, Guid) in Pcds:
            PcdInModule = Pcds[Name, Guid]
            if (Name, Guid) in self.Pcds:
                PcdInPlatform = self.Pcds[Name, Guid]
            else:
                PcdInPlatform = None
            self._OverridePcd(PcdInModule, PcdInPlatform, Module, Msg='DSC PCD sections', Library=Library)
            for SkuId in PcdInModule.SkuInfoList:
                Sku = PcdInModule.SkuInfoList[SkuId]
                if Sku.VariableGuid == '':
                    continue
                Sku.VariableGuidValue = GuidValue(Sku.VariableGuid, self.PackageList, self.MetaFile.Path)
                if Sku.VariableGuidValue is None:
                    PackageList = '\n\t'.join((str(P) for P in self.PackageList))
                    EdkLogger.error('build', RESOURCE_NOT_AVAILABLE, 'Value of GUID [%s] is not found in' % Sku.VariableGuid, ExtraData=PackageList + '\n\t(used with %s.%s from module %s)' % (Guid, Name, str(Module)), File=self.MetaFile)
        ModuleScopePcds = self.DataPipe.Get('MOL_PCDS')
        if Module in self.Platform.Modules:
            PlatformModule = self.Platform.Modules[str(Module)]
            PCD_DATA = ModuleScopePcds.get(Ma.Guid, {})
            mPcds = {(pcd.TokenCName, pcd.TokenSpaceGuidCName): pcd for pcd in PCD_DATA}
            for Key in mPcds:
                if self.BuildOptionPcd:
                    for pcd in self.BuildOptionPcd:
                        (TokenSpaceGuidCName, TokenCName, FieldName, pcdvalue, _) = pcd
                        if (TokenCName, TokenSpaceGuidCName) == Key and FieldName == '':
                            PlatformModule.Pcds[Key].DefaultValue = pcdvalue
                            PlatformModule.Pcds[Key].PcdValueFromComm = pcdvalue
                            break
                Flag = False
                if Key in Pcds:
                    ToPcd = Pcds[Key]
                    Flag = True
                elif Key in self.MixedPcd:
                    for PcdItem in self.MixedPcd[Key]:
                        if PcdItem in Pcds:
                            ToPcd = Pcds[PcdItem]
                            Flag = True
                            break
                if Flag:
                    self._OverridePcd(ToPcd, mPcds[Key], Module, Msg='DSC Components Module scoped PCD section', Library=Library)
        for (Name, Guid) in Pcds:
            Pcd = Pcds[Name, Guid]
            if Pcd.DatumType == TAB_VOID and (not Pcd.MaxDatumSize):
                Pcd.MaxSizeUserSet = None
                Value = Pcd.DefaultValue
                if not Value:
                    Pcd.MaxDatumSize = '1'
                elif Value[0] == 'L':
                    Pcd.MaxDatumSize = str((len(Value) - 2) * 2)
                elif Value[0] == '{':
                    Pcd.MaxDatumSize = str(len(Value.split(',')))
                else:
                    Pcd.MaxDatumSize = str(len(Value) - 1)
        return list(Pcds.values())

    @cached_property
    def Pcds(self):
        if False:
            print('Hello World!')
        PlatformPcdData = self.DataPipe.Get('PLA_PCD')
        return {(pcddata.TokenCName, pcddata.TokenSpaceGuidCName): pcddata for pcddata in PlatformPcdData}

    def CreateSkuInfoFromDict(self, SkuInfoDict):
        if False:
            return 10
        return SkuInfoClass(SkuInfoDict.get('SkuIdName'), SkuInfoDict.get('SkuId'), SkuInfoDict.get('VariableName'), SkuInfoDict.get('VariableGuid'), SkuInfoDict.get('VariableOffset'), SkuInfoDict.get('HiiDefaultValue'), SkuInfoDict.get('VpdOffset'), SkuInfoDict.get('DefaultValue'), SkuInfoDict.get('VariableGuidValue'), SkuInfoDict.get('VariableAttribute', ''), SkuInfoDict.get('DefaultStore', None))

    @cached_property
    def MixedPcd(self):
        if False:
            print('Hello World!')
        return self.DataPipe.Get('MixedPcd')

    @cached_property
    def _GuidDict(self):
        if False:
            for i in range(10):
                print('nop')
        RetVal = self.DataPipe.Get('GuidDict')
        if RetVal is None:
            RetVal = {}
        return RetVal

    @cached_property
    def BuildOptionPcd(self):
        if False:
            return 10
        return self.DataPipe.Get('BuildOptPcd')

    def ApplyBuildOption(self, module):
        if False:
            i = 10
            return i + 15
        PlatformOptions = self.DataPipe.Get('PLA_BO')
        ModuleBuildOptions = self.DataPipe.Get('MOL_BO')
        ModuleOptionFromDsc = ModuleBuildOptions.get((module.MetaFile.File, module.MetaFile.Root))
        if ModuleOptionFromDsc:
            (ModuleTypeOptions, PlatformModuleOptions) = (ModuleOptionFromDsc['ModuleTypeOptions'], ModuleOptionFromDsc['PlatformModuleOptions'])
        else:
            (ModuleTypeOptions, PlatformModuleOptions) = ({}, {})
        ToolDefinition = self.DataPipe.Get('TOOLDEF')
        ModuleOptions = self._ExpandBuildOption(module.BuildOptions)
        BuildRuleOrder = None
        for Options in [ToolDefinition, ModuleOptions, PlatformOptions, ModuleTypeOptions, PlatformModuleOptions]:
            for Tool in Options:
                for Attr in Options[Tool]:
                    if Attr == TAB_TOD_DEFINES_BUILDRULEORDER:
                        BuildRuleOrder = Options[Tool][Attr]
        AllTools = set(list(ModuleOptions.keys()) + list(PlatformOptions.keys()) + list(PlatformModuleOptions.keys()) + list(ModuleTypeOptions.keys()) + list(ToolDefinition.keys()))
        BuildOptions = defaultdict(lambda : defaultdict(str))
        for Tool in AllTools:
            for Options in [ToolDefinition, ModuleOptions, PlatformOptions, ModuleTypeOptions, PlatformModuleOptions]:
                if Tool not in Options:
                    continue
                for Attr in Options[Tool]:
                    if Attr == TAB_TOD_DEFINES_BUILDRULEORDER:
                        continue
                    Value = Options[Tool][Attr]
                    ToolList = [Tool]
                    if Tool == TAB_STAR:
                        ToolList = list(AllTools)
                        ToolList.remove(TAB_STAR)
                    for ExpandedTool in ToolList:
                        if Value.startswith('='):
                            BuildOptions[ExpandedTool][Attr] = mws.handleWsMacro(Value[1:])
                        elif Attr != 'PATH':
                            BuildOptions[ExpandedTool][Attr] += ' ' + mws.handleWsMacro(Value)
                        else:
                            BuildOptions[ExpandedTool][Attr] = mws.handleWsMacro(Value)
        return (BuildOptions, BuildRuleOrder)

    def ApplyLibraryInstance(self, module):
        if False:
            while True:
                i = 10
        alldeps = self.DataPipe.Get('DEPS')
        if alldeps is None:
            alldeps = {}
        mod_libs = alldeps.get((module.MetaFile.File, module.MetaFile.Root, module.Arch, module.MetaFile.Path), [])
        retVal = []
        for (file_path, root, arch, abs_path) in mod_libs:
            libMetaFile = PathClass(file_path, root)
            libMetaFile.OriginalPath = PathClass(file_path, root)
            libMetaFile.Path = abs_path
            retVal.append(self.Wa.BuildDatabase[libMetaFile, arch, self.Target, self.ToolChain])
        return retVal

    @cached_property
    def BuildRule(self):
        if False:
            i = 10
            return i + 15
        WInfo = self.DataPipe.Get('P_Info')
        RetVal = WInfo.get('BuildRuleFile')
        if RetVal._FileVersion == '':
            RetVal._FileVersion = AutoGenReqBuildRuleVerNum
        return RetVal