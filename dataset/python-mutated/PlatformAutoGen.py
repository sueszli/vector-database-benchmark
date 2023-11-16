from __future__ import print_function
from __future__ import absolute_import
import os.path as path
import copy
from collections import defaultdict
from .BuildEngine import BuildRule, gDefaultBuildRuleFile, AutoGenReqBuildRuleVerNum
from .GenVar import VariableMgr, var_info
from . import GenMake
from AutoGen.DataPipe import MemoryDataPipe
from AutoGen.ModuleAutoGen import ModuleAutoGen
from AutoGen.AutoGen import AutoGen
from AutoGen.AutoGen import CalculatePriorityValue
from Workspace.WorkspaceCommon import GetModuleLibInstances
from CommonDataClass.CommonClass import SkuInfoClass
from Common.caching import cached_class_function
from Common.Expression import ValueExpressionEx
from Common.StringUtils import StringToArray, NormPath
from Common.BuildToolError import *
from Common.DataType import *
from Common.Misc import *
import Common.VpdInfoFile as VpdInfoFile

def _SplitOption(OptionString):
    if False:
        for i in range(10):
            print('nop')
    OptionList = []
    LastChar = ' '
    OptionStart = 0
    QuotationMark = ''
    for Index in range(0, len(OptionString)):
        CurrentChar = OptionString[Index]
        if CurrentChar in ['"', "'"]:
            if QuotationMark == CurrentChar:
                QuotationMark = ''
            elif QuotationMark == '':
                QuotationMark = CurrentChar
            continue
        elif QuotationMark:
            continue
        if CurrentChar in ['/', '-'] and LastChar in [' ', '\t', '\r', '\n']:
            if Index > OptionStart:
                OptionList.append(OptionString[OptionStart:Index - 1])
            OptionStart = Index
        LastChar = CurrentChar
    OptionList.append(OptionString[OptionStart:])
    return OptionList

class PlatformAutoGen(AutoGen):

    def __init__(self, Workspace, MetaFile, Target, Toolchain, Arch, *args, **kwargs):
        if False:
            return 10
        if not hasattr(self, '_Init'):
            self._InitWorker(Workspace, MetaFile, Target, Toolchain, Arch)
            self._Init = True
    _DynaPcdList_ = []
    _NonDynaPcdList_ = []
    _PlatformPcds = {}

    def _InitWorker(self, Workspace, PlatformFile, Target, Toolchain, Arch):
        if False:
            for i in range(10):
                print('nop')
        EdkLogger.debug(EdkLogger.DEBUG_9, 'AutoGen platform [%s] [%s]' % (PlatformFile, Arch))
        GlobalData.gProcessingFile = '%s [%s, %s, %s]' % (PlatformFile, Arch, Toolchain, Target)
        self.MetaFile = PlatformFile
        self.Workspace = Workspace
        self.WorkspaceDir = Workspace.WorkspaceDir
        self.ToolChain = Toolchain
        self.BuildTarget = Target
        self.Arch = Arch
        self.SourceDir = PlatformFile.SubDir
        self.FdTargetList = self.Workspace.FdTargetList
        self.FvTargetList = self.Workspace.FvTargetList
        self.BuildDatabase = Workspace.BuildDatabase
        self.DscBuildDataObj = Workspace.Platform
        self.MakeFileName = ''
        self._DynamicPcdList = None
        self._NonDynamicPcdList = None
        self._AsBuildInfList = []
        self._AsBuildModuleList = []
        self.VariableInfo = None
        if GlobalData.gFdfParser is not None:
            self._AsBuildInfList = GlobalData.gFdfParser.Profile.InfList
            for Inf in self._AsBuildInfList:
                InfClass = PathClass(NormPath(Inf), GlobalData.gWorkspace, self.Arch)
                M = self.BuildDatabase[InfClass, self.Arch, self.BuildTarget, self.ToolChain]
                if not M.IsBinaryModule:
                    continue
                self._AsBuildModuleList.append(InfClass)
        self.LibraryBuildDirectoryList = []
        self.ModuleBuildDirectoryList = []
        self.DataPipe = MemoryDataPipe(self.BuildDir)
        self.DataPipe.FillData(self)
        return True

    def FillData_LibConstPcd(self):
        if False:
            i = 10
            return i + 15
        libConstPcd = {}
        for LibAuto in self.LibraryAutoGenList:
            if LibAuto.ConstPcd:
                libConstPcd[LibAuto.MetaFile.File, LibAuto.MetaFile.Root, LibAuto.Arch, LibAuto.MetaFile.Path] = LibAuto.ConstPcd
        self.DataPipe.DataContainer = {'LibConstPcd': libConstPcd}

    @cached_class_function
    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash((self.MetaFile, self.Arch, self.ToolChain, self.BuildTarget))

    @cached_class_function
    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '%s [%s]' % (self.MetaFile, self.Arch)

    @cached_class_function
    def CreateCodeFile(self, CreateModuleCodeFile=False):
        if False:
            for i in range(10):
                print('nop')
        if not CreateModuleCodeFile:
            return
        for Ma in self.ModuleAutoGenList:
            Ma.CreateCodeFile(CreateModuleCodeFile)

    @cached_property
    def GenFdsCommand(self):
        if False:
            return 10
        return self.Workspace.GenFdsCommand

    def CreateMakeFile(self, CreateModuleMakeFile=False, FfsCommand={}):
        if False:
            for i in range(10):
                print('nop')
        if CreateModuleMakeFile:
            for Ma in self._MaList:
                key = (Ma.MetaFile.File, self.Arch)
                if key in FfsCommand:
                    Ma.CreateMakeFile(CreateModuleMakeFile, FfsCommand[key])
                else:
                    Ma.CreateMakeFile(CreateModuleMakeFile)
        self.CreateLibModuelDirs()

    def CreateLibModuelDirs(self):
        if False:
            i = 10
            return i + 15
        if self.MakeFileName:
            return
        Makefile = GenMake.PlatformMakefile(self)
        self.LibraryBuildDirectoryList = Makefile.GetLibraryBuildDirectoryList()
        self.ModuleBuildDirectoryList = Makefile.GetModuleBuildDirectoryList()
        self.MakeFileName = Makefile.getMakefileName()

    @property
    def AllPcdList(self):
        if False:
            print('Hello World!')
        return self.DynamicPcdList + self.NonDynamicPcdList

    def CollectFixedAtBuildPcds(self):
        if False:
            return 10
        for LibAuto in self.LibraryAutoGenList:
            FixedAtBuildPcds = {}
            ShareFixedAtBuildPcdsSameValue = {}
            for Module in LibAuto.ReferenceModules:
                for Pcd in set(Module.FixedAtBuildPcds + LibAuto.FixedAtBuildPcds):
                    DefaultValue = Pcd.DefaultValue
                    if Pcd in Module.LibraryPcdList:
                        Index = Module.LibraryPcdList.index(Pcd)
                        DefaultValue = Module.LibraryPcdList[Index].DefaultValue
                    key = '.'.join((Pcd.TokenSpaceGuidCName, Pcd.TokenCName))
                    if key not in FixedAtBuildPcds:
                        ShareFixedAtBuildPcdsSameValue[key] = True
                        FixedAtBuildPcds[key] = DefaultValue
                    elif FixedAtBuildPcds[key] != DefaultValue:
                        ShareFixedAtBuildPcdsSameValue[key] = False
            for Pcd in LibAuto.FixedAtBuildPcds:
                key = '.'.join((Pcd.TokenSpaceGuidCName, Pcd.TokenCName))
                if (Pcd.TokenCName, Pcd.TokenSpaceGuidCName) not in self.NonDynamicPcdDict:
                    continue
                else:
                    DscPcd = self.NonDynamicPcdDict[Pcd.TokenCName, Pcd.TokenSpaceGuidCName]
                    if DscPcd.Type != TAB_PCDS_FIXED_AT_BUILD:
                        continue
                if key in ShareFixedAtBuildPcdsSameValue and ShareFixedAtBuildPcdsSameValue[key]:
                    LibAuto.ConstPcd[key] = FixedAtBuildPcds[key]

    def CollectVariables(self, DynamicPcdSet):
        if False:
            for i in range(10):
                print('nop')
        VpdRegionSize = 0
        VpdRegionBase = 0
        if self.Workspace.FdfFile:
            FdDict = self.Workspace.FdfProfile.FdDict[GlobalData.gFdfParser.CurrentFdName]
            for FdRegion in FdDict.RegionList:
                for item in FdRegion.RegionDataList:
                    if self.Platform.VpdToolGuid.strip() and self.Platform.VpdToolGuid in item:
                        VpdRegionSize = FdRegion.Size
                        VpdRegionBase = FdRegion.Offset
                        break
        VariableInfo = VariableMgr(self.DscBuildDataObj._GetDefaultStores(), self.DscBuildDataObj.SkuIds)
        VariableInfo.SetVpdRegionMaxSize(VpdRegionSize)
        VariableInfo.SetVpdRegionOffset(VpdRegionBase)
        Index = 0
        for Pcd in sorted(DynamicPcdSet):
            pcdname = '.'.join((Pcd.TokenSpaceGuidCName, Pcd.TokenCName))
            for SkuName in Pcd.SkuInfoList:
                Sku = Pcd.SkuInfoList[SkuName]
                SkuId = Sku.SkuId
                if SkuId is None or SkuId == '':
                    continue
                if len(Sku.VariableName) > 0:
                    if Sku.VariableAttribute and 'NV' not in Sku.VariableAttribute:
                        continue
                    VariableGuidStructure = Sku.VariableGuidValue
                    VariableGuid = GuidStructureStringToGuidString(VariableGuidStructure)
                    for StorageName in Sku.DefaultStoreDict:
                        VariableInfo.append_variable(var_info(Index, pcdname, StorageName, SkuName, StringToArray(Sku.VariableName), VariableGuid, Sku.VariableOffset, Sku.VariableAttribute, Sku.HiiDefaultValue, Sku.DefaultStoreDict[StorageName] if Pcd.DatumType in TAB_PCD_NUMERIC_TYPES else StringToArray(Sku.DefaultStoreDict[StorageName]), Pcd.DatumType, Pcd.CustomAttribute['DscPosition'], Pcd.CustomAttribute.get('IsStru', False)))
            Index += 1
        return VariableInfo

    def UpdateNVStoreMaxSize(self, OrgVpdFile):
        if False:
            print('Hello World!')
        if self.VariableInfo:
            VpdMapFilePath = os.path.join(self.BuildDir, TAB_FV_DIRECTORY, '%s.map' % self.Platform.VpdToolGuid)
            PcdNvStoreDfBuffer = [item for item in self._DynamicPcdList if item.TokenCName == 'PcdNvStoreDefaultValueBuffer' and item.TokenSpaceGuidCName == 'gEfiMdeModulePkgTokenSpaceGuid']
            if PcdNvStoreDfBuffer:
                try:
                    OrgVpdFile.Read(VpdMapFilePath)
                    PcdItems = OrgVpdFile.GetOffset(PcdNvStoreDfBuffer[0])
                    NvStoreOffset = list(PcdItems.values())[0].strip() if PcdItems else '0'
                except:
                    EdkLogger.error('build', FILE_READ_FAILURE, 'Can not find VPD map file %s to fix up VPD offset.' % VpdMapFilePath)
                NvStoreOffset = int(NvStoreOffset, 16) if NvStoreOffset.upper().startswith('0X') else int(NvStoreOffset)
                default_skuobj = PcdNvStoreDfBuffer[0].SkuInfoList.get(TAB_DEFAULT)
                maxsize = self.VariableInfo.VpdRegionSize - NvStoreOffset if self.VariableInfo.VpdRegionSize else len(default_skuobj.DefaultValue.split(','))
                var_data = self.VariableInfo.PatchNVStoreDefaultMaxSize(maxsize)
                if var_data and default_skuobj:
                    default_skuobj.DefaultValue = var_data
                    PcdNvStoreDfBuffer[0].DefaultValue = var_data
                    PcdNvStoreDfBuffer[0].SkuInfoList.clear()
                    PcdNvStoreDfBuffer[0].SkuInfoList[TAB_DEFAULT] = default_skuobj
                    PcdNvStoreDfBuffer[0].MaxDatumSize = str(len(default_skuobj.DefaultValue.split(',')))
        return OrgVpdFile

    def CollectPlatformDynamicPcds(self):
        if False:
            i = 10
            return i + 15
        self.CategoryPcds()
        self.SortDynamicPcd()

    def CategoryPcds(self):
        if False:
            return 10
        NoDatumTypePcdList = set()
        FdfModuleList = []
        for InfName in self._AsBuildInfList:
            InfName = mws.join(self.WorkspaceDir, InfName)
            FdfModuleList.append(os.path.normpath(InfName))
        for M in self._MbList:
            ModPcdList = self.ApplyPcdSetting(M, M.ModulePcdList)
            LibPcdList = []
            for lib in M.LibraryPcdList:
                LibPcdList.extend(self.ApplyPcdSetting(M, M.LibraryPcdList[lib], lib))
            for PcdFromModule in ModPcdList + LibPcdList:
                if PcdFromModule.DatumType == TAB_VOID and (not PcdFromModule.MaxDatumSize):
                    NoDatumTypePcdList.add('%s.%s [%s]' % (PcdFromModule.TokenSpaceGuidCName, PcdFromModule.TokenCName, M.MetaFile))
                if M.IsBinaryModule == True:
                    PcdFromModule.IsFromBinaryInf = True
                PcdFromModule.IsFromDsc = (PcdFromModule.TokenCName, PcdFromModule.TokenSpaceGuidCName) in self.Platform.Pcds
                if PcdFromModule.Type in PCD_DYNAMIC_TYPE_SET or PcdFromModule.Type in PCD_DYNAMIC_EX_TYPE_SET:
                    if M.MetaFile.Path not in FdfModuleList:
                        if PcdFromModule.Type in PCD_DYNAMIC_TYPE_SET and PcdFromModule.IsFromBinaryInf == False:
                            continue
                        if PcdFromModule.Type in PCD_DYNAMIC_EX_TYPE_SET:
                            continue
                    if M.ModuleType in SUP_MODULE_SET_PEI:
                        PcdFromModule.Phase = 'PEI'
                    if PcdFromModule not in self._DynaPcdList_:
                        self._DynaPcdList_.append(PcdFromModule)
                    elif PcdFromModule.Phase == 'PEI':
                        Index = self._DynaPcdList_.index(PcdFromModule)
                        self._DynaPcdList_[Index] = PcdFromModule
                elif PcdFromModule not in self._NonDynaPcdList_:
                    self._NonDynaPcdList_.append(PcdFromModule)
                elif PcdFromModule in self._NonDynaPcdList_ and PcdFromModule.IsFromBinaryInf == True:
                    Index = self._NonDynaPcdList_.index(PcdFromModule)
                    if self._NonDynaPcdList_[Index].IsFromBinaryInf == False:
                        self._NonDynaPcdList_.remove(self._NonDynaPcdList_[Index])
                        PcdFromModule.Pending = False
                        self._NonDynaPcdList_.append(PcdFromModule)
        DscModuleSet = {os.path.normpath(ModuleInf.Path) for ModuleInf in self.Platform.Modules}
        for InfName in FdfModuleList:
            if InfName not in DscModuleSet:
                InfClass = PathClass(InfName)
                M = self.BuildDatabase[InfClass, self.Arch, self.BuildTarget, self.ToolChain]
                if not M.IsBinaryModule:
                    continue
                ModulePcdList = self.ApplyPcdSetting(M, M.Pcds)
                for PcdFromModule in ModulePcdList:
                    PcdFromModule.IsFromBinaryInf = True
                    PcdFromModule.IsFromDsc = False
                    if PcdFromModule.Type not in PCD_DYNAMIC_EX_TYPE_SET and PcdFromModule.Type not in TAB_PCDS_PATCHABLE_IN_MODULE:
                        EdkLogger.error('build', AUTOGEN_ERROR, 'PCD setting error', File=self.MetaFile, ExtraData='\n\tExisted %s PCD %s in:\n\t\t%s\n' % (PcdFromModule.Type, PcdFromModule.TokenCName, InfName))
                    if PcdFromModule.DatumType == TAB_VOID and (not PcdFromModule.MaxDatumSize):
                        NoDatumTypePcdList.add('%s.%s [%s]' % (PcdFromModule.TokenSpaceGuidCName, PcdFromModule.TokenCName, InfName))
                    if M.ModuleType in SUP_MODULE_SET_PEI:
                        PcdFromModule.Phase = 'PEI'
                    if PcdFromModule not in self._DynaPcdList_ and PcdFromModule.Type in PCD_DYNAMIC_EX_TYPE_SET:
                        self._DynaPcdList_.append(PcdFromModule)
                    elif PcdFromModule not in self._NonDynaPcdList_ and PcdFromModule.Type in TAB_PCDS_PATCHABLE_IN_MODULE:
                        self._NonDynaPcdList_.append(PcdFromModule)
                    if PcdFromModule in self._DynaPcdList_ and PcdFromModule.Phase == 'PEI' and (PcdFromModule.Type in PCD_DYNAMIC_EX_TYPE_SET):
                        Index = self._DynaPcdList_.index(PcdFromModule)
                        self._DynaPcdList_[Index].Phase = PcdFromModule.Phase
                        self._DynaPcdList_[Index].Type = PcdFromModule.Type
        for PcdFromModule in self._NonDynaPcdList_:
            if PcdFromModule not in self._DynaPcdList_:
                continue
            Index = self._DynaPcdList_.index(PcdFromModule)
            if PcdFromModule.IsFromDsc == False and PcdFromModule.Type in TAB_PCDS_PATCHABLE_IN_MODULE and (PcdFromModule.IsFromBinaryInf == True) and (self._DynaPcdList_[Index].IsFromBinaryInf == False):
                Index = self._DynaPcdList_.index(PcdFromModule)
                self._DynaPcdList_.remove(self._DynaPcdList_[Index])
        if len(NoDatumTypePcdList) > 0:
            NoDatumTypePcdListString = '\n\t\t'.join(NoDatumTypePcdList)
            EdkLogger.error('build', AUTOGEN_ERROR, 'PCD setting error', File=self.MetaFile, ExtraData='\n\tPCD(s) without MaxDatumSize:\n\t\t%s\n' % NoDatumTypePcdListString)
        self._NonDynamicPcdList = sorted(self._NonDynaPcdList_)
        self._DynamicPcdList = self._DynaPcdList_

    def SortDynamicPcd(self):
        if False:
            print('Hello World!')
        UnicodePcdArray = set()
        HiiPcdArray = set()
        OtherPcdArray = set()
        VpdPcdDict = {}
        VpdFile = VpdInfoFile.VpdInfoFile()
        NeedProcessVpdMapFile = False
        for pcd in self.Platform.Pcds:
            if pcd not in self._PlatformPcds:
                self._PlatformPcds[pcd] = self.Platform.Pcds[pcd]
        for item in self._PlatformPcds:
            if self._PlatformPcds[item].DatumType and self._PlatformPcds[item].DatumType not in [TAB_UINT8, TAB_UINT16, TAB_UINT32, TAB_UINT64, TAB_VOID, 'BOOLEAN']:
                self._PlatformPcds[item].DatumType = TAB_VOID
        if self.Workspace.ArchList[-1] == self.Arch:
            for Pcd in self._DynamicPcdList:
                Sku = Pcd.SkuInfoList.get(TAB_DEFAULT)
                Sku.VpdOffset = Sku.VpdOffset.strip()
                if Pcd.DatumType not in [TAB_UINT8, TAB_UINT16, TAB_UINT32, TAB_UINT64, TAB_VOID, 'BOOLEAN']:
                    Pcd.DatumType = TAB_VOID
                if Pcd.Type in [TAB_PCDS_DYNAMIC_VPD, TAB_PCDS_DYNAMIC_EX_VPD]:
                    VpdPcdDict[Pcd.TokenCName, Pcd.TokenSpaceGuidCName] = Pcd
            PcdNvStoreDfBuffer = VpdPcdDict.get(('PcdNvStoreDefaultValueBuffer', 'gEfiMdeModulePkgTokenSpaceGuid'))
            if PcdNvStoreDfBuffer:
                self.VariableInfo = self.CollectVariables(self._DynamicPcdList)
                vardump = self.VariableInfo.dump()
                if vardump:
                    if len(vardump.split(',')) > 65535:
                        EdkLogger.error('build', RESOURCE_OVERFLOW, 'The current length of PCD %s value is %d, it exceeds to the max size of String PCD.' % ('.'.join([PcdNvStoreDfBuffer.TokenSpaceGuidCName, PcdNvStoreDfBuffer.TokenCName]), len(vardump.split(','))))
                    PcdNvStoreDfBuffer.DefaultValue = vardump
                    for skuname in PcdNvStoreDfBuffer.SkuInfoList:
                        PcdNvStoreDfBuffer.SkuInfoList[skuname].DefaultValue = vardump
                        PcdNvStoreDfBuffer.MaxDatumSize = str(len(vardump.split(',')))
            elif [Pcd for Pcd in self._DynamicPcdList if Pcd.UserDefinedDefaultStoresFlag]:
                EdkLogger.warn('build', 'PcdNvStoreDefaultValueBuffer should be defined as PcdsDynamicExVpd in dsc file since the DefaultStores is enabled for this platform.\n%s' % self.Platform.MetaFile.Path)
            PlatformPcds = sorted(self._PlatformPcds.keys())
            VpdSkuMap = {}
            for PcdKey in PlatformPcds:
                Pcd = self._PlatformPcds[PcdKey]
                if Pcd.Type in [TAB_PCDS_DYNAMIC_VPD, TAB_PCDS_DYNAMIC_EX_VPD] and PcdKey in VpdPcdDict:
                    Pcd = VpdPcdDict[PcdKey]
                    SkuValueMap = {}
                    DefaultSku = Pcd.SkuInfoList.get(TAB_DEFAULT)
                    if DefaultSku:
                        PcdValue = DefaultSku.DefaultValue
                        if PcdValue not in SkuValueMap:
                            SkuValueMap[PcdValue] = []
                            VpdFile.Add(Pcd, TAB_DEFAULT, DefaultSku.VpdOffset)
                        SkuValueMap[PcdValue].append(DefaultSku)
                    for (SkuName, Sku) in Pcd.SkuInfoList.items():
                        Sku.VpdOffset = Sku.VpdOffset.strip()
                        PcdValue = Sku.DefaultValue
                        if PcdValue == '':
                            PcdValue = Pcd.DefaultValue
                        if Sku.VpdOffset != TAB_STAR:
                            if PcdValue.startswith('{'):
                                Alignment = 8
                            elif PcdValue.startswith('L'):
                                Alignment = 2
                            else:
                                Alignment = 1
                            try:
                                VpdOffset = int(Sku.VpdOffset)
                            except:
                                try:
                                    VpdOffset = int(Sku.VpdOffset, 16)
                                except:
                                    EdkLogger.error('build', FORMAT_INVALID, 'Invalid offset value %s for PCD %s.%s.' % (Sku.VpdOffset, Pcd.TokenSpaceGuidCName, Pcd.TokenCName))
                            if VpdOffset % Alignment != 0:
                                if PcdValue.startswith('{'):
                                    EdkLogger.warn('build', 'The offset value of PCD %s.%s is not 8-byte aligned!' % (Pcd.TokenSpaceGuidCName, Pcd.TokenCName), File=self.MetaFile)
                                else:
                                    EdkLogger.error('build', FORMAT_INVALID, 'The offset value of PCD %s.%s should be %s-byte aligned.' % (Pcd.TokenSpaceGuidCName, Pcd.TokenCName, Alignment))
                        if PcdValue not in SkuValueMap:
                            SkuValueMap[PcdValue] = []
                            VpdFile.Add(Pcd, SkuName, Sku.VpdOffset)
                        SkuValueMap[PcdValue].append(Sku)
                        if not NeedProcessVpdMapFile and Sku.VpdOffset == TAB_STAR:
                            NeedProcessVpdMapFile = True
                            if self.Platform.VpdToolGuid is None or self.Platform.VpdToolGuid == '':
                                EdkLogger.error('Build', FILE_NOT_FOUND, 'Fail to find third-party BPDG tool to process VPD PCDs. BPDG Guid tool need to be defined in tools_def.txt and VPD_TOOL_GUID need to be provided in DSC file.')
                    VpdSkuMap[PcdKey] = SkuValueMap
            for DscPcd in PlatformPcds:
                DscPcdEntry = self._PlatformPcds[DscPcd]
                if DscPcdEntry.Type in [TAB_PCDS_DYNAMIC_VPD, TAB_PCDS_DYNAMIC_EX_VPD]:
                    if not (self.Platform.VpdToolGuid is None or self.Platform.VpdToolGuid == ''):
                        FoundFlag = False
                        for VpdPcd in VpdFile._VpdArray:
                            if VpdPcd.TokenSpaceGuidCName == DscPcdEntry.TokenSpaceGuidCName and VpdPcd.TokenCName == DscPcdEntry.TokenCName:
                                FoundFlag = True
                        if not FoundFlag:
                            SkuValueMap = {}
                            SkuObjList = list(DscPcdEntry.SkuInfoList.items())
                            DefaultSku = DscPcdEntry.SkuInfoList.get(TAB_DEFAULT)
                            if DefaultSku:
                                defaultindex = SkuObjList.index((TAB_DEFAULT, DefaultSku))
                                (SkuObjList[0], SkuObjList[defaultindex]) = (SkuObjList[defaultindex], SkuObjList[0])
                            for (SkuName, Sku) in SkuObjList:
                                Sku.VpdOffset = Sku.VpdOffset.strip()
                                for eachDec in self.PackageList:
                                    for DecPcd in eachDec.Pcds:
                                        DecPcdEntry = eachDec.Pcds[DecPcd]
                                        if DecPcdEntry.TokenSpaceGuidCName == DscPcdEntry.TokenSpaceGuidCName and DecPcdEntry.TokenCName == DscPcdEntry.TokenCName:
                                            EdkLogger.warn('build', 'Unreferenced vpd pcd used!', File=self.MetaFile, ExtraData='PCD: %s.%s used in the DSC file %s is unreferenced.' % (DscPcdEntry.TokenSpaceGuidCName, DscPcdEntry.TokenCName, self.Platform.MetaFile.Path))
                                            DscPcdEntry.DatumType = DecPcdEntry.DatumType
                                            DscPcdEntry.DefaultValue = DecPcdEntry.DefaultValue
                                            DscPcdEntry.TokenValue = DecPcdEntry.TokenValue
                                            DscPcdEntry.TokenSpaceGuidValue = eachDec.Guids[DecPcdEntry.TokenSpaceGuidCName]
                                            if not Sku.DefaultValue:
                                                DscPcdEntry.SkuInfoList[list(DscPcdEntry.SkuInfoList.keys())[0]].DefaultValue = DecPcdEntry.DefaultValue
                                if DscPcdEntry not in self._DynamicPcdList:
                                    self._DynamicPcdList.append(DscPcdEntry)
                                Sku.VpdOffset = Sku.VpdOffset.strip()
                                PcdValue = Sku.DefaultValue
                                if PcdValue == '':
                                    PcdValue = DscPcdEntry.DefaultValue
                                if Sku.VpdOffset != TAB_STAR:
                                    if PcdValue.startswith('{'):
                                        Alignment = 8
                                    elif PcdValue.startswith('L'):
                                        Alignment = 2
                                    else:
                                        Alignment = 1
                                    try:
                                        VpdOffset = int(Sku.VpdOffset)
                                    except:
                                        try:
                                            VpdOffset = int(Sku.VpdOffset, 16)
                                        except:
                                            EdkLogger.error('build', FORMAT_INVALID, 'Invalid offset value %s for PCD %s.%s.' % (Sku.VpdOffset, DscPcdEntry.TokenSpaceGuidCName, DscPcdEntry.TokenCName))
                                    if VpdOffset % Alignment != 0:
                                        if PcdValue.startswith('{'):
                                            EdkLogger.warn('build', 'The offset value of PCD %s.%s is not 8-byte aligned!' % (DscPcdEntry.TokenSpaceGuidCName, DscPcdEntry.TokenCName), File=self.MetaFile)
                                        else:
                                            EdkLogger.error('build', FORMAT_INVALID, 'The offset value of PCD %s.%s should be %s-byte aligned.' % (DscPcdEntry.TokenSpaceGuidCName, DscPcdEntry.TokenCName, Alignment))
                                if PcdValue not in SkuValueMap:
                                    SkuValueMap[PcdValue] = []
                                    VpdFile.Add(DscPcdEntry, SkuName, Sku.VpdOffset)
                                SkuValueMap[PcdValue].append(Sku)
                                if not NeedProcessVpdMapFile and Sku.VpdOffset == TAB_STAR:
                                    NeedProcessVpdMapFile = True
                            if DscPcdEntry.DatumType == TAB_VOID and PcdValue.startswith('L'):
                                UnicodePcdArray.add(DscPcdEntry)
                            elif len(Sku.VariableName) > 0:
                                HiiPcdArray.add(DscPcdEntry)
                            else:
                                OtherPcdArray.add(DscPcdEntry)
                            VpdSkuMap[DscPcd] = SkuValueMap
            if (self.Platform.FlashDefinition is None or self.Platform.FlashDefinition == '') and VpdFile.GetCount() != 0:
                EdkLogger.error('build', ATTRIBUTE_NOT_AVAILABLE, 'Fail to get FLASH_DEFINITION definition in DSC file %s which is required when DSC contains VPD PCD.' % str(self.Platform.MetaFile))
            if VpdFile.GetCount() != 0:
                self.FixVpdOffset(VpdFile)
                self.FixVpdOffset(self.UpdateNVStoreMaxSize(VpdFile))
                PcdNvStoreDfBuffer = [item for item in self._DynamicPcdList if item.TokenCName == 'PcdNvStoreDefaultValueBuffer' and item.TokenSpaceGuidCName == 'gEfiMdeModulePkgTokenSpaceGuid']
                if PcdNvStoreDfBuffer:
                    (PcdName, PcdGuid) = (PcdNvStoreDfBuffer[0].TokenCName, PcdNvStoreDfBuffer[0].TokenSpaceGuidCName)
                    if (PcdName, PcdGuid) in VpdSkuMap:
                        DefaultSku = PcdNvStoreDfBuffer[0].SkuInfoList.get(TAB_DEFAULT)
                        VpdSkuMap[PcdName, PcdGuid] = {DefaultSku.DefaultValue: [SkuObj for SkuObj in PcdNvStoreDfBuffer[0].SkuInfoList.values()]}
                if NeedProcessVpdMapFile:
                    VpdMapFilePath = os.path.join(self.BuildDir, TAB_FV_DIRECTORY, '%s.map' % self.Platform.VpdToolGuid)
                    try:
                        VpdFile.Read(VpdMapFilePath)
                        for pcd in VpdSkuMap:
                            vpdinfo = VpdFile.GetVpdInfo(pcd)
                            if vpdinfo is None:
                                continue
                            for pcdvalue in VpdSkuMap[pcd]:
                                for sku in VpdSkuMap[pcd][pcdvalue]:
                                    for item in vpdinfo:
                                        if item[2] == pcdvalue:
                                            sku.VpdOffset = item[1]
                    except:
                        EdkLogger.error('build', FILE_READ_FAILURE, 'Can not find VPD map file %s to fix up VPD offset.' % VpdMapFilePath)
            for Pcd in self._DynamicPcdList:
                Sku = Pcd.SkuInfoList.get(TAB_DEFAULT)
                Sku.VpdOffset = Sku.VpdOffset.strip()
                if Pcd.DatumType not in [TAB_UINT8, TAB_UINT16, TAB_UINT32, TAB_UINT64, TAB_VOID, 'BOOLEAN']:
                    Pcd.DatumType = TAB_VOID
                PcdValue = Sku.DefaultValue
                if Pcd.DatumType == TAB_VOID and PcdValue.startswith('L'):
                    UnicodePcdArray.add(Pcd)
                elif len(Sku.VariableName) > 0:
                    HiiPcdArray.add(Pcd)
                else:
                    OtherPcdArray.add(Pcd)
            del self._DynamicPcdList[:]
        self._DynamicPcdList.extend(list(UnicodePcdArray))
        self._DynamicPcdList.extend(list(HiiPcdArray))
        self._DynamicPcdList.extend(list(OtherPcdArray))
        self._DynamicPcdList.sort()
        allskuset = [(SkuName, Sku.SkuId) for pcd in self._DynamicPcdList for (SkuName, Sku) in pcd.SkuInfoList.items()]
        for pcd in self._DynamicPcdList:
            if len(pcd.SkuInfoList) == 1:
                for (SkuName, SkuId) in allskuset:
                    if isinstance(SkuId, str) and eval(SkuId) == 0 or SkuId == 0:
                        continue
                    pcd.SkuInfoList[SkuName] = copy.deepcopy(pcd.SkuInfoList[TAB_DEFAULT])
                    pcd.SkuInfoList[SkuName].SkuId = SkuId
                    pcd.SkuInfoList[SkuName].SkuIdName = SkuName

    def FixVpdOffset(self, VpdFile):
        if False:
            while True:
                i = 10
        FvPath = os.path.join(self.BuildDir, TAB_FV_DIRECTORY)
        if not os.path.exists(FvPath):
            try:
                os.makedirs(FvPath)
            except:
                EdkLogger.error('build', FILE_WRITE_FAILURE, 'Fail to create FV folder under %s' % self.BuildDir)
        VpdFilePath = os.path.join(FvPath, '%s.txt' % self.Platform.VpdToolGuid)
        if VpdFile.Write(VpdFilePath):
            BPDGToolName = None
            for ToolDef in self.ToolDefinition.values():
                if TAB_GUID in ToolDef and ToolDef[TAB_GUID] == self.Platform.VpdToolGuid:
                    if 'PATH' not in ToolDef:
                        EdkLogger.error('build', ATTRIBUTE_NOT_AVAILABLE, 'PATH attribute was not provided for BPDG guid tool %s in tools_def.txt' % self.Platform.VpdToolGuid)
                    BPDGToolName = ToolDef['PATH']
                    break
            if BPDGToolName is not None:
                VpdInfoFile.CallExtenalBPDGTool(BPDGToolName, VpdFilePath)
            else:
                EdkLogger.error('Build', FILE_NOT_FOUND, 'Fail to find third-party BPDG tool to process VPD PCDs. BPDG Guid tool need to be defined in tools_def.txt and VPD_TOOL_GUID need to be provided in DSC file.')

    @cached_property
    def Platform(self):
        if False:
            return 10
        return self.BuildDatabase[self.MetaFile, self.Arch, self.BuildTarget, self.ToolChain]

    @cached_property
    def Name(self):
        if False:
            while True:
                i = 10
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
    def FdfFile(self):
        if False:
            print('Hello World!')
        if self.Workspace.FdfFile:
            RetVal = mws.join(self.WorkspaceDir, self.Workspace.FdfFile)
        else:
            RetVal = ''
        return RetVal

    @cached_property
    def OutputDir(self):
        if False:
            while True:
                i = 10
        return self.Platform.OutputDirectory

    @cached_property
    def BuildDir(self):
        if False:
            i = 10
            return i + 15
        if os.path.isabs(self.OutputDir):
            GlobalData.gBuildDirectory = RetVal = path.join(path.abspath(self.OutputDir), self.BuildTarget + '_' + self.ToolChain)
        else:
            GlobalData.gBuildDirectory = RetVal = path.join(self.WorkspaceDir, self.OutputDir, self.BuildTarget + '_' + self.ToolChain)
        return RetVal

    @cached_property
    def MakeFileDir(self):
        if False:
            i = 10
            return i + 15
        return path.join(self.BuildDir, self.Arch)

    @cached_property
    def BuildCommand(self):
        if False:
            print('Hello World!')
        if 'MAKE' in self.EdkIIBuildOption and 'PATH' in self.EdkIIBuildOption['MAKE']:
            Path = self.EdkIIBuildOption['MAKE']['PATH']
            if Path.startswith('='):
                Path = Path[1:].strip()
            RetVal = _SplitOption(Path)
        elif 'MAKE' in self.ToolDefinition and 'PATH' in self.ToolDefinition['MAKE']:
            RetVal = _SplitOption(self.ToolDefinition['MAKE']['PATH'])
        else:
            return []
        if 'MAKE' in self.ToolDefinition and 'FLAGS' in self.ToolDefinition['MAKE']:
            NewOption = self.ToolDefinition['MAKE']['FLAGS'].strip()
            if NewOption != '':
                RetVal += _SplitOption(NewOption)
        if 'MAKE' in self.EdkIIBuildOption and 'FLAGS' in self.EdkIIBuildOption['MAKE']:
            Flags = self.EdkIIBuildOption['MAKE']['FLAGS']
            if Flags.startswith('='):
                RetVal = [RetVal[0]] + _SplitOption(Flags[1:].strip())
            else:
                RetVal = RetVal + _SplitOption(Flags.strip())
        return RetVal

    def ToolDefinitionPriority(self, Key):
        if False:
            return 10
        KeyList = Key.split('_')
        Priority = 0
        for Index in range(0, min(4, len(KeyList))):
            if KeyList[Index] != '*':
                Priority += 1 << Index
        return Priority

    @cached_property
    def ToolDefinition(self):
        if False:
            print('Hello World!')
        ToolDefinition = self.Workspace.ToolDef.ToolsDefTxtDictionary
        if TAB_TOD_DEFINES_COMMAND_TYPE not in self.Workspace.ToolDef.ToolsDefTxtDatabase:
            EdkLogger.error('build', RESOURCE_NOT_AVAILABLE, 'No tools found in configuration', ExtraData='[%s]' % self.MetaFile)
        RetVal = OrderedDict()
        DllPathList = set()
        PrioritizedDefList = sorted(ToolDefinition.keys(), key=self.ToolDefinitionPriority, reverse=True)
        for Def in PrioritizedDefList:
            (Target, Tag, Arch, Tool, Attr) = Def.split('_')
            if Target == TAB_STAR:
                Target = self.BuildTarget
            if Tag == TAB_STAR:
                Tag = self.ToolChain
            if Arch == TAB_STAR:
                Arch = self.Arch
            if Target != self.BuildTarget or Tag != self.ToolChain or Arch != self.Arch:
                continue
            Value = ToolDefinition[Def]
            if Attr == 'DLL':
                DllPathList.add(Value)
                continue
            if Tool not in RetVal:
                RetVal[Tool] = OrderedDict()
            if Attr not in RetVal[Tool]:
                RetVal[Tool][Attr] = Value
        ToolsDef = ''
        if GlobalData.gOptions.SilentMode and 'MAKE' in RetVal:
            if 'FLAGS' not in RetVal['MAKE']:
                RetVal['MAKE']['FLAGS'] = ''
            RetVal['MAKE']['FLAGS'] += ' -s'
        MakeFlags = ''
        ToolList = list(RetVal.keys())
        ToolList.sort()
        for Tool in ToolList:
            if Tool == TAB_STAR:
                continue
            AttrList = list(RetVal[Tool].keys())
            if TAB_STAR in ToolList:
                AttrList += list(RetVal[TAB_STAR])
            AttrList.sort()
            for Attr in AttrList:
                if Attr in RetVal[Tool]:
                    Value = RetVal[Tool][Attr]
                else:
                    Value = RetVal[TAB_STAR][Attr]
                if Tool in self._BuildOptionWithToolDef(RetVal) and Attr in self._BuildOptionWithToolDef(RetVal)[Tool]:
                    if self._BuildOptionWithToolDef(RetVal)[Tool][Attr].startswith('='):
                        Value = self._BuildOptionWithToolDef(RetVal)[Tool][Attr][1:].strip()
                    elif Attr != 'PATH' and Attr != 'GUID':
                        Value += ' ' + self._BuildOptionWithToolDef(RetVal)[Tool][Attr]
                    else:
                        Value = self._BuildOptionWithToolDef(RetVal)[Tool][Attr]
                if Attr == 'PATH':
                    if Tool != 'MAKE':
                        ToolsDef += '%s_%s = %s\n' % (Tool, Attr, Value)
                elif Attr != 'DLL':
                    if Tool == 'MAKE':
                        if Attr == 'FLAGS':
                            MakeFlags = Value
                    else:
                        ToolsDef += '%s_%s = %s\n' % (Tool, Attr, Value)
            ToolsDef += '\n'
        tool_def_file = os.path.join(self.MakeFileDir, 'TOOLS_DEF.' + self.Arch)
        SaveFileOnChange(tool_def_file, ToolsDef, False)
        for DllPath in DllPathList:
            os.environ['PATH'] = DllPath + os.pathsep + os.environ['PATH']
        os.environ['MAKE_FLAGS'] = MakeFlags
        return RetVal

    @cached_property
    def ToolDefinitionFile(self):
        if False:
            while True:
                i = 10
        tool_def_file = os.path.join(self.MakeFileDir, 'TOOLS_DEF.' + self.Arch)
        if not os.path.exists(tool_def_file):
            self.ToolDefinition
        return tool_def_file

    @cached_property
    def ToolChainFamily(self):
        if False:
            for i in range(10):
                print('nop')
        ToolDefinition = self.Workspace.ToolDef.ToolsDefTxtDatabase
        if TAB_TOD_DEFINES_FAMILY not in ToolDefinition or self.ToolChain not in ToolDefinition[TAB_TOD_DEFINES_FAMILY] or (not ToolDefinition[TAB_TOD_DEFINES_FAMILY][self.ToolChain]):
            EdkLogger.verbose('No tool chain family found in configuration for %s. Default to MSFT.' % self.ToolChain)
            RetVal = TAB_COMPILER_MSFT
        else:
            RetVal = ToolDefinition[TAB_TOD_DEFINES_FAMILY][self.ToolChain]
        return RetVal

    @cached_property
    def BuildRuleFamily(self):
        if False:
            while True:
                i = 10
        ToolDefinition = self.Workspace.ToolDef.ToolsDefTxtDatabase
        if TAB_TOD_DEFINES_BUILDRULEFAMILY not in ToolDefinition or self.ToolChain not in ToolDefinition[TAB_TOD_DEFINES_BUILDRULEFAMILY] or (not ToolDefinition[TAB_TOD_DEFINES_BUILDRULEFAMILY][self.ToolChain]):
            EdkLogger.verbose('No tool chain family found in configuration for %s. Default to MSFT.' % self.ToolChain)
            return TAB_COMPILER_MSFT
        return ToolDefinition[TAB_TOD_DEFINES_BUILDRULEFAMILY][self.ToolChain]

    @cached_property
    def BuildOption(self):
        if False:
            while True:
                i = 10
        return self._ExpandBuildOption(self.Platform.BuildOptions)

    def _BuildOptionWithToolDef(self, ToolDef):
        if False:
            for i in range(10):
                print('nop')
        return self._ExpandBuildOption(self.Platform.BuildOptions, ToolDef=ToolDef)

    @cached_property
    def EdkBuildOption(self):
        if False:
            return 10
        return self._ExpandBuildOption(self.Platform.BuildOptions, EDK_NAME)

    @cached_property
    def EdkIIBuildOption(self):
        if False:
            i = 10
            return i + 15
        return self._ExpandBuildOption(self.Platform.BuildOptions, EDKII_NAME)

    @cached_property
    def BuildRule(self):
        if False:
            while True:
                i = 10
        BuildRuleFile = None
        if TAB_TAT_DEFINES_BUILD_RULE_CONF in self.Workspace.TargetTxt.TargetTxtDictionary:
            BuildRuleFile = self.Workspace.TargetTxt.TargetTxtDictionary[TAB_TAT_DEFINES_BUILD_RULE_CONF]
        if not BuildRuleFile:
            BuildRuleFile = gDefaultBuildRuleFile
        RetVal = BuildRule(BuildRuleFile)
        if RetVal._FileVersion == '':
            RetVal._FileVersion = AutoGenReqBuildRuleVerNum
        elif RetVal._FileVersion < AutoGenReqBuildRuleVerNum:
            EdkLogger.error('build', AUTOGEN_ERROR, ExtraData='The version number [%s] of build_rule.txt is less than the version number required by the AutoGen.(the minimum required version number is [%s])' % (RetVal._FileVersion, AutoGenReqBuildRuleVerNum))
        return RetVal

    @cached_property
    def PackageList(self):
        if False:
            i = 10
            return i + 15
        RetVal = set()
        for Mb in self._MbList:
            RetVal.update(Mb.Packages)
            for lb in Mb.LibInstances:
                RetVal.update(lb.Packages)
        for ModuleFile in self._AsBuildModuleList:
            if ModuleFile in self.Platform.Modules:
                continue
            ModuleData = self.BuildDatabase[ModuleFile, self.Arch, self.BuildTarget, self.ToolChain]
            RetVal.update(ModuleData.Packages)
        RetVal.update(self.Platform.Packages)
        return list(RetVal)

    @cached_property
    def NonDynamicPcdDict(self):
        if False:
            while True:
                i = 10
        return {(Pcd.TokenCName, Pcd.TokenSpaceGuidCName): Pcd for Pcd in self.NonDynamicPcdList}

    @property
    def NonDynamicPcdList(self):
        if False:
            for i in range(10):
                print('nop')
        if not self._NonDynamicPcdList:
            self.CollectPlatformDynamicPcds()
        return self._NonDynamicPcdList

    @property
    def DynamicPcdList(self):
        if False:
            return 10
        if not self._DynamicPcdList:
            self.CollectPlatformDynamicPcds()
        return self._DynamicPcdList

    @cached_property
    def PcdTokenNumber(self):
        if False:
            while True:
                i = 10
        RetVal = OrderedDict()
        TokenNumber = 1
        for Pcd in self.DynamicPcdList:
            if Pcd.Phase == 'PEI' and Pcd.Type in PCD_DYNAMIC_TYPE_SET:
                EdkLogger.debug(EdkLogger.DEBUG_5, '%s %s (%s) -> %d' % (Pcd.TokenCName, Pcd.TokenSpaceGuidCName, Pcd.Phase, TokenNumber))
                RetVal[Pcd.TokenCName, Pcd.TokenSpaceGuidCName] = TokenNumber
                TokenNumber += 1
        for Pcd in self.DynamicPcdList:
            if Pcd.Phase == 'PEI' and Pcd.Type in PCD_DYNAMIC_EX_TYPE_SET:
                EdkLogger.debug(EdkLogger.DEBUG_5, '%s %s (%s) -> %d' % (Pcd.TokenCName, Pcd.TokenSpaceGuidCName, Pcd.Phase, TokenNumber))
                RetVal[Pcd.TokenCName, Pcd.TokenSpaceGuidCName] = TokenNumber
                TokenNumber += 1
        for Pcd in self.DynamicPcdList:
            if Pcd.Phase == 'DXE' and Pcd.Type in PCD_DYNAMIC_TYPE_SET:
                EdkLogger.debug(EdkLogger.DEBUG_5, '%s %s (%s) -> %d' % (Pcd.TokenCName, Pcd.TokenSpaceGuidCName, Pcd.Phase, TokenNumber))
                RetVal[Pcd.TokenCName, Pcd.TokenSpaceGuidCName] = TokenNumber
                TokenNumber += 1
        for Pcd in self.DynamicPcdList:
            if Pcd.Phase == 'DXE' and Pcd.Type in PCD_DYNAMIC_EX_TYPE_SET:
                EdkLogger.debug(EdkLogger.DEBUG_5, '%s %s (%s) -> %d' % (Pcd.TokenCName, Pcd.TokenSpaceGuidCName, Pcd.Phase, TokenNumber))
                RetVal[Pcd.TokenCName, Pcd.TokenSpaceGuidCName] = TokenNumber
                TokenNumber += 1
        for Pcd in self.NonDynamicPcdList:
            RetVal[Pcd.TokenCName, Pcd.TokenSpaceGuidCName] = 0
        return RetVal

    @cached_property
    def _MbList(self):
        if False:
            for i in range(10):
                print('nop')
        ModuleList = []
        for m in self.Platform.Modules:
            component = self.Platform.Modules[m]
            module = self.BuildDatabase[m, self.Arch, self.BuildTarget, self.ToolChain]
            module.Guid = component.Guid
            ModuleList.append(module)
        return ModuleList

    @cached_property
    def _MaList(self):
        if False:
            print('Hello World!')
        for ModuleFile in self.Platform.Modules:
            Ma = ModuleAutoGen(self.Workspace, ModuleFile, self.BuildTarget, self.ToolChain, self.Arch, self.MetaFile, self.DataPipe)
            self.Platform.Modules[ModuleFile].M = Ma
        return [x.M for x in self.Platform.Modules.values()]

    @cached_property
    def ModuleAutoGenList(self):
        if False:
            print('Hello World!')
        RetVal = []
        for Ma in self._MaList:
            if Ma not in RetVal:
                RetVal.append(Ma)
        return RetVal

    @cached_property
    def LibraryAutoGenList(self):
        if False:
            while True:
                i = 10
        RetVal = []
        for Ma in self._MaList:
            for La in Ma.LibraryAutoGenList:
                if La not in RetVal:
                    RetVal.append(La)
                if Ma not in La.ReferenceModules:
                    La.ReferenceModules.append(Ma)
        return RetVal

    def ValidModule(self, Module):
        if False:
            print('Hello World!')
        return Module in self.Platform.Modules or Module in self.Platform.LibraryInstances or Module in self._AsBuildModuleList

    @cached_property
    def GetAllModuleInfo(self, WithoutPcd=True):
        if False:
            print('Hello World!')
        ModuleLibs = set()
        for m in self.Platform.Modules:
            module_obj = self.BuildDatabase[m, self.Arch, self.BuildTarget, self.ToolChain]
            if not bool(module_obj.LibraryClass):
                Libs = GetModuleLibInstances(module_obj, self.Platform, self.BuildDatabase, self.Arch, self.BuildTarget, self.ToolChain, self.MetaFile, EdkLogger)
            else:
                Libs = []
            ModuleLibs.update(set([(l.MetaFile.File, l.MetaFile.Root, l.MetaFile.Path, l.MetaFile.BaseName, l.MetaFile.OriginalPath, l.Arch, True) for l in Libs]))
            if WithoutPcd and module_obj.PcdIsDriver:
                continue
            ModuleLibs.add((m.File, m.Root, m.Path, m.BaseName, m.OriginalPath, module_obj.Arch, bool(module_obj.LibraryClass)))
        return ModuleLibs

    def ApplyLibraryInstance(self, Module):
        if False:
            print('Hello World!')
        if str(Module) not in self.Platform.Modules:
            return []
        return GetModuleLibInstances(Module, self.Platform, self.BuildDatabase, self.Arch, self.BuildTarget, self.ToolChain, self.MetaFile, EdkLogger)

    def _OverridePcd(self, ToPcd, FromPcd, Module='', Msg='', Library=''):
        if False:
            for i in range(10):
                print('nop')
        TokenCName = ToPcd.TokenCName
        for PcdItem in GlobalData.MixedPcd:
            if (ToPcd.TokenCName, ToPcd.TokenSpaceGuidCName) in GlobalData.MixedPcd[PcdItem]:
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
                    ToPcd.DefaultValue = ValueExpressionEx(ToPcd.DefaultValue, ToPcd.DatumType, self.Platform._GuidDict)(True)
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

    def ApplyPcdSetting(self, Module, Pcds, Library=''):
        if False:
            print('Hello World!')
        for (Name, Guid) in Pcds:
            PcdInModule = Pcds[Name, Guid]
            if (Name, Guid) in self.Platform.Pcds:
                PcdInPlatform = self.Platform.Pcds[Name, Guid]
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
        if Module in self.Platform.Modules:
            PlatformModule = self.Platform.Modules[str(Module)]
            for Key in PlatformModule.Pcds:
                if GlobalData.BuildOptionPcd:
                    for pcd in GlobalData.BuildOptionPcd:
                        (TokenSpaceGuidCName, TokenCName, FieldName, pcdvalue, _) = pcd
                        if (TokenCName, TokenSpaceGuidCName) == Key and FieldName == '':
                            PlatformModule.Pcds[Key].DefaultValue = pcdvalue
                            PlatformModule.Pcds[Key].PcdValueFromComm = pcdvalue
                            break
                Flag = False
                if Key in Pcds:
                    ToPcd = Pcds[Key]
                    Flag = True
                elif Key in GlobalData.MixedPcd:
                    for PcdItem in GlobalData.MixedPcd[Key]:
                        if PcdItem in Pcds:
                            ToPcd = Pcds[PcdItem]
                            Flag = True
                            break
                if Flag:
                    self._OverridePcd(ToPcd, PlatformModule.Pcds[Key], Module, Msg='DSC Components Module scoped PCD section', Library=Library)
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

    def ApplyBuildOption(self, Module):
        if False:
            i = 10
            return i + 15
        PlatformOptions = self.EdkIIBuildOption
        ModuleTypeOptions = self.Platform.GetBuildOptionsByModuleType(EDKII_NAME, Module.ModuleType)
        ModuleTypeOptions = self._ExpandBuildOption(ModuleTypeOptions)
        ModuleOptions = self._ExpandBuildOption(Module.BuildOptions)
        if Module in self.Platform.Modules:
            PlatformModule = self.Platform.Modules[str(Module)]
            PlatformModuleOptions = self._ExpandBuildOption(PlatformModule.BuildOptions)
        else:
            PlatformModuleOptions = {}
        BuildRuleOrder = None
        for Options in [self.ToolDefinition, ModuleOptions, PlatformOptions, ModuleTypeOptions, PlatformModuleOptions]:
            for Tool in Options:
                for Attr in Options[Tool]:
                    if Attr == TAB_TOD_DEFINES_BUILDRULEORDER:
                        BuildRuleOrder = Options[Tool][Attr]
        AllTools = set(list(ModuleOptions.keys()) + list(PlatformOptions.keys()) + list(PlatformModuleOptions.keys()) + list(ModuleTypeOptions.keys()) + list(self.ToolDefinition.keys()))
        BuildOptions = defaultdict(lambda : defaultdict(str))
        for Tool in AllTools:
            for Options in [self.ToolDefinition, ModuleOptions, PlatformOptions, ModuleTypeOptions, PlatformModuleOptions]:
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

    def GetGlobalBuildOptions(self, Module):
        if False:
            while True:
                i = 10
        ModuleTypeOptions = self.Platform.GetBuildOptionsByModuleType(EDKII_NAME, Module.ModuleType)
        ModuleTypeOptions = self._ExpandBuildOption(ModuleTypeOptions)
        if Module in self.Platform.Modules:
            PlatformModule = self.Platform.Modules[str(Module)]
            PlatformModuleOptions = self._ExpandBuildOption(PlatformModule.BuildOptions)
        else:
            PlatformModuleOptions = {}
        return (ModuleTypeOptions, PlatformModuleOptions)

    def ModuleGuid(self, Module):
        if False:
            return 10
        if os.path.basename(Module.MetaFile.File) != os.path.basename(Module.MetaFile.Path):
            return os.path.basename(Module.MetaFile.Path)[:36]
        return Module.Guid

    @cached_property
    def UniqueBaseName(self):
        if False:
            print('Hello World!')
        retVal = {}
        ModuleNameDict = {}
        UniqueName = {}
        for Module in self._MbList:
            unique_base_name = '%s_%s' % (Module.BaseName, self.ModuleGuid(Module))
            if unique_base_name not in ModuleNameDict:
                ModuleNameDict[unique_base_name] = []
            ModuleNameDict[unique_base_name].append(Module.MetaFile)
            if Module.BaseName not in UniqueName:
                UniqueName[Module.BaseName] = set()
            UniqueName[Module.BaseName].add((self.ModuleGuid(Module), Module.MetaFile))
        for module_paths in ModuleNameDict.values():
            if len(set(module_paths)) > 1:
                samemodules = list(set(module_paths))
                EdkLogger.error('build', FILE_DUPLICATED, 'Modules have same BaseName and FILE_GUID:\n  %s\n  %s' % (samemodules[0], samemodules[1]))
        for name in UniqueName:
            Guid_Path = UniqueName[name]
            if len(Guid_Path) > 1:
                for (guid, mpath) in Guid_Path:
                    retVal[name, mpath] = '%s_%s' % (name, guid)
        return retVal

    def _ExpandBuildOption(self, Options, ModuleStyle=None, ToolDef=None):
        if False:
            while True:
                i = 10
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