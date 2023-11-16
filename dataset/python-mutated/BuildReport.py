import json
from pathlib import Path
import Common.LongFilePathOs as os
import re
import platform
import textwrap
import traceback
import sys
import time
import struct
import hashlib
import subprocess
import threading
from datetime import datetime
from io import BytesIO
from Common import EdkLogger
from Common.Misc import SaveFileOnChange
from Common.Misc import GuidStructureByteArrayToGuidString
from Common.Misc import GuidStructureStringToGuidString
from Common.BuildToolError import FILE_WRITE_FAILURE
from Common.BuildToolError import CODE_ERROR
from Common.BuildToolError import COMMAND_FAILURE
from Common.BuildToolError import FORMAT_INVALID
from Common.LongFilePathSupport import OpenLongFilePath as open
from Common.MultipleWorkspace import MultipleWorkspace as mws
import Common.GlobalData as GlobalData
from AutoGen.ModuleAutoGen import ModuleAutoGen
from Common.Misc import PathClass
from Common.StringUtils import NormPath
from Common.DataType import *
import collections
from Common.Expression import *
from GenFds.AprioriSection import DXE_APRIORI_GUID, PEI_APRIORI_GUID
from AutoGen.IncludesAutoGen import IncludesAutoGen
gDxsDependencyPattern = re.compile('DEPENDENCY_START(.+)DEPENDENCY_END', re.DOTALL)
gFvTotalSizePattern = re.compile('EFI_FV_TOTAL_SIZE = (0x[0-9a-fA-F]+)')
gFvTakenSizePattern = re.compile('EFI_FV_TAKEN_SIZE = (0x[0-9a-fA-F]+)')
gModuleSizePattern = re.compile('MODULE_SIZE = (\\d+)')
gTimeStampPattern = re.compile('TIME_STAMP = (\\d+)')
gPcdGuidPattern = re.compile('PCD\\((\\w+)[.](\\w+)\\)')
gOffsetGuidPattern = re.compile('(0x[0-9A-Fa-f]+) ([-A-Fa-f0-9]+)')
gModulePattern = '\\n[-\\w]+\\s*\\(([^,]+),\\s*BaseAddress=%(Address)s,\\s*EntryPoint=%(Address)s,\\s*Type=\\w+\\)\\s*\\(GUID=([-0-9A-Fa-f]+)[^)]*\\)'
gMapFileItemPattern = re.compile(gModulePattern % {'Address': '(-?0[xX][0-9A-Fa-f]+)'})
gIncludePattern = re.compile('#include\\s*["<]([^">]+)[">]')
gIncludePattern2 = re.compile('#include\\s+EFI_([A-Z_]+)\\s*[(]\\s*(\\w+)\\s*[)]')
gGlueLibEntryPoint = re.compile('__EDKII_GLUE_MODULE_ENTRY_POINT__\\s*=\\s*(\\w+)')
gLineMaxLength = 120
gEndOfLine = '\r\n'
gSectionStart = '>' + '=' * (gLineMaxLength - 2) + '<'
gSectionEnd = '<' + '=' * (gLineMaxLength - 2) + '>' + '\n'
gSectionSep = '=' * gLineMaxLength
gSubSectionStart = '>' + '-' * (gLineMaxLength - 2) + '<'
gSubSectionEnd = '<' + '-' * (gLineMaxLength - 2) + '>'
gSubSectionSep = '-' * gLineMaxLength
gPcdTypeMap = {TAB_PCDS_FIXED_AT_BUILD: ('FIXED', TAB_PCDS_FIXED_AT_BUILD), TAB_PCDS_PATCHABLE_IN_MODULE: ('PATCH', TAB_PCDS_PATCHABLE_IN_MODULE), TAB_PCDS_FEATURE_FLAG: ('FLAG', TAB_PCDS_FEATURE_FLAG), TAB_PCDS_DYNAMIC: ('DYN', TAB_PCDS_DYNAMIC), TAB_PCDS_DYNAMIC_HII: ('DYNHII', TAB_PCDS_DYNAMIC), TAB_PCDS_DYNAMIC_VPD: ('DYNVPD', TAB_PCDS_DYNAMIC), TAB_PCDS_DYNAMIC_EX: ('DEX', TAB_PCDS_DYNAMIC_EX), TAB_PCDS_DYNAMIC_EX_HII: ('DEXHII', TAB_PCDS_DYNAMIC_EX), TAB_PCDS_DYNAMIC_EX_VPD: ('DEXVPD', TAB_PCDS_DYNAMIC_EX)}
gDriverTypeMap = {SUP_MODULE_SEC: '0x3 (SECURITY_CORE)', SUP_MODULE_PEI_CORE: '0x4 (PEI_CORE)', SUP_MODULE_PEIM: '0x6 (PEIM)', SUP_MODULE_DXE_CORE: '0x5 (DXE_CORE)', SUP_MODULE_DXE_DRIVER: '0x7 (DRIVER)', SUP_MODULE_DXE_SAL_DRIVER: '0x7 (DRIVER)', SUP_MODULE_DXE_SMM_DRIVER: '0x7 (DRIVER)', SUP_MODULE_DXE_RUNTIME_DRIVER: '0x7 (DRIVER)', SUP_MODULE_UEFI_DRIVER: '0x7 (DRIVER)', SUP_MODULE_UEFI_APPLICATION: '0x9 (APPLICATION)', SUP_MODULE_SMM_CORE: '0xD (SMM_CORE)', 'SMM_DRIVER': '0xA (SMM)', SUP_MODULE_MM_STANDALONE: '0xE (MM_STANDALONE)', SUP_MODULE_MM_CORE_STANDALONE: '0xF (MM_CORE_STANDALONE)'}
gOpCodeList = ['BEFORE', 'AFTER', 'PUSH', 'AND', 'OR', 'NOT', 'TRUE', 'FALSE', 'END', 'SOR']
VPDPcdList = []

def FileWrite(File, String, Wrapper=False):
    if False:
        return 10
    if Wrapper:
        String = textwrap.fill(String, 120)
    File.append(String + gEndOfLine)

def ByteArrayForamt(Value):
    if False:
        while True:
            i = 10
    IsByteArray = False
    SplitNum = 16
    ArrayList = []
    if Value.startswith('{') and Value.endswith('}') and (not Value.startswith('{CODE(')):
        Value = Value[1:-1]
        ValueList = Value.split(',')
        if len(ValueList) >= SplitNum:
            IsByteArray = True
    if IsByteArray:
        if ValueList:
            Len = len(ValueList) / SplitNum
            for (i, element) in enumerate(ValueList):
                ValueList[i] = '0x%02X' % int(element.strip(), 16)
        if Len:
            Id = 0
            while Id <= Len:
                End = min(SplitNum * (Id + 1), len(ValueList))
                Str = ','.join(ValueList[SplitNum * Id:End])
                if End == len(ValueList):
                    Str += '}'
                    ArrayList.append(Str)
                    break
                else:
                    Str += ','
                    ArrayList.append(Str)
                Id += 1
        else:
            ArrayList = [Value + '}']
    return (IsByteArray, ArrayList)

def FindIncludeFiles(Source, IncludePathList, IncludeFiles):
    if False:
        print('Hello World!')
    FileContents = open(Source).read()
    for Match in gIncludePattern.finditer(FileContents):
        FileName = Match.group(1).strip()
        for Dir in [os.path.dirname(Source)] + IncludePathList:
            FullFileName = os.path.normpath(os.path.join(Dir, FileName))
            if os.path.exists(FullFileName):
                IncludeFiles[FullFileName.lower().replace('\\', '/')] = FullFileName
                break
    for Match in gIncludePattern2.finditer(FileContents):
        Key = Match.group(2)
        Type = Match.group(1)
        if 'ARCH_PROTOCOL' in Type:
            FileName = 'ArchProtocol/%(Key)s/%(Key)s.h' % {'Key': Key}
        elif 'PROTOCOL' in Type:
            FileName = 'Protocol/%(Key)s/%(Key)s.h' % {'Key': Key}
        elif 'PPI' in Type:
            FileName = 'Ppi/%(Key)s/%(Key)s.h' % {'Key': Key}
        elif TAB_GUID in Type:
            FileName = 'Guid/%(Key)s/%(Key)s.h' % {'Key': Key}
        else:
            continue
        for Dir in IncludePathList:
            FullFileName = os.path.normpath(os.path.join(Dir, FileName))
            if os.path.exists(FullFileName):
                IncludeFiles[FullFileName.lower().replace('\\', '/')] = FullFileName
                break

def FileLinesSplit(Content=None, MaxLength=None):
    if False:
        i = 10
        return i + 15
    ContentList = Content.split(TAB_LINE_BREAK)
    NewContent = ''
    NewContentList = []
    for Line in ContentList:
        while len(Line.rstrip()) > MaxLength:
            LineSpaceIndex = Line.rfind(TAB_SPACE_SPLIT, 0, MaxLength)
            LineSlashIndex = Line.rfind(TAB_SLASH, 0, MaxLength)
            LineBackSlashIndex = Line.rfind(TAB_BACK_SLASH, 0, MaxLength)
            if max(LineSpaceIndex, LineSlashIndex, LineBackSlashIndex) > 0:
                LineBreakIndex = max(LineSpaceIndex, LineSlashIndex, LineBackSlashIndex)
            else:
                LineBreakIndex = MaxLength
            NewContentList.append(Line[:LineBreakIndex])
            Line = Line[LineBreakIndex:]
        if Line:
            NewContentList.append(Line)
    for NewLine in NewContentList:
        NewContent += NewLine + TAB_LINE_BREAK
    NewContent = NewContent.replace(gEndOfLine, TAB_LINE_BREAK).replace('\r\r\n', gEndOfLine)
    return NewContent

class DepexParser(object):

    def __init__(self, Wa):
        if False:
            for i in range(10):
                print('nop')
        self._GuidDb = {}
        for Pa in Wa.AutoGenObjectList:
            for Package in Pa.PackageList:
                for Protocol in Package.Protocols:
                    GuidValue = GuidStructureStringToGuidString(Package.Protocols[Protocol])
                    self._GuidDb[GuidValue.upper()] = Protocol
                for Ppi in Package.Ppis:
                    GuidValue = GuidStructureStringToGuidString(Package.Ppis[Ppi])
                    self._GuidDb[GuidValue.upper()] = Ppi
                for Guid in Package.Guids:
                    GuidValue = GuidStructureStringToGuidString(Package.Guids[Guid])
                    self._GuidDb[GuidValue.upper()] = Guid
            for Ma in Pa.ModuleAutoGenList:
                for Pcd in Ma.FixedVoidTypePcds:
                    PcdValue = Ma.FixedVoidTypePcds[Pcd]
                    if len(PcdValue.split(',')) == 16:
                        GuidValue = GuidStructureByteArrayToGuidString(PcdValue)
                        self._GuidDb[GuidValue.upper()] = Pcd

    def ParseDepexFile(self, DepexFileName):
        if False:
            return 10
        DepexFile = open(DepexFileName, 'rb')
        DepexStatement = []
        OpCode = DepexFile.read(1)
        while OpCode:
            Statement = gOpCodeList[struct.unpack('B', OpCode)[0]]
            if Statement in ['BEFORE', 'AFTER', 'PUSH']:
                GuidValue = '%08X-%04X-%04X-%02X%02X-%02X%02X%02X%02X%02X%02X' % struct.unpack(PACK_PATTERN_GUID, DepexFile.read(16))
                GuidString = self._GuidDb.get(GuidValue, GuidValue)
                Statement = '%s %s' % (Statement, GuidString)
            DepexStatement.append(Statement)
            OpCode = DepexFile.read(1)
        return DepexStatement

class LibraryReport(object):

    def __init__(self, M):
        if False:
            print('Hello World!')
        self.LibraryList = []
        for Lib in M.DependentLibraryList:
            LibInfPath = str(Lib)
            LibClassList = Lib.LibraryClass[0].LibraryClass
            LibConstructorList = Lib.ConstructorList
            LibDesstructorList = Lib.DestructorList
            LibDepexList = Lib.DepexExpression[M.Arch, M.ModuleType]
            for LibAutoGen in M.LibraryAutoGenList:
                if LibInfPath == LibAutoGen.MetaFile.Path:
                    LibTime = LibAutoGen.BuildTime
                    break
            self.LibraryList.append((LibInfPath, LibClassList, LibConstructorList, LibDesstructorList, LibDepexList, LibTime))

    def GenerateReport(self, File):
        if False:
            return 10
        if len(self.LibraryList) > 0:
            FileWrite(File, gSubSectionStart)
            FileWrite(File, TAB_BRG_LIBRARY)
            FileWrite(File, gSubSectionSep)
            for LibraryItem in self.LibraryList:
                LibInfPath = LibraryItem[0]
                FileWrite(File, LibInfPath)
                LibClass = LibraryItem[1]
                EdkIILibInfo = ''
                LibConstructor = ' '.join(LibraryItem[2])
                if LibConstructor:
                    EdkIILibInfo += ' C = ' + LibConstructor
                LibDestructor = ' '.join(LibraryItem[3])
                if LibDestructor:
                    EdkIILibInfo += ' D = ' + LibDestructor
                LibDepex = ' '.join(LibraryItem[4])
                if LibDepex:
                    EdkIILibInfo += ' Depex = ' + LibDepex
                if LibraryItem[5]:
                    EdkIILibInfo += ' Time = ' + LibraryItem[5]
                if EdkIILibInfo:
                    FileWrite(File, '{%s: %s}' % (LibClass, EdkIILibInfo))
                else:
                    FileWrite(File, '{%s}' % LibClass)
            FileWrite(File, gSubSectionEnd)

class DepexReport(object):

    def __init__(self, M):
        if False:
            for i in range(10):
                print('nop')
        self.Depex = ''
        self._DepexFileName = os.path.join(M.BuildDir, 'OUTPUT', M.Module.BaseName + '.depex')
        ModuleType = M.ModuleType
        if not ModuleType:
            ModuleType = COMPONENT_TO_MODULE_MAP_DICT.get(M.ComponentType, '')
        if ModuleType in [SUP_MODULE_SEC, SUP_MODULE_PEI_CORE, SUP_MODULE_DXE_CORE, SUP_MODULE_SMM_CORE, SUP_MODULE_MM_CORE_STANDALONE, SUP_MODULE_UEFI_APPLICATION]:
            return
        for Source in M.SourceFileList:
            if os.path.splitext(Source.Path)[1].lower() == '.dxs':
                Match = gDxsDependencyPattern.search(open(Source.Path).read())
                if Match:
                    self.Depex = Match.group(1).strip()
                    self.Source = 'DXS'
                    break
        else:
            self.Depex = M.DepexExpressionDict.get(M.ModuleType, '')
            self.ModuleDepex = ' '.join(M.Module.DepexExpression[M.Arch, M.ModuleType])
            if not self.ModuleDepex:
                self.ModuleDepex = '(None)'
            LibDepexList = []
            for Lib in M.DependentLibraryList:
                LibDepex = ' '.join(Lib.DepexExpression[M.Arch, M.ModuleType]).strip()
                if LibDepex != '':
                    LibDepexList.append('(' + LibDepex + ')')
            self.LibraryDepex = ' AND '.join(LibDepexList)
            if not self.LibraryDepex:
                self.LibraryDepex = '(None)'
            self.Source = 'INF'

    def GenerateReport(self, File, GlobalDepexParser):
        if False:
            i = 10
            return i + 15
        if not self.Depex:
            return
        FileWrite(File, gSubSectionStart)
        if os.path.isfile(self._DepexFileName):
            try:
                DepexStatements = GlobalDepexParser.ParseDepexFile(self._DepexFileName)
                FileWrite(File, 'Final Dependency Expression (DEPEX) Instructions')
                for DepexStatement in DepexStatements:
                    FileWrite(File, '  %s' % DepexStatement)
                FileWrite(File, gSubSectionSep)
            except:
                EdkLogger.warn(None, 'Dependency expression file is corrupted', self._DepexFileName)
        FileWrite(File, 'Dependency Expression (DEPEX) from %s' % self.Source)
        if self.Source == 'INF':
            FileWrite(File, self.Depex, True)
            FileWrite(File, gSubSectionSep)
            FileWrite(File, 'From Module INF:  %s' % self.ModuleDepex, True)
            FileWrite(File, 'From Library INF: %s' % self.LibraryDepex, True)
        else:
            FileWrite(File, self.Depex)
        FileWrite(File, gSubSectionEnd)

class BuildFlagsReport(object):

    def __init__(self, M):
        if False:
            for i in range(10):
                print('nop')
        BuildOptions = {}
        for Source in M.SourceFileList:
            Ext = os.path.splitext(Source.File)[1].lower()
            if Ext in ['.c', '.cc', '.cpp']:
                BuildOptions['CC'] = 1
            elif Ext in ['.s', '.asm']:
                BuildOptions['PP'] = 1
                BuildOptions['ASM'] = 1
            elif Ext in ['.vfr']:
                BuildOptions['VFRPP'] = 1
                BuildOptions['VFR'] = 1
            elif Ext in ['.dxs']:
                BuildOptions['APP'] = 1
                BuildOptions['CC'] = 1
            elif Ext in ['.asl']:
                BuildOptions['ASLPP'] = 1
                BuildOptions['ASL'] = 1
            elif Ext in ['.aslc']:
                BuildOptions['ASLCC'] = 1
                BuildOptions['ASLDLINK'] = 1
                BuildOptions['CC'] = 1
            elif Ext in ['.asm16']:
                BuildOptions['ASMLINK'] = 1
            BuildOptions['SLINK'] = 1
            BuildOptions['DLINK'] = 1
        self.ToolChainTag = M.ToolChain
        self.BuildFlags = {}
        for Tool in BuildOptions:
            self.BuildFlags[Tool + '_FLAGS'] = M.BuildOption.get(Tool, {}).get('FLAGS', '')

    def GenerateReport(self, File):
        if False:
            for i in range(10):
                print('nop')
        FileWrite(File, gSubSectionStart)
        FileWrite(File, 'Build Flags')
        FileWrite(File, 'Tool Chain Tag: %s' % self.ToolChainTag)
        for Tool in self.BuildFlags:
            FileWrite(File, gSubSectionSep)
            FileWrite(File, '%s = %s' % (Tool, self.BuildFlags[Tool]), True)
        FileWrite(File, gSubSectionEnd)

class ModuleReport(object):

    def __init__(self, M, ReportType):
        if False:
            while True:
                i = 10
        self.ModuleName = M.Module.BaseName
        self.ModuleInfPath = M.MetaFile.File
        self.ModuleArch = M.Arch
        self.FileGuid = M.Guid
        self.Size = 0
        self.BuildTimeStamp = None
        self.Hash = 0
        self.DriverType = ''
        if not M.IsLibrary:
            ModuleType = M.ModuleType
            if not ModuleType:
                ModuleType = COMPONENT_TO_MODULE_MAP_DICT.get(M.ComponentType, '')
            if ModuleType == SUP_MODULE_DXE_SMM_DRIVER:
                PiSpec = M.Module.Specification.get('PI_SPECIFICATION_VERSION', '0x00010000')
                if int(PiSpec, 0) >= 65546:
                    ModuleType = 'SMM_DRIVER'
            self.DriverType = gDriverTypeMap.get(ModuleType, '0x2 (FREE_FORM)')
        self.UefiSpecVersion = M.Module.Specification.get('UEFI_SPECIFICATION_VERSION', '')
        self.PiSpecVersion = M.Module.Specification.get('PI_SPECIFICATION_VERSION', '')
        self.PciDeviceId = M.Module.Defines.get('PCI_DEVICE_ID', '')
        self.PciVendorId = M.Module.Defines.get('PCI_VENDOR_ID', '')
        self.PciClassCode = M.Module.Defines.get('PCI_CLASS_CODE', '')
        self.BuildTime = M.BuildTime
        self._BuildDir = M.BuildDir
        self.ModulePcdSet = {}
        if 'PCD' in ReportType:
            for Pcd in M.ModulePcdList + M.LibraryPcdList:
                self.ModulePcdSet.setdefault((Pcd.TokenCName, Pcd.TokenSpaceGuidCName, Pcd.Type), (Pcd.InfDefaultValue, Pcd.DefaultValue))
        self.LibraryReport = None
        if 'LIBRARY' in ReportType:
            self.LibraryReport = LibraryReport(M)
        self.DepexReport = None
        if 'DEPEX' in ReportType:
            self.DepexReport = DepexReport(M)
        if 'BUILD_FLAGS' in ReportType:
            self.BuildFlagsReport = BuildFlagsReport(M)

    def GenerateReport(self, File, GlobalPcdReport, GlobalPredictionReport, GlobalDepexParser, ReportType):
        if False:
            return 10
        FileWrite(File, gSectionStart)
        FwReportFileName = os.path.join(self._BuildDir, 'OUTPUT', self.ModuleName + '.txt')
        if os.path.isfile(FwReportFileName):
            try:
                FileContents = open(FwReportFileName).read()
                Match = gModuleSizePattern.search(FileContents)
                if Match:
                    self.Size = int(Match.group(1))
                Match = gTimeStampPattern.search(FileContents)
                if Match:
                    self.BuildTimeStamp = datetime.utcfromtimestamp(int(Match.group(1)))
            except IOError:
                EdkLogger.warn(None, 'Fail to read report file', FwReportFileName)
        if 'HASH' in ReportType:
            OutputDir = os.path.join(self._BuildDir, 'OUTPUT')
            DefaultEFIfile = os.path.join(OutputDir, self.ModuleName + '.efi')
            if os.path.isfile(DefaultEFIfile):
                Tempfile = os.path.join(OutputDir, self.ModuleName + '_hash.tmp')
                cmd = ['GenFw', '--rebase', str(0), '-o', Tempfile, DefaultEFIfile]
                try:
                    PopenObject = subprocess.Popen(' '.join(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                except Exception as X:
                    EdkLogger.error('GenFw', COMMAND_FAILURE, ExtraData='%s: %s' % (str(X), cmd[0]))
                EndOfProcedure = threading.Event()
                EndOfProcedure.clear()
                if PopenObject.stderr:
                    StdErrThread = threading.Thread(target=ReadMessage, args=(PopenObject.stderr, EdkLogger.quiet, EndOfProcedure))
                    StdErrThread.setName('STDERR-Redirector')
                    StdErrThread.setDaemon(False)
                    StdErrThread.start()
                PopenObject.wait()
                if PopenObject.stderr:
                    StdErrThread.join()
                if PopenObject.returncode != 0:
                    EdkLogger.error('GenFw', COMMAND_FAILURE, 'Failed to generate firmware hash image for %s' % DefaultEFIfile)
                if os.path.isfile(Tempfile):
                    self.Hash = hashlib.sha1()
                    buf = open(Tempfile, 'rb').read()
                    if self.Hash.update(buf):
                        self.Hash = self.Hash.update(buf)
                    self.Hash = self.Hash.hexdigest()
                    os.remove(Tempfile)
        FileWrite(File, 'Module Summary')
        FileWrite(File, 'Module Name:          %s' % self.ModuleName)
        FileWrite(File, 'Module Arch:          %s' % self.ModuleArch)
        FileWrite(File, 'Module INF Path:      %s' % self.ModuleInfPath)
        FileWrite(File, 'File GUID:            %s' % self.FileGuid)
        if self.Size:
            FileWrite(File, 'Size:                 0x%X (%.2fK)' % (self.Size, self.Size / 1024.0))
        if self.Hash:
            FileWrite(File, 'SHA1 HASH:            %s *%s' % (self.Hash, self.ModuleName + '.efi'))
        if self.BuildTimeStamp:
            FileWrite(File, 'Build Time Stamp:     %s' % self.BuildTimeStamp)
        if self.BuildTime:
            FileWrite(File, 'Module Build Time:    %s' % self.BuildTime)
        if self.DriverType:
            FileWrite(File, 'Driver Type:          %s' % self.DriverType)
        if self.UefiSpecVersion:
            FileWrite(File, 'UEFI Spec Version:    %s' % self.UefiSpecVersion)
        if self.PiSpecVersion:
            FileWrite(File, 'PI Spec Version:      %s' % self.PiSpecVersion)
        if self.PciDeviceId:
            FileWrite(File, 'PCI Device ID:        %s' % self.PciDeviceId)
        if self.PciVendorId:
            FileWrite(File, 'PCI Vendor ID:        %s' % self.PciVendorId)
        if self.PciClassCode:
            FileWrite(File, 'PCI Class Code:       %s' % self.PciClassCode)
        FileWrite(File, gSectionSep)
        if 'PCD' in ReportType:
            GlobalPcdReport.GenerateReport(File, self.ModulePcdSet, self.FileGuid)
        if 'LIBRARY' in ReportType:
            self.LibraryReport.GenerateReport(File)
        if 'DEPEX' in ReportType:
            self.DepexReport.GenerateReport(File, GlobalDepexParser)
        if 'BUILD_FLAGS' in ReportType:
            self.BuildFlagsReport.GenerateReport(File)
        if 'FIXED_ADDRESS' in ReportType and self.FileGuid:
            GlobalPredictionReport.GenerateReport(File, self.FileGuid)
        FileWrite(File, gSectionEnd)

def ReadMessage(From, To, ExitFlag):
    if False:
        return 10
    while True:
        Line = From.readline()
        if Line is not None and Line != b'':
            To(Line.rstrip().decode(encoding='utf-8', errors='ignore'))
        else:
            break
        if ExitFlag.isSet():
            break

class PcdReport(object):

    def __init__(self, Wa):
        if False:
            print('Hello World!')
        self.AllPcds = {}
        self.UnusedPcds = {}
        self.ConditionalPcds = {}
        self.MaxLen = 0
        self.Arch = None
        if Wa.FdfProfile:
            self.FdfPcdSet = Wa.FdfProfile.PcdDict
        else:
            self.FdfPcdSet = {}
        self.DefaultStoreSingle = True
        self.SkuSingle = True
        if GlobalData.gDefaultStores and len(GlobalData.gDefaultStores) > 1:
            self.DefaultStoreSingle = False
        if GlobalData.gSkuids and len(GlobalData.gSkuids) > 1:
            self.SkuSingle = False
        self.ModulePcdOverride = {}
        for Pa in Wa.AutoGenObjectList:
            self.Arch = Pa.Arch
            for Pcd in Pa.AllPcdList:
                PcdList = self.AllPcds.setdefault(Pcd.TokenSpaceGuidCName, {}).setdefault(Pcd.Type, [])
                if Pcd not in PcdList:
                    PcdList.append(Pcd)
                if len(Pcd.TokenCName) > self.MaxLen:
                    self.MaxLen = len(Pcd.TokenCName)
            UnusedPcdFullList = []
            StructPcdDict = GlobalData.gStructurePcd.get(self.Arch, collections.OrderedDict())
            for (Name, Guid) in StructPcdDict:
                if (Name, Guid) not in Pa.Platform.Pcds:
                    Pcd = StructPcdDict[Name, Guid]
                    PcdList = self.AllPcds.setdefault(Guid, {}).setdefault(Pcd.Type, [])
                    if Pcd not in PcdList and Pcd not in UnusedPcdFullList:
                        UnusedPcdFullList.append(Pcd)
            for item in Pa.Platform.Pcds:
                Pcd = Pa.Platform.Pcds[item]
                if not Pcd.Type:
                    for T in PCD_TYPE_LIST:
                        PcdList = self.AllPcds.setdefault(Pcd.TokenSpaceGuidCName, {}).setdefault(T, [])
                        if Pcd in PcdList:
                            Pcd.Type = T
                            break
                if not Pcd.Type:
                    PcdTypeFlag = False
                    for package in Pa.PackageList:
                        for T in PCD_TYPE_LIST:
                            if (Pcd.TokenCName, Pcd.TokenSpaceGuidCName, T) in package.Pcds:
                                Pcd.Type = T
                                PcdTypeFlag = True
                                if not Pcd.DatumType:
                                    Pcd.DatumType = package.Pcds[Pcd.TokenCName, Pcd.TokenSpaceGuidCName, T].DatumType
                                break
                        if PcdTypeFlag:
                            break
                if not Pcd.DatumType:
                    PcdType = Pcd.Type
                    if PcdType.startswith(TAB_PCDS_DYNAMIC_EX):
                        PcdType = TAB_PCDS_DYNAMIC_EX
                    elif PcdType.startswith(TAB_PCDS_DYNAMIC):
                        PcdType = TAB_PCDS_DYNAMIC
                    for package in Pa.PackageList:
                        if (Pcd.TokenCName, Pcd.TokenSpaceGuidCName, PcdType) in package.Pcds:
                            Pcd.DatumType = package.Pcds[Pcd.TokenCName, Pcd.TokenSpaceGuidCName, PcdType].DatumType
                            break
                PcdList = self.AllPcds.setdefault(Pcd.TokenSpaceGuidCName, {}).setdefault(Pcd.Type, [])
                UnusedPcdList = self.UnusedPcds.setdefault(Pcd.TokenSpaceGuidCName, {}).setdefault(Pcd.Type, [])
                if Pcd in UnusedPcdList:
                    UnusedPcdList.remove(Pcd)
                if Pcd not in PcdList and Pcd not in UnusedPcdFullList:
                    UnusedPcdFullList.append(Pcd)
                if len(Pcd.TokenCName) > self.MaxLen:
                    self.MaxLen = len(Pcd.TokenCName)
            if GlobalData.gConditionalPcds:
                for PcdItem in GlobalData.gConditionalPcds:
                    if '.' in PcdItem:
                        (TokenSpaceGuidCName, TokenCName) = PcdItem.split('.')
                        if (TokenCName, TokenSpaceGuidCName) in Pa.Platform.Pcds:
                            Pcd = Pa.Platform.Pcds[TokenCName, TokenSpaceGuidCName]
                            PcdList = self.ConditionalPcds.setdefault(Pcd.TokenSpaceGuidCName, {}).setdefault(Pcd.Type, [])
                            if Pcd not in PcdList:
                                PcdList.append(Pcd)
            UnusedPcdList = []
            if UnusedPcdFullList:
                for Pcd in UnusedPcdFullList:
                    if Pcd.TokenSpaceGuidCName + '.' + Pcd.TokenCName in GlobalData.gConditionalPcds:
                        continue
                    UnusedPcdList.append(Pcd)
            for Pcd in UnusedPcdList:
                PcdList = self.UnusedPcds.setdefault(Pcd.TokenSpaceGuidCName, {}).setdefault(Pcd.Type, [])
                if Pcd not in PcdList:
                    PcdList.append(Pcd)
            for Module in Pa.Platform.Modules.values():
                for ModulePcd in Module.M.ModulePcdList + Module.M.LibraryPcdList:
                    TokenCName = ModulePcd.TokenCName
                    TokenSpaceGuid = ModulePcd.TokenSpaceGuidCName
                    ModuleDefault = ModulePcd.DefaultValue
                    ModulePath = os.path.basename(Module.M.MetaFile.File)
                    self.ModulePcdOverride.setdefault((TokenCName, TokenSpaceGuid), {})[ModulePath] = ModuleDefault
        self.DecPcdDefault = {}
        self._GuidDict = {}
        for Pa in Wa.AutoGenObjectList:
            for Package in Pa.PackageList:
                Guids = Package.Guids
                self._GuidDict.update(Guids)
                for (TokenCName, TokenSpaceGuidCName, DecType) in Package.Pcds:
                    DecDefaultValue = Package.Pcds[TokenCName, TokenSpaceGuidCName, DecType].DefaultValue
                    self.DecPcdDefault.setdefault((TokenCName, TokenSpaceGuidCName, DecType), DecDefaultValue)
        self.DscPcdDefault = {}
        for Pa in Wa.AutoGenObjectList:
            for (TokenCName, TokenSpaceGuidCName) in Pa.Platform.Pcds:
                DscDefaultValue = Pa.Platform.Pcds[TokenCName, TokenSpaceGuidCName].DscDefaultValue
                if DscDefaultValue:
                    self.DscPcdDefault[TokenCName, TokenSpaceGuidCName] = DscDefaultValue

    def GenerateReport(self, File, ModulePcdSet, ModuleGuid=None):
        if False:
            return 10
        if not ModulePcdSet:
            if self.ConditionalPcds:
                self.GenerateReportDetail(File, ModulePcdSet, 1)
            if self.UnusedPcds:
                IsEmpty = True
                for Token in self.UnusedPcds:
                    TokenDict = self.UnusedPcds[Token]
                    for Type in TokenDict:
                        if TokenDict[Type]:
                            IsEmpty = False
                            break
                    if not IsEmpty:
                        break
                if not IsEmpty:
                    self.GenerateReportDetail(File, ModulePcdSet, 2)
        self.GenerateReportDetail(File, ModulePcdSet, ModuleGuid=ModuleGuid)

    def GenerateReportDetail(self, File, ModulePcdSet, ReportSubType=0, ModuleGuid=None):
        if False:
            while True:
                i = 10
        PcdDict = self.AllPcds
        if ReportSubType == 1:
            PcdDict = self.ConditionalPcds
        elif ReportSubType == 2:
            PcdDict = self.UnusedPcds
        if not ModulePcdSet:
            FileWrite(File, gSectionStart)
            if ReportSubType == 1:
                FileWrite(File, 'Conditional Directives used by the build system')
            elif ReportSubType == 2:
                FileWrite(File, 'PCDs not used by modules or in conditional directives')
            else:
                FileWrite(File, 'Platform Configuration Database Report')
            FileWrite(File, '  *B  - PCD override in the build option')
            FileWrite(File, '  *P  - Platform scoped PCD override in DSC file')
            FileWrite(File, '  *F  - Platform scoped PCD override in FDF file')
            if not ReportSubType:
                FileWrite(File, '  *M  - Module scoped PCD override')
            FileWrite(File, gSectionSep)
        elif not ReportSubType and ModulePcdSet:
            FileWrite(File, gSubSectionStart)
            FileWrite(File, TAB_BRG_PCD)
            FileWrite(File, gSubSectionSep)
        AllPcdDict = {}
        for Key in PcdDict:
            AllPcdDict[Key] = {}
            for Type in PcdDict[Key]:
                for Pcd in PcdDict[Key][Type]:
                    AllPcdDict[Key][Pcd.TokenCName, Type] = Pcd
        for Key in sorted(AllPcdDict):
            First = True
            for (PcdTokenCName, Type) in sorted(AllPcdDict[Key]):
                Pcd = AllPcdDict[Key][PcdTokenCName, Type]
                (TypeName, DecType) = gPcdTypeMap.get(Type, ('', Type))
                MixedPcdFlag = False
                if GlobalData.MixedPcd:
                    for PcdKey in GlobalData.MixedPcd:
                        if (Pcd.TokenCName, Pcd.TokenSpaceGuidCName) in GlobalData.MixedPcd[PcdKey]:
                            PcdTokenCName = PcdKey[0]
                            MixedPcdFlag = True
                    if MixedPcdFlag and (not ModulePcdSet):
                        continue
                DecDefaultValue = self.DecPcdDefault.get((Pcd.TokenCName, Pcd.TokenSpaceGuidCName, DecType))
                DscDefaultValue = self.DscPcdDefault.get((Pcd.TokenCName, Pcd.TokenSpaceGuidCName))
                DscDefaultValBak = DscDefaultValue
                Field = ''
                for (CName, Guid, Field) in self.FdfPcdSet:
                    if CName == PcdTokenCName and Guid == Key:
                        DscDefaultValue = self.FdfPcdSet[CName, Guid, Field]
                        break
                if DscDefaultValue != DscDefaultValBak:
                    try:
                        DscDefaultValue = ValueExpressionEx(DscDefaultValue, Pcd.DatumType, self._GuidDict)(True)
                    except BadExpression as DscDefaultValue:
                        EdkLogger.error('BuildReport', FORMAT_INVALID, 'PCD Value: %s, Type: %s' % (DscDefaultValue, Pcd.DatumType))
                InfDefaultValue = None
                PcdValue = DecDefaultValue
                if DscDefaultValue:
                    PcdValue = DscDefaultValue
                if not self.IsStructurePcd(Pcd.TokenCName, Pcd.TokenSpaceGuidCName):
                    Pcd.DefaultValue = PcdValue
                PcdComponentValue = None
                if ModulePcdSet is not None:
                    if (Pcd.TokenCName, Pcd.TokenSpaceGuidCName, Type) not in ModulePcdSet:
                        continue
                    (InfDefaultValue, PcdComponentValue) = ModulePcdSet[Pcd.TokenCName, Pcd.TokenSpaceGuidCName, Type]
                    PcdValue = PcdComponentValue
                    if not self.IsStructurePcd(Pcd.TokenCName, Pcd.TokenSpaceGuidCName):
                        Pcd.DefaultValue = PcdValue
                    if InfDefaultValue:
                        try:
                            InfDefaultValue = ValueExpressionEx(InfDefaultValue, Pcd.DatumType, self._GuidDict)(True)
                        except BadExpression as InfDefaultValue:
                            EdkLogger.error('BuildReport', FORMAT_INVALID, 'PCD Value: %s, Type: %s' % (InfDefaultValue, Pcd.DatumType))
                    if InfDefaultValue == '':
                        InfDefaultValue = None
                BuildOptionMatch = False
                if GlobalData.BuildOptionPcd:
                    for pcd in GlobalData.BuildOptionPcd:
                        if (Pcd.TokenSpaceGuidCName, Pcd.TokenCName) == (pcd[0], pcd[1]):
                            if pcd[2]:
                                continue
                            PcdValue = pcd[3]
                            if not self.IsStructurePcd(Pcd.TokenCName, Pcd.TokenSpaceGuidCName):
                                Pcd.DefaultValue = PcdValue
                            BuildOptionMatch = True
                            break
                if First:
                    if ModulePcdSet is None:
                        FileWrite(File, '')
                    FileWrite(File, Key)
                    First = False
                if Pcd.DatumType in TAB_PCD_NUMERIC_TYPES:
                    if PcdValue.startswith('0') and (not PcdValue.lower().startswith('0x')) and (len(PcdValue) > 1) and PcdValue.lstrip('0'):
                        PcdValue = PcdValue.lstrip('0')
                    PcdValueNumber = int(PcdValue.strip(), 0)
                    if DecDefaultValue is None:
                        DecMatch = True
                    else:
                        if DecDefaultValue.startswith('0') and (not DecDefaultValue.lower().startswith('0x')) and (len(DecDefaultValue) > 1) and DecDefaultValue.lstrip('0'):
                            DecDefaultValue = DecDefaultValue.lstrip('0')
                        DecDefaultValueNumber = int(DecDefaultValue.strip(), 0)
                        DecMatch = DecDefaultValueNumber == PcdValueNumber
                    if InfDefaultValue is None:
                        InfMatch = True
                    else:
                        if InfDefaultValue.startswith('0') and (not InfDefaultValue.lower().startswith('0x')) and (len(InfDefaultValue) > 1) and InfDefaultValue.lstrip('0'):
                            InfDefaultValue = InfDefaultValue.lstrip('0')
                        InfDefaultValueNumber = int(InfDefaultValue.strip(), 0)
                        InfMatch = InfDefaultValueNumber == PcdValueNumber
                    if DscDefaultValue is None:
                        DscMatch = True
                    else:
                        if DscDefaultValue.startswith('0') and (not DscDefaultValue.lower().startswith('0x')) and (len(DscDefaultValue) > 1) and DscDefaultValue.lstrip('0'):
                            DscDefaultValue = DscDefaultValue.lstrip('0')
                        DscDefaultValueNumber = int(DscDefaultValue.strip(), 0)
                        DscMatch = DscDefaultValueNumber == PcdValueNumber
                else:
                    if DecDefaultValue is None:
                        DecMatch = True
                    else:
                        DecMatch = DecDefaultValue.strip() == PcdValue.strip()
                    if InfDefaultValue is None:
                        InfMatch = True
                    else:
                        InfMatch = InfDefaultValue.strip() == PcdValue.strip()
                    if DscDefaultValue is None:
                        DscMatch = True
                    else:
                        DscMatch = DscDefaultValue.strip() == PcdValue.strip()
                IsStructure = False
                if self.IsStructurePcd(Pcd.TokenCName, Pcd.TokenSpaceGuidCName):
                    IsStructure = True
                    if TypeName in ('DYNVPD', 'DEXVPD'):
                        SkuInfoList = Pcd.SkuInfoList
                    Pcd = GlobalData.gStructurePcd[self.Arch][Pcd.TokenCName, Pcd.TokenSpaceGuidCName]
                    if ModulePcdSet and ModulePcdSet.get((Pcd.TokenCName, Pcd.TokenSpaceGuidCName, Type)):
                        (InfDefaultValue, PcdComponentValue) = ModulePcdSet[Pcd.TokenCName, Pcd.TokenSpaceGuidCName, Type]
                        DscDefaultValBak = Pcd.DefaultValue
                        Pcd.DefaultValue = PcdComponentValue
                    Pcd.DatumType = Pcd.StructName
                    if TypeName in ('DYNVPD', 'DEXVPD'):
                        Pcd.SkuInfoList = SkuInfoList
                    if Pcd.PcdValueFromComm or Pcd.PcdFieldValueFromComm:
                        BuildOptionMatch = True
                        DecMatch = False
                    elif Pcd.PcdValueFromFdf or Pcd.PcdFieldValueFromFdf:
                        DscDefaultValue = True
                        DscMatch = True
                        DecMatch = False
                    elif Pcd.Type in PCD_DYNAMIC_TYPE_SET | PCD_DYNAMIC_EX_TYPE_SET:
                        DscOverride = False
                        if Pcd.DefaultFromDSC:
                            DscOverride = True
                        else:
                            DictLen = 0
                            for item in Pcd.SkuOverrideValues:
                                DictLen += len(Pcd.SkuOverrideValues[item])
                            if not DictLen:
                                DscOverride = False
                            elif not Pcd.SkuInfoList:
                                OverrideValues = Pcd.SkuOverrideValues
                                if OverrideValues:
                                    for Data in OverrideValues.values():
                                        Struct = list(Data.values())
                                        if Struct:
                                            DscOverride = self.ParseStruct(Struct[0])
                                            break
                            else:
                                SkuList = sorted(Pcd.SkuInfoList.keys())
                                for Sku in SkuList:
                                    SkuInfo = Pcd.SkuInfoList[Sku]
                                    if SkuInfo.DefaultStoreDict:
                                        DefaultStoreList = sorted(SkuInfo.DefaultStoreDict.keys())
                                        for DefaultStore in DefaultStoreList:
                                            OverrideValues = Pcd.SkuOverrideValues.get(Sku)
                                            if OverrideValues:
                                                DscOverride = self.ParseStruct(OverrideValues[DefaultStore])
                                                if DscOverride:
                                                    break
                                    if DscOverride:
                                        break
                        if DscOverride:
                            DscDefaultValue = True
                            DscMatch = True
                            DecMatch = False
                        else:
                            DecMatch = True
                    elif Pcd.DscRawValue or (ModuleGuid and ModuleGuid.replace('-', 'S') in Pcd.PcdValueFromComponents):
                        DscDefaultValue = True
                        DscMatch = True
                        DecMatch = False
                    else:
                        DscDefaultValue = False
                        DecMatch = True
                if Pcd.DatumType == 'BOOLEAN':
                    if DscDefaultValue:
                        DscDefaultValue = str(int(DscDefaultValue, 0))
                    if DecDefaultValue:
                        DecDefaultValue = str(int(DecDefaultValue, 0))
                    if InfDefaultValue:
                        InfDefaultValue = str(int(InfDefaultValue, 0))
                    if Pcd.DefaultValue:
                        Pcd.DefaultValue = str(int(Pcd.DefaultValue, 0))
                if DecMatch:
                    self.PrintPcdValue(File, Pcd, PcdTokenCName, TypeName, IsStructure, DscMatch, DscDefaultValBak, InfMatch, InfDefaultValue, DecMatch, DecDefaultValue, '  ')
                elif InfDefaultValue and InfMatch:
                    self.PrintPcdValue(File, Pcd, PcdTokenCName, TypeName, IsStructure, DscMatch, DscDefaultValBak, InfMatch, InfDefaultValue, DecMatch, DecDefaultValue, '*M')
                elif BuildOptionMatch:
                    self.PrintPcdValue(File, Pcd, PcdTokenCName, TypeName, IsStructure, DscMatch, DscDefaultValBak, InfMatch, InfDefaultValue, DecMatch, DecDefaultValue, '*B')
                elif PcdComponentValue:
                    self.PrintPcdValue(File, Pcd, PcdTokenCName, TypeName, IsStructure, DscMatch, DscDefaultValBak, InfMatch, PcdComponentValue, DecMatch, DecDefaultValue, '*M', ModuleGuid)
                elif DscDefaultValue and DscMatch:
                    if (Pcd.TokenCName, Key, Field) in self.FdfPcdSet:
                        self.PrintPcdValue(File, Pcd, PcdTokenCName, TypeName, IsStructure, DscMatch, DscDefaultValBak, InfMatch, InfDefaultValue, DecMatch, DecDefaultValue, '*F')
                    else:
                        self.PrintPcdValue(File, Pcd, PcdTokenCName, TypeName, IsStructure, DscMatch, DscDefaultValBak, InfMatch, InfDefaultValue, DecMatch, DecDefaultValue, '*P')
                if ModulePcdSet is None:
                    if IsStructure:
                        continue
                    if not TypeName in ('PATCH', 'FLAG', 'FIXED'):
                        continue
                    if not BuildOptionMatch:
                        ModuleOverride = self.ModulePcdOverride.get((Pcd.TokenCName, Pcd.TokenSpaceGuidCName), {})
                        for ModulePath in ModuleOverride:
                            ModuleDefault = ModuleOverride[ModulePath]
                            if Pcd.DatumType in TAB_PCD_NUMERIC_TYPES:
                                if ModuleDefault.startswith('0') and (not ModuleDefault.lower().startswith('0x')) and (len(ModuleDefault) > 1) and ModuleDefault.lstrip('0'):
                                    ModuleDefault = ModuleDefault.lstrip('0')
                                ModulePcdDefaultValueNumber = int(ModuleDefault.strip(), 0)
                                Match = ModulePcdDefaultValueNumber == PcdValueNumber
                                if Pcd.DatumType == 'BOOLEAN':
                                    ModuleDefault = str(ModulePcdDefaultValueNumber)
                            else:
                                Match = ModuleDefault.strip() == PcdValue.strip()
                            if Match:
                                continue
                            (IsByteArray, ArrayList) = ByteArrayForamt(ModuleDefault.strip())
                            if IsByteArray:
                                FileWrite(File, ' *M     %-*s = %s' % (self.MaxLen + 15, ModulePath, '{'))
                                for Array in ArrayList:
                                    FileWrite(File, Array)
                            else:
                                Value = ModuleDefault.strip()
                                if Pcd.DatumType in TAB_PCD_CLEAN_NUMERIC_TYPES:
                                    if Value.startswith(('0x', '0X')):
                                        Value = '{} ({:d})'.format(Value, int(Value, 0))
                                    else:
                                        Value = '0x{:X} ({})'.format(int(Value, 0), Value)
                                FileWrite(File, ' *M     %-*s = %s' % (self.MaxLen + 15, ModulePath, Value))
        if ModulePcdSet is None:
            FileWrite(File, gSectionEnd)
        elif not ReportSubType and ModulePcdSet:
            FileWrite(File, gSubSectionEnd)

    def ParseStruct(self, struct):
        if False:
            print('Hello World!')
        HasDscOverride = False
        if struct:
            for (_, Values) in list(struct.items()):
                for (Key, value) in Values.items():
                    if value[1] and value[1].endswith('.dsc'):
                        HasDscOverride = True
                        break
                if HasDscOverride == True:
                    break
        return HasDscOverride

    def PrintPcdDefault(self, File, Pcd, IsStructure, DscMatch, DscDefaultValue, InfMatch, InfDefaultValue, DecMatch, DecDefaultValue):
        if False:
            i = 10
            return i + 15
        if not DscMatch and DscDefaultValue is not None:
            Value = DscDefaultValue.strip()
            (IsByteArray, ArrayList) = ByteArrayForamt(Value)
            if IsByteArray:
                FileWrite(File, '    %*s = %s' % (self.MaxLen + 19, 'DSC DEFAULT', '{'))
                for Array in ArrayList:
                    FileWrite(File, Array)
            else:
                if Pcd.DatumType in TAB_PCD_CLEAN_NUMERIC_TYPES:
                    if Value.startswith(('0x', '0X')):
                        Value = '{} ({:d})'.format(Value, int(Value, 0))
                    else:
                        Value = '0x{:X} ({})'.format(int(Value, 0), Value)
                FileWrite(File, '    %*s = %s' % (self.MaxLen + 19, 'DSC DEFAULT', Value))
        if not InfMatch and InfDefaultValue is not None:
            Value = InfDefaultValue.strip()
            (IsByteArray, ArrayList) = ByteArrayForamt(Value)
            if IsByteArray:
                FileWrite(File, '    %*s = %s' % (self.MaxLen + 19, 'INF DEFAULT', '{'))
                for Array in ArrayList:
                    FileWrite(File, Array)
            else:
                if Pcd.DatumType in TAB_PCD_CLEAN_NUMERIC_TYPES:
                    if Value.startswith(('0x', '0X')):
                        Value = '{} ({:d})'.format(Value, int(Value, 0))
                    else:
                        Value = '0x{:X} ({})'.format(int(Value, 0), Value)
                FileWrite(File, '    %*s = %s' % (self.MaxLen + 19, 'INF DEFAULT', Value))
        if not DecMatch and DecDefaultValue is not None:
            Value = DecDefaultValue.strip()
            (IsByteArray, ArrayList) = ByteArrayForamt(Value)
            if IsByteArray:
                FileWrite(File, '    %*s = %s' % (self.MaxLen + 19, 'DEC DEFAULT', '{'))
                for Array in ArrayList:
                    FileWrite(File, Array)
            else:
                if Pcd.DatumType in TAB_PCD_CLEAN_NUMERIC_TYPES:
                    if Value.startswith(('0x', '0X')):
                        Value = '{} ({:d})'.format(Value, int(Value, 0))
                    else:
                        Value = '0x{:X} ({})'.format(int(Value, 0), Value)
                FileWrite(File, '    %*s = %s' % (self.MaxLen + 19, 'DEC DEFAULT', Value))
            if IsStructure:
                for filedvalues in Pcd.DefaultValues.values():
                    self.PrintStructureInfo(File, filedvalues)
        if DecMatch and IsStructure:
            for filedvalues in Pcd.DefaultValues.values():
                self.PrintStructureInfo(File, filedvalues)

    def PrintPcdValue(self, File, Pcd, PcdTokenCName, TypeName, IsStructure, DscMatch, DscDefaultValue, InfMatch, InfDefaultValue, DecMatch, DecDefaultValue, Flag='  ', ModuleGuid=None):
        if False:
            print('Hello World!')
        if not Pcd.SkuInfoList:
            Value = Pcd.DefaultValue
            (IsByteArray, ArrayList) = ByteArrayForamt(Value)
            if IsByteArray:
                FileWrite(File, ' %-*s   : %6s %10s = %s' % (self.MaxLen, Flag + ' ' + PcdTokenCName, TypeName, '(' + Pcd.DatumType + ')', '{'))
                for Array in ArrayList:
                    FileWrite(File, Array)
            else:
                if Pcd.DatumType in TAB_PCD_CLEAN_NUMERIC_TYPES:
                    if Value.startswith('0') and (not Value.lower().startswith('0x')) and (len(Value) > 1) and Value.lstrip('0'):
                        Value = Value.lstrip('0')
                    if Value.startswith(('0x', '0X')):
                        Value = '{} ({:d})'.format(Value, int(Value, 0))
                    else:
                        Value = '0x{:X} ({})'.format(int(Value, 0), Value)
                FileWrite(File, ' %-*s   : %6s %10s = %s' % (self.MaxLen, Flag + ' ' + PcdTokenCName, TypeName, '(' + Pcd.DatumType + ')', Value))
            if IsStructure:
                FiledOverrideFlag = False
                if (Pcd.TokenCName, Pcd.TokenSpaceGuidCName) in GlobalData.gPcdSkuOverrides:
                    OverrideValues = GlobalData.gPcdSkuOverrides[Pcd.TokenCName, Pcd.TokenSpaceGuidCName]
                else:
                    OverrideValues = Pcd.SkuOverrideValues
                FieldOverrideValues = None
                if OverrideValues:
                    for Data in OverrideValues.values():
                        Struct = list(Data.values())
                        if Struct:
                            FieldOverrideValues = Struct[0]
                            FiledOverrideFlag = True
                            break
                if Pcd.PcdFiledValueFromDscComponent and ModuleGuid and (ModuleGuid.replace('-', 'S') in Pcd.PcdFiledValueFromDscComponent):
                    FieldOverrideValues = Pcd.PcdFiledValueFromDscComponent[ModuleGuid.replace('-', 'S')]
                if FieldOverrideValues:
                    OverrideFieldStruct = self.OverrideFieldValue(Pcd, FieldOverrideValues)
                    self.PrintStructureInfo(File, OverrideFieldStruct)
                if not FiledOverrideFlag and (Pcd.PcdFieldValueFromComm or Pcd.PcdFieldValueFromFdf):
                    OverrideFieldStruct = self.OverrideFieldValue(Pcd, {})
                    self.PrintStructureInfo(File, OverrideFieldStruct)
            self.PrintPcdDefault(File, Pcd, IsStructure, DscMatch, DscDefaultValue, InfMatch, InfDefaultValue, DecMatch, DecDefaultValue)
        else:
            FirstPrint = True
            SkuList = sorted(Pcd.SkuInfoList.keys())
            for Sku in SkuList:
                SkuInfo = Pcd.SkuInfoList[Sku]
                SkuIdName = SkuInfo.SkuIdName
                if TypeName in ('DYNHII', 'DEXHII'):
                    if SkuInfo.DefaultStoreDict:
                        DefaultStoreList = sorted(SkuInfo.DefaultStoreDict.keys())
                        for DefaultStore in DefaultStoreList:
                            Value = SkuInfo.DefaultStoreDict[DefaultStore]
                            (IsByteArray, ArrayList) = ByteArrayForamt(Value)
                            if Pcd.DatumType == 'BOOLEAN':
                                Value = str(int(Value, 0))
                            if FirstPrint:
                                FirstPrint = False
                                if IsByteArray:
                                    if self.DefaultStoreSingle and self.SkuSingle:
                                        FileWrite(File, ' %-*s   : %6s %10s = %s' % (self.MaxLen, Flag + ' ' + PcdTokenCName, TypeName, '(' + Pcd.DatumType + ')', '{'))
                                    elif self.DefaultStoreSingle and (not self.SkuSingle):
                                        FileWrite(File, ' %-*s   : %6s %10s %10s = %s' % (self.MaxLen, Flag + ' ' + PcdTokenCName, TypeName, '(' + Pcd.DatumType + ')', '(' + SkuIdName + ')', '{'))
                                    elif not self.DefaultStoreSingle and self.SkuSingle:
                                        FileWrite(File, ' %-*s   : %6s %10s %10s = %s' % (self.MaxLen, Flag + ' ' + PcdTokenCName, TypeName, '(' + Pcd.DatumType + ')', '(' + DefaultStore + ')', '{'))
                                    else:
                                        FileWrite(File, ' %-*s   : %6s %10s %10s %10s = %s' % (self.MaxLen, Flag + ' ' + PcdTokenCName, TypeName, '(' + Pcd.DatumType + ')', '(' + SkuIdName + ')', '(' + DefaultStore + ')', '{'))
                                    for Array in ArrayList:
                                        FileWrite(File, Array)
                                else:
                                    if Pcd.DatumType in TAB_PCD_CLEAN_NUMERIC_TYPES:
                                        if Value.startswith(('0x', '0X')):
                                            Value = '{} ({:d})'.format(Value, int(Value, 0))
                                        else:
                                            Value = '0x{:X} ({})'.format(int(Value, 0), Value)
                                    if self.DefaultStoreSingle and self.SkuSingle:
                                        FileWrite(File, ' %-*s   : %6s %10s = %s' % (self.MaxLen, Flag + ' ' + PcdTokenCName, TypeName, '(' + Pcd.DatumType + ')', Value))
                                    elif self.DefaultStoreSingle and (not self.SkuSingle):
                                        FileWrite(File, ' %-*s   : %6s %10s %10s = %s' % (self.MaxLen, Flag + ' ' + PcdTokenCName, TypeName, '(' + Pcd.DatumType + ')', '(' + SkuIdName + ')', Value))
                                    elif not self.DefaultStoreSingle and self.SkuSingle:
                                        FileWrite(File, ' %-*s   : %6s %10s %10s = %s' % (self.MaxLen, Flag + ' ' + PcdTokenCName, TypeName, '(' + Pcd.DatumType + ')', '(' + DefaultStore + ')', Value))
                                    else:
                                        FileWrite(File, ' %-*s   : %6s %10s %10s %10s = %s' % (self.MaxLen, Flag + ' ' + PcdTokenCName, TypeName, '(' + Pcd.DatumType + ')', '(' + SkuIdName + ')', '(' + DefaultStore + ')', Value))
                            elif IsByteArray:
                                if self.DefaultStoreSingle and self.SkuSingle:
                                    FileWrite(File, ' %-*s   : %6s %10s = %s' % (self.MaxLen, ' ', TypeName, '(' + Pcd.DatumType + ')', '{'))
                                elif self.DefaultStoreSingle and (not self.SkuSingle):
                                    FileWrite(File, ' %-*s   : %6s %10s %10s = %s' % (self.MaxLen, ' ', TypeName, '(' + Pcd.DatumType + ')', '(' + SkuIdName + ')', '{'))
                                elif not self.DefaultStoreSingle and self.SkuSingle:
                                    FileWrite(File, ' %-*s   : %6s %10s %10s = %s' % (self.MaxLen, ' ', TypeName, '(' + Pcd.DatumType + ')', '(' + DefaultStore + ')', '{'))
                                else:
                                    FileWrite(File, ' %-*s   : %6s %10s %10s %10s = %s' % (self.MaxLen, ' ', TypeName, '(' + Pcd.DatumType + ')', '(' + SkuIdName + ')', '(' + DefaultStore + ')', '{'))
                                for Array in ArrayList:
                                    FileWrite(File, Array)
                            else:
                                if Pcd.DatumType in TAB_PCD_CLEAN_NUMERIC_TYPES:
                                    if Value.startswith(('0x', '0X')):
                                        Value = '{} ({:d})'.format(Value, int(Value, 0))
                                    else:
                                        Value = '0x{:X} ({})'.format(int(Value, 0), Value)
                                if self.DefaultStoreSingle and self.SkuSingle:
                                    FileWrite(File, ' %-*s   : %6s %10s = %s' % (self.MaxLen, ' ', TypeName, '(' + Pcd.DatumType + ')', Value))
                                elif self.DefaultStoreSingle and (not self.SkuSingle):
                                    FileWrite(File, ' %-*s   : %6s %10s %10s = %s' % (self.MaxLen, ' ', TypeName, '(' + Pcd.DatumType + ')', '(' + SkuIdName + ')', Value))
                                elif not self.DefaultStoreSingle and self.SkuSingle:
                                    FileWrite(File, ' %-*s   : %6s %10s %10s = %s' % (self.MaxLen, ' ', TypeName, '(' + Pcd.DatumType + ')', '(' + DefaultStore + ')', Value))
                                else:
                                    FileWrite(File, ' %-*s   : %6s %10s %10s %10s = %s' % (self.MaxLen, ' ', TypeName, '(' + Pcd.DatumType + ')', '(' + SkuIdName + ')', '(' + DefaultStore + ')', Value))
                            FileWrite(File, '%*s: %s: %s' % (self.MaxLen + 4, SkuInfo.VariableGuid, SkuInfo.VariableName, SkuInfo.VariableOffset))
                            if IsStructure:
                                OverrideValues = Pcd.SkuOverrideValues.get(Sku)
                                if OverrideValues:
                                    OverrideFieldStruct = self.OverrideFieldValue(Pcd, OverrideValues[DefaultStore])
                                    self.PrintStructureInfo(File, OverrideFieldStruct)
                            self.PrintPcdDefault(File, Pcd, IsStructure, DscMatch, DscDefaultValue, InfMatch, InfDefaultValue, DecMatch, DecDefaultValue)
                else:
                    Value = SkuInfo.DefaultValue
                    (IsByteArray, ArrayList) = ByteArrayForamt(Value)
                    if Pcd.DatumType == 'BOOLEAN':
                        Value = str(int(Value, 0))
                    if FirstPrint:
                        FirstPrint = False
                        if IsByteArray:
                            if self.SkuSingle:
                                FileWrite(File, ' %-*s   : %6s %10s = %s' % (self.MaxLen, Flag + ' ' + PcdTokenCName, TypeName, '(' + Pcd.DatumType + ')', '{'))
                            else:
                                FileWrite(File, ' %-*s   : %6s %10s %10s = %s' % (self.MaxLen, Flag + ' ' + PcdTokenCName, TypeName, '(' + Pcd.DatumType + ')', '(' + SkuIdName + ')', '{'))
                            for Array in ArrayList:
                                FileWrite(File, Array)
                        else:
                            if Pcd.DatumType in TAB_PCD_CLEAN_NUMERIC_TYPES:
                                if Value.startswith(('0x', '0X')):
                                    Value = '{} ({:d})'.format(Value, int(Value, 0))
                                else:
                                    Value = '0x{:X} ({})'.format(int(Value, 0), Value)
                            if self.SkuSingle:
                                FileWrite(File, ' %-*s   : %6s %10s = %s' % (self.MaxLen, Flag + ' ' + PcdTokenCName, TypeName, '(' + Pcd.DatumType + ')', Value))
                            else:
                                FileWrite(File, ' %-*s   : %6s %10s %10s = %s' % (self.MaxLen, Flag + ' ' + PcdTokenCName, TypeName, '(' + Pcd.DatumType + ')', '(' + SkuIdName + ')', Value))
                    elif IsByteArray:
                        if self.SkuSingle:
                            FileWrite(File, ' %-*s   : %6s %10s = %s' % (self.MaxLen, ' ', TypeName, '(' + Pcd.DatumType + ')', '{'))
                        else:
                            FileWrite(File, ' %-*s   : %6s %10s %10s = %s' % (self.MaxLen, ' ', TypeName, '(' + Pcd.DatumType + ')', '(' + SkuIdName + ')', '{'))
                        for Array in ArrayList:
                            FileWrite(File, Array)
                    else:
                        if Pcd.DatumType in TAB_PCD_CLEAN_NUMERIC_TYPES:
                            if Value.startswith(('0x', '0X')):
                                Value = '{} ({:d})'.format(Value, int(Value, 0))
                            else:
                                Value = '0x{:X} ({})'.format(int(Value, 0), Value)
                        if self.SkuSingle:
                            FileWrite(File, ' %-*s   : %6s %10s = %s' % (self.MaxLen, ' ', TypeName, '(' + Pcd.DatumType + ')', Value))
                        else:
                            FileWrite(File, ' %-*s   : %6s %10s %10s = %s' % (self.MaxLen, ' ', TypeName, '(' + Pcd.DatumType + ')', '(' + SkuIdName + ')', Value))
                    if TypeName in ('DYNVPD', 'DEXVPD'):
                        FileWrite(File, '%*s' % (self.MaxLen + 4, SkuInfo.VpdOffset))
                        VPDPcdItem = (Pcd.TokenSpaceGuidCName + '.' + PcdTokenCName, SkuIdName, SkuInfo.VpdOffset, Pcd.MaxDatumSize, SkuInfo.DefaultValue)
                        if VPDPcdItem not in VPDPcdList:
                            PcdGuidList = self.UnusedPcds.get(Pcd.TokenSpaceGuidCName)
                            if PcdGuidList:
                                PcdList = PcdGuidList.get(Pcd.Type)
                                if not PcdList:
                                    VPDPcdList.append(VPDPcdItem)
                                for VpdPcd in PcdList:
                                    if PcdTokenCName == VpdPcd.TokenCName:
                                        break
                                else:
                                    VPDPcdList.append(VPDPcdItem)
                    if IsStructure:
                        FiledOverrideFlag = False
                        OverrideValues = Pcd.SkuOverrideValues.get(Sku)
                        if OverrideValues:
                            Keys = list(OverrideValues.keys())
                            OverrideFieldStruct = self.OverrideFieldValue(Pcd, OverrideValues[Keys[0]])
                            self.PrintStructureInfo(File, OverrideFieldStruct)
                            FiledOverrideFlag = True
                        if not FiledOverrideFlag and (Pcd.PcdFieldValueFromComm or Pcd.PcdFieldValueFromFdf):
                            OverrideFieldStruct = self.OverrideFieldValue(Pcd, {})
                            self.PrintStructureInfo(File, OverrideFieldStruct)
                    self.PrintPcdDefault(File, Pcd, IsStructure, DscMatch, DscDefaultValue, InfMatch, InfDefaultValue, DecMatch, DecDefaultValue)

    def OverrideFieldValue(self, Pcd, OverrideStruct):
        if False:
            return 10
        OverrideFieldStruct = collections.OrderedDict()
        if OverrideStruct:
            for (_, Values) in OverrideStruct.items():
                for (Key, value) in Values.items():
                    if value[1] and value[1].endswith('.dsc'):
                        OverrideFieldStruct[Key] = value
        if Pcd.PcdFieldValueFromFdf:
            for (Key, Values) in Pcd.PcdFieldValueFromFdf.items():
                if Key in OverrideFieldStruct and Values[0] == OverrideFieldStruct[Key][0]:
                    continue
                OverrideFieldStruct[Key] = Values
        if Pcd.PcdFieldValueFromComm:
            for (Key, Values) in Pcd.PcdFieldValueFromComm.items():
                if Key in OverrideFieldStruct and Values[0] == OverrideFieldStruct[Key][0]:
                    continue
                OverrideFieldStruct[Key] = Values
        return OverrideFieldStruct

    def PrintStructureInfo(self, File, Struct):
        if False:
            print('Hello World!')
        for (Key, Value) in sorted(Struct.items(), key=lambda x: x[0]):
            if Value[1] and 'build command options' in Value[1]:
                FileWrite(File, '    *B  %-*s = %s' % (self.MaxLen + 4, '.' + Key, Value[0]))
            elif Value[1] and Value[1].endswith('.fdf'):
                FileWrite(File, '    *F  %-*s = %s' % (self.MaxLen + 4, '.' + Key, Value[0]))
            else:
                FileWrite(File, '        %-*s = %s' % (self.MaxLen + 4, '.' + Key, Value[0]))

    def StrtoHex(self, value):
        if False:
            for i in range(10):
                print('nop')
        try:
            value = hex(int(value))
            return value
        except:
            if value.startswith('L"') and value.endswith('"'):
                valuelist = []
                for ch in value[2:-1]:
                    valuelist.append(hex(ord(ch)))
                    valuelist.append('0x00')
                return valuelist
            elif value.startswith('"') and value.endswith('"'):
                return hex(ord(value[1:-1]))
            elif value.startswith('{') and value.endswith('}'):
                valuelist = []
                if ',' not in value:
                    return value[1:-1]
                for ch in value[1:-1].split(','):
                    ch = ch.strip()
                    if ch.startswith('0x') or ch.startswith('0X'):
                        valuelist.append(ch)
                        continue
                    try:
                        valuelist.append(hex(int(ch.strip())))
                    except:
                        pass
                return valuelist
            else:
                return value

    def IsStructurePcd(self, PcdToken, PcdTokenSpaceGuid):
        if False:
            while True:
                i = 10
        if GlobalData.gStructurePcd and self.Arch in GlobalData.gStructurePcd and ((PcdToken, PcdTokenSpaceGuid) in GlobalData.gStructurePcd[self.Arch]):
            return True
        else:
            return False

class PredictionReport(object):

    def __init__(self, Wa):
        if False:
            i = 10
            return i + 15
        self._MapFileName = os.path.join(Wa.BuildDir, Wa.Name + '.map')
        self._MapFileParsed = False
        self._EotToolInvoked = False
        self._FvDir = Wa.FvDir
        self._EotDir = Wa.BuildDir
        self._FfsEntryPoint = {}
        self._GuidMap = {}
        self._SourceList = []
        self.FixedMapDict = {}
        self.ItemList = []
        self.MaxLen = 0
        for Pa in Wa.AutoGenObjectList:
            for Module in Pa.LibraryAutoGenList + Pa.ModuleAutoGenList:
                if Module.ModuleType == SUP_MODULE_BASE:
                    continue
                self._SourceList.append(str(Module))
                IncludeList = {}
                for Source in Module.SourceFileList:
                    if os.path.splitext(str(Source))[1].lower() == '.c':
                        self._SourceList.append('  ' + str(Source))
                        FindIncludeFiles(Source.Path, Module.IncludePathList, IncludeList)
                for IncludeFile in IncludeList.values():
                    self._SourceList.append('  ' + IncludeFile)
                for Guid in Module.PpiList:
                    self._GuidMap[Guid] = GuidStructureStringToGuidString(Module.PpiList[Guid])
                for Guid in Module.ProtocolList:
                    self._GuidMap[Guid] = GuidStructureStringToGuidString(Module.ProtocolList[Guid])
                for Guid in Module.GuidList:
                    self._GuidMap[Guid] = GuidStructureStringToGuidString(Module.GuidList[Guid])
                if Module.Guid and (not Module.IsLibrary):
                    EntryPoint = ' '.join(Module.Module.ModuleEntryPointList)
                    RealEntryPoint = '_ModuleEntryPoint'
                    self._FfsEntryPoint[Module.Guid.upper()] = (EntryPoint, RealEntryPoint)
        self._FvList = []
        if Wa.FdfProfile:
            for Fd in Wa.FdfProfile.FdDict:
                for FdRegion in Wa.FdfProfile.FdDict[Fd].RegionList:
                    if FdRegion.RegionType != BINARY_FILE_TYPE_FV:
                        continue
                    for FvName in FdRegion.RegionDataList:
                        if FvName in self._FvList:
                            continue
                        self._FvList.append(FvName)
                        for Ffs in Wa.FdfProfile.FvDict[FvName.upper()].FfsList:
                            for Section in Ffs.SectionList:
                                try:
                                    for FvSection in Section.SectionList:
                                        if FvSection.FvName in self._FvList:
                                            continue
                                        self._FvList.append(FvSection.FvName)
                                except AttributeError:
                                    pass

    def _ParseMapFile(self):
        if False:
            i = 10
            return i + 15
        if self._MapFileParsed:
            return
        self._MapFileParsed = True
        if os.path.isfile(self._MapFileName):
            try:
                FileContents = open(self._MapFileName).read()
                for Match in gMapFileItemPattern.finditer(FileContents):
                    AddressType = Match.group(1)
                    BaseAddress = Match.group(2)
                    EntryPoint = Match.group(3)
                    Guid = Match.group(4).upper()
                    List = self.FixedMapDict.setdefault(Guid, [])
                    List.append((AddressType, BaseAddress, '*I'))
                    List.append((AddressType, EntryPoint, '*E'))
            except:
                EdkLogger.warn(None, 'Cannot open file to read', self._MapFileName)

    def _InvokeEotTool(self):
        if False:
            return 10
        if self._EotToolInvoked:
            return
        self._EotToolInvoked = True
        FvFileList = []
        for FvName in self._FvList:
            FvFile = os.path.join(self._FvDir, FvName + '.Fv')
            if os.path.isfile(FvFile):
                FvFileList.append(FvFile)
        if len(FvFileList) == 0:
            return
        SourceList = os.path.join(self._EotDir, 'SourceFile.txt')
        GuidList = os.path.join(self._EotDir, 'GuidList.txt')
        DispatchList = os.path.join(self._EotDir, 'Dispatch.txt')
        TempFile = []
        for Item in self._SourceList:
            FileWrite(TempFile, Item)
        SaveFileOnChange(SourceList, ''.join(TempFile), False)
        TempFile = []
        for Key in self._GuidMap:
            FileWrite(TempFile, '%s %s' % (Key, self._GuidMap[Key]))
        SaveFileOnChange(GuidList, ''.join(TempFile), False)
        try:
            from Eot.EotMain import Eot
            EotStartTime = time.time()
            Eot(CommandLineOption=False, SourceFileList=SourceList, GuidList=GuidList, FvFileList=' '.join(FvFileList), Dispatch=DispatchList, IsInit=True)
            EotEndTime = time.time()
            EotDuration = time.strftime('%H:%M:%S', time.gmtime(int(round(EotEndTime - EotStartTime))))
            EdkLogger.quiet('EOT run time: %s\n' % EotDuration)
            for Line in open(DispatchList):
                if len(Line.split()) < 4:
                    continue
                (Guid, Phase, FfsName, FilePath) = Line.split()
                Symbol = self._FfsEntryPoint.get(Guid, [FfsName, ''])[0]
                if len(Symbol) > self.MaxLen:
                    self.MaxLen = len(Symbol)
                self.ItemList.append((Phase, Symbol, FilePath))
        except:
            EdkLogger.quiet('(Python %s on %s\n%s)' % (platform.python_version(), sys.platform, traceback.format_exc()))
            EdkLogger.warn(None, 'Failed to generate execution order prediction report, for some error occurred in executing EOT.')

    def _GenerateExecutionOrderReport(self, File):
        if False:
            print('Hello World!')
        self._InvokeEotTool()
        if len(self.ItemList) == 0:
            return
        FileWrite(File, gSectionStart)
        FileWrite(File, 'Execution Order Prediction')
        FileWrite(File, '*P PEI phase')
        FileWrite(File, '*D DXE phase')
        FileWrite(File, '*E Module INF entry point name')
        FileWrite(File, '*N Module notification function name')
        FileWrite(File, 'Type %-*s %s' % (self.MaxLen, 'Symbol', 'Module INF Path'))
        FileWrite(File, gSectionSep)
        for Item in self.ItemList:
            FileWrite(File, '*%sE  %-*s %s' % (Item[0], self.MaxLen, Item[1], Item[2]))
        FileWrite(File, gSectionStart)

    def _GenerateFixedAddressReport(self, File, Guid, NotifyList):
        if False:
            while True:
                i = 10
        self._ParseMapFile()
        FixedAddressList = self.FixedMapDict.get(Guid)
        if not FixedAddressList:
            return
        FileWrite(File, gSubSectionStart)
        FileWrite(File, 'Fixed Address Prediction')
        FileWrite(File, '*I  Image Loading Address')
        FileWrite(File, '*E  Entry Point Address')
        FileWrite(File, '*N  Notification Function Address')
        FileWrite(File, '*F  Flash Address')
        FileWrite(File, '*M  Memory Address')
        FileWrite(File, '*S  SMM RAM Offset')
        FileWrite(File, 'TOM Top of Memory')
        FileWrite(File, 'Type Address           Name')
        FileWrite(File, gSubSectionSep)
        for Item in FixedAddressList:
            Type = Item[0]
            Value = Item[1]
            Symbol = Item[2]
            if Symbol == '*I':
                Name = '(Image Base)'
            elif Symbol == '*E':
                Name = self._FfsEntryPoint.get(Guid, ['', '_ModuleEntryPoint'])[1]
            elif Symbol in NotifyList:
                Name = Symbol
                Symbol = '*N'
            else:
                continue
            if 'Flash' in Type:
                Symbol += 'F'
            elif 'Memory' in Type:
                Symbol += 'M'
            else:
                Symbol += 'S'
            if Value[0] == '-':
                Value = 'TOM' + Value
            FileWrite(File, '%s  %-16s  %s' % (Symbol, Value, Name))

    def GenerateReport(self, File, Guid):
        if False:
            i = 10
            return i + 15
        if Guid:
            self._GenerateFixedAddressReport(File, Guid.upper(), [])
        else:
            self._GenerateExecutionOrderReport(File)

class FdRegionReport(object):

    def _DiscoverNestedFvList(self, FvName, Wa):
        if False:
            while True:
                i = 10
        FvDictKey = FvName.upper()
        if FvDictKey in Wa.FdfProfile.FvDict:
            for Ffs in Wa.FdfProfile.FvDict[FvName.upper()].FfsList:
                for Section in Ffs.SectionList:
                    try:
                        for FvSection in Section.SectionList:
                            if FvSection.FvName in self.FvList:
                                continue
                            self._GuidsDb[Ffs.NameGuid.upper()] = FvSection.FvName
                            self.FvList.append(FvSection.FvName)
                            self.FvInfo[FvSection.FvName] = ('Nested FV', 0, 0)
                            self._DiscoverNestedFvList(FvSection.FvName, Wa)
                    except AttributeError:
                        pass

    def __init__(self, FdRegion, Wa):
        if False:
            print('Hello World!')
        self.Type = FdRegion.RegionType
        self.BaseAddress = FdRegion.Offset
        self.Size = FdRegion.Size
        self.FvList = []
        self.FvInfo = {}
        self._GuidsDb = {}
        self._FvDir = Wa.FvDir
        self._WorkspaceDir = Wa.WorkspaceDir
        if self.Type != BINARY_FILE_TYPE_FV:
            return
        for FvName in FdRegion.RegionDataList:
            if FvName in self.FvList:
                continue
            self.FvList.append(FvName)
            self.FvInfo[FvName] = ('Fd Region', self.BaseAddress, self.Size)
            self._DiscoverNestedFvList(FvName, Wa)
        PlatformPcds = {}
        for Pa in Wa.AutoGenObjectList:
            for Package in Pa.PackageList:
                for (TokenCName, TokenSpaceGuidCName, DecType) in Package.Pcds:
                    DecDefaultValue = Package.Pcds[TokenCName, TokenSpaceGuidCName, DecType].DefaultValue
                    PlatformPcds[TokenCName, TokenSpaceGuidCName] = DecDefaultValue
        for Pa in Wa.AutoGenObjectList:
            for (TokenCName, TokenSpaceGuidCName) in Pa.Platform.Pcds:
                DscDefaultValue = Pa.Platform.Pcds[TokenCName, TokenSpaceGuidCName].DefaultValue
                PlatformPcds[TokenCName, TokenSpaceGuidCName] = DscDefaultValue
        self._GuidsDb[PEI_APRIORI_GUID] = 'PEI Apriori'
        self._GuidsDb[DXE_APRIORI_GUID] = 'DXE Apriori'
        self._GuidsDb['7E374E25-8E01-4FEE-87F2-390C23C606CD'] = 'ACPI table storage'
        for Pa in Wa.AutoGenObjectList:
            for ModuleKey in Pa.Platform.Modules:
                M = Pa.Platform.Modules[ModuleKey].M
                InfPath = mws.join(Wa.WorkspaceDir, M.MetaFile.File)
                self._GuidsDb[M.Guid.upper()] = '%s (%s)' % (M.Module.BaseName, InfPath)
        for FvName in self.FvList:
            FvDictKey = FvName.upper()
            if FvDictKey in Wa.FdfProfile.FvDict:
                for Ffs in Wa.FdfProfile.FvDict[FvName.upper()].FfsList:
                    try:
                        Guid = Ffs.NameGuid.upper()
                        Match = gPcdGuidPattern.match(Ffs.NameGuid)
                        if Match:
                            PcdTokenspace = Match.group(1)
                            PcdToken = Match.group(2)
                            if (PcdToken, PcdTokenspace) in PlatformPcds:
                                GuidValue = PlatformPcds[PcdToken, PcdTokenspace]
                                Guid = GuidStructureByteArrayToGuidString(GuidValue).upper()
                        for Section in Ffs.SectionList:
                            try:
                                ModuleSectFile = mws.join(Wa.WorkspaceDir, Section.SectFileName)
                                self._GuidsDb[Guid] = ModuleSectFile
                            except AttributeError:
                                pass
                    except AttributeError:
                        pass

    def _GenerateReport(self, File, Title, Type, BaseAddress, Size=0, FvName=None):
        if False:
            while True:
                i = 10
        FileWrite(File, gSubSectionStart)
        FileWrite(File, Title)
        FileWrite(File, 'Type:               %s' % Type)
        FileWrite(File, 'Base Address:       0x%X' % BaseAddress)
        if self.Type == BINARY_FILE_TYPE_FV:
            FvTotalSize = 0
            FvTakenSize = 0
            FvFreeSize = 0
            if FvName.upper().endswith('.FV'):
                FileExt = FvName + '.txt'
            else:
                FileExt = FvName + '.Fv.txt'
            if not os.path.isfile(FileExt):
                FvReportFileName = mws.join(self._WorkspaceDir, FileExt)
                if not os.path.isfile(FvReportFileName):
                    FvReportFileName = os.path.join(self._FvDir, FileExt)
            try:
                FvReport = open(FvReportFileName).read()
                Match = gFvTotalSizePattern.search(FvReport)
                if Match:
                    FvTotalSize = int(Match.group(1), 16)
                Match = gFvTakenSizePattern.search(FvReport)
                if Match:
                    FvTakenSize = int(Match.group(1), 16)
                FvFreeSize = FvTotalSize - FvTakenSize
                FileWrite(File, 'Size:               0x%X (%.0fK)' % (FvTotalSize, FvTotalSize / 1024.0))
                FileWrite(File, 'Fv Name:            %s (%.1f%% Full)' % (FvName, FvTakenSize * 100.0 / FvTotalSize))
                FileWrite(File, 'Occupied Size:      0x%X (%.0fK)' % (FvTakenSize, FvTakenSize / 1024.0))
                FileWrite(File, 'Free Size:          0x%X (%.0fK)' % (FvFreeSize, FvFreeSize / 1024.0))
                FileWrite(File, 'Offset     Module')
                FileWrite(File, gSubSectionSep)
                OffsetInfo = {}
                for Match in gOffsetGuidPattern.finditer(FvReport):
                    Guid = Match.group(2).upper()
                    OffsetInfo[Match.group(1)] = self._GuidsDb.get(Guid, Guid)
                OffsetList = sorted(OffsetInfo.keys())
                for Offset in OffsetList:
                    FileWrite(File, '%s %s' % (Offset, OffsetInfo[Offset]))
            except IOError:
                EdkLogger.warn(None, 'Fail to read report file', FvReportFileName)
        else:
            FileWrite(File, 'Size:               0x%X (%.0fK)' % (Size, Size / 1024.0))
        FileWrite(File, gSubSectionEnd)

    def GenerateReport(self, File):
        if False:
            return 10
        if len(self.FvList) > 0:
            for FvItem in self.FvList:
                Info = self.FvInfo[FvItem]
                self._GenerateReport(File, Info[0], TAB_FV_DIRECTORY, Info[1], Info[2], FvItem)
        else:
            self._GenerateReport(File, 'FD Region', self.Type, self.BaseAddress, self.Size)

class FdReport(object):

    def __init__(self, Fd, Wa):
        if False:
            while True:
                i = 10
        self.FdName = Fd.FdUiName
        self.BaseAddress = Fd.BaseAddress
        self.Size = Fd.Size
        self.FdRegionList = [FdRegionReport(FdRegion, Wa) for FdRegion in Fd.RegionList]
        self.FvPath = os.path.join(Wa.BuildDir, TAB_FV_DIRECTORY)
        self.VPDBaseAddress = 0
        self.VPDSize = 0
        for (index, FdRegion) in enumerate(Fd.RegionList):
            if str(FdRegion.RegionType) == 'FILE' and Wa.Platform.VpdToolGuid in str(FdRegion.RegionDataList):
                self.VPDBaseAddress = self.FdRegionList[index].BaseAddress
                self.VPDSize = self.FdRegionList[index].Size
                break

    def GenerateReport(self, File):
        if False:
            return 10
        FileWrite(File, gSectionStart)
        FileWrite(File, 'Firmware Device (FD)')
        FileWrite(File, 'FD Name:            %s' % self.FdName)
        FileWrite(File, 'Base Address:       %s' % self.BaseAddress)
        FileWrite(File, 'Size:               0x%X (%.0fK)' % (self.Size, self.Size / 1024.0))
        if len(self.FdRegionList) > 0:
            FileWrite(File, gSectionSep)
            for FdRegionItem in self.FdRegionList:
                FdRegionItem.GenerateReport(File)
        if VPDPcdList:
            VPDPcdList.sort(key=lambda x: int(x[2], 0))
            FileWrite(File, gSubSectionStart)
            FileWrite(File, 'FD VPD Region')
            FileWrite(File, 'Base Address:       0x%X' % self.VPDBaseAddress)
            FileWrite(File, 'Size:               0x%X (%.0fK)' % (self.VPDSize, self.VPDSize / 1024.0))
            FileWrite(File, gSubSectionSep)
            for item in VPDPcdList:
                Offset = '0x%08X' % (int(item[2], 16) + self.VPDBaseAddress)
                (IsByteArray, ArrayList) = ByteArrayForamt(item[-1])
                Skuinfo = item[1]
                if len(GlobalData.gSkuids) == 1:
                    Skuinfo = GlobalData.gSkuids[0]
                if IsByteArray:
                    FileWrite(File, '%s | %s | %s | %s | %s' % (item[0], Skuinfo, Offset, item[3], '{'))
                    for Array in ArrayList:
                        FileWrite(File, Array)
                else:
                    FileWrite(File, '%s | %s | %s | %s | %s' % (item[0], Skuinfo, Offset, item[3], item[-1]))
            FileWrite(File, gSubSectionEnd)
        FileWrite(File, gSectionEnd)

class PlatformReport(object):

    def __init__(self, Wa, MaList, ReportType):
        if False:
            for i in range(10):
                print('nop')
        self._WorkspaceDir = Wa.WorkspaceDir
        self.PlatformName = Wa.Name
        self.PlatformDscPath = Wa.Platform
        self.Architectures = ' '.join(Wa.ArchList)
        self.ToolChain = Wa.ToolChain
        self.Target = Wa.BuildTarget
        self.OutputPath = os.path.join(Wa.WorkspaceDir, Wa.OutputDir)
        self.BuildEnvironment = platform.platform()
        self.PcdReport = None
        if 'PCD' in ReportType:
            self.PcdReport = PcdReport(Wa)
        self.FdReportList = []
        if 'FLASH' in ReportType and Wa.FdfProfile and (MaList is None):
            for Fd in Wa.FdfProfile.FdDict:
                self.FdReportList.append(FdReport(Wa.FdfProfile.FdDict[Fd], Wa))
        self.PredictionReport = None
        if 'FIXED_ADDRESS' in ReportType or 'EXECUTION_ORDER' in ReportType:
            self.PredictionReport = PredictionReport(Wa)
        self.DepexParser = None
        if 'DEPEX' in ReportType:
            self.DepexParser = DepexParser(Wa)
        self.ModuleReportList = []
        if MaList is not None:
            self._IsModuleBuild = True
            for Ma in MaList:
                self.ModuleReportList.append(ModuleReport(Ma, ReportType))
        else:
            self._IsModuleBuild = False
            for Pa in Wa.AutoGenObjectList:
                ModuleAutoGenList = []
                for ModuleKey in Pa.Platform.Modules:
                    ModuleAutoGenList.append(Pa.Platform.Modules[ModuleKey].M)
                if GlobalData.gFdfParser is not None:
                    if Pa.Arch in GlobalData.gFdfParser.Profile.InfDict:
                        INFList = GlobalData.gFdfParser.Profile.InfDict[Pa.Arch]
                        for InfName in INFList:
                            InfClass = PathClass(NormPath(InfName), Wa.WorkspaceDir, Pa.Arch)
                            Ma = ModuleAutoGen(Wa, InfClass, Pa.BuildTarget, Pa.ToolChain, Pa.Arch, Wa.MetaFile, Pa.DataPipe)
                            if Ma is None:
                                continue
                            if Ma not in ModuleAutoGenList:
                                ModuleAutoGenList.append(Ma)
                for MGen in ModuleAutoGenList:
                    self.ModuleReportList.append(ModuleReport(MGen, ReportType))

    def GenerateReport(self, File, BuildDuration, AutoGenTime, MakeTime, GenFdsTime, ReportType):
        if False:
            i = 10
            return i + 15
        FileWrite(File, 'Platform Summary')
        FileWrite(File, 'Platform Name:        %s' % self.PlatformName)
        FileWrite(File, 'Platform DSC Path:    %s' % self.PlatformDscPath)
        FileWrite(File, 'Architectures:        %s' % self.Architectures)
        FileWrite(File, 'Tool Chain:           %s' % self.ToolChain)
        FileWrite(File, 'Target:               %s' % self.Target)
        if GlobalData.gSkuids:
            FileWrite(File, 'SKUID:                %s' % ' '.join(GlobalData.gSkuids))
        if GlobalData.gDefaultStores:
            FileWrite(File, 'DefaultStore:         %s' % ' '.join(GlobalData.gDefaultStores))
        FileWrite(File, 'Output Path:          %s' % self.OutputPath)
        FileWrite(File, 'Build Environment:    %s' % self.BuildEnvironment)
        FileWrite(File, 'Build Duration:       %s' % BuildDuration)
        if AutoGenTime:
            FileWrite(File, 'AutoGen Duration:     %s' % AutoGenTime)
        if MakeTime:
            FileWrite(File, 'Make Duration:        %s' % MakeTime)
        if GenFdsTime:
            FileWrite(File, 'GenFds Duration:      %s' % GenFdsTime)
        FileWrite(File, 'Report Content:       %s' % ', '.join(ReportType))
        if GlobalData.MixedPcd:
            FileWrite(File, gSectionStart)
            FileWrite(File, 'The following PCDs use different access methods:')
            FileWrite(File, gSectionSep)
            for PcdItem in GlobalData.MixedPcd:
                FileWrite(File, '%s.%s' % (str(PcdItem[1]), str(PcdItem[0])))
            FileWrite(File, gSectionEnd)
        if not self._IsModuleBuild:
            if 'PCD' in ReportType:
                self.PcdReport.GenerateReport(File, None)
            if 'FLASH' in ReportType:
                for FdReportListItem in self.FdReportList:
                    FdReportListItem.GenerateReport(File)
        for ModuleReportItem in self.ModuleReportList:
            ModuleReportItem.GenerateReport(File, self.PcdReport, self.PredictionReport, self.DepexParser, ReportType)
        if not self._IsModuleBuild:
            if 'EXECUTION_ORDER' in ReportType:
                self.PredictionReport.GenerateReport(File, None)

class BuildReport(object):

    def __init__(self, ReportFile, ReportType):
        if False:
            while True:
                i = 10
        self.ReportFile = ReportFile
        if ReportFile:
            self.ReportList = []
            self.ReportType = []
            if ReportType:
                for ReportTypeItem in ReportType:
                    if ReportTypeItem not in self.ReportType:
                        self.ReportType.append(ReportTypeItem)
            else:
                self.ReportType = ['PCD', 'LIBRARY', 'BUILD_FLAGS', 'DEPEX', 'HASH', 'FLASH', 'FIXED_ADDRESS']

    def AddPlatformReport(self, Wa, MaList=None):
        if False:
            for i in range(10):
                print('nop')
        if self.ReportFile:
            self.ReportList.append((Wa, MaList))

    def GenerateReport(self, BuildDuration, AutoGenTime, MakeTime, GenFdsTime):
        if False:
            return 10
        if self.ReportFile:
            try:
                if 'COMPILE_INFO' in self.ReportType:
                    self.GenerateCompileInfo()
                File = []
                for (Wa, MaList) in self.ReportList:
                    PlatformReport(Wa, MaList, self.ReportType).GenerateReport(File, BuildDuration, AutoGenTime, MakeTime, GenFdsTime, self.ReportType)
                Content = FileLinesSplit(''.join(File), gLineMaxLength)
                SaveFileOnChange(self.ReportFile, Content, False)
                EdkLogger.quiet('Build report can be found at %s' % os.path.abspath(self.ReportFile))
            except IOError:
                EdkLogger.error(None, FILE_WRITE_FAILURE, ExtraData=self.ReportFile)
            except:
                EdkLogger.error('BuildReport', CODE_ERROR, 'Unknown fatal error when generating build report', ExtraData=self.ReportFile, RaiseError=False)
                EdkLogger.quiet('(Python %s on %s\n%s)' % (platform.python_version(), sys.platform, traceback.format_exc()))

    def GenerateCompileInfo(self):
        if False:
            i = 10
            return i + 15
        try:
            compile_commands = []
            used_files = set()
            module_report = []
            for (Wa, MaList) in self.ReportList:
                for file_path in Wa._GetMetaFiles(Wa.BuildTarget, Wa.ToolChain):
                    used_files.add(file_path)
                for autoGen in Wa.AutoGenObjectList:
                    for module in autoGen.LibraryAutoGenList + autoGen.ModuleAutoGenList:
                        used_files.add(module.MetaFile.Path)
                        module_report_data = {}
                        module_report_data['Name'] = module.Name
                        module_report_data['Arch'] = module.Arch
                        module_report_data['Path'] = module.MetaFile.Path
                        module_report_data['Guid'] = module.Guid
                        module_report_data['BuildType'] = module.BuildType
                        module_report_data['IsLibrary'] = module.IsLibrary
                        module_report_data['SourceDir'] = module.SourceDir
                        module_report_data['Files'] = []
                        module_report_data['LibraryClass'] = module.Module.LibraryClass
                        module_report_data['ModuleEntryPointList'] = module.Module.ModuleEntryPointList
                        module_report_data['ConstructorList'] = module.Module.ConstructorList
                        module_report_data['DestructorList'] = module.Module.DestructorList
                        for data_file in module.SourceFileList:
                            module_report_data['Files'].append({'Name': data_file.Name, 'Path': data_file.Path})
                        module_report_data['Libraries'] = []
                        for data_library in module.LibraryAutoGenList:
                            module_report_data['Libraries'].append({'Path': data_library.MetaFile.Path})
                        module_report_data['Packages'] = []
                        for data_package in module.PackageList:
                            module_report_data['Packages'].append({'Path': data_package.MetaFile.Path, 'Includes': []})
                            for data_package_include in data_package.Includes:
                                module_report_data['Packages'][-1]['Includes'].append(data_package_include.Path)
                        module_report_data['PPI'] = []
                        for data_ppi in module.PpiList.keys():
                            module_report_data['PPI'].append({'Name': data_ppi, 'Guid': module.PpiList[data_ppi]})
                        module_report_data['Protocol'] = []
                        for data_protocol in module.ProtocolList.keys():
                            module_report_data['Protocol'].append({'Name': data_protocol, 'Guid': module.ProtocolList[data_protocol]})
                        module_report_data['Pcd'] = []
                        for data_pcd in module.LibraryPcdList:
                            module_report_data['Pcd'].append({'Space': data_pcd.TokenSpaceGuidCName, 'Name': data_pcd.TokenCName, 'Value': data_pcd.TokenValue, 'Guid': data_pcd.TokenSpaceGuidValue, 'DatumType': data_pcd.DatumType, 'Type': data_pcd.Type, 'DefaultValue': data_pcd.DefaultValue})
                        module_report.append(module_report_data)
                        includes_autogen = IncludesAutoGen(module.MakeFileDir, module)
                        for dep in includes_autogen.DepsCollection:
                            used_files.add(dep)
                        inc_flag = '-I'
                        if module.BuildRuleFamily == TAB_COMPILER_MSFT:
                            inc_flag = '/I'
                        for source in module.SourceFileList:
                            used_files.add(source.Path)
                            compile_command = {}
                            if source.Ext in ['.c', '.cc', '.cpp']:
                                compile_command['file'] = source.Path
                                compile_command['directory'] = source.Dir
                                build_command = module.BuildRules[source.Ext].CommandList[0]
                                build_command_variables = re.findall('\\$\\((.*?)\\)', build_command)
                                for var in build_command_variables:
                                    var_tokens = var.split('_')
                                    var_main = var_tokens[0]
                                    if len(var_tokens) == 1:
                                        var_value = module.BuildOption[var_main]['PATH']
                                    else:
                                        var_value = module.BuildOption[var_main][var_tokens[1]]
                                    build_command = build_command.replace(f'$({var})', var_value)
                                    include_files = f' {inc_flag}'.join(module.IncludePathList)
                                    build_command = build_command.replace('${src}', include_files)
                                    build_command = build_command.replace('${dst}', module.OutputDir)
                                compile_command['command'] = re.sub('\\$\\(.*?\\)', '', build_command)
                                compile_commands.append(compile_command)
                compile_info_folder = Path(Wa.BuildDir).joinpath('CompileInfo')
                compile_info_folder.mkdir(exist_ok=True)
                compile_commands.sort(key=lambda x: x['file'])
                SaveFileOnChange(compile_info_folder.joinpath(f'compile_commands.json'), json.dumps(compile_commands, indent=2), False)
                SaveFileOnChange(compile_info_folder.joinpath(f'cscope.files'), '\n'.join(sorted(used_files)), False)
                module_report.sort(key=lambda x: x['Path'])
                SaveFileOnChange(compile_info_folder.joinpath(f'module_report.json'), json.dumps(module_report, indent=2), False)
        except:
            EdkLogger.error('BuildReport', CODE_ERROR, 'Unknown fatal error when generating build report compile information', ExtraData=self.ReportFile, RaiseError=False)
            EdkLogger.quiet('(Python %s on %s\n%s)' % (platform.python_version(), sys.platform, traceback.format_exc()))
if __name__ == '__main__':
    pass