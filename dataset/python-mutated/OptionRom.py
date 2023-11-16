from __future__ import absolute_import
import Common.LongFilePathOs as os
import subprocess
from . import OptRomInfStatement
from .GenFdsGlobalVariable import GenFdsGlobalVariable
from CommonDataClass.FdfClass import OptionRomClassObject
from Common.Misc import SaveFileOnChange
from Common import EdkLogger
from Common.BuildToolError import *

class OPTIONROM(OptionRomClassObject):

    def __init__(self, Name=''):
        if False:
            print('Hello World!')
        OptionRomClassObject.__init__(self)
        self.DriverName = Name

    def AddToBuffer(self, Buffer, Flag=False):
        if False:
            i = 10
            return i + 15
        if not Flag:
            GenFdsGlobalVariable.InfLogger('\nGenerating %s Option ROM ...' % self.DriverName)
        EfiFileList = []
        BinFileList = []
        for FfsFile in self.FfsList:
            if isinstance(FfsFile, OptRomInfStatement.OptRomInfStatement):
                FilePathNameList = FfsFile.GenFfs(IsMakefile=Flag)
                if len(FilePathNameList) == 0:
                    EdkLogger.error('GenFds', GENFDS_ERROR, 'Module %s not produce .efi files, so NO file could be put into option ROM.' % FfsFile.InfFileName)
                if FfsFile.OverrideAttribs is None:
                    EfiFileList.extend(FilePathNameList)
                else:
                    FileName = os.path.basename(FilePathNameList[0])
                    TmpOutputDir = os.path.join(GenFdsGlobalVariable.FvDir, self.DriverName, FfsFile.CurrentArch)
                    if not os.path.exists(TmpOutputDir):
                        os.makedirs(TmpOutputDir)
                    TmpOutputFile = os.path.join(TmpOutputDir, FileName + '.tmp')
                    GenFdsGlobalVariable.GenerateOptionRom(TmpOutputFile, FilePathNameList, [], FfsFile.OverrideAttribs.NeedCompress, FfsFile.OverrideAttribs.PciClassCode, FfsFile.OverrideAttribs.PciRevision, FfsFile.OverrideAttribs.PciDeviceId, FfsFile.OverrideAttribs.PciVendorId, IsMakefile=Flag)
                    BinFileList.append(TmpOutputFile)
            else:
                FilePathName = FfsFile.GenFfs(IsMakefile=Flag)
                if FfsFile.OverrideAttribs is not None:
                    FileName = os.path.basename(FilePathName)
                    TmpOutputDir = os.path.join(GenFdsGlobalVariable.FvDir, self.DriverName, FfsFile.CurrentArch)
                    if not os.path.exists(TmpOutputDir):
                        os.makedirs(TmpOutputDir)
                    TmpOutputFile = os.path.join(TmpOutputDir, FileName + '.tmp')
                    GenFdsGlobalVariable.GenerateOptionRom(TmpOutputFile, [FilePathName], [], FfsFile.OverrideAttribs.NeedCompress, FfsFile.OverrideAttribs.PciClassCode, FfsFile.OverrideAttribs.PciRevision, FfsFile.OverrideAttribs.PciDeviceId, FfsFile.OverrideAttribs.PciVendorId, IsMakefile=Flag)
                    BinFileList.append(TmpOutputFile)
                elif FfsFile.FileType == 'EFI':
                    EfiFileList.append(FilePathName)
                else:
                    BinFileList.append(FilePathName)
        OutputFile = os.path.join(GenFdsGlobalVariable.FvDir, self.DriverName)
        OutputFile = OutputFile + '.rom'
        GenFdsGlobalVariable.GenerateOptionRom(OutputFile, EfiFileList, BinFileList, IsMakefile=Flag)
        if not Flag:
            GenFdsGlobalVariable.InfLogger('\nGenerate %s Option ROM Successfully' % self.DriverName)
        GenFdsGlobalVariable.SharpCounter = 0
        return OutputFile

class OverrideAttribs:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.PciVendorId = None
        self.PciClassCode = None
        self.PciDeviceId = None
        self.PciRevision = None
        self.NeedCompress = None