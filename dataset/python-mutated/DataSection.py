from __future__ import absolute_import
from . import Section
from .GenFdsGlobalVariable import GenFdsGlobalVariable
import subprocess
from .Ffs import SectionSuffix
import Common.LongFilePathOs as os
from CommonDataClass.FdfClass import DataSectionClassObject
from Common.Misc import PeImageClass
from Common.LongFilePathSupport import CopyLongFilePath
from Common.DataType import *

class DataSection(DataSectionClassObject):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        DataSectionClassObject.__init__(self)

    def GenSection(self, OutputPath, ModuleName, SecNum, keyStringList, FfsFile=None, Dict=None, IsMakefile=False):
        if False:
            return 10
        if Dict is None:
            Dict = {}
        if FfsFile is not None:
            self.SectFileName = GenFdsGlobalVariable.ReplaceWorkspaceMacro(self.SectFileName)
            self.SectFileName = GenFdsGlobalVariable.MacroExtend(self.SectFileName, Dict, FfsFile.CurrentArch)
        else:
            self.SectFileName = GenFdsGlobalVariable.ReplaceWorkspaceMacro(self.SectFileName)
            self.SectFileName = GenFdsGlobalVariable.MacroExtend(self.SectFileName, Dict)
        'Check Section file exist or not !'
        if not os.path.exists(self.SectFileName):
            self.SectFileName = os.path.join(GenFdsGlobalVariable.WorkSpaceDir, self.SectFileName)
        'Copy Map file to Ffs output'
        Filename = GenFdsGlobalVariable.MacroExtend(self.SectFileName)
        if Filename[len(Filename) - 4:] == '.efi':
            MapFile = Filename.replace('.efi', '.map')
            CopyMapFile = os.path.join(OutputPath, ModuleName + '.map')
            if IsMakefile:
                if GenFdsGlobalVariable.CopyList == []:
                    GenFdsGlobalVariable.CopyList = [(MapFile, CopyMapFile)]
                else:
                    GenFdsGlobalVariable.CopyList.append((MapFile, CopyMapFile))
            elif os.path.exists(MapFile):
                if not os.path.exists(CopyMapFile) or os.path.getmtime(MapFile) > os.path.getmtime(CopyMapFile):
                    CopyLongFilePath(MapFile, CopyMapFile)
        if self.Alignment == 'Auto' and self.SecType in (BINARY_FILE_TYPE_TE, BINARY_FILE_TYPE_PE32):
            self.Alignment = '0'
        NoStrip = True
        if self.SecType in (BINARY_FILE_TYPE_TE, BINARY_FILE_TYPE_PE32):
            if self.KeepReloc is not None:
                NoStrip = self.KeepReloc
        if not NoStrip:
            FileBeforeStrip = os.path.join(OutputPath, ModuleName + '.efi')
            if not os.path.exists(FileBeforeStrip) or os.path.getmtime(self.SectFileName) > os.path.getmtime(FileBeforeStrip):
                CopyLongFilePath(self.SectFileName, FileBeforeStrip)
            StrippedFile = os.path.join(OutputPath, ModuleName + '.stripped')
            GenFdsGlobalVariable.GenerateFirmwareImage(StrippedFile, [GenFdsGlobalVariable.MacroExtend(self.SectFileName, Dict)], Strip=True, IsMakefile=IsMakefile)
            self.SectFileName = StrippedFile
        if self.SecType == BINARY_FILE_TYPE_TE:
            TeFile = os.path.join(OutputPath, ModuleName + 'Te.raw')
            GenFdsGlobalVariable.GenerateFirmwareImage(TeFile, [GenFdsGlobalVariable.MacroExtend(self.SectFileName, Dict)], Type='te', IsMakefile=IsMakefile)
            self.SectFileName = TeFile
        OutputFile = os.path.join(OutputPath, ModuleName + SUP_MODULE_SEC + SecNum + SectionSuffix.get(self.SecType))
        OutputFile = os.path.normpath(OutputFile)
        GenFdsGlobalVariable.GenerateSection(OutputFile, [self.SectFileName], Section.Section.SectionType.get(self.SecType), IsMakefile=IsMakefile)
        FileList = [OutputFile]
        return (FileList, self.Alignment)