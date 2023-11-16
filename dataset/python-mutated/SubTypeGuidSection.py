from __future__ import absolute_import
from . import Section
import subprocess
from .Ffs import SectionSuffix
import Common.LongFilePathOs as os
from .GenFdsGlobalVariable import GenFdsGlobalVariable
from .GenFdsGlobalVariable import FindExtendTool
from CommonDataClass.FdfClass import SubTypeGuidSectionClassObject
import sys
from Common import EdkLogger
from Common.BuildToolError import *
from .FvImageSection import FvImageSection
from Common.LongFilePathSupport import OpenLongFilePath as open
from Common.DataType import *

class SubTypeGuidSection(SubTypeGuidSectionClassObject):

    def __init__(self):
        if False:
            return 10
        SubTypeGuidSectionClassObject.__init__(self)

    def GenSection(self, OutputPath, ModuleName, SecNum, KeyStringList, FfsInf=None, Dict=None, IsMakefile=False):
        if False:
            while True:
                i = 10
        self.KeyStringList = KeyStringList
        self.CurrentArchList = GenFdsGlobalVariable.ArchList
        if FfsInf is not None:
            self.Alignment = FfsInf.__ExtendMacro__(self.Alignment)
            self.SubTypeGuid = FfsInf.__ExtendMacro__(self.SubTypeGuid)
            self.SectionType = FfsInf.__ExtendMacro__(self.SectionType)
            self.CurrentArchList = [FfsInf.CurrentArch]
        if Dict is None:
            Dict = {}
        self.SectFileName = GenFdsGlobalVariable.ReplaceWorkspaceMacro(self.SectFileName)
        self.SectFileName = GenFdsGlobalVariable.MacroExtend(self.SectFileName, Dict)
        OutputFile = os.path.join(OutputPath, ModuleName + SUP_MODULE_SEC + SecNum + SectionSuffix.get('SUBTYPE_GUID'))
        GenFdsGlobalVariable.GenerateSection(OutputFile, [self.SectFileName], 'EFI_SECTION_FREEFORM_SUBTYPE_GUID', Guid=self.SubTypeGuid, IsMakefile=IsMakefile)
        OutputFileList = []
        OutputFileList.append(OutputFile)
        return (OutputFileList, self.Alignment)