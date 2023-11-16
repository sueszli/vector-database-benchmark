from __future__ import absolute_import
from . import Section
from .Ffs import SectionSuffix
import subprocess
import Common.LongFilePathOs as os
from .GenFdsGlobalVariable import GenFdsGlobalVariable
from CommonDataClass.FdfClass import UiSectionClassObject
from Common.LongFilePathSupport import OpenLongFilePath as open
from Common.DataType import *

class UiSection(UiSectionClassObject):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        UiSectionClassObject.__init__(self)

    def GenSection(self, OutputPath, ModuleName, SecNum, KeyStringList, FfsInf=None, Dict=None, IsMakefile=False):
        if False:
            for i in range(10):
                print('nop')
        if FfsInf is not None:
            self.Alignment = FfsInf.__ExtendMacro__(self.Alignment)
            self.StringData = FfsInf.__ExtendMacro__(self.StringData)
            self.FileName = FfsInf.__ExtendMacro__(self.FileName)
        OutputFile = os.path.join(OutputPath, ModuleName + SUP_MODULE_SEC + SecNum + SectionSuffix.get(BINARY_FILE_TYPE_UI))
        if self.StringData is not None:
            NameString = self.StringData
        elif self.FileName is not None:
            if Dict is None:
                Dict = {}
            FileNameStr = GenFdsGlobalVariable.ReplaceWorkspaceMacro(self.FileName)
            FileNameStr = GenFdsGlobalVariable.MacroExtend(FileNameStr, Dict)
            FileObj = open(FileNameStr, 'r')
            NameString = FileObj.read()
            FileObj.close()
        else:
            NameString = ''
        GenFdsGlobalVariable.GenerateSection(OutputFile, None, 'EFI_SECTION_USER_INTERFACE', Ui=NameString, IsMakefile=IsMakefile)
        OutputFileList = []
        OutputFileList.append(OutputFile)
        return (OutputFileList, self.Alignment)