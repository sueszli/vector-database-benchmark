from __future__ import absolute_import
from .Ffs import SectionSuffix
import Common.LongFilePathOs as os
from .GenFdsGlobalVariable import GenFdsGlobalVariable
from CommonDataClass.FdfClass import VerSectionClassObject
from Common.LongFilePathSupport import OpenLongFilePath as open
from Common.DataType import SUP_MODULE_SEC

class VerSection(VerSectionClassObject):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        VerSectionClassObject.__init__(self)

    def GenSection(self, OutputPath, ModuleName, SecNum, KeyStringList, FfsInf=None, Dict=None, IsMakefile=False):
        if False:
            while True:
                i = 10
        if FfsInf:
            self.Alignment = FfsInf.__ExtendMacro__(self.Alignment)
            self.BuildNum = FfsInf.__ExtendMacro__(self.BuildNum)
            self.StringData = FfsInf.__ExtendMacro__(self.StringData)
            self.FileName = FfsInf.__ExtendMacro__(self.FileName)
        OutputFile = os.path.join(OutputPath, ModuleName + SUP_MODULE_SEC + SecNum + SectionSuffix.get('VERSION'))
        OutputFile = os.path.normpath(OutputFile)
        StringData = ''
        if self.StringData:
            StringData = self.StringData
        elif self.FileName:
            if Dict is None:
                Dict = {}
            FileNameStr = GenFdsGlobalVariable.ReplaceWorkspaceMacro(self.FileName)
            FileNameStr = GenFdsGlobalVariable.MacroExtend(FileNameStr, Dict)
            FileObj = open(FileNameStr, 'r')
            StringData = FileObj.read()
            StringData = '"' + StringData + '"'
            FileObj.close()
        GenFdsGlobalVariable.GenerateSection(OutputFile, [], 'EFI_SECTION_VERSION', Ver=StringData, BuildNumber=self.BuildNum, IsMakefile=IsMakefile)
        OutputFileList = []
        OutputFileList.append(OutputFile)
        return (OutputFileList, self.Alignment)