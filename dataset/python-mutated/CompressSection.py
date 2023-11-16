from __future__ import absolute_import
from .Ffs import SectionSuffix
from . import Section
import subprocess
import Common.LongFilePathOs as os
from .GenFdsGlobalVariable import GenFdsGlobalVariable
from CommonDataClass.FdfClass import CompressSectionClassObject
from Common.DataType import *

class CompressSection(CompressSectionClassObject):
    CompTypeDict = {'PI_STD': 'PI_STD', 'PI_NONE': 'PI_NONE'}

    def __init__(self):
        if False:
            i = 10
            return i + 15
        CompressSectionClassObject.__init__(self)

    def GenSection(self, OutputPath, ModuleName, SecNum, KeyStringList, FfsInf=None, Dict=None, IsMakefile=False):
        if False:
            for i in range(10):
                print('nop')
        if FfsInf is not None:
            self.CompType = FfsInf.__ExtendMacro__(self.CompType)
            self.Alignment = FfsInf.__ExtendMacro__(self.Alignment)
        SectFiles = tuple()
        SectAlign = []
        Index = 0
        MaxAlign = None
        if Dict is None:
            Dict = {}
        for Sect in self.SectionList:
            Index = Index + 1
            SecIndex = '%s.%d' % (SecNum, Index)
            (ReturnSectList, AlignValue) = Sect.GenSection(OutputPath, ModuleName, SecIndex, KeyStringList, FfsInf, Dict, IsMakefile=IsMakefile)
            if AlignValue is not None:
                if MaxAlign is None:
                    MaxAlign = AlignValue
                if GenFdsGlobalVariable.GetAlignment(AlignValue) > GenFdsGlobalVariable.GetAlignment(MaxAlign):
                    MaxAlign = AlignValue
            if ReturnSectList != []:
                if AlignValue is None:
                    AlignValue = '1'
                for FileData in ReturnSectList:
                    SectFiles += (FileData,)
                    SectAlign.append(AlignValue)
        OutputFile = OutputPath + os.sep + ModuleName + SUP_MODULE_SEC + SecNum + SectionSuffix['COMPRESS']
        OutputFile = os.path.normpath(OutputFile)
        DummyFile = OutputFile + '.dummy'
        GenFdsGlobalVariable.GenerateSection(DummyFile, SectFiles, InputAlign=SectAlign, IsMakefile=IsMakefile)
        GenFdsGlobalVariable.GenerateSection(OutputFile, [DummyFile], Section.Section.SectionType['COMPRESS'], CompressionType=self.CompTypeDict[self.CompType], IsMakefile=IsMakefile)
        OutputFileList = []
        OutputFileList.append(OutputFile)
        return (OutputFileList, self.Alignment)