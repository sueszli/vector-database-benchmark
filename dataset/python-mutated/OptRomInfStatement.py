from __future__ import absolute_import
from . import RuleSimpleFile
from . import RuleComplexFile
from . import Section
import Common.GlobalData as GlobalData
from Common.DataType import *
from Common.StringUtils import *
from .FfsInfStatement import FfsInfStatement
from .GenFdsGlobalVariable import GenFdsGlobalVariable

class OptRomInfStatement(FfsInfStatement):

    def __init__(self):
        if False:
            while True:
                i = 10
        FfsInfStatement.__init__(self)
        self.OverrideAttribs = None

    def __GetOptRomParams(self):
        if False:
            while True:
                i = 10
        if self.OverrideAttribs is None:
            self.OverrideAttribs = OverrideAttribs()
        if self.OverrideAttribs.NeedCompress is None:
            self.OverrideAttribs.NeedCompress = self.OptRomDefs.get('PCI_COMPRESS')
            if self.OverrideAttribs.NeedCompress is not None:
                if self.OverrideAttribs.NeedCompress.upper() not in ('TRUE', 'FALSE'):
                    GenFdsGlobalVariable.ErrorLogger('Expected TRUE/FALSE for PCI_COMPRESS: %s' % self.InfFileName)
                self.OverrideAttribs.NeedCompress = self.OverrideAttribs.NeedCompress.upper() == 'TRUE'
        if self.OverrideAttribs.PciVendorId is None:
            self.OverrideAttribs.PciVendorId = self.OptRomDefs.get('PCI_VENDOR_ID')
        if self.OverrideAttribs.PciClassCode is None:
            self.OverrideAttribs.PciClassCode = self.OptRomDefs.get('PCI_CLASS_CODE')
        if self.OverrideAttribs.PciDeviceId is None:
            self.OverrideAttribs.PciDeviceId = self.OptRomDefs.get('PCI_DEVICE_ID')
        if self.OverrideAttribs.PciRevision is None:
            self.OverrideAttribs.PciRevision = self.OptRomDefs.get('PCI_REVISION')

    def GenFfs(self, IsMakefile=False):
        if False:
            for i in range(10):
                print('nop')
        self.__InfParse__()
        self.__GetOptRomParams()
        Rule = self.__GetRule__()
        GenFdsGlobalVariable.VerboseLogger('Packing binaries from inf file : %s' % self.InfFileName)
        if isinstance(Rule, RuleSimpleFile.RuleSimpleFile):
            EfiOutputList = self.__GenSimpleFileSection__(Rule, IsMakefile=IsMakefile)
            return EfiOutputList
        elif isinstance(Rule, RuleComplexFile.RuleComplexFile):
            EfiOutputList = self.__GenComplexFileSection__(Rule, IsMakefile=IsMakefile)
            return EfiOutputList

    def __GenSimpleFileSection__(self, Rule, IsMakefile=False):
        if False:
            return 10
        OutputFileList = []
        if Rule.FileName is not None:
            GenSecInputFile = self.__ExtendMacro__(Rule.FileName)
            OutputFileList.append(GenSecInputFile)
        else:
            (OutputFileList, IsSect) = Section.Section.GetFileList(self, '', Rule.FileExtension)
        return OutputFileList

    def __GenComplexFileSection__(self, Rule, IsMakefile=False):
        if False:
            while True:
                i = 10
        OutputFileList = []
        for Sect in Rule.SectionList:
            if Sect.SectionType == BINARY_FILE_TYPE_PE32:
                if Sect.FileName is not None:
                    GenSecInputFile = self.__ExtendMacro__(Sect.FileName)
                    OutputFileList.append(GenSecInputFile)
                else:
                    (FileList, IsSect) = Section.Section.GetFileList(self, '', Sect.FileExtension)
                    OutputFileList.extend(FileList)
        return OutputFileList

class OverrideAttribs:

    def __init__(self):
        if False:
            return 10
        self.PciVendorId = None
        self.PciClassCode = None
        self.PciDeviceId = None
        self.PciRevision = None
        self.NeedCompress = None