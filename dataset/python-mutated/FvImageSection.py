from __future__ import absolute_import
from . import Section
from io import BytesIO
from .Ffs import SectionSuffix
import subprocess
from .GenFdsGlobalVariable import GenFdsGlobalVariable
import Common.LongFilePathOs as os
from CommonDataClass.FdfClass import FvImageSectionClassObject
from Common.MultipleWorkspace import MultipleWorkspace as mws
from Common import EdkLogger
from Common.BuildToolError import *
from Common.DataType import *

class FvImageSection(FvImageSectionClassObject):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        FvImageSectionClassObject.__init__(self)

    def GenSection(self, OutputPath, ModuleName, SecNum, KeyStringList, FfsInf=None, Dict=None, IsMakefile=False):
        if False:
            return 10
        OutputFileList = []
        if Dict is None:
            Dict = {}
        if self.FvFileType is not None:
            (FileList, IsSect) = Section.Section.GetFileList(FfsInf, self.FvFileType, self.FvFileExtension)
            if IsSect:
                return (FileList, self.Alignment)
            Num = SecNum
            MaxFvAlignment = 0
            for FvFileName in FileList:
                FvAlignmentValue = 0
                if os.path.isfile(FvFileName):
                    FvFileObj = open(FvFileName, 'rb')
                    FvFileObj.seek(0)
                    FvHeaderBuffer = FvFileObj.read(72)
                    if isinstance(FvHeaderBuffer[46], str):
                        FvAlignmentValue = 1 << (ord(FvHeaderBuffer[46]) & 31)
                    else:
                        FvAlignmentValue = 1 << (FvHeaderBuffer[46] & 31)
                    FvFileObj.close()
                if FvAlignmentValue > MaxFvAlignment:
                    MaxFvAlignment = FvAlignmentValue
                OutputFile = os.path.join(OutputPath, ModuleName + SUP_MODULE_SEC + Num + SectionSuffix.get('FV_IMAGE'))
                GenFdsGlobalVariable.GenerateSection(OutputFile, [FvFileName], 'EFI_SECTION_FIRMWARE_VOLUME_IMAGE', IsMakefile=IsMakefile)
                OutputFileList.append(OutputFile)
            if MaxFvAlignment >= 1024:
                if MaxFvAlignment >= 1048576:
                    if MaxFvAlignment >= 16777216:
                        self.Alignment = '16M'
                    else:
                        self.Alignment = str(MaxFvAlignment // 1048576) + 'M'
                else:
                    self.Alignment = str(MaxFvAlignment // 1024) + 'K'
            else:
                self.Alignment = str(MaxFvAlignment)
            return (OutputFileList, self.Alignment)
        if self.FvName is not None:
            Buffer = BytesIO()
            Fv = GenFdsGlobalVariable.FdfParser.Profile.FvDict.get(self.FvName)
            if Fv is not None:
                self.Fv = Fv
                if not self.FvAddr and self.Fv.BaseAddress:
                    self.FvAddr = self.Fv.BaseAddress
                FvFileName = Fv.AddToBuffer(Buffer, self.FvAddr, MacroDict=Dict, Flag=IsMakefile)
                if Fv.FvAlignment is not None:
                    if self.Alignment is None:
                        self.Alignment = Fv.FvAlignment
                    elif GenFdsGlobalVariable.GetAlignment(Fv.FvAlignment) > GenFdsGlobalVariable.GetAlignment(self.Alignment):
                        self.Alignment = Fv.FvAlignment
            elif self.FvFileName is not None:
                FvFileName = GenFdsGlobalVariable.ReplaceWorkspaceMacro(self.FvFileName)
                if os.path.isfile(FvFileName):
                    FvFileObj = open(FvFileName, 'rb')
                    FvFileObj.seek(0)
                    FvHeaderBuffer = FvFileObj.read(72)
                    if isinstance(FvHeaderBuffer[46], str):
                        FvAlignmentValue = 1 << (ord(FvHeaderBuffer[46]) & 31)
                    else:
                        FvAlignmentValue = 1 << (FvHeaderBuffer[46] & 31)
                    if FvAlignmentValue >= 1024:
                        if FvAlignmentValue >= 1048576:
                            if FvAlignmentValue >= 16777216:
                                self.Alignment = '16M'
                            else:
                                self.Alignment = str(FvAlignmentValue // 1048576) + 'M'
                        else:
                            self.Alignment = str(FvAlignmentValue // 1024) + 'K'
                    else:
                        self.Alignment = str(FvAlignmentValue)
                    FvFileObj.close()
                elif len(mws.getPkgPath()) == 0:
                    EdkLogger.error('GenFds', FILE_NOT_FOUND, '%s is not found in WORKSPACE: %s' % self.FvFileName, GenFdsGlobalVariable.WorkSpaceDir)
                else:
                    EdkLogger.error('GenFds', FILE_NOT_FOUND, '%s is not found in packages path:\n\t%s' % (self.FvFileName, '\n\t'.join(mws.getPkgPath())))
            else:
                EdkLogger.error('GenFds', GENFDS_ERROR, 'FvImageSection Failed! %s NOT found in FDF' % self.FvName)
            OutputFile = os.path.join(OutputPath, ModuleName + SUP_MODULE_SEC + SecNum + SectionSuffix.get('FV_IMAGE'))
            GenFdsGlobalVariable.GenerateSection(OutputFile, [FvFileName], 'EFI_SECTION_FIRMWARE_VOLUME_IMAGE', IsMakefile=IsMakefile)
            OutputFileList.append(OutputFile)
            return (OutputFileList, self.Alignment)