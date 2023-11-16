"""
InfParser
"""
import re
import os
from copy import deepcopy
from Library.StringUtils import GetSplitValueList
from Library.StringUtils import ConvertSpecialChar
from Library.Misc import ProcessLineExtender
from Library.Misc import ProcessEdkComment
from Library.Parsing import NormPath
from Library.ParserValidate import IsValidInfMoudleTypeList
from Library.ParserValidate import IsValidArch
from Library import DataType as DT
from Library import GlobalData
import Logger.Log as Logger
from Logger import StringTable as ST
from Logger.ToolError import FORMAT_INVALID
from Logger.ToolError import FILE_READ_FAILURE
from Logger.ToolError import PARSER_ERROR
from Object.Parser.InfCommonObject import InfSectionCommonDef
from Parser.InfSectionParser import InfSectionParser
from Parser.InfParserMisc import gINF_SECTION_DEF
from Parser.InfParserMisc import IsBinaryInf

def OpenInfFile(Filename):
    if False:
        i = 10
        return i + 15
    FileLinesList = []
    try:
        FInputfile = open(Filename, 'r')
        try:
            FileLinesList = FInputfile.readlines()
        except BaseException:
            Logger.Error('InfParser', FILE_READ_FAILURE, ST.ERR_FILE_OPEN_FAILURE, File=Filename)
        finally:
            FInputfile.close()
    except BaseException:
        Logger.Error('InfParser', FILE_READ_FAILURE, ST.ERR_FILE_OPEN_FAILURE, File=Filename)
    return FileLinesList

class InfParser(InfSectionParser):

    def __init__(self, Filename=None, WorkspaceDir=None):
        if False:
            return 10
        InfSectionParser.__init__()
        self.WorkspaceDir = WorkspaceDir
        self.SupArchList = DT.ARCH_LIST
        self.EventList = []
        self.HobList = []
        self.BootModeList = []
        if Filename is not None:
            self.ParseInfFile(Filename)

    def ParseInfFile(self, Filename):
        if False:
            for i in range(10):
                print('nop')
        Filename = NormPath(Filename)
        (Path, Name) = os.path.split(Filename)
        self.FullPath = Filename
        self.RelaPath = Path
        self.FileName = Name
        GlobalData.gINF_MODULE_DIR = Path
        GlobalData.gINF_MODULE_NAME = self.FullPath
        GlobalData.gIS_BINARY_INF = False
        LineNo = 0
        CurrentSection = DT.MODEL_UNKNOWN
        SectionLines = []
        HeaderCommentStart = False
        HeaderCommentEnd = False
        HeaderStarLineNo = -1
        BinaryHeaderCommentStart = False
        BinaryHeaderCommentEnd = False
        BinaryHeaderStarLineNo = -1
        NewSectionStartFlag = False
        FirstSectionStartFlag = False
        CommentBlock = []
        self.EventList = []
        self.HobList = []
        self.BootModeList = []
        SectionType = ''
        FileLinesList = OpenInfFile(Filename)
        DefineSectionParsedFlag = False
        FileLinesList = ConvertSpecialChar(FileLinesList)
        FileLinesList = ProcessLineExtender(FileLinesList)
        OrigLines = [Line for Line in FileLinesList]
        (FileLinesList, EdkCommentStartPos) = ProcessEdkComment(FileLinesList)
        if IsBinaryInf(FileLinesList):
            GlobalData.gIS_BINARY_INF = True
        InfSectionCommonDefObj = None
        for Line in FileLinesList:
            LineNo = LineNo + 1
            Line = Line.strip()
            if LineNo < len(FileLinesList) - 1:
                NextLine = FileLinesList[LineNo].strip()
            if (Line == '' or not Line) and LineNo == len(FileLinesList):
                LastSectionFalg = True
            if Line.startswith(DT.TAB_SPECIAL_COMMENT) and Line.find(DT.TAB_HEADER_COMMENT) > -1 and (not HeaderCommentStart) and (not HeaderCommentEnd):
                CurrentSection = DT.MODEL_META_DATA_FILE_HEADER
                HeaderStarLineNo = LineNo
                SectionLines.append((Line, LineNo))
                HeaderCommentStart = True
                continue
            if (Line.startswith(DT.TAB_COMMENT_SPLIT) and CurrentSection == DT.MODEL_META_DATA_FILE_HEADER) and HeaderCommentStart and (not Line.startswith(DT.TAB_SPECIAL_COMMENT)) and (not HeaderCommentEnd) and (NextLine != ''):
                SectionLines.append((Line, LineNo))
                continue
            if (Line.startswith(DT.TAB_SPECIAL_COMMENT) or not Line.strip().startswith('#')) and HeaderCommentStart and (not HeaderCommentEnd):
                HeaderCommentEnd = True
                BinaryHeaderCommentStart = False
                BinaryHeaderCommentEnd = False
                HeaderCommentStart = False
                if Line.find(DT.TAB_BINARY_HEADER_COMMENT) > -1:
                    self.InfHeaderParser(SectionLines, self.InfHeader, self.FileName)
                    SectionLines = []
                else:
                    SectionLines.append((Line, LineNo))
                    self.InfHeaderParser(SectionLines, self.InfHeader, self.FileName)
                    SectionLines = []
                    continue
            if Line.startswith(DT.TAB_SPECIAL_COMMENT) and Line.find(DT.TAB_BINARY_HEADER_COMMENT) > -1 and (not BinaryHeaderCommentStart):
                SectionLines = []
                CurrentSection = DT.MODEL_META_DATA_FILE_HEADER
                BinaryHeaderStarLineNo = LineNo
                SectionLines.append((Line, LineNo))
                BinaryHeaderCommentStart = True
                HeaderCommentEnd = True
                continue
            if Line.startswith(DT.TAB_SPECIAL_COMMENT) and BinaryHeaderCommentStart and (not BinaryHeaderCommentEnd) and (Line.find(DT.TAB_BINARY_HEADER_COMMENT) > -1):
                Logger.Error('Parser', FORMAT_INVALID, ST.ERR_MULTIPLE_BINARYHEADER_EXIST, File=Filename)
            if (Line.startswith(DT.TAB_COMMENT_SPLIT) and CurrentSection == DT.MODEL_META_DATA_FILE_HEADER) and BinaryHeaderCommentStart and (not Line.startswith(DT.TAB_SPECIAL_COMMENT)) and (not BinaryHeaderCommentEnd) and (NextLine != ''):
                SectionLines.append((Line, LineNo))
                continue
            if (Line.startswith(DT.TAB_SPECIAL_COMMENT) or not Line.strip().startswith(DT.TAB_COMMENT_SPLIT)) and BinaryHeaderCommentStart and (not BinaryHeaderCommentEnd):
                SectionLines.append((Line, LineNo))
                BinaryHeaderCommentStart = False
                self.InfHeaderParser(SectionLines, self.InfBinaryHeader, self.FileName, True)
                SectionLines = []
                BinaryHeaderCommentEnd = True
                continue
            LastSectionFalg = False
            if LineNo == len(FileLinesList):
                LastSectionFalg = True
            if Line.startswith(DT.TAB_COMMENT_SPLIT) and (not Line.startswith(DT.TAB_SPECIAL_COMMENT)):
                SectionLines.append((Line, LineNo))
                if not LastSectionFalg:
                    continue
            if Line.startswith(DT.TAB_SECTION_START) and Line.find(DT.TAB_SECTION_END) > -1 or LastSectionFalg:
                HeaderCommentEnd = True
                BinaryHeaderCommentEnd = True
                if not LastSectionFalg:
                    HeaderContent = Line[1:Line.find(DT.TAB_SECTION_END)]
                    if HeaderContent.find(DT.TAB_COMMENT_SPLIT) != -1:
                        Logger.Error('InfParser', FORMAT_INVALID, ST.ERR_INF_PARSER_DEFINE_SECTION_HEADER_INVALID, File=self.FullPath, Line=LineNo, ExtraData=Line)
                    self.LastSectionHeaderContent = deepcopy(self.SectionHeaderContent)
                    TailComments = ''
                    CommentIndex = Line.find(DT.TAB_COMMENT_SPLIT)
                    if CommentIndex > -1:
                        TailComments = Line[CommentIndex:]
                        Line = Line[:CommentIndex]
                    InfSectionCommonDefObj = InfSectionCommonDef()
                    if TailComments != '':
                        InfSectionCommonDefObj.SetTailComments(TailComments)
                    if CommentBlock != '':
                        InfSectionCommonDefObj.SetHeaderComments(CommentBlock)
                        CommentBlock = []
                    if CurrentSection == DT.MODEL_META_DATA_DEFINE:
                        DefineSectionParsedFlag = self._CallSectionParsers(CurrentSection, DefineSectionParsedFlag, SectionLines, InfSectionCommonDefObj, LineNo)
                    self.SectionHeaderParser(Line, self.FileName, LineNo)
                    self._CheckSectionHeaders(Line, LineNo)
                    SectionType = _ConvertSecNameToType(self.SectionHeaderContent[0][0])
                if not FirstSectionStartFlag:
                    CurrentSection = SectionType
                    FirstSectionStartFlag = True
                else:
                    NewSectionStartFlag = True
            else:
                SectionLines.append((Line, LineNo))
                continue
            if LastSectionFalg:
                (SectionLines, CurrentSection) = self._ProcessLastSection(SectionLines, Line, LineNo, CurrentSection)
            if NewSectionStartFlag or LastSectionFalg:
                if CurrentSection != DT.MODEL_META_DATA_DEFINE or (LastSectionFalg and CurrentSection == DT.MODEL_META_DATA_DEFINE):
                    DefineSectionParsedFlag = self._CallSectionParsers(CurrentSection, DefineSectionParsedFlag, SectionLines, InfSectionCommonDefObj, LineNo)
                CurrentSection = SectionType
                SectionLines = []
        if HeaderStarLineNo == -1:
            Logger.Error('InfParser', FORMAT_INVALID, ST.ERR_NO_SOURCE_HEADER, File=self.FullPath)
        if BinaryHeaderStarLineNo > -1 and HeaderStarLineNo > -1 and (HeaderStarLineNo > BinaryHeaderStarLineNo):
            Logger.Error('InfParser', FORMAT_INVALID, ST.ERR_BINARY_HEADER_ORDER, File=self.FullPath)
        if EdkCommentStartPos != -1:
            Logger.Error('InfParser', FORMAT_INVALID, ST.ERR_INF_PARSER_EDKI_COMMENT_IN_EDKII, File=self.FullPath, Line=EdkCommentStartPos + 1, ExtraData=OrigLines[EdkCommentStartPos])
        self._ExtractEventHobBootMod(FileLinesList)

    def _CheckSectionHeaders(self, Line, LineNo):
        if False:
            while True:
                i = 10
        if len(self.SectionHeaderContent) == 0:
            Logger.Error('InfParser', FORMAT_INVALID, ST.ERR_INF_PARSER_DEFINE_SECTION_HEADER_INVALID, File=self.FullPath, Line=LineNo, ExtraData=Line)
        else:
            for SectionItem in self.SectionHeaderContent:
                ArchList = []
                if SectionItem[0].strip().upper() == DT.TAB_INF_FIXED_PCD.upper() or SectionItem[0].strip().upper() == DT.TAB_INF_PATCH_PCD.upper() or SectionItem[0].strip().upper() == DT.TAB_INF_PCD_EX.upper() or (SectionItem[0].strip().upper() == DT.TAB_INF_PCD.upper()) or (SectionItem[0].strip().upper() == DT.TAB_INF_FEATURE_PCD.upper()):
                    ArchList = GetSplitValueList(SectionItem[1].strip(), ' ')
                else:
                    ArchList = [SectionItem[1].strip()]
                for Arch in ArchList:
                    if not IsValidArch(Arch) and SectionItem[0].strip().upper() != DT.TAB_DEPEX.upper() and (SectionItem[0].strip().upper() != DT.TAB_USER_EXTENSIONS.upper()) and (SectionItem[0].strip().upper() != DT.TAB_COMMON_DEFINES.upper()):
                        Logger.Error('InfParser', FORMAT_INVALID, ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % SectionItem[1], File=self.FullPath, Line=LineNo, ExtraData=Line)
                ChkModSectionList = ['LIBRARYCLASSES']
                if self.SectionHeaderContent[0][0].upper() in ChkModSectionList:
                    if SectionItem[2].strip().upper():
                        MoudleTypeList = GetSplitValueList(SectionItem[2].strip().upper())
                        if not IsValidInfMoudleTypeList(MoudleTypeList):
                            Logger.Error('InfParser', FORMAT_INVALID, ST.ERR_INF_PARSER_DEFINE_FROMAT_INVALID % SectionItem[2], File=self.FullPath, Line=LineNo, ExtraData=Line)

    def _CallSectionParsers(self, CurrentSection, DefineSectionParsedFlag, SectionLines, InfSectionCommonDefObj, LineNo):
        if False:
            for i in range(10):
                print('nop')
        if CurrentSection == DT.MODEL_META_DATA_DEFINE:
            if not DefineSectionParsedFlag:
                self.InfDefineParser(SectionLines, self.InfDefSection, self.FullPath, InfSectionCommonDefObj)
                DefineSectionParsedFlag = True
            else:
                Logger.Error('Parser', PARSER_ERROR, ST.ERR_INF_PARSER_MULTI_DEFINE_SECTION, File=self.FullPath, RaiseError=Logger.IS_RAISE_ERROR)
        elif CurrentSection == DT.MODEL_META_DATA_BUILD_OPTION:
            self.InfBuildOptionParser(SectionLines, self.InfBuildOptionSection, self.FullPath)
        elif CurrentSection == DT.MODEL_EFI_LIBRARY_CLASS:
            self.InfLibraryParser(SectionLines, self.InfLibraryClassSection, self.FullPath)
        elif CurrentSection == DT.MODEL_META_DATA_PACKAGE:
            self.InfPackageParser(SectionLines, self.InfPackageSection, self.FullPath)
        elif CurrentSection == DT.MODEL_PCD_FIXED_AT_BUILD or CurrentSection == DT.MODEL_PCD_PATCHABLE_IN_MODULE or CurrentSection == DT.MODEL_PCD_FEATURE_FLAG or (CurrentSection == DT.MODEL_PCD_DYNAMIC_EX) or (CurrentSection == DT.MODEL_PCD_DYNAMIC):
            self.InfPcdParser(SectionLines, self.InfPcdSection, self.FullPath)
        elif CurrentSection == DT.MODEL_EFI_SOURCE_FILE:
            self.InfSourceParser(SectionLines, self.InfSourcesSection, self.FullPath)
        elif CurrentSection == DT.MODEL_META_DATA_USER_EXTENSION:
            self.InfUserExtensionParser(SectionLines, self.InfUserExtensionSection, self.FullPath)
        elif CurrentSection == DT.MODEL_EFI_PROTOCOL:
            self.InfProtocolParser(SectionLines, self.InfProtocolSection, self.FullPath)
        elif CurrentSection == DT.MODEL_EFI_PPI:
            self.InfPpiParser(SectionLines, self.InfPpiSection, self.FullPath)
        elif CurrentSection == DT.MODEL_EFI_GUID:
            self.InfGuidParser(SectionLines, self.InfGuidSection, self.FullPath)
        elif CurrentSection == DT.MODEL_EFI_DEPEX:
            self.InfDepexParser(SectionLines, self.InfDepexSection, self.FullPath)
        elif CurrentSection == DT.MODEL_EFI_BINARY_FILE:
            self.InfBinaryParser(SectionLines, self.InfBinariesSection, self.FullPath)
        elif len(self.SectionHeaderContent) >= 1:
            Logger.Error('Parser', PARSER_ERROR, ST.ERR_INF_PARSER_UNKNOWN_SECTION, File=self.FullPath, Line=LineNo, RaiseError=Logger.IS_RAISE_ERROR)
        else:
            Logger.Error('Parser', PARSER_ERROR, ST.ERR_INF_PARSER_NO_SECTION_ERROR, File=self.FullPath, Line=LineNo, RaiseError=Logger.IS_RAISE_ERROR)
        return DefineSectionParsedFlag

    def _ExtractEventHobBootMod(self, FileLinesList):
        if False:
            print('Hello World!')
        SpecialSectionStart = False
        CheckLocation = False
        GFindSpecialCommentRe = re.compile('#(?:\\s*)\\[(.*?)\\](?:.*)', re.DOTALL)
        GFindNewSectionRe2 = re.compile('#?(\\s*)\\[(.*?)\\](.*)', re.DOTALL)
        LineNum = 0
        Element = []
        for Line in FileLinesList:
            Line = Line.strip()
            LineNum += 1
            MatchObject = GFindSpecialCommentRe.search(Line)
            if MatchObject:
                SpecialSectionStart = True
                Element = []
                if MatchObject.group(1).upper().startswith('EVENT'):
                    List = self.EventList
                elif MatchObject.group(1).upper().startswith('HOB'):
                    List = self.HobList
                elif MatchObject.group(1).upper().startswith('BOOTMODE'):
                    List = self.BootModeList
                else:
                    SpecialSectionStart = False
                    CheckLocation = False
                if SpecialSectionStart:
                    Element.append([Line, LineNum])
                    List.append(Element)
            else:
                MatchObject = GFindNewSectionRe2.search(Line)
                if SpecialSectionStart:
                    if MatchObject:
                        SpecialSectionStart = False
                        CheckLocation = False
                        Element = []
                    elif not Line:
                        SpecialSectionStart = False
                        CheckLocation = True
                        Element = []
                    elif not Line.startswith(DT.TAB_COMMENT_SPLIT):
                        Logger.Warn('Parser', ST.WARN_SPECIAL_SECTION_LOCATION_WRONG, File=self.FullPath, Line=LineNum)
                        SpecialSectionStart = False
                        CheckLocation = False
                        Element = []
                    else:
                        Element.append([Line, LineNum])
                elif CheckLocation:
                    if MatchObject:
                        CheckLocation = False
                    elif Line:
                        Logger.Warn('Parser', ST.WARN_SPECIAL_SECTION_LOCATION_WRONG, File=self.FullPath, Line=LineNum)
                        CheckLocation = False
        if len(self.BootModeList) >= 1:
            self.InfSpecialCommentParser(self.BootModeList, self.InfSpecialCommentSection, self.FileName, DT.TYPE_BOOTMODE_SECTION)
        if len(self.EventList) >= 1:
            self.InfSpecialCommentParser(self.EventList, self.InfSpecialCommentSection, self.FileName, DT.TYPE_EVENT_SECTION)
        if len(self.HobList) >= 1:
            self.InfSpecialCommentParser(self.HobList, self.InfSpecialCommentSection, self.FileName, DT.TYPE_HOB_SECTION)

    def _ProcessLastSection(self, SectionLines, Line, LineNo, CurrentSection):
        if False:
            return 10
        if not (Line.startswith(DT.TAB_SECTION_START) and Line.find(DT.TAB_SECTION_END) > -1):
            SectionLines.append((Line, LineNo))
        if len(self.SectionHeaderContent) >= 1:
            TemSectionName = self.SectionHeaderContent[0][0].upper()
            if TemSectionName.upper() not in gINF_SECTION_DEF.keys():
                Logger.Error('InfParser', FORMAT_INVALID, ST.ERR_INF_PARSER_UNKNOWN_SECTION, File=self.FullPath, Line=LineNo, ExtraData=Line, RaiseError=Logger.IS_RAISE_ERROR)
            else:
                CurrentSection = gINF_SECTION_DEF[TemSectionName]
                self.LastSectionHeaderContent = self.SectionHeaderContent
        return (SectionLines, CurrentSection)

def _ConvertSecNameToType(SectionName):
    if False:
        i = 10
        return i + 15
    SectionType = ''
    if SectionName.upper() not in gINF_SECTION_DEF.keys():
        SectionType = DT.MODEL_UNKNOWN
    else:
        SectionType = gINF_SECTION_DEF[SectionName.upper()]
    return SectionType