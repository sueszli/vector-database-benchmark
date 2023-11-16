"""
InfPcdSectionParser
"""
import Logger.Log as Logger
from Logger import StringTable as ST
from Logger.ToolError import FORMAT_INVALID
from Parser.InfParserMisc import InfExpandMacro
from Library import DataType as DT
from Library.Parsing import MacroParser
from Library.Misc import GetSplitValueList
from Library import GlobalData
from Library.StringUtils import SplitPcdEntry
from Parser.InfParserMisc import InfParserSectionRoot

class InfPcdSectionParser(InfParserSectionRoot):

    def InfPcdParser(self, SectionString, InfSectionObject, FileName):
        if False:
            for i in range(10):
                print('nop')
        KeysList = []
        PcdList = []
        CommentsList = []
        ValueList = []
        LineIndex = -1
        for Item in self.LastSectionHeaderContent:
            if (Item[0], Item[1], Item[3]) not in KeysList:
                KeysList.append((Item[0], Item[1], Item[3]))
                LineIndex = Item[3]
            if (Item[0].upper() == DT.TAB_INF_FIXED_PCD.upper() or Item[0].upper() == DT.TAB_INF_FEATURE_PCD.upper() or Item[0].upper() == DT.TAB_INF_PCD.upper()) and GlobalData.gIS_BINARY_INF:
                Logger.Error('InfParser', FORMAT_INVALID, ST.ERR_ASBUILD_PCD_SECTION_TYPE % ('"' + Item[0] + '"'), File=FileName, Line=LineIndex)
        if not GlobalData.gIS_BINARY_INF:
            SectionMacros = {}
            for Line in SectionString:
                PcdLineContent = Line[0]
                PcdLineNo = Line[1]
                if PcdLineContent.strip() == '':
                    CommentsList = []
                    continue
                if PcdLineContent.strip().startswith(DT.TAB_COMMENT_SPLIT):
                    CommentsList.append(Line)
                    continue
                elif PcdLineContent.find(DT.TAB_COMMENT_SPLIT) > -1:
                    CommentsList.append((PcdLineContent[PcdLineContent.find(DT.TAB_COMMENT_SPLIT):], PcdLineNo))
                    PcdLineContent = PcdLineContent[:PcdLineContent.find(DT.TAB_COMMENT_SPLIT)]
                if PcdLineContent != '':
                    (Name, Value) = MacroParser((PcdLineContent, PcdLineNo), FileName, DT.MODEL_EFI_PCD, self.FileLocalMacros)
                    if Name is not None:
                        SectionMacros[Name] = Value
                        ValueList = []
                        CommentsList = []
                        continue
                    PcdEntryReturn = SplitPcdEntry(PcdLineContent)
                    if not PcdEntryReturn[1]:
                        TokenList = ['']
                    else:
                        TokenList = PcdEntryReturn[0]
                    ValueList[0:len(TokenList)] = TokenList
                    ValueList = [InfExpandMacro(Value, (FileName, PcdLineContent, PcdLineNo), self.FileLocalMacros, SectionMacros, True) for Value in ValueList]
                if len(ValueList) >= 1:
                    PcdList.append((ValueList, CommentsList, (PcdLineContent, PcdLineNo, FileName)))
                    ValueList = []
                    CommentsList = []
                continue
        else:
            for Line in SectionString:
                LineContent = Line[0].strip()
                LineNo = Line[1]
                if LineContent == '':
                    CommentsList = []
                    continue
                if LineContent.startswith(DT.TAB_COMMENT_SPLIT):
                    CommentsList.append(LineContent)
                    continue
                CommentIndex = LineContent.find(DT.TAB_COMMENT_SPLIT)
                if CommentIndex > -1:
                    CommentsList.append(LineContent[CommentIndex + 1:])
                    LineContent = LineContent[:CommentIndex]
                TokenList = GetSplitValueList(LineContent, DT.TAB_VALUE_SPLIT)
                if KeysList[0][0].upper() == DT.TAB_INF_PATCH_PCD.upper():
                    if len(TokenList) != 3:
                        Logger.Error('InfParser', FORMAT_INVALID, ST.ERR_ASBUILD_PATCHPCD_FORMAT_INVALID, File=FileName, Line=LineNo, ExtraData=LineContent)
                elif KeysList[0][0].upper() == DT.TAB_INF_PCD_EX.upper():
                    if len(TokenList) != 1:
                        Logger.Error('InfParser', FORMAT_INVALID, ST.ERR_ASBUILD_PCDEX_FORMAT_INVALID, File=FileName, Line=LineNo, ExtraData=LineContent)
                ValueList[0:len(TokenList)] = TokenList
                if len(ValueList) >= 1:
                    PcdList.append((ValueList, CommentsList, (LineContent, LineNo, FileName)))
                    ValueList = []
                    CommentsList = []
                continue
        if not InfSectionObject.SetPcds(PcdList, KeysList=KeysList, PackageInfo=self.InfPackageSection.GetPackages()):
            Logger.Error('InfParser', FORMAT_INVALID, ST.ERR_INF_PARSER_MODULE_SECTION_TYPE_ERROR % '[PCD]', File=FileName, Line=LineIndex)