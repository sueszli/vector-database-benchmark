"""
InfBinarySectionParser
"""
import Logger.Log as Logger
from Logger import StringTable as ST
from Logger.ToolError import FORMAT_INVALID
from Parser.InfParserMisc import InfExpandMacro
from Library import DataType as DT
from Library.Parsing import MacroParser
from Library.Misc import GetSplitValueList
from Object.Parser.InfCommonObject import InfLineCommentObject
from Object.Parser.InfCommonObject import CurrentLine
from Parser.InfParserMisc import InfParserSectionRoot

class InfBinarySectionParser(InfParserSectionRoot):

    def InfBinaryParser(self, SectionString, InfSectionObject, FileName):
        if False:
            while True:
                i = 10
        SectionMacros = {}
        ValueList = []
        UiBinaryList = []
        VerBinaryList = []
        ComBinaryList = []
        StillCommentFalg = False
        HeaderComments = []
        LineComment = None
        AllSectionContent = ''
        for Line in SectionString:
            BinLineContent = Line[0]
            BinLineNo = Line[1]
            if BinLineContent.strip() == '':
                continue
            CurrentLineObj = CurrentLine()
            CurrentLineObj.FileName = FileName
            CurrentLineObj.LineString = BinLineContent
            CurrentLineObj.LineNo = BinLineNo
            if BinLineContent.strip().startswith(DT.TAB_COMMENT_SPLIT):
                if StillCommentFalg:
                    HeaderComments.append(Line)
                    AllSectionContent += BinLineContent + DT.END_OF_LINE
                    continue
                else:
                    HeaderComments = []
                    HeaderComments.append(Line)
                    AllSectionContent += BinLineContent + DT.END_OF_LINE
                    StillCommentFalg = True
                    continue
            else:
                StillCommentFalg = False
            if len(HeaderComments) >= 1:
                LineComment = InfLineCommentObject()
                LineCommentContent = ''
                for Item in HeaderComments:
                    LineCommentContent += Item[0] + DT.END_OF_LINE
                LineComment.SetHeaderComments(LineCommentContent)
            if BinLineContent.find(DT.TAB_COMMENT_SPLIT) > -1:
                TailComments = BinLineContent[BinLineContent.find(DT.TAB_COMMENT_SPLIT):]
                BinLineContent = BinLineContent[:BinLineContent.find(DT.TAB_COMMENT_SPLIT)]
                if LineComment is None:
                    LineComment = InfLineCommentObject()
                LineComment.SetTailComments(TailComments)
            MacroDef = MacroParser((BinLineContent, BinLineNo), FileName, DT.MODEL_EFI_BINARY_FILE, self.FileLocalMacros)
            if MacroDef[0] is not None:
                SectionMacros[MacroDef[0]] = MacroDef[1]
                LineComment = None
                HeaderComments = []
                continue
            LineContent = InfExpandMacro(BinLineContent, (FileName, BinLineContent, BinLineNo), self.FileLocalMacros, SectionMacros, True)
            AllSectionContent += LineContent + DT.END_OF_LINE
            TokenList = GetSplitValueList(LineContent, DT.TAB_VALUE_SPLIT, 1)
            ValueList[0:len(TokenList)] = TokenList
            ValueList[0] = ValueList[0].strip()
            if ValueList[0] == DT.BINARY_FILE_TYPE_UNI_UI or ValueList[0] == DT.BINARY_FILE_TYPE_SEC_UI or ValueList[0] == DT.BINARY_FILE_TYPE_UI:
                if len(ValueList) == 2:
                    TokenList = GetSplitValueList(ValueList[1], DT.TAB_VALUE_SPLIT, 2)
                    NewValueList = []
                    NewValueList.append(ValueList[0])
                    for Item in TokenList:
                        NewValueList.append(Item)
                    UiBinaryList.append((NewValueList, LineComment, CurrentLineObj))
            elif ValueList[0] == DT.BINARY_FILE_TYPE_UNI_VER or ValueList[0] == DT.BINARY_FILE_TYPE_SEC_VER or ValueList[0] == DT.BINARY_FILE_TYPE_VER:
                if len(ValueList) == 2:
                    TokenList = GetSplitValueList(ValueList[1], DT.TAB_VALUE_SPLIT, 2)
                    NewValueList = []
                    NewValueList.append(ValueList[0])
                    for Item in TokenList:
                        NewValueList.append(Item)
                    VerBinaryList.append((NewValueList, LineComment, CurrentLineObj))
            elif len(ValueList) == 2:
                if ValueList[0].strip() == 'SUBTYPE_GUID':
                    TokenList = GetSplitValueList(ValueList[1], DT.TAB_VALUE_SPLIT, 5)
                else:
                    TokenList = GetSplitValueList(ValueList[1], DT.TAB_VALUE_SPLIT, 4)
                NewValueList = []
                NewValueList.append(ValueList[0])
                for Item in TokenList:
                    NewValueList.append(Item)
                ComBinaryList.append((NewValueList, LineComment, CurrentLineObj))
            elif len(ValueList) == 1:
                NewValueList = []
                NewValueList.append(ValueList[0])
                ComBinaryList.append((NewValueList, LineComment, CurrentLineObj))
            ValueList = []
            LineComment = None
            TailComments = ''
            HeaderComments = []
            continue
        ArchList = []
        for Item in self.LastSectionHeaderContent:
            if Item[1] not in ArchList:
                ArchList.append(Item[1])
                InfSectionObject.SetSupArchList(Item[1])
        InfSectionObject.SetAllContent(AllSectionContent)
        if not InfSectionObject.SetBinary(UiBinaryList, VerBinaryList, ComBinaryList, ArchList):
            Logger.Error('InfParser', FORMAT_INVALID, ST.ERR_INF_PARSER_MODULE_SECTION_TYPE_ERROR % '[Binaries]', File=FileName, Line=Item[3])